"""Fit normal references, set thresholds, and evaluate cached observations."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import CACHE, DATA, RUNS, digest, file_hash, read_json, write_json
from .extract import VIEWS, cache_file, select_rows
from .index import CONDITIONS, ROOTS
from .scoring import Components, compare_objects, normal_bank, scores
from .report import report

METHODS = ["hard", "joint", "forced", "visual", "normal_cost", "free"]


def audit_groups(reference, evaluation, required):
    if required and any(not row["group"] for row in reference + evaluation):
        raise ValueError("Cross-condition evaluation needs verified instance/session IDs; index with --groups")
    source = {r["group"] for r in reference if r["group"]}
    overlap = {r["group"] for r in evaluation} & source
    if overlap:
        raise ValueError(f"Instance/session overlap between fit/calibration and test: {sorted(overlap)[:5]}")


def oracle_components(annotation, names):
    if (not annotation.get("reviewer", "").strip() or not annotation.get("violation", "").strip()
            or not isinstance(annotation.get("objects"), list)):
        raise ValueError("Oracle requires reviewer, violation and complete object list")
    # Extra oracle-only type precedes background; reference foreground IDs stay stable.
    fore = [*names[:-1], "__other__"]
    objects = []
    for obj in annotation["objects"]:
        label = fore.index(obj["name"])
        p = np.zeros(len(fore) + 1)
        p[label] = 1
        objects.append(dict(p=p.tolist(), box=obj["box"]))
    return Components.load(objects, len(fore) + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=ROOTS, required=True)
    parser.add_argument("--protocol", choices=["within", "cross"], default="within")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=["clean"])
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--penalty", type=float, default=3.)
    parser.add_argument("--spatial", type=float, default=1.)
    parser.add_argument("--quantile", type=float, default=.95)
    parser.add_argument("--name", help="Optional descriptive run directory name")
    parser.add_argument("--oracle", type=Path, help="Reviewed JSONL subset of test images")
    parser.add_argument("--clean", action="store_true")
    a = parser.parse_args()
    if not 0 < a.quantile < 1 or (a.protocol == "cross" and a.dataset != "vid"):
        parser.error("Invalid quantile or dataset/protocol combination")
    views = list(dict.fromkeys(["clean", *a.views]))
    rows = select_rows(read_json(DATA / "index" / f"{a.dataset}.json"), a.categories, a.limit)
    annotations = {}
    if a.oracle:
        for line in a.oracle.read_text(encoding="utf-8-sig").splitlines():
            item = json.loads(line)
            if item["id"] in annotations:
                raise ValueError(f"Duplicate oracle ID: {item['id']}")
            annotations[item["id"]] = item
        if not annotations or set(annotations) - {r["id"] for r in rows if r["split"] == "test"}:
            raise ValueError("Oracle IDs must belong to the selected test cohort")
    root = a.cache / a.dataset
    settings = read_json(root / "settings.json")
    sig = digest(settings)
    config = dict(dataset=a.dataset, protocol=a.protocol, categories=a.categories, views=views,
                  limit=a.limit, top_k=a.top_k, penalty=a.penalty, spatial=a.spatial,
                  quantile=a.quantile, frontend=settings, cohort=digest(rows), seed=42,
                  oracle_digest=digest(annotations), supervision="object-vocabulary + normal references",
                  code=[file_hash(Path(__file__).with_name(name)) for name in
                        ("experiment.py", "scoring.py", "report.py")])
    name = a.name or "_".join([
        "-".join(a.categories or ["all"]), "-".join(views), f"k{a.top_k}",
        f"cost{a.penalty:g}", f"position{a.spatial:g}", f"q{a.quantile:g}",
        f"n{a.limit}" if a.limit else "full", "reviewed" if annotations else "auto"])
    if not name or name in (".", "..") or Path(name).name != name:
        parser.error("--name must be a directory name")
    out = RUNS / a.dataset / a.protocol / name
    if (out / "settings.json").exists() and read_json(out / "settings.json") != config:
        raise ValueError(f"Inputs or code changed. Choose a new --name for this run: {out}")
    write_json(out / "settings.json", config)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    records, calibration_rows = [], []

    def load(row, view):
        item = read_json(cache_file(root, view, row))
        if item["signature"] != sig or item["sha256"] != row["sha256"]:
            raise ValueError(f"Candidate provenance mismatch: {row['id']}")
        return item, Components.load(item["objects"], len(item["names"]))

    def measure(row, view, bank, stage):
        dest = out / "partial" / stage / view / (row["id"] + ".json")
        if dest.exists() and not a.clean:
            return read_json(dest)
        item, comp = load(row, view)
        start = time.perf_counter()
        result = scores(comp, bank, a.top_k, a.penalty, a.spatial)
        result["inference_ms"] = (time.perf_counter() - start) * 1000
        result["frontend_ms"] = item["elapsed_ms"]
        if row["id"] in annotations:
            oracle = oracle_components(annotations[row["id"]], item["names"])
            result["oracle"] = scores(oracle, bank, a.top_k, a.penalty, a.spatial)["hard"]
            result["oracle_count"] = len(oracle.p)
            result.update(compare_objects(comp, oracle))
            result["review_type"] = annotations[row["id"]].get("violation", "unspecified")
        write_json(dest, result)
        return result

    for category, cohort in grouped.items():
        conditions = CONDITIONS if a.dataset == "vid" else ["Original"]
        sources = ["Original"] if a.protocol == "cross" else conditions
        for source in sources:
            local = [r for r in cohort if r["condition"] == source]
            train = [r for r in local if r["split"] == "train" and not r["label"]]
            val = [r for r in local if r["split"] == "validation" and not r["label"]]
            test = [r for r in (cohort if a.protocol == "cross" else local) if r["split"] == "test"]
            audit_groups(train + val, test, a.protocol == "cross")
            if val:
                audit_groups(train, val, False)
            bank = normal_bank([(r, load(r, "clean")[1]) for r in train])
            for view in views:
                cal = []
                for i, row in enumerate(tqdm(val or train, desc=f"{category}/{source}/{view} calibration")):
                    refs = bank
                    if not val:
                        # VID has no validation split. Exclude self and all known siblings.
                        refs = [ref for j, ref in enumerate(bank) if j != i and
                                (not row["group"] or train[j]["group"] != row["group"]) and
                                train[j]["sha256"] != row["sha256"]]
                        if not refs:
                            raise ValueError("Need at least two independent normal reference groups")
                    value = measure(row, view, refs, "calibration")
                    cal.append(value)
                    calibration_rows.append({**row, "view": view, **value})
                threshold = {m: float(np.quantile([c[m] for c in cal], a.quantile, method="higher"))
                             for m in METHODS}
                if view == "clean":
                    clean_threshold = threshold.copy()
                for row in tqdm(test, desc=f"{category}/{source}/{view} test"):
                    result = measure(row, view, bank, "test")
                    for method in [*METHODS, "hard_cal", *(["oracle"] if "oracle" in result else [])]:
                        key = "hard" if method == "hard_cal" else method
                        t = (threshold["hard"] if method == "hard_cal" else
                             clean_threshold["hard" if method == "oracle" else method])
                        score = result[key]
                        records.append(dict(**row, view=view, source=source, method=method,
                                            score=score, threshold=t, margin=score-t, alarm=int(score>t),
                                            entropy=result["entropy"], count=result["count"],
                                            oracle_count=result.get("oracle_count"),
                                            **{k: result.get(k) for k in ("gt_objects", "matched", "missed",
                                                                           "spurious", "wrong_type")},
                                            review_type=result.get("review_type", ""),
                                            frontend_ms=result["frontend_ms"],
                                            all_methods_ms=result["inference_ms"]))
    table = pd.DataFrame(records)
    table.to_csv(out / "scores.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(out / "calibration.csv", index=False)
    report(table, out, a.protocol)
    print(out)


if __name__ == "__main__":
    main()
