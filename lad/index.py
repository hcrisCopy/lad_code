"""Index official image splits; preserve original subtype and relative image ID."""
import argparse
import csv
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from .common import ROOTS, file_hash, index_path, write_json, read_json

LOCO = "breakfast_box juice_bottle pushpins screw_bag splicing_connectors".split()
ASPECTS = {
    "Sticks": ("Quantity", "Length"), "Fruits": ("Quantity", "Type"),
    "Tools": ("Quantity", "Placement"), "Cookies": ("Quantity", "Relation"),
    "Tapes": ("Length", "Type"), "Stationery": ("Length", "Placement"),
    "Ropes": ("Length", "Relation"), "Blocks": ("Type", "Placement"),
    "Dishes": ("Type", "Relation"), "Balls": ("Placement", "Relation"),
}
CONDITIONS = ["Original", "Cable_BG", "Mesh_BG", "Low-light_CD", "Blurry_CD"]


def folders(dataset, category):
    yield "train", "good"
    if dataset == "loco":
        yield "validation", "good"
        yield "test", "structural_anomalies"
        yield "test", "logical_anomalies"
    else:
        for aspect in (*ASPECTS[category], "Mix"):
            yield "test", f"logical_anomalies/{aspect}"
    yield "test", "good"


def list_images(dataset):
    root = ROOTS[dataset]
    suffix = ".png" if dataset == "loco" else ".jpg"
    for category in LOCO if dataset == "loco" else ASPECTS:
        for condition in ["Original"] if dataset == "loco" else CONDITIONS:
            task = category if condition == "Original" else f"{category}_{condition}"
            seen = set()
            for split, subtype in folders(dataset, category):
                folder = root / task / split / subtype
                images = sorted(p for p in folder.iterdir() if p.suffix.lower() == suffix)
                if not images:
                    raise ValueError(f"No {suffix} images in {folder}")
                for path in images:
                    seen.add(path)
                    yield dict(id=path.relative_to(root).as_posix(), category=category,
                               condition=condition, split=split, subtype=subtype,
                               label=int(subtype != "good"))
            actual = {p for split in ("train", "test", "validation")
                      for p in (root / task / split).rglob("*") if p.suffix.lower() == suffix}
            if seen != actual:
                raise ValueError(f"Unexpected image layout in {task}: {sorted(actual-seen)[:5]}")


def group_map(path):
    if path is None:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        groups = {}
        for row in csv.DictReader(stream):
            if not row["group"].strip() or row["id"] in groups:
                raise ValueError(f"Missing or duplicate group assignment: {row}")
            groups[row["id"]] = row["group"].strip()
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=ROOTS, required=True)
    p.add_argument("--groups", type=Path, help="CSV: id,group (physical instance/session)")
    p.add_argument("--clean", action="store_true")
    a = p.parse_args()
    rows = list(list_images(a.dataset))
    # Count images in train, validation and test; exclude ground-truth masks.
    expected = {"vid": 10395, "loco": 3651}[a.dataset]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} images, found {len(rows)}")
    groups = group_map(a.groups)
    if a.groups and set(groups) != {r["id"] for r in rows}:
        raise ValueError("Group CSV must cover each official image ID exactly once")
    output = index_path(a.dataset)
    partial = output.with_suffix(".progress.json")
    cache = {} if a.clean or not partial.exists() else read_json(partial)
    owners = defaultdict(set)
    for n, row in enumerate(tqdm(rows, desc="Index images"), 1):
        path = ROOTS[a.dataset] / row["id"]
        stat = path.stat()
        key = [stat.st_size, stat.st_mtime_ns]
        entry = cache.get(row["id"])
        if entry is None or entry[:2] != key:
            entry = cache[row["id"]] = [*key, file_hash(path)]
        row.update(sha256=entry[2], group=groups.get(row["id"]))
        owners[(row["category"], entry[2])].add(row["split"])
        if n % 200 == 0:
            write_json(partial, cache)
    write_json(partial, cache)
    duplicates = [key for key, splits in owners.items() if len(splits) > 1]
    if duplicates:
        raise ValueError(f"Identical files cross official splits: {duplicates[:5]}")
    write_json(output, rows)
    print(f"{len(rows)} images -> {output}")


if __name__ == "__main__":
    main()
