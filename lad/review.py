"""Prepare a reproducible stratified list for independent component review."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .common import DATA, read_json
from .extract import vocabulary
from .index import ROOTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=ROOTS)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--per-stratum", type=int, default=3)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.per_stratum < 1:
        p.error("--per-stratum must be positive")
    groups = defaultdict(list)
    rows = read_json(DATA / "index" / f"{a.dataset}.json")
    if set(a.categories) - {r["category"] for r in rows}:
        p.error("Unknown category")
    for r in rows:
        if r["split"] == "test" and r["category"] in a.categories and r["subtype"] != "structural_anomalies":
            groups[r["category"], r["condition"], r["subtype"]].append(r)
    rng = random.Random(42)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as stream:
        for group in groups.values():
            for row in rng.sample(group, min(a.per_stratum, len(group))):
                item = dict(id=row["id"], image=(ROOTS[a.dataset] / row["id"]).as_posix(),
                            vocabulary=[*vocabulary(a.dataset, row["category"])[0][:-1], "__other__"],
                            reviewer="", violation="", objects=None)
                stream.write(json.dumps(item) + "\n")
    print(a.output)


if __name__ == "__main__":
    main()
