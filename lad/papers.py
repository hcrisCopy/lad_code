import csv
import subprocess
from pathlib import Path

from .common import file_hash


def main():
    root = Path("../lad_research/papers")
    root.mkdir(parents=True, exist_ok=True)
    with Path("papers.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            path = root / row["file"]
            if not path.exists():
                tmp = path.with_suffix(".part")
                subprocess.run(["curl", "--fail", "--location", "--retry", "3",
                                "--output", str(tmp), row["url"]], check=True)
                if file_hash(tmp) != row["sha256"]:
                    raise ValueError(f"PDF differs from reviewed version: {path}")
                tmp.replace(path)
            if file_hash(path) != row["sha256"]:
                raise ValueError(f"Invalid PDF: {path}")
            print(path)


if __name__ == "__main__":
    main()
