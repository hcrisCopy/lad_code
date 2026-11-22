"""Extract an official archive, stripping only its unique dataset wrapper."""
import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from tqdm import tqdm

from .common import DATA
from .index import LOCO, ROOTS, ASPECTS


def unpack(kind, archive, clean):
    vid = kind == "vid"
    marker = "Fruits" if vid else "breakfast_box"
    output = ROOTS["vid"] if vid else DATA / ("mvtec_loco" if kind == "loco" else "mvtec_loco_segmentations")
    categories = set(ASPECTS) if vid else set(LOCO)
    is_zip = zipfile.is_zipfile(archive)
    source = zipfile.ZipFile(archive) if is_zip else tarfile.open(archive, "r:*")
    with source:
        members = ([m for m in source.infolist() if not m.is_dir()] if is_zip
                   else [m for m in source.getmembers() if m.isfile()])
        names = [PurePosixPath(m.filename if is_zip else m.name) for m in members]
        roots = {p.parts[:i] for p in names for i, name in enumerate(p.parts)
                 if name == marker and p.parts[i+1:i+3] == ("train", "good")
                 and "__MACOSX" not in p.parts}
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one {marker}/train/good tree: {roots}")
        prefix = roots.pop()
        for member, path in tqdm(zip(members, names), total=len(names), desc=output.name):
            rest = path.parts[len(prefix):]
            if path.parts[:len(prefix)] != prefix or not rest:
                continue
            if (rest[0].split("_")[0] if vid else rest[0]) not in categories:
                continue
            dest = output.joinpath(*rest)
            if ".." in rest or not dest.resolve().is_relative_to(output.resolve()):
                raise ValueError(f"Unsafe archive entry: {path}")
            size = member.file_size if is_zip else member.size
            if dest.exists() and not clean:
                if dest.stat().st_size != size:
                    raise ValueError(f"Size mismatch: {dest}; re-extract with --clean")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name("." + dest.name + ".part")
            with (source.open(member) if is_zip else source.extractfile(member)) as src, tmp.open("wb") as target:
                shutil.copyfileobj(src, target)
            if tmp.stat().st_size != size:
                raise IOError(f"Incomplete archive entry: {path}")
            tmp.replace(dest)
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=["vid", "loco", "maps"])
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--clean", action="store_true")
    a = p.parse_args()
    unpack(a.kind, a.archive, a.clean)
