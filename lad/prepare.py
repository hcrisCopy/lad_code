"""Fetch original archives and checkpoints ahead of offline runs."""
import argparse
import getpass
import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import gdown

from .common import BACKBONE, TEXT_HEAD, WEIGHTS, DATA, REPO, REVISION, file_hash
from .unpack import unpack

BASE = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
ARCHIVES = DATA / "downloads"
DRIVE = {"vid": ("1_UaWAuylvaErnvOq0uxq4gIg_NeSUNdz", "VID-AD.zip"),
         "maps": ("1oUcC3O_2-T6TccEdCEC4VBk9mChIffAm", "mvtec_loco_segmentations.zip")}


def download(env, dest, checksum=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        url = os.environ.get(env) or getpass.getpass(f"{env} (paste the full authorized file URL): ")
        if urlsplit(url).scheme != "https" or "*" in urlsplit(url).path:
            raise ValueError(f"{env} requires an HTTPS file URL")
        part = dest.with_suffix(dest.suffix + ".part")
        cmd = ["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
               "--output", str(part), url]
        # Avoid putting private signed URLs into exception text.
        if subprocess.run(cmd).returncode:
            raise RuntimeError(f"Download failed for {dest.name}")
        if checksum and not file_hash(part).startswith(checksum):
            part.unlink()
            raise ValueError(f"Wrong checkpoint: {dest.name}")
        part.replace(dest)
    if checksum and not file_hash(dest).startswith(checksum):
        raise ValueError(f"Wrong checkpoint: {dest}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("assets", nargs="+", choices=["repo", "weights", "baseline-weight", "vid", "loco", "maps"])
    a = p.parse_args()
    for name in a.assets:
        if name == "repo":
            if not REPO.exists():
                REPO.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "clone", "https://github.com/cnulab/LogiCo.git", str(REPO)], check=True)
                subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", REVISION], check=True)
            rev = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
            if rev != REVISION:
                raise ValueError(f"Expected LogiCo {REVISION} at {REPO}")
        elif name in DRIVE:
            file_id, filename = DRIVE[name]
            ARCHIVES.mkdir(parents=True, exist_ok=True)
            dest = ARCHIVES / filename
            if not dest.is_file():
                part = str(dest) + ".part"
                if not gdown.download(id=file_id, output=part, resume=True):
                    raise RuntimeError(f"Google Drive download failed: {filename}")
                Path(part).replace(dest)
            unpack(name, dest, clean=False)
        elif name == "loco":
            dest = ARCHIVES / "mvtec_loco_anomaly_detection.tar.xz"
            download("LOCO_URL", dest)
            unpack(name, dest, clean=False)
        else:
            pairs = [("DINOV3_B_URL", BASE)] if name == "baseline-weight" else [
                ("DINOV3_L_URL", BACKBONE), ("DINOTXT_URL", TEXT_HEAD)]
            for env, filename in pairs:
                download(env, WEIGHTS / filename, Path(filename).stem.split("-")[-1])


if __name__ == "__main__":
    main()
