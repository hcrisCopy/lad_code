import hashlib
import json
import os
from pathlib import Path

DATA = Path("../lad_data")
REPO = Path("../lad_research/repos/LogiCo")
WEIGHTS = DATA / "weights"
RUNS = DATA / "runs"
CACHE = DATA / "candidates"
REVISION = "ac724ffb0c36392b443eecc41b58ba152b4298a0"
BACKBONE = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
TEXT_HEAD = "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
