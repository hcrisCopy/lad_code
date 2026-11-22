"""Read dense dino.txt scores through LogiCo before hard segmentation."""
import argparse
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

from .common import (BACKBONE, CACHE, DATA, REPO, REVISION, TEXT_HEAD, WEIGHTS,
                     digest, file_hash, read_json, write_json)
from .index import ROOTS

# Object vocabulary only. No counts, positions or allowed configurations.
VID_NAMES = {
    "Fruits": ["orange fruit", "kiwi fruit", "other fruit"],
    "Tools": ["metal bolt", "metal washer", "metal nut"],
    "Cookies": ["yellow cookie", "black cookie", "square dish", "round dish"],
    "Balls": ["orange ball", "white ball"],
    "Blocks": ["round block", "triangular block", "square block"],
    "Dishes": ["fork", "plate", "spoon", "knife"],
    "Sticks": ["red stick", "blue stick"],
    "Tapes": ["green tape strip", "red tape strip"],
    "Stationery": ["black pencil", "red pencil", "blue eraser", "red eraser"],
    "Ropes": ["rope", "reference stick", "text label"],
}
VIEWS = ("clean", "dark", "blur")


def select_rows(rows, categories, limit):
    if limit < 0:
        raise ValueError("limit must be >= 0")
    if categories and set(categories) - {r["category"] for r in rows}:
        raise ValueError("Unknown category")
    groups = defaultdict(list)
    for row in rows:
        if not categories or row["category"] in categories:
            groups[row["category"], row["condition"], row["split"], row["subtype"]].append(row)
    return [row for group in groups.values() for row in group[:limit or None]]


def cache_file(root, view, row):
    return root / view / (row["id"] + ".json")


def vocabulary(dataset, category):
    configs = REPO / "configs/segmentations/mvtec_loco"
    config = yaml.safe_load((configs / (category + ".yaml" if dataset == "loco"
                                        else "breakfast_box.yaml")).read_text())
    names = config["prompts"] if dataset == "loco" else [*VID_NAMES[category], "background"]
    # In the five LOCO YAMLs the background (including the connector grid) is last.
    return names, config["prompt_templates"]


def make_candidates(prob, min_area, proposal_ratio):
    """One disjoint proposal per connected foreground winner, including weak winners."""
    foreground = prob[:-1]
    winner = foreground.argmax(0)
    supported = foreground.max(0) >= proposal_ratio * prob[-1]
    height, width = winner.shape
    objects = []
    for label in range(len(foreground)):
        mask = ((winner == label) & supported).astype(np.uint8)
        count, ids, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, count):
            x, y, w, h, area = map(int, stats[i])
            if area < min_area * height * width:
                continue
            scores = prob[:, ids == i].mean(axis=1, dtype=np.float64)
            scores /= scores.sum()
            objects.append({"p": scores.tolist(),
                            "box": [x / width, y / height, (x+w) / width, (y+h) / height]})
    return objects


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=ROOTS)
    p.add_argument("--categories", nargs="+")
    p.add_argument("--views", choices=VIEWS, nargs="+", default=["clean"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--crop-batch", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--min-area", type=float, default=0.001)
    p.add_argument("--proposal-ratio", type=float, default=0.8)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--clean", action="store_true")
    a = p.parse_args()
    if (a.size < 16 or a.size % 16 or a.crop_batch < 1 or a.temperature <= 0
            or not 0 < a.min_area < 1 or not 0 < a.proposal_ratio <= 1):
        p.error("Invalid extraction parameters")
    rank = int(os.getenv("LOCAL_RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    random.seed(42)
    torch.manual_seed(42)
    sys.path.insert(0, str(REPO.resolve()))
    import create_segmentation_maps as logico

    rev = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION:
        raise ValueError(f"LogiCo revision mismatch: {rev}")
    weights = [WEIGHTS / BACKBONE, WEIGHTS / TEXT_HEAD]
    signature = dict(schema=2, dataset=a.dataset, revision=rev, size=a.size,
                     crop_batch=a.crop_batch, temperature=a.temperature, seed=42,
                     min_area=a.min_area, proposal_ratio=a.proposal_ratio,
                     weights=[file_hash(w) for w in weights], adapter=file_hash(Path(__file__)))
    for path, sha in zip(weights, signature["weights"]):
        if not sha.startswith(path.stem.split("-")[-1]):
            raise ValueError(f"Incorrect official checkpoint: {path}")
    root = a.cache / a.dataset
    config = root / "settings.json"
    if config.exists() and read_json(config) != signature:
        raise ValueError("Use a new --cache directory for a different frontend")
    write_json(config, signature)
    sig = digest(signature)
    rows = select_rows(read_json(DATA / "index" / f"{a.dataset}.json"), a.categories, a.limit)
    jobs = [(r, v) for r in rows for v in dict.fromkeys(a.views)]
    device = torch.device("cuda", rank)
    logico.device = device
    model, tokenizer = logico.dinov3_vitl16_dinotxt_tet1280d20h24l(
        pretrained=False, vision_backbone_pretrained_path=weights[0],
        txt_backbone_pretrained_path=weights[1],
        bpe_path_or_url=str(REPO / "dinov3/bpe_simple_vocab_16e6.txt"))
    model = model.eval().to(device)
    transform = logico.make_classification_eval_transform(resize_size=a.size, crop_size=a.size)
    embeddings = {}
    with torch.inference_mode():
        for row, view in tqdm(jobs[rank::world], desc="Candidates", disable=rank != 0):
            out = cache_file(root, view, row)
            if out.exists() and not a.clean:
                saved = read_json(out)
                if (saved["signature"], saved["sha256"]) != (sig, row["sha256"]):
                    raise ValueError(f"Stale cache: {out}")
                continue
            category = row["category"]
            names, templates = vocabulary(a.dataset, category)
            if category not in embeddings:
                embeddings[category] = logico.build_text_embeddings(model, tokenizer, names, templates)
            path = ROOTS[a.dataset] / row["id"]
            if file_hash(path) != row["sha256"]:
                raise ValueError(f"Image changed since indexing: {path}")
            start = time.perf_counter()
            with Image.open(path) as source:
                img = source.convert("RGB").resize((a.size, a.size))
            if view == "dark":
                img = ImageEnhance.Brightness(img).enhance(0.4)
            elif view == "blur":
                img = img.filter(ImageFilter.GaussianBlur(radius=a.size / 128))
            fmap, backbone_map = logico.aggregate_features(model, transform, img, a.crop_batch)
            del backbone_map
            logits = embeddings[category] @ F.normalize(fmap.flatten(1).float(), dim=0)
            prob = (logits / a.temperature).softmax(0).reshape(-1, a.size, a.size).cpu().numpy()
            objects = make_candidates(prob, a.min_area, a.proposal_ratio)
            write_json(out, dict(signature=sig, sha256=row["sha256"], names=names, objects=objects,
                                 elapsed_ms=1000*(time.perf_counter()-start)))


if __name__ == "__main__":
    main()
