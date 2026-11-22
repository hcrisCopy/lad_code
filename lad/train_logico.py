"""Train the author's LOCO model for a fixed epoch count, with resumable epochs."""
import argparse
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import DATA, REPO, ROOTS, WEIGHTS, digest, file_hash, index_path, read_json, write_json
from .index import LOCO

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", choices=LOCO, default="breakfast_box")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=14)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, default=DATA / "logico")
    p.add_argument("--clean", action="store_true")
    a = p.parse_args()
    if a.epochs < 1 or a.limit < 0 or a.batch_size < 1:
        p.error("Invalid training arguments")
    sys.path.insert(0, str(REPO.resolve()))
    import train_logico as upstream

    upstream.setup_seed(42)
    device = torch.device("cuda", 0)
    root = ROOTS["loco"] / a.category
    configs = REPO / "configs"
    cfg = EasyDict(yaml.safe_load((configs / "train/mvtec_loco.yaml").read_text()))
    vocab = yaml.safe_load((configs / "segmentations/mvtec_loco" / f"{a.category}.yaml").read_text())["prompts"]
    ckpt = WEIGHTS / upstream.get_dinov3_weight_name(cfg.encoder)
    transform, gt_transform = upstream.get_data_transforms(cfg.image_size)
    rows = read_json(index_path("loco"))
    data = {}
    sources = {}
    for split in ("train", "validation"):
        ds = upstream.LOCODatasetWithSeg(root=str(root), image_size=cfg.image_size,
                                         transform=transform, gt_transform=gt_transform, phase=split)
        ds.samples = sorted(ds.samples, key=lambda sample: sample[0])[:a.limit or None]
        expected = {r["id"]: r["sha256"] for r in rows if r["category"] == a.category
                    and r["split"] == split and r["subtype"] == "good"}
        selected = sorted(expected)[:a.limit or None]
        if not ds.samples or [Path(s[0]).relative_to(ROOTS["loco"]).as_posix() for s in ds.samples] != selected:
            raise ValueError(f"Official {split} loader disagrees with index")
        for sample, key in zip(ds.samples, selected):
            if sample[4] != 0 or file_hash(Path(sample[0])) != expected[key]:
                raise ValueError(f"Normal source mismatch: {key}")
            sources[key] = [expected[key], file_hash(Path(sample[1]))]
        data[split] = ds
    if len(data["train"]) < a.batch_size:
        p.error("Training needs at least one full batch; lower --batch-size for debugging")
    out = a.out / "mvtec_loco" / a.category
    settings = dict(epochs=a.epochs, batch_size=a.batch_size, limit=a.limit, seed=42, config=cfg,
                    source=digest(sources), weight=file_hash(ckpt), trainer=file_hash(Path(__file__)))
    meta = out / "settings.json"
    if meta.exists() and read_json(meta) != settings:
        raise ValueError("Use a new --out for a different baseline configuration")
    write_json(meta, settings)
    encoder = upstream.load_dinov3_model(cfg.encoder, pretrained_weight_path=str(ckpt))
    encoder.name = cfg.encoder
    encoder.to(device).eval()
    for ds in data.values():
        ds.extract_feats(encoder, cfg.encoder_target_layers, cfg.encoder_feat_stride,
                         cfg.encoder_feat_dim, device=device, use_aggregate_features=cfg.use_aggregate_features)
    train = DataLoader(data["train"], batch_size=a.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val = DataLoader(data["validation"], batch_size=a.batch_size, num_workers=4)
    model = upstream.LogiCo(**cfg.model, encoder=encoder, comp_num=len(vocab),
                            encoder_feat_dim=cfg.encoder_feat_dim, device=device)
    synth = upstream.CombSynthesizer(**cfg.synthesizer, num_cls=len(vocab))
    opt = torch.optim.AdamW(model.get_trainable_params(), lr=2e-4, weight_decay=1e-4)
    steps = a.epochs * len(train)
    scheduler = upstream.WarmCosineScheduler(opt, base_value=2e-4, final_value=1e-4,
                                              total_iters=steps, warmup_iters=min(100, steps))
    heads = {"CRN": model.CRN, "SRN": model.SRN, "SMD": model.SMD}
    history = []
    resume = out / "training_state.pt"
    if resume.is_file() and not a.clean:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        for name, head in heads.items():
            head.load_state_dict(state["heads"][name])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        del state
    for epoch in range(len(history), a.epochs):
        model.train()
        losses = []
        for batch in tqdm(train, desc=f"Epoch {epoch+1}/{a.epochs}"):
            batch["anomaly_seg_map"], batch["masks"] = synth(batch["seg_map"])
            loss = sum(model.get_losses(batch))
            opt.zero_grad()
            loss.backward()
            for head in (model.CRN, model.SRN):
                torch.nn.utils.clip_grad_norm_(head.parameters(), 0.1)
            opt.step()
            scheduler.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
        tmp = out / "training_state.tmp"
        torch.save(dict(heads={k: h.state_dict() for k, h in heads.items()}, optimizer=opt.state_dict(),
                        scheduler=scheduler.state_dict(), history=history, rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(), python_rng=random.getstate(),
                        numpy_rng=np.random.get_state()), tmp)
        tmp.replace(resume)
        print(f"Epoch {epoch+1} loss={history[-1]:.4f}", flush=True)
    model.eval()
    model.get_scores_mean_and_std(val)
    tmp = out / "LogiCo_final.tmp"
    model.save_ckpt(str(tmp))
    tmp.replace(out / f"LogiCo_epoch_{a.epochs}_ckpt.pkl")
    fig, ax = plt.subplots()
    ax.plot(range(1, len(history)+1), history)
    ax.set(xlabel="Epoch", ylabel="Training loss")
    fig.savefig(out / "loss.png", bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
