"""Partial one-to-one assignment to normal exemplars; all costs are per image sums."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass
class Components:
    p: np.ndarray
    boxes: np.ndarray

    @classmethod
    def load(cls, objects, classes):
        p = np.array([o["p"] for o in objects], dtype=float).reshape(-1, classes)
        boxes = np.array([o["box"] for o in objects], dtype=float).reshape(-1, 4)
        if (len(p) != len(objects) or len(boxes) != len(p) or not np.isfinite(p).all()
                or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5)):
            raise ValueError("Invalid component probabilities")
        if (not np.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any()
                or (boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError("Expected normalized boxes [left, top, right, bottom]")
        return cls(p, boxes)

    def reference(self):
        labels = self.p.argmax(1)
        keep = labels != self.p.shape[1] - 1
        return labels[keep], self.boxes[keep]


def normal_bank(samples):
    refs = []
    for row, comp in samples:
        if row["label"] != 0 or row["split"] != "train":
            raise ValueError("Normal reference bank only accepts train/good")
        refs.append(comp.reference())
    if not refs or not any(len(labels) for labels, _ in refs):
        raise ValueError("No foreground components in normal reference bank")
    return refs


def compare_objects(observed, oracle, iou_threshold=0.5):
    """Prioritize number of IoU-qualified pairs, then total IoU, independently of type."""
    types, boxes = observed.reference()
    truth, targets = oracle.reference()
    lo = np.maximum(boxes[:, None, :2], targets[None, :, :2])
    hi = np.minimum(boxes[:, None, 2:], targets[None, :, 2:])
    overlap = np.maximum(hi - lo, 0).prod(2)
    area = (boxes[:, 2:] - boxes[:, :2]).prod(1)
    target_area = (targets[:, 2:] - targets[:, :2]).prod(1)
    iou = overlap / (area[:, None] + target_area[None, :] - overlap)
    eligible = iou >= iou_threshold
    rows, cols = linear_sum_assignment(-(eligible * (1 + min(iou.shape)) + iou))
    valid = eligible[rows, cols]
    matched = int(valid.sum())
    return dict(gt_objects=len(truth), missed=len(truth)-matched, spurious=len(types)-matched,
                matched=matched, wrong_type=int((types[rows[valid]] != truth[cols[valid]]).sum()))


def assign(pair, extra, missing):
    """Dummy slots allow unmatched observations and missing reference objects."""
    n, m = pair.shape
    if not n:
        return m * missing
    if not m:
        return float(extra.sum())
    cost = np.zeros((n + m, m + n))
    cost[:n, :m] = pair
    cost[:n, m:] = extra[:, None]
    cost[n:, :m] = missing
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum())


def scores(comp, refs, top_k=3, penalty=3.0, spatial=1.0):
    if top_k < 1 or penalty <= 0 or spatial < 0:
        raise ValueError("Invalid scoring parameters")
    p = comp.p
    background = p.shape[1] - 1
    n = len(p)
    visual = -np.log(np.clip(p, 1e-12, 1))
    free = visual.min(1)
    delta = visual - free[:, None]
    hard = p.argmax(1)
    present = hard != background
    order = np.argsort(-p[:, :-1], axis=1, kind="stable")
    allowed = np.zeros_like(p[:, :-1], dtype=bool)
    np.put_along_axis(allowed, order[:, :top_k], True, axis=1)
    fg_delta = delta[:, :-1].min(1)
    extra = np.minimum(delta[:, -1], fg_delta + penalty)
    best_hard = best_joint = best_forced = float("inf")
    for labels, boxes in refs:
        # Mean absolute box-coordinate error represents placement and extent.
        geometry = spatial * cdist(comp.boxes, boxes, metric="cityblock") / 4
        mismatch = (hard[present, None] != labels[None, :]) * penalty
        hc = assign(mismatch + geometry[present], np.full(present.sum(), penalty), penalty)
        # A mis-typed object may stay as observed and pay one type violation.
        switch = np.where(allowed[:, labels], delta[:, labels], np.inf)
        # If background wins, retaining any object must pay its visual evidence cost.
        pairing = np.minimum(switch, fg_delta[:, None] + penalty)
        jc = assign(pairing + geometry, extra, penalty)
        # Ignore visual costs within candidate support. Even confident real extras
        # can be dismissed as background in this deliberately over-strong control.
        fc = assign((~allowed[:, labels])*penalty + geometry, np.zeros(n), penalty)
        best_hard, best_joint, best_forced = min(best_hard, hc), min(best_joint, jc), min(best_forced, fc)
    entropy = float((p * visual).sum(1).mean()) if n else 0.0
    free_cost = float(free.sum())
    # Visual-only top-K mean surprisal; no reference or rule enters this control.
    top_probs = np.take_along_axis(p[:, :-1], order[:, :top_k], axis=1)
    mass = np.column_stack((top_probs, p[:, -1]))
    normalized = mass / np.maximum(mass.sum(1, keepdims=True), 1e-12)
    ensemble = float((-normalized * np.log(np.maximum(mass, 1e-12))).sum(1).mean()) if n else 0.
    return dict(hard=best_hard, joint=best_joint, forced=best_forced, visual=ensemble,
                normal_cost=free_cost + best_joint, free=free_cost, entropy=entropy,
                count=int(present.sum()))
