import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm


def metrics(frame, score="score"):
    y = frame.label.to_numpy()
    alarm = frame.alarm.to_numpy()
    if set(y) != {0, 1}:
        raise ValueError("AUROC needs both good and anomalous images in the cohort")
    return dict(n_good=int((y == 0).sum()), n_anomaly=int(y.sum()),
                auroc=roc_auc_score(y, frame[score]), ap=average_precision_score(y, frame[score]),
                f1=f1_score(y, alarm, zero_division=0), fpr=alarm[y == 0].mean(),
                recall=alarm[y == 1].mean(), fnr=1-alarm[y == 1].mean())


def correlation(a, b):
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return spearmanr(a, b).statistic


def paired_delta(frame, method, baseline, draws=500):
    paired = frame.pivot(index="id", columns="method", values="score")[[method, baseline]].dropna()
    meta = frame.drop_duplicates("id").set_index("id").loc[paired.index]
    labels = meta.label.to_numpy()
    if len(set(labels)) != 2:
        raise ValueError(f"Reviewed subset requires good and anomalous images for {method}/{baseline}")
    units = meta.group if meta.group.notna().all() else pd.Series(meta.index, index=meta.index)
    blocks = list(units.groupby(units).indices.values())
    values = paired.to_numpy()

    def delta(idx):
        return roc_auc_score(labels[idx], values[idx, 0]) - roc_auc_score(labels[idx], values[idx, 1])

    rng = np.random.default_rng(42)
    samples = []
    for _ in range(draws):
        idx = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        if len(set(labels[idx])) == 2:
            samples.append(delta(idx))
    interval = np.quantile(samples, [.025, .975]) if len(blocks) > 1 and samples else [np.nan]*2
    return dict(method=method, baseline=baseline, images=len(meta), groups=len(blocks),
                unit="group" if meta.group.notna().all() else "image", draws=len(samples),
                delta_auroc=delta(slice(None)), lo=interval[0], hi=interval[1])


def report(table, out, protocol):
    tasks, pooled, paired, ambiguity, reviews = [], [], [], [], []
    groups = table.groupby(["category", "condition", "view"])
    for (category, condition, view), data in tqdm(groups, total=len(groups), desc="Paired intervals"):
        task = dict(category=category, condition=condition, view=view)
        good = data[data.label == 0]
        logical = data[data.subtype != "structural_anomalies"]
        for method, rows in data.groupby("method"):
            if method == "oracle":
                continue
            for scope, part in [("overall", rows), ("logical", rows[rows.subtype != "structural_anomalies"])]:
                tasks.append(dict(**task, method=method, scope=scope, **metrics(part)))
            normal = good[good.method == method]
            for subtype, abnormal in rows[rows.label == 1].groupby("subtype"):
                tasks.append(dict(**task, method=method, scope=subtype,
                                  **metrics(pd.concat([normal, abnormal]))))
        for method, base in [("joint", "hard"), ("joint", "visual"), ("joint", "forced")]:
            paired.append(dict(**task, **paired_delta(logical, method, base)))
        reviewed = logical[logical.review_type != ""]
        if not reviewed.empty:
            # Recall on a missing/wrong/mismatched subset remains defined without controls.
            for (kind, method), part in reviewed[reviewed.label == 1].groupby(["review_type", "method"]):
                reviews.append(dict(**task, violation=kind, method=method, n=len(part),
                                    recall=part.alarm.mean(), fnr=1-part.alarm.mean()))
            if reviewed.label.nunique() == 2:
                paired.append(dict(**task, **paired_delta(reviewed, "oracle", "hard")))
                for method, part in reviewed.groupby("method"):
                    tasks.append(dict(**task, method=method, scope="reviewed_logical", **metrics(part)))
        clean = table[(table.category == category) & (table.condition == condition)
                      & (table.view == "clean") & (table.method == "hard")].set_index("id")
        for method in ("hard", "hard_cal", "joint"):
            normals = good[good.method == method].set_index("id")
            counts = normals["count"] - clean.loc[normals.index, "count"]
            ambiguity.append(dict(**task, method=method, n=len(normals), fpr=normals.alarm.mean(),
                                  entropy=normals.entropy.mean(), mean_count_change=counts.mean(),
                                  rho_entropy_alarm=correlation(normals.entropy, normals.alarm),
                                  rho_entropy_score=correlation(normals.entropy, normals.score)))
    for (category, view, method), part in table[table.method != "oracle"].groupby(["category", "view", "method"]):
        if protocol == "cross":
            for scope, selected in [("mixed_all", part), ("unseen_only", part[part.condition != "Original"])]:
                pooled.append(dict(category=category, view=view, method=method, scope=scope,
                                   **metrics(selected)))
    tasks = pd.DataFrame(tasks)
    tasks.to_csv(out / "per_task.csv", index=False)
    cols = ["auroc", "ap", "f1", "fpr", "recall", "fnr"]
    comparable = tasks[tasks.scope.isin(["logical", "overall"])]
    macro = comparable.groupby(["scope", "view", "method"])[cols].mean()
    macro.to_csv(out / "macro.csv")
    by_cond = comparable.groupby(["scope", "view", "method", "condition"])[cols].mean()
    by_cond.to_csv(out / "per_condition.csv")
    by_cond.groupby(level=["scope", "view", "method"]).auroc.min().to_csv(out / "worst_condition.csv")
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    pd.DataFrame(ambiguity).to_csv(out / "ambiguity.csv", index=False)
    if pooled:
        pd.DataFrame(pooled).to_csv(out / "pooled.csv", index=False)
    if reviews:
        pd.DataFrame(reviews).to_csv(out / "reviewed_recall.csv", index=False)
    reviewed = table[(table.method == "hard") & table.gt_objects.notna()].copy()
    if not reviewed.empty:
        by = ["category", "condition", "view", "label"]
        totals = reviewed.groupby(by)[["missed", "matched", "gt_objects", "wrong_type", "spurious"]].sum()
        totals["miss_rate"] = totals.missed / totals.gt_objects.replace(0, np.nan)
        totals["type_error_given_match"] = totals.wrong_type / totals.matched.replace(0, np.nan)
        totals.to_csv(out / "perception.csv")
    print(macro.xs(("logical", "clean"), level=("scope", "view")).round(4).to_string())
