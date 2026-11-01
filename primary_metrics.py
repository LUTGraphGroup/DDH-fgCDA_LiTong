from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_METRICS = {
    "Accuracy": "accuracy",
    "Macro-F1": "macro_f1",
    "MCC": "mcc",
    "Macro-AUC": "macro_auc_ovr",
    "Macro-AUPR": "macro_aupr_ovr",
}
PRIMARY_COLUMNS = ["Dataset", *PRIMARY_METRICS]


def primary_row(metrics, dataset):

    missing = [key for key in PRIMARY_METRICS.values() if key not in metrics]
    if missing:
        raise ValueError(f"Missing primary metrics: {missing}; use the common three-class evaluator")
    row = {"Dataset": str(dataset)}
    for title, key in PRIMARY_METRICS.items():
        value = metrics[key]
        row[title] = np.nan if value is None else float(value)


    support_keys = ["support_unknown", "support_up-regulated", "support_down-regulated"]
    if any(key in metrics and float(metrics[key]) == 0 for key in support_keys):
        row["Macro-AUC"] = np.nan
        row["Macro-AUPR"] = np.nan
    return row


def write_table(rows, output_path):

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=PRIMARY_COLUMNS)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NA")
    lines = ["| " + " | ".join(PRIMARY_COLUMNS) + " |", "|---|---:|---:|---:|---:|---:|"]
    for row in frame.to_dict("records"):
        cells = [str(row["Dataset"]).replace("|", "\\|")]
        for key in PRIMARY_METRICS:
            value = row[key]
            cells.append("NA" if pd.isna(value) else f"{value:.6f}" if isinstance(value, (float, int, np.number)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def save_primary_metrics(metrics, output_dir, dataset):

    return write_table([primary_row(metrics, dataset)], Path(output_dir) / "primary_metrics.csv")


def save_primary_fold_metrics(fold_metrics, output_dir, dataset):

    records = pd.DataFrame(fold_metrics).to_dict("records")

    records = [row for row in records if str(row.get("fold", "")).strip().lower() not in {"mean", "std"}]
    if not records:
        raise ValueError("No fold evaluations to summarize")
    rows = []
    for index, record in enumerate(records, 1):
        fold = record.get("fold", index)
        if isinstance(fold, float) and fold.is_integer():
            fold = int(fold)
        rows.append(primary_row(record, f"{dataset} / fold_{fold}"))
    values = pd.DataFrame(rows)[list(PRIMARY_METRICS)].astype(float)

    mean = values.mean(skipna=False)
    std = values.std(ddof=1, skipna=False)
    out = Path(output_dir)
    fold_rows = rows + [dict(Dataset=f"{dataset} / mean", **mean.to_dict()),
                        dict(Dataset=f"{dataset} / std", **std.to_dict())]
    write_table(fold_rows, out / "primary_metrics_folds.csv")
    write_table([dict(Dataset=dataset, **mean.to_dict())], out / "primary_metrics.csv")
    paper_row = {"Dataset": dataset}
    for key in PRIMARY_METRICS:
        paper_row[key] = f"{mean[key]:.6f} ± {std[key]:.6f}" if np.isfinite(mean[key]) and np.isfinite(std[key]) else "NA"
    write_table([paper_row], out / "primary_metrics_mean_std.csv")
    return mean, std
