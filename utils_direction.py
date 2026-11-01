import hashlib
import itertools
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    matthews_corrcoef, confusion_matrix, roc_auc_score, average_precision_score)
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parent
DATASETS = {
    'Data1': {'suffix': '603_83', 'shape': (603,83), 'mirnas':190,
              'a_cm':'A_cm_603x190.xlsx', 'a_md':'A_md_190x83.xlsx'},
    'Data2': {'suffix': '2738_275', 'shape': (2738,275), 'mirnas':364,
              'a_cm':'A_cm_2738_364.xlsx', 'a_md':'A_md_364_275.xlsx'},
}
DEFAULT_GRID = {'lr':[.0001,.0005], 'drop_rate':[.1,.3], 'epochs':[100,300,500]}
METRICS = {'accuracy':'Accuracy', 'macro_precision':'Macro-Precision',
    'macro_recall':'Macro-Recall', 'macro_f1':'Macro-F1', 'mcc':'MCC',
    'macro_auc_ovr':'Macro-AUC', 'macro_aupr_ovr':'Macro-AUPR'}
LABEL_ID_TO_NAME = {0:'unknown', 1:'up-regulated', 2:'down-regulated'}

def sha256(path):

    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_matrix(path, shape):

    path = Path(path)
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
        attempts = [value]
    else:
        reader = pd.read_excel if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv
        attempts = []
        for options in ({"header": None}, {"header": 0, "index_col": 0}, {"header": 0}):
            try:
                value = reader(path, **options).to_numpy(dtype=np.float64)
                if value.shape == tuple(shape) and np.isfinite(value).all():
                    return value
                attempts.append(value.shape)
            except (ValueError, TypeError):
                pass
        raise ValueError(f"Matrix shape/values invalid: {path}; expected {shape}; attempts {attempts}")
    if value.shape != tuple(shape) or not np.isfinite(value).all():
        raise ValueError(f"Invalid matrix: {path}, expected {shape}")
    return value


def read_json(path):

    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):

    def clean(item):

        if isinstance(item, dict):
            return {str(k):clean(v) for k,v in item.items()}
        if isinstance(item, (list,tuple)):
            return [clean(v) for v in item]
        if isinstance(item, np.generic):
            item=item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_pairs(folder,name,pairs):

    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/f'{name}_pairs.npy',pairs,allow_pickle=False)
    pd.DataFrame(pairs,columns=['circ_index','disease_index','label']).to_csv(folder/f'{name}_pairs.csv',index=False)


def parse_grid(grid):

    if not isinstance(grid,dict) or set(grid)!=set(DEFAULT_GRID):
        raise ValueError('Grid must contain epochs, lr and drop_rate')
    for key,values in grid.items():
        if not isinstance(values,list) or not values:
            raise ValueError(f'Empty or invalid grid: {key}')
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and np.isfinite(v) for v in values):
            raise ValueError(f'Non-numeric grid: {key}')
        if len(set(values))!=len(values) or not set(values)<=set(DEFAULT_GRID[key]):
            raise ValueError(f'Use unique candidates from Table 3: {key}')
        if key=='epochs' and not all(isinstance(v,int) for v in values):
            raise ValueError('Epoch counts must be integers')
    keys=list(grid)
    return [dict(zip(keys,values)) for values in itertools.product(*(grid[k] for k in keys))]


def verify_sources(inventory):

    for path,expected in inventory.items():
        if sha256(path)!=expected:
            raise RuntimeError(f'Protected source changed: {path}')

def set_random_seed(seed=2022, deterministic=True):


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def tensor_to_numpy(x):


    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def logits_to_probs(logits):


    if isinstance(logits, torch.Tensor):
        probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return probs.astype(np.float64)

def probs_to_preds(probs):


    probs = np.asarray(probs)
    return np.argmax(probs, axis=1).astype(np.int64)

def evaluate_multiclass_from_logits(
    y_true,
    logits,
    num_classes=3,
    labels=(0, 1, 2)
):


    y_true = tensor_to_numpy(y_true).astype(np.int64).reshape(-1)

    probs = logits_to_probs(logits)
    y_pred = probs_to_preds(probs)

    metrics, cm = evaluate_multiclass_from_probs(
        y_true=y_true,
        probs=probs,
        y_pred=y_pred,
        num_classes=num_classes,
        labels=labels
    )

    return metrics, cm, probs, y_pred

def evaluate_multiclass_from_probs(
    y_true,
    probs,
    y_pred=None,
    num_classes=3,
    labels=(0, 1, 2)
):


    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)

    if y_pred is None:
        y_pred = probs_to_preds(probs)
    else:
        y_pred = np.asarray(y_pred).astype(np.int64).reshape(-1)

    if probs.ndim != 2 or probs.shape[1] != num_classes:
        raise ValueError(f"probs shape 应为 [N, {num_classes}]，但得到 {probs.shape}")

    if y_true.shape[0] != probs.shape[0]:
        raise ValueError(f"y_true 数量 {y_true.shape[0]} 与 probs 数量 {probs.shape[0]} 不一致")

    acc = accuracy_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        average="macro",
        zero_division=0
    )

    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        average="weighted",
        zero_division=0
    )

    p_each, r_each, f1_each, support_each = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        average=None,
        zero_division=0
    )

    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))


    y_true_bin = label_binarize(y_true, classes=list(labels))

    try:
        macro_auc_ovr = roc_auc_score(
            y_true,
            probs,
            labels=list(labels),
            multi_class="ovr",
            average="macro"
        )
    except Exception:
        macro_auc_ovr = np.nan

    try:
        weighted_auc_ovr = roc_auc_score(
            y_true,
            probs,
            labels=list(labels),
            multi_class="ovr",
            average="weighted"
        )
    except Exception:
        weighted_auc_ovr = np.nan

    try:
        macro_aupr_ovr = average_precision_score(
            y_true_bin,
            probs,
            average="macro"
        )
    except Exception:
        macro_aupr_ovr = np.nan

    try:
        weighted_aupr_ovr = average_precision_score(
            y_true_bin,
            probs,
            average="weighted"
        )
    except Exception:
        weighted_aupr_ovr = np.nan


    auc_each = {}
    aupr_each = {}

    for idx, label_id in enumerate(labels):
        label_name = LABEL_ID_TO_NAME[int(label_id)]

        y_true_binary = y_true_bin[:, idx]
        y_score = probs[:, idx]


        if len(np.unique(y_true_binary)) < 2:
            auc_each[label_name] = np.nan
        else:
            try:
                auc_each[label_name] = roc_auc_score(
                    y_true_binary,
                    y_score
                )
            except Exception:
                auc_each[label_name] = np.nan


        if len(np.unique(y_true_binary)) < 2:
            aupr_each[label_name] = np.nan
        else:
            try:
                aupr_each[label_name] = average_precision_score(
                    y_true_binary,
                    y_score
                )
            except Exception:
                aupr_each[label_name] = np.nan

    metrics = {
        "accuracy": float(acc),

        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),

        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),

        "mcc": float(mcc),

        "macro_auc_ovr": float(macro_auc_ovr) if not np.isnan(macro_auc_ovr) else np.nan,
        "weighted_auc_ovr": float(weighted_auc_ovr) if not np.isnan(weighted_auc_ovr) else np.nan,

        "macro_aupr_ovr": float(macro_aupr_ovr) if not np.isnan(macro_aupr_ovr) else np.nan,
        "weighted_aupr_ovr": float(weighted_aupr_ovr) if not np.isnan(weighted_aupr_ovr) else np.nan,
    }


    for idx, label_id in enumerate(labels):
        label_name = LABEL_ID_TO_NAME[int(label_id)]

        metrics[f"precision_{label_name}"] = float(p_each[idx])
        metrics[f"recall_{label_name}"] = float(r_each[idx])
        metrics[f"f1_{label_name}"] = float(f1_each[idx])
        metrics[f"support_{label_name}"] = int(support_each[idx])

        metrics[f"auc_{label_name}"] = (
            float(auc_each[label_name])
            if not np.isnan(auc_each[label_name])
            else np.nan
        )

        metrics[f"aupr_{label_name}"] = (
            float(aupr_each[label_name])
            if not np.isnan(aupr_each[label_name])
            else np.nan
        )

    return metrics, cm
