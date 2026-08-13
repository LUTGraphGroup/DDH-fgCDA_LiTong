# -*- coding: utf-8 -*-
# @File : utils_direction.py
# 三分类细粒度 circRNA-disease 表达方向预测工具函数
#
# 标签定义：
#   0 = unknown / unconfirmed
#   1 = up-regulated
#   2 = down-regulated

import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report
)
from sklearn.preprocessing import label_binarize


# =========================
# 1. 全局标签定义
# =========================
LABEL_ID_TO_NAME = {
    0: "unknown",
    1: "up-regulated",
    2: "down-regulated"
}

LABEL_NAME_TO_ID = {
    "unknown": 0,
    "up-regulated": 1,
    "down-regulated": 2
}


# =========================
# 2. 随机种子
# =========================
def set_random_seed(seed=2022, deterministic=True):
    """
    固定随机种子，保证实验尽量可复现。

    参数：
        seed: int
        deterministic: bool
            True 时会让 cudnn 尽量确定性运行，但可能稍微降低速度。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================
# 3. 安全读写工具
# =========================
def ensure_dir(path):
    """
    如果目录不存在，则创建。
    """
    os.makedirs(path, exist_ok=True)
    return path


def read_csv_auto(path, header=0):
    """
    自动尝试常见编码读取 CSV。
    """
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, header=header, encoding=enc)
        except Exception as e:
            last_error = e

    raise last_error


# =========================
# 4. 读取方向标签矩阵
# =========================
def load_direction_label_matrix(processed_dir):
    """
    读取 step4 生成的 2738 × 275 方向标签矩阵。

    返回：
        label_matrix: np.ndarray, shape = [num_circ, num_disease]

    标签：
        -1 = ignore
         0 = unknown
         1 = up-regulated
         2 = down-regulated
    """
    matrix_path = os.path.join(
        processed_dir,
        "step4_direction_label_matrix_2738_275.npy"
    )

    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"找不到方向标签矩阵: {matrix_path}")

    label_matrix = np.load(matrix_path)

    if label_matrix.ndim != 2:
        raise ValueError(f"方向标签矩阵必须是二维矩阵，当前 shape={label_matrix.shape}")

    return label_matrix


def load_positive_up_down_pairs(processed_dir):
    """
    读取 step4 生成的上调/下调正样本 pair 表。
    """
    path = os.path.join(processed_dir, "step4_positive_up_down_pairs.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到上调/下调正样本文件: {path}")

    return read_csv_auto(path)


def load_unknown_candidate_pairs(processed_dir):
    """
    读取 step4 生成的 unknown 候选样本池。
    """
    path = os.path.join(processed_dir, "step4_unknown_candidate_pairs.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 unknown 候选样本文件: {path}")

    return read_csv_auto(path)


# =========================
# 5. 读取 step5 五折样本
# =========================
def get_fold_dir(processed_dir, fold_id):
    """
    返回某一折的目录：
        processed_direction_dataset/step5_training_samples_2738_275/fold_{fold_id}
    """
    return os.path.join(
        processed_dir,
        "step5_training_samples_2738_275",
        f"fold_{fold_id}"
    )


def load_fold_pairs(processed_dir, fold_id):
    """
    读取某一折的 train_pairs.npy 和 test_pairs.npy。

    参数：
        processed_dir:
            processed_direction_dataset 的路径

        fold_id:
            0, 1, 2, 3, 4

    返回：
        train_pairs: np.ndarray, shape = [N_train, 3]
        test_pairs:  np.ndarray, shape = [N_test, 3]

    每一行：
        [circ_index, disease_index, label]
    """
    fold_dir = get_fold_dir(processed_dir, fold_id)

    train_path = os.path.join(fold_dir, "train_pairs.npy")
    test_path = os.path.join(fold_dir, "test_pairs.npy")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练样本: {train_path}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试样本: {test_path}")

    train_pairs = np.load(train_path)
    test_pairs = np.load(test_path)

    check_pairs_array(train_pairs, name=f"fold_{fold_id} train_pairs")
    check_pairs_array(test_pairs, name=f"fold_{fold_id} test_pairs")

    return train_pairs, test_pairs


def load_all_training_pairs(processed_dir, negative_ratio=1.0):
    """
    读取 step5 生成的完整三分类样本表。

    默认文件名：
        step5_training_pairs_neg_ratio_1.0.npy
    """
    path = os.path.join(
        processed_dir,
        "step5_training_samples_2738_275",
        f"step5_training_pairs_neg_ratio_{negative_ratio}.npy"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到完整训练样本文件: {path}")

    pairs = np.load(path)
    check_pairs_array(pairs, name="all_training_pairs")

    return pairs


def check_pairs_array(pairs, name="pairs"):
    """
    检查 pair 数组是否合法。

    pairs:
        shape = [N, 3]
        col0 = circ_index
        col1 = disease_index
        col2 = label
    """
    if not isinstance(pairs, np.ndarray):
        raise TypeError(f"{name} 必须是 numpy.ndarray，但得到 {type(pairs)}")

    if pairs.ndim != 2 or pairs.shape[1] != 3:
        raise ValueError(f"{name} shape 应为 [N, 3]，但得到 {pairs.shape}")

    labels = pairs[:, 2].astype(int)
    bad_labels = sorted(set(labels.tolist()) - {0, 1, 2})

    if len(bad_labels) > 0:
        raise ValueError(f"{name} 中存在非法 label: {bad_labels}")

    return True


# =========================
# 6. numpy pair 转 torch tensor
# =========================
def pairs_to_torch(pairs, device):
    """
    将 [circ_index, disease_index, label] 转成模型输入。

    参数：
        pairs: np.ndarray, shape = [N, 3]
        device: torch.device

    返回：
        pair_indices: torch.LongTensor, shape = [N, 2]
        labels:       torch.LongTensor, shape = [N]

    用法：
        pair_indices, labels = pairs_to_torch(train_pairs, device)
        logits = model(..., pair_indices)
        loss = criterion(logits, labels)
    """
    check_pairs_array(pairs)

    pair_indices = torch.LongTensor(pairs[:, 0:2]).to(device)
    labels = torch.LongTensor(pairs[:, 2]).to(device)

    return pair_indices, labels


def split_pair_array(pairs):
    """
    把 numpy pair 数组拆成 circ_index、disease_index、label。

    返回：
        circ_idx: np.ndarray, shape = [N]
        dis_idx:  np.ndarray, shape = [N]
        labels:   np.ndarray, shape = [N]
    """
    check_pairs_array(pairs)

    circ_idx = pairs[:, 0].astype(np.int64)
    dis_idx = pairs[:, 1].astype(np.int64)
    labels = pairs[:, 2].astype(np.int64)

    return circ_idx, dis_idx, labels


# =========================
# 7. 类别统计与类别权重
# =========================
def count_labels(pairs, num_classes=3):
    """
    统计 pair 数组中的标签数量。
    """
    check_pairs_array(pairs)

    labels = pairs[:, 2].astype(int)
    counts = np.bincount(labels, minlength=num_classes)

    return counts


def compute_class_weights_from_pairs(
    train_pairs,
    num_classes=3,
    device=None,
    mode="balanced"
):
    """
    根据训练集标签数量计算 CrossEntropyLoss 的类别权重。

    当前推荐：
        mode = "balanced"

    公式：
        weight_c = total / (num_classes * count_c)

    对你当前数据，通常会得到类似：
        [0.6667, 0.9177, 2.4371]

    其中 down-regulated 样本少，所以权重大。
    """
    counts = count_labels(train_pairs, num_classes=num_classes)
    safe_counts = np.maximum(counts, 1)

    total = safe_counts.sum()

    if mode == "balanced":
        weights = total / (num_classes * safe_counts)
    elif mode == "sqrt_balanced":
        weights = np.sqrt(total / (num_classes * safe_counts))
    elif mode == "none":
        weights = np.ones(num_classes, dtype=np.float32)
    else:
        raise ValueError(f"不支持的 class weight mode: {mode}")

    weights = weights.astype(np.float32)
    weights_tensor = torch.FloatTensor(weights)

    if device is not None:
        weights_tensor = weights_tensor.to(device)

    return weights_tensor, counts


def print_label_distribution(pairs, name="pairs"):
    """
    打印标签分布。
    """
    counts = count_labels(pairs, num_classes=3)

    print(f"\n{name} 标签分布:")
    for label_id in range(3):
        label_name = LABEL_ID_TO_NAME[label_id]
        print(f"  label {label_id} ({label_name}): {int(counts[label_id])}")

    return counts


# =========================
# 8. 构建每折训练图，避免测试边泄漏
# =========================
def build_train_graph_matrix_for_fold(
    A,
    test_pairs,
    remove_labels=(1, 2),
    verbose=True
):
    """
    为当前 fold 构建训练用 circRNA-disease 二值关联矩阵 A_graph_train。

    背景：
        你的超图 H_circ/H_dis 会使用 A_cd 作为一部分结构先验。
        如果直接用完整 A 构建超图，测试集中原本已知的 up/down pair
        可能已经作为边进入图结构，造成信息泄漏。

    做法：
        A_graph_train = A.copy()
        对 test_pairs 中 label 为 1 或 2 的 pair，将 A_graph_train[r, d] 置 0。

    参数：
        A:
            原始 circRNA-disease 关联矩阵，shape = [R, D]

        test_pairs:
            当前 fold 的测试样本，shape = [N_test, 3]

        remove_labels:
            默认只移除测试集中的上调和下调正方向样本，即 label 1 和 label 2。
            label 0 是 unknown，不需要移除。

    返回：
        A_graph_train: np.ndarray, shape = [R, D]
    """
    A_graph_train = np.asarray(A, dtype=np.float32).copy()

    check_pairs_array(test_pairs, name="test_pairs")

    removed = 0
    already_zero = 0

    for r, d, label in test_pairs.astype(int):
        if int(label) in remove_labels:
            if A_graph_train[r, d] != 0:
                A_graph_train[r, d] = 0.0
                removed += 1
            else:
                already_zero += 1

    if verbose:
        print("\n[Graph Leakage Control]")
        print(f"  从 A_cd 中移除测试正方向边数量: {removed}")
        print(f"  测试正方向边原本已为 0 的数量: {already_zero}")
        print(f"  A 原始边数量: {int(np.sum(np.asarray(A) == 1))}")
        print(f"  A_graph_train 边数量: {int(np.sum(A_graph_train == 1))}")

    return A_graph_train


# =========================
# 9. 多分类 Focal Loss，可选
# =========================
class MulticlassFocalLoss(nn.Module):
    """
    多分类 Focal Loss。

    第一版建议先用 CrossEntropyLoss。
    如果 down-regulated 类召回率很低，可以再尝试该损失。

    参数：
        alpha:
            None 或 shape=[num_classes] 的类别权重 tensor。
            可以直接传 compute_class_weights_from_pairs 得到的 class_weights。

        gamma:
            越大越关注难分类样本，常用 1.0 或 2.0。
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(MulticlassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits:  [B, C]
        targets: [B]
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError(f"不支持的 reduction: {self.reduction}")


def build_criterion(
    train_pairs,
    device,
    loss_type="ce",
    class_weight_mode="balanced",
    focal_gamma=2.0
):
    """
    构建三分类损失函数。

    推荐第一版：
        loss_type = "ce"

    可选：
        loss_type = "focal"

    返回：
        criterion
        class_weights
        class_counts
    """
    class_weights, class_counts = compute_class_weights_from_pairs(
        train_pairs,
        num_classes=3,
        device=device,
        mode=class_weight_mode
    )

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "focal":
        criterion = MulticlassFocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            reduction="mean"
        )
    else:
        raise ValueError(f"不支持的 loss_type: {loss_type}")

    return criterion, class_weights, class_counts


# =========================
# 10. logits / probs / preds 转换
# =========================
def tensor_to_numpy(x):
    """
    torch.Tensor 或 numpy.ndarray 统一转 numpy。
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def logits_to_probs(logits):
    """
    logits -> softmax probabilities。

    支持 torch.Tensor 或 np.ndarray。
    返回 np.ndarray, shape = [N, 3]
    """
    if isinstance(logits, torch.Tensor):
        probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return probs.astype(np.float64)


def probs_to_preds(probs):
    """
    softmax probabilities -> class predictions。
    """
    probs = np.asarray(probs)
    return np.argmax(probs, axis=1).astype(np.int64)


# =========================
# 11. 三分类评价指标
# =========================
def evaluate_multiclass_from_logits(
    y_true,
    logits,
    num_classes=3,
    labels=(0, 1, 2)
):
    """
    根据 logits 计算三分类指标。

    参数：
        y_true:
            shape = [N]，真实标签 0/1/2

        logits:
            shape = [N, 3]，模型输出 logits

    返回：
        metrics: dict
        cm: np.ndarray, shape = [3, 3]
        probs: np.ndarray, shape = [N, 3]
        y_pred: np.ndarray, shape = [N]
    """
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
    """
    根据 softmax 概率计算三分类指标。

    主要指标：
        accuracy
        macro_precision
        macro_recall
        macro_f1
        weighted_f1
        mcc
        macro_auc_ovr
        macro_aupr_ovr

    同时返回每个类别的 precision/recall/f1/support。
    """
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

    # OvR AUC / AUPR
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

    # 每个类别的 OvR AUC / AUPR
    # 对每一类都构造一个二分类任务：
    #   当前类 = 正类
    #   其他类 = 负类
    auc_each = {}
    aupr_each = {}

    for idx, label_id in enumerate(labels):
        label_name = LABEL_ID_TO_NAME[int(label_id)]

        y_true_binary = y_true_bin[:, idx]
        y_score = probs[:, idx]

        # AUC 要求正负样本都存在。
        # 如果某一折中某个类别没有正样本或没有负样本，AUC 没有定义，记为 nan。
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

        # AUPR 也建议在正负样本都存在时计算。
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

    # 每个类别的指标
    # 每个类别的指标
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


def print_metrics(metrics, prefix=""):
    """
    打印常用三分类指标。
    """
    if prefix:
        print(prefix)

    keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "mcc",
        "macro_auc_ovr",
        "weighted_auc_ovr",
        "macro_aupr_ovr",
        "weighted_aupr_ovr",

        "auc_unknown",
        "auc_up-regulated",
        "auc_down-regulated",
        "aupr_unknown",
        "aupr_up-regulated",
        "aupr_down-regulated",

        "precision_unknown",
        "precision_up-regulated",
        "precision_down-regulated",
        "recall_unknown",
        "recall_up-regulated",
        "recall_down-regulated",
        "f1_unknown",
        "f1_up-regulated",
        "f1_down-regulated"
    ]

    for k in keys:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")


def get_classification_report(y_true, y_pred):
    """
    返回 sklearn classification_report 字符串。
    """
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            LABEL_ID_TO_NAME[0],
            LABEL_ID_TO_NAME[1],
            LABEL_ID_TO_NAME[2]
        ],
        digits=6,
        zero_division=0
    )


# =========================
# 12. 保存结果
# =========================
def confusion_matrix_to_dataframe(cm):
    """
    混淆矩阵转成带行列名的 DataFrame。
    行是真实标签，列是预测标签。
    """
    names = [LABEL_ID_TO_NAME[i] for i in range(3)]

    return pd.DataFrame(
        cm,
        index=[f"true_{n}" for n in names],
        columns=[f"pred_{n}" for n in names]
    )


def save_fold_outputs(
    out_dir,
    fold_id,
    metrics,
    cm,
    test_pairs,
    probs,
    y_pred
):
    """
    保存单折预测结果。

    保存内容：
        metrics.json
        metrics.csv
        confusion_matrix.csv
        test_predictions.csv
        classification_report.txt
    """
    ensure_dir(out_dir)

    y_true = test_pairs[:, 2].astype(np.int64)

    # 1. metrics json
    metrics_json_path = os.path.join(out_dir, f"fold_{fold_id}_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # 2. metrics csv
    metrics_csv_path = os.path.join(out_dir, f"fold_{fold_id}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

    # 3. confusion matrix
    cm_df = confusion_matrix_to_dataframe(cm)
    cm_path = os.path.join(out_dir, f"fold_{fold_id}_confusion_matrix.csv")
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    # 4. prediction details
    pred_df = pd.DataFrame({
        "circ_index": test_pairs[:, 0].astype(np.int64),
        "disease_index": test_pairs[:, 1].astype(np.int64),
        "true_label": y_true,
        "true_label_name": [LABEL_ID_TO_NAME[int(x)] for x in y_true],
        "pred_label": y_pred.astype(np.int64),
        "pred_label_name": [LABEL_ID_TO_NAME[int(x)] for x in y_pred],
        "prob_unknown": probs[:, 0],
        "prob_up-regulated": probs[:, 1],
        "prob_down-regulated": probs[:, 2],
    })

    pred_path = os.path.join(out_dir, f"fold_{fold_id}_test_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 5. classification report
    report = get_classification_report(y_true, y_pred)
    report_path = os.path.join(out_dir, f"fold_{fold_id}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "metrics_json": metrics_json_path,
        "metrics_csv": metrics_csv_path,
        "confusion_matrix": cm_path,
        "predictions": pred_path,
        "classification_report": report_path,
    }


def save_all_fold_metrics(fold_metrics, save_path):
    """
    保存五折指标，并追加 mean 和 std 行。

    参数：
        fold_metrics:
            list[dict]，每个 dict 是一折的 metrics，并且建议包含 fold 字段。

        save_path:
            保存路径，例如 results_direction/fold_metrics_5fold.csv
    """
    df = pd.DataFrame(fold_metrics)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    mean_row = {}
    std_row = {}

    for col in df.columns:
        if col in numeric_cols and col != "fold":
            mean_row[col] = df[col].mean()
            std_row[col] = df[col].std()
        elif col == "fold":
            mean_row[col] = "mean"
            std_row[col] = "std"
        else:
            mean_row[col] = ""
            std_row[col] = ""

    df = pd.concat(
        [df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True
    )

    ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else ".")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return df


# =========================
# 13. 训练日志保存
# =========================
def save_loss_curve_csv(loss_records, save_path):
    """
    保存训练 loss 记录。

    loss_records:
        list[dict]
        每个元素例如：
        {
            "epoch": 1,
            "loss": 0.9,
            "cls_loss": 0.8,
            "reg_loss": 0.01,
            ...
        }
    """
    df = pd.DataFrame(loss_records)

    ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else ".")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return df


# =========================
# 14. 简单运行检查
# =========================
def sanity_check_fold_data(processed_dir, fold_id, device=None):
    """
    快速检查某一折数据是否能正常读取，并打印类别权重。

    用法：
        sanity_check_fold_data("./processed_direction_dataset", 0, device)
    """
    train_pairs, test_pairs = load_fold_pairs(processed_dir, fold_id)

    print_label_distribution(train_pairs, name=f"Fold {fold_id} train")
    print_label_distribution(test_pairs, name=f"Fold {fold_id} test")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pair_indices, labels = pairs_to_torch(train_pairs, device)

    class_weights, counts = compute_class_weights_from_pairs(
        train_pairs,
        num_classes=3,
        device=device
    )

    print(f"\nFold {fold_id} tensor shape:")
    print(f"  pair_indices: {tuple(pair_indices.shape)}")
    print(f"  labels:       {tuple(labels.shape)}")

    print(f"\nFold {fold_id} class counts:")
    print(counts)

    print(f"\nFold {fold_id} class weights:")
    print(class_weights)

    return {
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "pair_indices": pair_indices,
        "labels": labels,
        "class_weights": class_weights,
        "class_counts": counts
    }