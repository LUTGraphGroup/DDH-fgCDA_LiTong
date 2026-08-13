#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fold-local, leakage-free training entry point for DDH-fgCDA.

This script deliberately does not load the precomputed integrated similarity
matrices, because they may contain GIP similarities computed from the complete
association matrix.  For every fold it reconstructs all target-dependent
inputs from that fold's training pairs only.
"""

import argparse
import importlib
import json
import os
from pathlib import Path

import colorama
import numpy as np
import pandas as pd
import torch
from colorama import Fore

from hypergrl_module import (
    prepare_hypergraph_features_direction_branches,
    prepare_hypergraph_features_full_direction,
    prepare_hypergraph_features_multi,
)


DATASET_CONFIGS = {
    "603_83": {
        "module": "train_direction_603_83_with_auc_aupr_plot_margin",
        "shape": (603, 83),
        "mirna_count": 190,
        "association": "Association Matrix_603_83.csv",
        "a_cm": "A_cm_603x190.xlsx",
        "a_md": "A_md_190x83.xlsx",
    },
    "2738_275": {
        "module": "train_direction_2738_275_with_auc_aupr_plot_margin",
        "shape": (2738, 275),
        "mirna_count": 364,
        "association": "Association Matrix_2738_275.csv",
        "a_cm": "A_cm_2738_364.xlsx",
        "a_md": "A_md_364_275.xlsx",
    },
}


def _read_matrix_auto(path, expected_shape):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required matrix does not exist: {path}")

    reader = pd.read_excel if path.suffix.lower() in {".xls", ".xlsx"} else pd.read_csv
    attempts = (
        {"header": 0, "index_col": 0},
        {"header": None},
        {"header": 0},
    )
    diagnostics = []
    for kwargs in attempts:
        try:
            matrix = reader(path, **kwargs).values.astype(np.float32)
            diagnostics.append((kwargs, matrix.shape))
            if matrix.shape == expected_shape:
                if not np.isfinite(matrix).all():
                    raise ValueError(f"Matrix contains NaN/Inf: {path}")
                return matrix
        except Exception as exc:
            diagnostics.append((kwargs, repr(exc)))

    raise ValueError(
        f"Cannot read {path} with expected shape {expected_shape}. "
        f"Attempts: {diagnostics}"
    )


def load_leakage_safe_sources(args, config):
    """Load only raw target associations and external miRNA relations."""
    raw_dir = Path(args.raw_data_dir)
    num_circ, num_dis = config["shape"]
    num_mirna = config["mirna_count"]

    association = _read_matrix_auto(
        raw_dir / config["association"], (num_circ, num_dis)
    )
    a_cm = _read_matrix_auto(raw_dir / config["a_cm"], (num_circ, num_mirna))
    a_md = _read_matrix_auto(raw_dir / config["a_md"], (num_mirna, num_dis))

    return (association > 0).astype(np.float32), a_cm, a_md


def build_fold_relation_matrices(train_pairs, test_pairs, shape):
    """Construct binary/up/down matrices exclusively from training positives."""
    num_circ, num_dis = shape
    train_pairs = np.asarray(train_pairs)
    test_pairs = np.asarray(test_pairs)
    if train_pairs.ndim != 2 or train_pairs.shape[1] < 3:
        raise ValueError(f"train_pairs must have shape [N, >=3], got {train_pairs.shape}")
    if test_pairs.ndim != 2 or test_pairs.shape[1] < 3:
        raise ValueError(f"test_pairs must have shape [N, >=3], got {test_pairs.shape}")

    a_train = np.zeros(shape, dtype=np.float32)
    direction_train = np.zeros(shape, dtype=np.int64)

    for row in train_pairs:
        circ_idx, dis_idx, label = map(int, row[:3])
        if not (0 <= circ_idx < num_circ and 0 <= dis_idx < num_dis):
            raise IndexError(f"Training pair index out of range: {row[:3]}")
        if label not in (0, 1, 2):
            raise ValueError(f"Unexpected training label {label}: {row[:3]}")
        if label == 0:
            continue
        old_label = int(direction_train[circ_idx, dis_idx])
        if old_label not in (0, label):
            raise ValueError(
                f"Conflicting direction labels for pair ({circ_idx}, {dis_idx}): "
                f"{old_label} and {label}"
            )
        a_train[circ_idx, dis_idx] = 1.0
        direction_train[circ_idx, dis_idx] = label

    leaked = []
    for row in test_pairs:
        circ_idx, dis_idx, label = map(int, row[:3])
        if label in (1, 2) and a_train[circ_idx, dis_idx] != 0:
            leaked.append((circ_idx, dis_idx, label))
            if len(leaked) >= 10:
                break
    if leaked:
        raise RuntimeError(
            "Positive train/test overlap detected; leakage-free construction aborted. "
            f"Examples: {leaked}"
        )

    a_up = (direction_train == 1).astype(np.float32)
    a_down = (direction_train == 2).astype(np.float32)
    if not np.array_equal(a_train, ((a_up + a_down) > 0).astype(np.float32)):
        raise AssertionError("A_train is inconsistent with A_up/A_down")
    return a_train, a_up, a_down


def gaussian_interaction_profile_similarity(profiles):
    """Compute the standard GIP kernel without an NxNxF broadcast."""
    profiles = np.asarray(profiles, dtype=np.float64)
    squared_norm = np.einsum("ij,ij->i", profiles, profiles)
    mean_squared_norm = float(squared_norm.mean())
    gamma = 1.0 / mean_squared_norm if mean_squared_norm > 0 else 1.0
    distance_squared = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * profiles.dot(profiles.T)
    )
    np.maximum(distance_squared, 0.0, out=distance_squared)
    similarity = np.exp(-gamma * distance_squared)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32), gamma


def cosine_profile_similarity(profiles):
    """External similarity derived only from fold-independent miRNA relations."""
    profiles = np.asarray(profiles, dtype=np.float64)
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    normalized = np.divide(
        profiles,
        norms,
        out=np.zeros_like(profiles),
        where=norms > 0,
    )
    similarity = normalized.dot(normalized.T)
    np.clip(similarity, 0.0, 1.0, out=similarity)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def fuse_fold_similarities(gip_similarity, external_similarity, gip_weight):
    if not 0.0 <= gip_weight <= 1.0:
        raise ValueError("--gip_weight must be in [0, 1]")
    fused = gip_weight * gip_similarity + (1.0 - gip_weight) * external_similarity
    fused = 0.5 * (fused + fused.T)
    np.fill_diagonal(fused, 1.0)
    return fused.astype(np.float32)


def prepare_fold_inputs(base, args, device, a_train, a_up, a_down, a_cm, a_md):
    circ_gip, circ_gamma = gaussian_interaction_profile_similarity(a_train)
    dis_gip, dis_gamma = gaussian_interaction_profile_similarity(a_train.T)

    circ_external = cosine_profile_similarity(a_cm)
    dis_external = cosine_profile_similarity(a_md.T)
    circ_similarity = fuse_fold_similarities(circ_gip, circ_external, args.gip_weight)
    dis_similarity = fuse_fold_similarities(dis_gip, dis_external, args.gip_weight)

    circ_input, dis_input, _, _, _ = base.build_node_input_features(
        A=a_train,
        circSimi_feat=circ_similarity,
        disSimi_feat=dis_similarity,
        A_cm=a_cm,
        A_md=a_md,
    )
    base.adjust_pca_dim(args, circ_input, dis_input)

    common = dict(
        circ_input_feat=circ_input,
        dis_input_feat=dis_input,
        A_cm=a_cm,
        A_md=a_md,
        device=device,
        pca_dim=args.hyper_pca_dim,
        var_show=True,
        w_cm=args.w_cm,
        w_md=args.w_md,
        w_bridge=args.w_bridge,
        use_bridge=True,
        normalize_each_block=True,
    )
    if args.graph_build_mode == "full_direction_prior":
        if args.direction_branch_mode == "three_branch":
            circ_pca, dis_pca, h_circ, h_dis = prepare_hypergraph_features_direction_branches(
                A_cd=a_train,
                A_up=a_up,
                A_down=a_down,
                w_all=args.w_cd,
                w_up=args.w_up,
                w_down=args.w_down,
                **common,
            )
        else:
            circ_pca, dis_pca, h_circ, h_dis = prepare_hypergraph_features_full_direction(
                A_cd=a_train,
                A_up=a_up,
                A_down=a_down,
                w_cd=args.w_cd,
                w_up=args.w_up,
                w_down=args.w_down,
                **common,
            )
    elif args.graph_build_mode == "full_binary_prior":
        circ_pca, dis_pca, h_circ, h_dis = prepare_hypergraph_features_multi(
            A_cd=a_train,
            w_cd=args.w_cd,
            **common,
        )
    else:
        raise ValueError(f"Unsupported graph_build_mode: {args.graph_build_mode}")

    audit = {
        "circ_gip_gamma": circ_gamma,
        "disease_gip_gamma": dis_gamma,
        "gip_weight": float(args.gip_weight),
        "external_similarity": "cosine similarity of A_cm / transpose(A_md) miRNA profiles",
        "precomputed_integrated_similarity_used": False,
    }
    return circ_pca, dis_pca, h_circ, h_dis, circ_similarity, dis_similarity, audit


def save_fold_audit(args, fold_id, train_pairs, test_pairs, a_full, a_train, a_up, a_down, audit):
    fold_dir = Path(args.results_dir) / f"fold_{fold_id + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    test_positive = np.asarray(test_pairs)[:, 2].astype(int) > 0
    test_positive_pairs = np.asarray(test_pairs)[test_positive, :2].astype(int)
    test_edges_present = sum(a_train[r, d] != 0 for r, d in test_positive_pairs)
    payload = {
        "fold": fold_id + 1,
        "train_samples": int(len(train_pairs)),
        "test_samples": int(len(test_pairs)),
        "full_association_edges_for_audit_only": int(a_full.sum()),
        "training_association_edges": int(a_train.sum()),
        "training_up_edges": int(a_up.sum()),
        "training_down_edges": int(a_down.sum()),
        "positive_test_edges_present_in_training_graph": int(test_edges_present),
        **audit,
    }
    with (fold_dir / "leakage_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def build_parser_and_module():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="603_83")
    known, _ = bootstrap.parse_known_args()
    config = DATASET_CONFIGS[known.dataset]
    base = importlib.import_module(config["module"])
    base.parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default=known.dataset)
    base.parser.add_argument(
        "--gip_weight",
        type=float,
        default=0.5,
        help="Weight of fold-specific GIP in GIP/miRNA-profile similarity fusion.",
    )
    base.parser.set_defaults(
        results_dir=f"./results_direction_{known.dataset}_leak_free",
        use_relation_scorer=0,
    )
    return base, config, base.parser


def main():
    colorama.init(autoreset=True)
    base, config, parser = build_parser_and_module()
    args = parser.parse_args()
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    base.set_random_seed(args.seed, deterministic=True)
    base.ensure_dir(args.results_dir)

    print(Fore.CYAN + f"Leakage-free fold-local training: dataset={args.dataset}, device={device}")
    print(Fore.CYAN + "Precomputed integrated similarity matrices will not be loaded.")
    a_full, a_cm, a_md = load_leakage_safe_sources(args, config)

    all_metrics = []
    all_curves = []
    for fold_id in range(5):
        train_pairs, test_pairs = base.load_fold_pairs(args.direction_data_dir, fold_id)
        a_train, a_up, a_down = build_fold_relation_matrices(
            train_pairs, test_pairs, config["shape"]
        )
        prepared = prepare_fold_inputs(
            base, args, device, a_train, a_up, a_down, a_cm, a_md
        )
        circ_pca, dis_pca, h_circ, h_dis, circ_sim, dis_sim, audit = prepared
        save_fold_audit(
            args, fold_id, train_pairs, test_pairs,
            a_full, a_train, a_up, a_down, audit,
        )
        metrics, curves = base.train_one_fold(
            args=args,
            fold_id=fold_id,
            device=device,
            circ_feat_pca=circ_pca,
            dis_feat_pca=dis_pca,
            H_circ=h_circ,
            H_dis=h_dis,
            circSimi_feat=circ_sim,
            disSimi_feat=dis_sim,
        )
        all_metrics.append(metrics)
        all_curves.append(curves)

    metrics_path = os.path.join(args.results_dir, "fold_metrics_5fold_direction.csv")
    metrics_df = base.save_all_fold_metrics(all_metrics, metrics_path)
    for extension in ("png", "svg"):
        base.plot_macro_roc_across_folds(
            all_curves,
            os.path.join(args.results_dir, f"roc_curve_macro_5fold.{extension}"),
            "ROC Curve (5-fold macro-average, OvR)",
        )
        base.plot_macro_pr_across_folds(
            all_curves,
            os.path.join(args.results_dir, f"pr_curve_macro_5fold.{extension}"),
            "PR Curve (5-fold macro-average, OvR)",
        )
    base.print_final_summary(metrics_df)
    print(Fore.GREEN + "Leakage-free five-fold training completed.")


if __name__ == "__main__":
    main()
