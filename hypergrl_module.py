# hypergrl_module.py
# 用于封装超图神经网络模块，供主模型中调用

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional


# hypergrl_module.py 仅替换这个类，其他不动
class SymmetricHypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = nn.LayerNorm(out_channels)

    @torch.no_grad()
    def _compute_H_norm(self, H: torch.Tensor):
        H = H.float()
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        H = H.clamp_min(0.0)

        dv = H.sum(dim=1)   # [V]
        de = H.sum(dim=0)   # [E]

        dv_inv_sqrt = torch.zeros_like(dv)
        mask_v = dv > 0
        dv_inv_sqrt[mask_v] = 1.0 / torch.sqrt(dv[mask_v])

        de_inv = torch.zeros_like(de)
        mask_e = de > 0
        de_inv[mask_e] = 1.0 / de[mask_e]

        H_scaled = H * de_inv.unsqueeze(0)      # [V, E]
        T = H_scaled @ H.t()                    # [V, V]
        H_norm = (dv_inv_sqrt.unsqueeze(1) * T) * dv_inv_sqrt.unsqueeze(0)
        H_norm = torch.nan_to_num(H_norm, nan=0.0, posinf=0.0, neginf=0.0)
        return H_norm

    def forward(self, x, H):
        H_norm = self._compute_H_norm(H)
        x = self.linear(x.float())
        x = H_norm @ x
        x = self.norm(x)
        return F.relu(x)


def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


def _safe_l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def _edge_feature_from_H(H: torch.Tensor, X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    H: [V, E]
    X: [V, d]
    return: [E, d]
    """
    deg_e = H.sum(dim=0, keepdim=True).t().clamp_min(eps)   # [E, 1]
    return (H.t() @ X) / deg_e


def _row_topk_sparsify(H: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    if k is None or k <= 0 or k >= H.size(1):
        return H
    vals, idx = torch.topk(H, k=k, dim=1)
    mask = torch.zeros_like(H)
    mask.scatter_(1, idx, 1.0)
    return H * mask


def _adaptive_topk(num_edges: int, ratio: float = 0.05, min_k: int = 8, max_k: int = 32) -> int:
    k = int(round(num_edges * ratio))
    k = max(1, k)
    return max(min_k, min(max_k, k))



def _binarize_support(H: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (H > eps).float()


def _masked_row_topk(scores: torch.Tensor, mask: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    """
    scores: [V, E]
    mask:   [V, E] in {0,1}
    return: masked top-k scores (others = 0)
    """
    masked_scores = scores * mask
    if k is None or k <= 0 or k >= masked_scores.size(1):
        return masked_scores

    vals, idx = torch.topk(masked_scores, k=k, dim=1)
    keep = torch.zeros_like(masked_scores)
    keep.scatter_(1, idx, 1.0)
    return masked_scores * keep


def _build_local_candidate_mask(H_static: torch.Tensor,
                                sim_scores: torch.Tensor,
                                extra_k: int = 8) -> torch.Tensor:
    """
    H_static:   [V, E]
    sim_scores: [V, E]
    返回局部候选掩码：
    - 保留静态已有连接
    - 再补充每行额外 top-k 候选
    """
    base_mask = _binarize_support(H_static)   # [V, E]

    if extra_k is None or extra_k <= 0:
        return base_mask

    vals, idx = torch.topk(sim_scores, k=min(extra_k, sim_scores.size(1)), dim=1)
    aug_mask = torch.zeros_like(sim_scores)
    aug_mask.scatter_(1, idx, 1.0)

    candidate_mask = torch.clamp(base_mask + aug_mask, min=0.0, max=1.0)
    return candidate_mask

class PairwiseEditGate(nn.Module):
    """
    输入:
        node_q:    [V, d]
        edge_embed:[E, d]
        sim:       [V, E] or None
    输出:
        gate:      [V, E] in [0, 1]
    """
    def __init__(self, feat_dim, hidden_dim=32, init_p=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.edge_proj = nn.Linear(feat_dim, hidden_dim, bias=False)

        # sim 的影响系数
        self.sim_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # 让初始门值可控
        self.bias = nn.Parameter(torch.tensor(_logit(init_p), dtype=torch.float32))

    def forward(self, node_q: torch.Tensor, edge_embed: torch.Tensor,
                sim: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.node_proj(node_q)           # [V, h]
        k = self.edge_proj(edge_embed)       # [E, h]

        logits = (q @ k.t()) / math.sqrt(self.hidden_dim)   # [V, E]

        if sim is not None:
            # sim ∈ [0,1] -> 映射到 [-1,1]
            logits = logits + self.sim_scale * (2.0 * sim - 1.0)

        logits = logits + self.bias
        gate = torch.sigmoid(logits)
        gate = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0)
        return gate


class HypergraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(HypergraphEncoder, self).__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            self.layers.append(SymmetricHypergraphConv(dims[i], dims[i+1]))

    def forward(self, x, H):
        for layer in self.layers:
            x = layer(x, H)
        return x

class ResidualDynamicSideEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 embed_dim=128,
                 hidden_dims=(256, 128),
                 edge_dim=64,
                 tau=0.9,
                 topk_ratio=0.05,
                 min_topk=8,
                 max_topk=32,
                 refine_ratio=0.10,
                 local_extra_k=8,
                 init_eta=0.50,
                 attr_dropout=0.0,
                 edit_gate_dim=32,
                 init_keep=0.60,
                 init_prune=0.05,
                 init_recruit=0.12,
                 init_mix=0.85,
                 prune_floor=0.20,
                 num_coevo_steps=3,
                 node_refine_ratio=0.05,
                 lambda_edit=1e-3,
                 lambda_prior=5e-3,
                 lambda_smooth=1e-3,
                 lambda_consistency=1e-2):
        super().__init__()

        self.tau = tau
        self.topk_ratio = topk_ratio
        self.min_topk = min_topk
        self.max_topk = max_topk
        self.refine_ratio = refine_ratio
        self.local_extra_k = local_extra_k

        # 节点 / 超边原型投影
        self.node_proj = nn.Linear(input_dim, edge_dim, bias=False)
        self.edge_seed_proj = nn.Linear(input_dim, edge_dim, bias=False)
        self.edge_target_proj = nn.Linear(embed_dim, edge_dim, bias=False)
        self.node_target_proj = nn.Linear(embed_dim, edge_dim, bias=False)
        self.attr_dropout = nn.Dropout(attr_dropout)
        self.eta_logit = nn.Parameter(torch.tensor(_logit(init_eta), dtype=torch.float32))
        self.edge_attr_base = None

        # 最终编码器
        self.encoder = HypergraphEncoder(input_dim, list(hidden_dims) + [embed_dim])

        self.prune_floor = prune_floor

        self.num_coevo_steps = max(1, num_coevo_steps)
        self.node_refine_ratio = node_refine_ratio

        self.lambda_edit = lambda_edit
        self.lambda_prior = lambda_prior
        self.lambda_smooth = lambda_smooth
        self.lambda_consistency = lambda_consistency

        # 一步预编码
        self.coevo_convs = nn.ModuleList()
        self.coevo_convs.append(SymmetricHypergraphConv(input_dim, embed_dim))
        for _ in range(self.num_coevo_steps - 1):
            self.coevo_convs.append(SymmetricHypergraphConv(embed_dim, embed_dim))

        # 局部结构编辑门：全部输出 [V, E]
        self.keep_gate = PairwiseEditGate(
            feat_dim=edge_dim,
            hidden_dim=edit_gate_dim,
            init_p=init_keep
        )
        self.prune_gate = PairwiseEditGate(
            feat_dim=edge_dim,
            hidden_dim=edit_gate_dim,
            init_p=init_prune
        )
        self.recruit_gate = PairwiseEditGate(
            feat_dim=edge_dim,
            hidden_dim=edit_gate_dim,
            init_p=init_recruit
        )
        self.mix_gate = PairwiseEditGate(
            feat_dim=edge_dim,
            hidden_dim=edit_gate_dim,
            init_p=init_mix
        )

        # 调试缓存
        self.last_alpha = None
        self.last_beta = None
        self.last_gamma = None
        self.last_eta = None

        self.last_keep_gate = None
        self.last_prune_gate = None
        self.last_recruit_gate = None
        self.last_mix_gate = None

        self.last_H_dynamic = None
        self.last_H_mixed = None
        self.last_consistency = None

        self.last_edit_loss = None
        self.last_prior_loss = None
        self.last_smooth_loss = None
        self.last_structure_loss = None
        self.last_step_graphs = None
        self.last_step_proto = None

    def _edit_hypergraph_once(self,
                              H_base: torch.Tensor,
                              H_anchor: torch.Tensor,
                              H_anchor_support: torch.Tensor,
                              node_q: torch.Tensor,
                              edge_embed: torch.Tensor,
                              sim: torch.Tensor):
        """
        H_base: 当前步编辑基图（上一轮的结果）
        H_anchor: 静态先验骨架
        H_anchor_support: 静态 support
        """
        base_support = _binarize_support(H_base)

        # 候选集合：静态 support + 当前 base_support + 局部 top-k
        candidate_mask = _build_local_candidate_mask(
            H_static=H_anchor_support,
            sim_scores=sim,
            extra_k=self.local_extra_k
        )
        candidate_mask = torch.clamp(candidate_mask + base_support, min=0.0, max=1.0)

        add_mask = (candidate_mask - base_support).clamp_min(0.0)

        # 局部门
        keep_gate = self.keep_gate(node_q, edge_embed, sim) * base_support
        prune_gate = self.prune_gate(node_q, edge_embed, 1.0 - sim) * base_support
        recruit_gate = self.recruit_gate(node_q, edge_embed, sim) * add_mask
        mix_gate = self.mix_gate(node_q, edge_embed, sim)

        # 基图编辑：增强 + 抑制
        keep_factor = 1.0 + keep_gate * sim
        prune_factor = 1.0 - (1.0 - self.prune_floor) * prune_gate
        prune_factor = prune_factor.clamp_min(self.prune_floor)

        H_base_edit = H_base * keep_factor * prune_factor

        # 招募新边
        k = _adaptive_topk(
            num_edges=sim.size(1),
            ratio=self.topk_ratio,
            min_k=self.min_topk,
            max_k=self.max_topk
        )
        add_scores = recruit_gate * sim
        H_add = _masked_row_topk(add_scores, add_mask, k)

        # 动态图
        H_dynamic = H_base_edit + H_add

        # 与静态锚点做局部融合
        H_final = mix_gate * H_anchor + (1.0 - mix_gate) * H_dynamic

        cache = {
            "keep_gate": keep_gate,
            "prune_gate": prune_gate,
            "recruit_gate": recruit_gate,
            "mix_gate": mix_gate,
            "H_add": H_add,
            "H_base_edit": H_base_edit,
            "delta_H": H_final - H_base
        }
        return H_final, H_dynamic, cache

    def _clean_H(self, H: torch.Tensor) -> torch.Tensor:
        H = H.float()
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        H = H.clamp_min(0.0)
        return H

    def _compute_similarity_scores(self, node_q: torch.Tensor, edge_embed: torch.Tensor) -> torch.Tensor:
        edge_embed = _safe_l2_normalize(edge_embed)
        logits = (node_q @ edge_embed.t()) / self.tau
        scores = torch.sigmoid(logits)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores

    def _init_edge_attr_base(self, num_edges: int, edge_dim: int, device: torch.device):
        if (self.edge_attr_base is None) or \
                (self.edge_attr_base.shape[0] != num_edges) or \
                (self.edge_attr_base.shape[1] != edge_dim) or \
                (self.edge_attr_base.device != device):
            param = torch.empty(num_edges, edge_dim, device=device)
            nn.init.xavier_uniform_(param)
            self.edge_attr_base = nn.Parameter(param)

    def _build_edge_attr(self, edge_seed: torch.Tensor) -> torch.Tensor:
        self._init_edge_attr_base(
            num_edges=edge_seed.size(0),
            edge_dim=edge_seed.size(1),
            device=edge_seed.device
        )
        eta = torch.sigmoid(self.eta_logit)
        edge_attr = (1.0 - eta) * edge_seed + eta * self.edge_attr_base
        edge_attr = self.attr_dropout(edge_attr)
        edge_attr = _safe_l2_normalize(edge_attr)
        return edge_attr, eta

    def _compute_structure_losses(self,
                                  H_anchor: torch.Tensor,
                                  H_anchor_support: torch.Tensor,
                                  H_prev: torch.Tensor,
                                  H_curr: torch.Tensor,
                                  cache: dict):
        # 1) 编辑稀疏：希望每一步别改太猛
        loss_edit = cache["delta_H"].abs().mean() + 0.5 * cache["H_add"].abs().mean()

        # 2) 先验保持：在静态 support 上不要偏离太远
        loss_prior = F.mse_loss(
            H_curr * H_anchor_support,
            H_anchor * H_anchor_support
        )

        # 3) 步间平滑：避免相邻两步图结构跳变过大
        loss_smooth = F.mse_loss(H_curr, H_prev.detach())

        return loss_edit, loss_prior, loss_smooth

    def get_weighted_structure_loss(self):
        if self.last_structure_loss is None:
            return None
        return self.last_structure_loss

    def forward(self, x: torch.Tensor, H_static: torch.Tensor) -> torch.Tensor:
        # 1) 静态锚点
        H_anchor = self._clean_H(H_static)
        H_anchor_support = _binarize_support(H_anchor)

        # 2) 初始节点查询
        node_q = _safe_l2_normalize(self.node_proj(x))  # [V, de]

        # 3) 初始超边原型
        edge_seed_feat = _edge_feature_from_H(H_anchor, x)  # [E, d]
        edge_seed = _safe_l2_normalize(self.edge_seed_proj(edge_seed_feat))  # [E, de]
        edge_proto, eta = self._build_edge_attr(edge_seed)

        # 4) 初始化循环状态
        H_prev = H_anchor
        x_state = x

        zero = x.new_zeros(())
        loss_edit_acc = zero.clone()
        loss_prior_acc = zero.clone()
        loss_smooth_acc = zero.clone()
        loss_consistency_acc = zero.clone()

        step_graphs = []
        step_proto = []

        final_cache = None
        final_H_dynamic = None

        # 5) 多步共演化
        for step in range(self.num_coevo_steps):
            # 5.1 当前 prototype 与 node_q 生成当前图
            sim_t = self._compute_similarity_scores(node_q, edge_proto)

            H_curr, H_dynamic, cache = self._edit_hypergraph_once(
                H_base=H_prev,
                H_anchor=H_anchor,
                H_anchor_support=H_anchor_support,
                node_q=node_q,
                edge_embed=edge_proto,
                sim=sim_t
            )

            # 5.2 当前图上做一步卷积，得到当前状态表示
            z_t = self.coevo_convs[step](x_state, H_curr)

            # 5.3 用 z_t 反推下一步的超边目标原型
            edge_target_feat = _edge_feature_from_H(H_curr, z_t)  # [E, embed_dim]
            edge_target = _safe_l2_normalize(self.edge_target_proj(edge_target_feat))  # [E, de]

            # 5.4 用 z_t 反推下一步的节点目标查询
            node_target = _safe_l2_normalize(self.node_target_proj(z_t))  # [V, de]

            # 5.5 原型 / 节点查询渐进更新
            edge_next = _safe_l2_normalize(
                (1.0 - self.refine_ratio) * edge_proto + self.refine_ratio * edge_target
            )
            node_next = _safe_l2_normalize(
                (1.0 - self.node_refine_ratio) * node_q + self.node_refine_ratio * node_target
            )

            # 5.6 累积结构损失
            loss_edit_t, loss_prior_t, loss_smooth_t = self._compute_structure_losses(
                H_anchor=H_anchor,
                H_anchor_support=H_anchor_support,
                H_prev=H_prev,
                H_curr=H_curr,
                cache=cache
            )

            loss_edit_acc = loss_edit_acc + loss_edit_t
            loss_prior_acc = loss_prior_acc + loss_prior_t
            loss_smooth_acc = loss_smooth_acc + loss_smooth_t

            # 5.7 原型一致性损失
            loss_consistency_acc = loss_consistency_acc + F.mse_loss(
                edge_next, edge_target.detach()
            )

            # 5.8 额外的状态平滑：让原型和节点查询不要跳太猛
            loss_smooth_acc = loss_smooth_acc + F.mse_loss(edge_next, edge_proto.detach())
            loss_smooth_acc = loss_smooth_acc + F.mse_loss(node_next, node_q.detach())

            # 5.9 更新下一步状态
            H_prev = H_curr
            x_state = z_t
            edge_proto = edge_next
            node_q = node_next

            step_graphs.append(H_curr.detach())
            step_proto.append(edge_proto.detach())

            final_cache = cache
            final_H_dynamic = H_dynamic

        # 6) 最终图编码
        H_final = H_prev
        z = self.encoder(x, H_final)

        # 7) 平均化各项损失
        denom = float(self.num_coevo_steps)
        loss_edit = loss_edit_acc / denom
        loss_prior = loss_prior_acc / denom
        loss_smooth = loss_smooth_acc / denom
        loss_consistency = loss_consistency_acc / denom

        structure_loss = (
                self.lambda_edit * loss_edit +
                self.lambda_prior * loss_prior +
                self.lambda_smooth * loss_smooth
        )

        # 8) 调试缓存
        self.last_alpha = final_cache["mix_gate"].mean().detach()
        self.last_beta = final_cache["keep_gate"].mean().detach()
        self.last_gamma = final_cache["recruit_gate"].mean().detach()
        self.last_eta = eta.detach()

        self.last_keep_gate = final_cache["keep_gate"].detach()
        self.last_prune_gate = final_cache["prune_gate"].detach()
        self.last_recruit_gate = final_cache["recruit_gate"].detach()
        self.last_mix_gate = final_cache["mix_gate"].detach()

        self.last_H_dynamic = final_H_dynamic.detach()
        self.last_H_mixed = H_final.detach()

        self.last_edit_loss = loss_edit
        self.last_prior_loss = loss_prior
        self.last_smooth_loss = loss_smooth
        self.last_structure_loss = structure_loss
        self.last_consistency = loss_consistency

        self.last_step_graphs = step_graphs
        self.last_step_proto = step_proto

        return z

class HyperGRLEncoder(nn.Module):
    def __init__(self, circ_feat_dim, dis_feat_dim, embed_dim=128):
        super(HyperGRLEncoder, self).__init__()

        self.circ_encoder = ResidualDynamicSideEncoder(
            input_dim=circ_feat_dim,
            embed_dim=embed_dim,
            edge_dim=min(64, circ_feat_dim),
            tau=0.9,
            topk_ratio=0.03,
            min_topk=6,
            max_topk=24,
            refine_ratio=0.08,
            local_extra_k=6,
            init_eta=0.35,
            attr_dropout=0.0,
            edit_gate_dim=32,
            init_keep=0.60,
            init_prune=0.05,
            init_recruit=0.12,
            init_mix=0.88,
            prune_floor=0.20,
            num_coevo_steps=3,
            node_refine_ratio=0.05,
            lambda_edit=1e-3,
            lambda_prior=5e-3,
            lambda_smooth=1e-3,
            lambda_consistency=1e-2
        )

        self.dis_encoder = ResidualDynamicSideEncoder(
            input_dim=dis_feat_dim,
            embed_dim=embed_dim,
            edge_dim=min(64, dis_feat_dim),
            tau=0.9,
            topk_ratio=0.03,
            min_topk=6,
            max_topk=24,
            refine_ratio=0.08,
            local_extra_k=6,
            init_eta=0.35,
            attr_dropout=0.0,
            edit_gate_dim=32,
            init_keep=0.60,
            init_prune=0.05,
            init_recruit=0.12,
            init_mix=0.88,
            prune_floor=0.20,
            num_coevo_steps=3,
            node_refine_ratio=0.05,
            lambda_edit=1e-3,
            lambda_prior=5e-3,
            lambda_smooth=1e-3,
            lambda_consistency=1e-2
        )

    def forward(self, circ_feat, dis_feat, H_circ, H_dis):
        Z_circ = self.circ_encoder(circ_feat, H_circ)
        Z_dis  = self.dis_encoder(dis_feat, H_dis)
        return Z_circ, Z_dis

    def get_consistency_loss(self):
        loss_c = getattr(self.circ_encoder, "last_consistency", None)
        loss_d = getattr(self.dis_encoder, "last_consistency", None)

        if loss_c is None and loss_d is None:
            return None
        if loss_c is None:
            return loss_d
        if loss_d is None:
            return loss_c
        return loss_c + loss_d

    def get_structure_loss(self):
        loss_c = getattr(self.circ_encoder, "last_structure_loss", None)
        loss_d = getattr(self.dis_encoder, "last_structure_loss", None)

        if loss_c is None and loss_d is None:
            return None
        if loss_c is None:
            return loss_d
        if loss_d is None:
            return loss_c
        return loss_c + loss_d

    def get_regularization_loss(self):
        reg = None

        cons = self.get_consistency_loss()
        struct = self.get_structure_loss()

        if cons is not None:
            reg = self.circ_encoder.lambda_consistency * cons

        if struct is not None:
            reg = struct if reg is None else reg + struct

        return reg

# ======================== 数据预处理部分 ========================

def integrate_circ_features(cfs1, clev, cfs2):
    combined = (cfs1 + clev + cfs2) / 3.0
    return combined

def integrate_dis_features(dmesh, dgipk, ddo):
    combined = (dmesh + dgipk + ddo) / 3.0
    return combined

def pca_reduce(features, dim=256, var_show=True):
    pca = PCA(n_components=dim)
    reduced = pca.fit_transform(features)
    if var_show and hasattr(pca, "explained_variance_ratio_"):
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        eff_k = int((evr > 1e-8).sum())
        print(
            f"[PCA] n_components={dim} | "
            f"cum@16={cum[min(15, len(cum)-1)]:.6f} "
            f"cum@32={cum[min(31, len(cum)-1)]:.6f} "
            f"cum@64={cum[min(63, len(cum)-1)]:.6f} "
            f"cum@96={cum[min(95, len(cum)-1)]:.6f} "
            f"cum@128={cum[min(127, len(cum)-1)]:.6f} "
            f"cum@256={cum[min(255, len(cum)-1)]:.6f} "
            f"cum@{dim}={cum[-1]:.6f} | "
            f"effective_k≈{eff_k}"
        )
    print(f"[PCA] reduced dim = {reduced.shape[1]}")
    return torch.tensor(reduced, dtype=torch.float32)


def _to_clean_tensor(x):
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x.clamp_min(0.0)
    return x


def _row_normalize_block(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    row_sum = x.sum(dim=1, keepdim=True).clamp_min(eps)
    return x / row_sum


def build_bridge_matrix(A_cm, A_md, normalize=True):
    """
    A_cm: [603, 190]
    A_md: [190, 83]
    return B_cd: [603, 83]
    """
    A_cm = _to_clean_tensor(A_cm)
    A_md = _to_clean_tensor(A_md)
    B_cd = A_cm @ A_md
    if normalize:
        B_cd = _row_normalize_block(B_cd)
    return B_cd.float()


def build_incidence_matrix(assoc_matrix, mode='circ'):
    assoc_matrix = _to_clean_tensor(assoc_matrix)

    if mode == 'circ':
        H = assoc_matrix.clone()      # [R, D]
    elif mode == 'dis':
        H = assoc_matrix.T.clone()    # [D, R]
    else:
        raise ValueError("mode must be 'circ' or 'dis'")

    return H.float()


def build_multi_relation_incidence(A_cd, A_cm=None, A_md=None, A_bridge=None,
                                   mode='circ',
                                   w_cd=1.0, w_cm=1.0, w_md=1.0, w_bridge=1.0,
                                   normalize_each_block=True):
    A_cd = _to_clean_tensor(A_cd)
    A_cm = _to_clean_tensor(A_cm)
    A_md = _to_clean_tensor(A_md)
    A_bridge = _to_clean_tensor(A_bridge)

    def norm_block(x):
        if x is None:
            return None
        return _row_normalize_block(x) if normalize_each_block else x

    blocks = []

    if mode == 'circ':
        # [603, 83]
        blocks.append(w_cd * norm_block(A_cd))

        # [603, 190]
        if A_cm is not None:
            blocks.append(w_cm * norm_block(A_cm))

        # [603, 83]
        if A_bridge is not None:
            blocks.append(w_bridge * norm_block(A_bridge))

    elif mode == 'dis':
        # [83, 603]
        blocks.append(w_cd * norm_block(A_cd.T))

        # [83, 190]
        if A_md is not None:
            blocks.append(w_md * norm_block(A_md.T))

        # [83, 603]
        if A_bridge is not None:
            blocks.append(w_bridge * norm_block(A_bridge.T))
    else:
        raise ValueError("mode must be 'circ' or 'dis'")

    H = torch.cat(blocks, dim=1)
    return H.float()


def prepare_hypergraph_features_multi(
        circ_input_feat, dis_input_feat,
        A_cd, A_cm, A_md,
        device,
        pca_dim: int = 96,
        var_show: bool = True,
        w_cd: float = 1.0,
        w_cm: float = 1.0,
        w_md: float = 1.0,
        w_bridge: float = 1.0,
        use_bridge: bool = True,
        normalize_each_block: bool = True):

    circ_input_feat = np.asarray(circ_input_feat, dtype=np.float32)
    dis_input_feat  = np.asarray(dis_input_feat, dtype=np.float32)

    circ_feat_pca = pca_reduce(circ_input_feat, dim=pca_dim, var_show=var_show)
    dis_feat_pca  = pca_reduce(dis_input_feat, dim=pca_dim, var_show=var_show)

    A_bridge = None
    if use_bridge and (A_cm is not None) and (A_md is not None):
        A_bridge = build_bridge_matrix(A_cm, A_md, normalize=True)

    H_circ = build_multi_relation_incidence(
        A_cd=A_cd, A_cm=A_cm, A_md=A_md, A_bridge=A_bridge,
        mode='circ',
        w_cd=w_cd, w_cm=w_cm, w_md=w_md, w_bridge=w_bridge,
        normalize_each_block=normalize_each_block
    )
    H_dis = build_multi_relation_incidence(
        A_cd=A_cd, A_cm=A_cm, A_md=A_md, A_bridge=A_bridge,
        mode='dis',
        w_cd=w_cd, w_cm=w_cm, w_md=w_md, w_bridge=w_bridge,
        normalize_each_block=normalize_each_block
    )

    circ_feat_pca = circ_feat_pca.to(torch.float32).to(device)
    dis_feat_pca  = dis_feat_pca.to(torch.float32).to(device)
    H_circ = H_circ.to(torch.float32).to(device)
    H_dis  = H_dis.to(torch.float32).to(device)

    return circ_feat_pca, dis_feat_pca, H_circ, H_dis

#  构建 all / up / down 三个独立 H。
def prepare_hypergraph_features_direction_branches(
        circ_input_feat,
        dis_input_feat,
        A_cd,
        A_up,
        A_down,
        A_cm,
        A_md,
        device,
        pca_dim: int = 96,
        var_show: bool = True,
        w_all: float = 1.0,
        w_up: float = 1.0,
        w_down: float = 1.0,
        w_cm: float = 1.0,
        w_md: float = 1.0,
        w_bridge: float = 1.0,
        use_bridge: bool = True,
        normalize_each_block: bool = True):
    """
    构建方向专属动态超图分支。

    返回:
        circ_feat_pca: [R, pca_dim]
        dis_feat_pca:  [D, pca_dim]

        H_circ_dict:
            {
                "all":  [R, E_all],
                "up":   [R, E_up],
                "down": [R, E_down]
            }

        H_dis_dict:
            {
                "all":  [D, E_all],
                "up":   [D, E_up],
                "down": [D, E_down]
            }

    三个分支含义:
        all  分支: [A_cd   | A_cm | A_bridge]
        up   分支: [A_up   | A_cm | A_bridge]
        down 分支: [A_down | A_cm | A_bridge]
    """
    circ_input_feat = np.asarray(circ_input_feat, dtype=np.float32)
    dis_input_feat = np.asarray(dis_input_feat, dtype=np.float32)

    # 1. PCA 只做一次，三个分支共享同一套节点输入特征
    circ_feat_pca = pca_reduce(
        circ_input_feat,
        dim=pca_dim,
        var_show=var_show
    )

    dis_feat_pca = pca_reduce(
        dis_input_feat,
        dim=pca_dim,
        var_show=var_show
    )

    # 2. miRNA 桥接矩阵只构建一次
    A_bridge = None
    if use_bridge and (A_cm is not None) and (A_md is not None):
        A_bridge = build_bridge_matrix(
            A_cm,
            A_md,
            normalize=True
        )

    # 3. all 分支：普通关联结构 + miRNA 桥接
    H_circ_all = build_multi_relation_incidence(
        A_cd=A_cd,
        A_cm=A_cm,
        A_md=A_md,
        A_bridge=A_bridge,
        mode='circ',
        w_cd=w_all,
        w_cm=w_cm,
        w_md=w_md,
        w_bridge=w_bridge,
        normalize_each_block=normalize_each_block
    )

    H_dis_all = build_multi_relation_incidence(
        A_cd=A_cd,
        A_cm=A_cm,
        A_md=A_md,
        A_bridge=A_bridge,
        mode='dis',
        w_cd=w_all,
        w_cm=w_cm,
        w_md=w_md,
        w_bridge=w_bridge,
        normalize_each_block=normalize_each_block
    )

    # 4. up 分支：上调结构 + miRNA 桥接
    H_circ_up = build_multi_relation_incidence(
        A_cd=A_up,
        A_cm=A_cm,
        A_md=A_md,
        A_bridge=A_bridge,
        mode='circ',
        w_cd=w_up,
        w_cm=w_cm,
        w_md=w_md,
        w_bridge=w_bridge,
        normalize_each_block=normalize_each_block
    )

    H_dis_up = build_multi_relation_incidence(
        A_cd=A_up,
        A_cm=A_cm,
        A_md=A_md,
        A_bridge=A_bridge,
        mode='dis',
        w_cd=w_up,
        w_cm=w_cm,
        w_md=w_md,
        w_bridge=w_bridge,
        normalize_each_block=normalize_each_block
    )

    # 5. down 分支：下调结构 + miRNA 桥接
    H_circ_down = build_multi_relation_incidence(
        A_cd=A_down,
        A_cm=A_cm,
        A_md=A_md,
        A_bridge=A_bridge,
        mode='circ',
        w_cd=w_down,
        w_cm=w_cm,
        w_md=w_md,
        w_bridge=w_bridge,
        normalize_each_block=normalize_each_block
    )

    H_dis_down = build_multi_relation_incidence(
        A_cd=A_down,
        A_cm=A_cm,
        A_md=A_md,
        A_bridge=A_bridge,
        mode='dis',
        w_cd=w_down,
        w_cm=w_cm,
        w_md=w_md,
        w_bridge=w_bridge,
        normalize_each_block=normalize_each_block
    )

    # 6. 移动到 device
    circ_feat_pca = circ_feat_pca.to(torch.float32).to(device)
    dis_feat_pca = dis_feat_pca.to(torch.float32).to(device)

    H_circ_dict = {
        "all": H_circ_all.to(torch.float32).to(device),
        "up": H_circ_up.to(torch.float32).to(device),
        "down": H_circ_down.to(torch.float32).to(device),
    }

    H_dis_dict = {
        "all": H_dis_all.to(torch.float32).to(device),
        "up": H_dis_up.to(torch.float32).to(device),
        "down": H_dis_down.to(torch.float32).to(device),
    }

    return circ_feat_pca, dis_feat_pca, H_circ_dict, H_dis_dict

def build_full_direction_incidence(
        A_cd,
        A_up=None,
        A_down=None,
        A_cm=None,
        A_md=None,
        A_bridge=None,
        mode='circ',
        w_cd=1.0,
        w_up=1.5,
        w_down=2.0,
        w_cm=1.0,
        w_md=1.0,
        w_bridge=1.0,
        normalize_each_block=True):
    """
    全图方向先验超图构建。

    A_cd:
        [R, D] 普通 circRNA-disease 关联矩阵。

    A_up:
        [R, D] 上调方向矩阵，direction_label_matrix == 1。

    A_down:
        [R, D] 下调方向矩阵，direction_label_matrix == 2。

    mode='circ':
        返回 H_circ，shape = [R, E_circ]

    mode='dis':
        返回 H_dis，shape = [D, E_dis]
    """
    A_cd = _to_clean_tensor(A_cd)
    A_up = _to_clean_tensor(A_up)
    A_down = _to_clean_tensor(A_down)
    A_cm = _to_clean_tensor(A_cm)
    A_md = _to_clean_tensor(A_md)
    A_bridge = _to_clean_tensor(A_bridge)

    def norm_block(x):
        if x is None:
            return None
        return _row_normalize_block(x) if normalize_each_block else x

    blocks = []

    if mode == 'circ':
        # 1) 普通 circRNA-disease 关联结构
        if A_cd is not None:
            blocks.append(w_cd * norm_block(A_cd))

        # 2) 上调方向结构
        if A_up is not None:
            blocks.append(w_up * norm_block(A_up))

        # 3) 下调方向结构
        if A_down is not None:
            blocks.append(w_down * norm_block(A_down))

        # 4) circRNA-miRNA 结构
        if A_cm is not None:
            blocks.append(w_cm * norm_block(A_cm))

        # 5) circRNA-miRNA-disease 桥接结构
        if A_bridge is not None:
            blocks.append(w_bridge * norm_block(A_bridge))

    elif mode == 'dis':
        # 1) disease-circRNA 普通关联结构
        if A_cd is not None:
            blocks.append(w_cd * norm_block(A_cd.T))

        # 2) disease-circRNA 上调方向结构
        if A_up is not None:
            blocks.append(w_up * norm_block(A_up.T))

        # 3) disease-circRNA 下调方向结构
        if A_down is not None:
            blocks.append(w_down * norm_block(A_down.T))

        # 4) disease-miRNA 结构
        if A_md is not None:
            blocks.append(w_md * norm_block(A_md.T))

        # 5) disease-miRNA-circRNA 桥接结构
        if A_bridge is not None:
            blocks.append(w_bridge * norm_block(A_bridge.T))

    else:
        raise ValueError("mode must be 'circ' or 'dis'")

    if len(blocks) == 0:
        raise ValueError("没有任何可用于构建超图的结构块。")

    H = torch.cat(blocks, dim=1)
    H = torch.nan_to_num(H.float(), nan=0.0, posinf=0.0, neginf=0.0)
    H = H.clamp_min(0.0)

    return H

def prepare_hypergraph_features_full_direction(
        circ_input_feat,
        dis_input_feat,
        A_cd,
        A_up,
        A_down,
        A_cm,
        A_md,
        device,
        pca_dim: int = 96,
        var_show: bool = True,
        w_cd: float = 1.0,
        w_up: float = 1.5,
        w_down: float = 2.0,
        w_cm: float = 1.0,
        w_md: float = 1.0,
        w_bridge: float = 1.0,
        use_bridge: bool = True,
        normalize_each_block: bool = True):
    """
    全图方向先验构图版本。

    与 prepare_hypergraph_features_multi 的区别：
    原来只使用 A_cd；
    现在额外使用 A_up 和 A_down 作为方向结构块。

    返回：
        circ_feat_pca: [R, pca_dim]
        dis_feat_pca:  [D, pca_dim]
        H_circ:        [R, E_circ]
        H_dis:         [D, E_dis]
    """
    circ_input_feat = np.asarray(circ_input_feat, dtype=np.float32)
    dis_input_feat = np.asarray(dis_input_feat, dtype=np.float32)

    circ_feat_pca = pca_reduce(
        circ_input_feat,
        dim=pca_dim,
        var_show=var_show
    )

    dis_feat_pca = pca_reduce(
        dis_input_feat,
        dim=pca_dim,
        var_show=var_show
    )

    A_bridge = None
    if use_bridge and (A_cm is not None) and (A_md is not None):
        A_bridge = build_bridge_matrix(
            A_cm,
            A_md,
            normalize=True
        )

    H_circ = build_full_direction_incidence(
        A_cd=A_cd,
        A_up=A_up,
        A_down=A_down,
        A_cm=A_cm,
        A_md=A_md,
        A_bridge=A_bridge,
        mode='circ',
        w_cd=w_cd,
        w_up=w_up,
        w_down=w_down,
        w_cm=w_cm,
        w_md=w_md,
        w_bridge=w_bridge,
        normalize_each_block=normalize_each_block
    )

    H_dis = build_full_direction_incidence(
        A_cd=A_cd,
        A_up=A_up,
        A_down=A_down,
        A_cm=A_cm,
        A_md=A_md,
        A_bridge=A_bridge,
        mode='dis',
        w_cd=w_cd,
        w_up=w_up,
        w_down=w_down,
        w_cm=w_cm,
        w_md=w_md,
        w_bridge=w_bridge,
        normalize_each_block=normalize_each_block
    )

    circ_feat_pca = circ_feat_pca.to(torch.float32).to(device)
    dis_feat_pca = dis_feat_pca.to(torch.float32).to(device)
    H_circ = H_circ.to(torch.float32).to(device)
    H_dis = H_dis.to(torch.float32).to(device)

    return circ_feat_pca, dis_feat_pca, H_circ, H_dis