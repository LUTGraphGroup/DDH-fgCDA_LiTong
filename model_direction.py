# -*- coding: utf-8 -*-
# @File : model_direction.py
# 细粒度 circRNA-disease 表达方向预测模型
# label:
#   0 = unknown / unconfirmed
#   1 = up-regulated
#   2 = down-regulated

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from hypergrl_module import HyperGRLEncoder


def build_pair_features_from_indices(final_embedding, pair_indices, num_circ):
    """
    根据 pair_indices 构造 circRNA-disease pair 特征。

    参数：
        final_embedding: [R + D, E]
            前 R 行是 circRNA 表示，后 D 行是 disease 表示。

        pair_indices: [B, 2]
            第 0 列是 circ_index
            第 1 列是 disease_index

        num_circ:
            circRNA 数量 R

    返回：
        pair_feat: [B, 4E]
            [zc, zd, |zc - zd|, zc * zd]

        zc_pair: [B, E]
            当前 batch 中 circRNA 节点表示

        zd_pair: [B, E]
            当前 batch 中 disease 节点表示
    """
    assert pair_indices.dim() == 2 and pair_indices.size(1) == 2, \
        f"pair_indices shape 应为 [B, 2]，但得到 {tuple(pair_indices.shape)}"

    circ_idx = pair_indices[:, 0].long()
    dis_idx = pair_indices[:, 1].long()

    Zc = final_embedding[:num_circ, :]
    Zd = final_embedding[num_circ:, :]

    zc_pair = Zc.index_select(0, circ_idx)
    zd_pair = Zd.index_select(0, dis_idx)

    pair_feat = torch.cat(
        [
            zc_pair,
            zd_pair,
            torch.abs(zc_pair - zd_pair),
            zc_pair * zd_pair
        ],
        dim=-1
    )

    return pair_feat, zc_pair, zd_pair


class ExpertRoutedPairHead(nn.Module):
    """
    专家路由 pair 分类头。

    与原二分类版本相比，核心变化是：
    原来每个 expert 输出 1 个 logit；
    现在每个 expert 输出 num_classes 个 logits。

    输入：
        x: [B, 4E]
           [zc, zd, |zc-zd|, zc*zd]

    输出：
        logits: [B, num_classes]
    """
    def __init__(self,
                 embed_dim,
                 hidden_dim=256,
                 gate_hidden=128,
                 dropout_rate=0.5,
                 num_experts=4,
                 topk_experts=2,
                 num_classes=3):
        super(ExpertRoutedPairHead, self).__init__()

        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.topk_experts = min(topk_experts, num_experts)
        self.num_classes = num_classes

        def make_branch():
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )

        # 四路异质证据编码：
        # 1) circRNA 表示
        # 2) disease 表示
        # 3) 差异信息 |zc-zd|
        # 4) 乘积交互 zc*zd
        self.circ_branch = make_branch()
        self.dis_branch = make_branch()
        self.diff_branch = make_branch()
        self.prod_branch = make_branch()

        self.role_gate_net = nn.Sequential(
            nn.Linear(embed_dim * 4, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gate_hidden, 4)
        )

        expert_in_dim = embed_dim * 3

        # 专家路由器
        self.router = nn.Sequential(
            nn.Linear(expert_in_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gate_hidden, num_experts)
        )

        # 多个专家：
        # 原来最后是 nn.Linear(hidden_dim, 1)
        # 现在改成 nn.Linear(hidden_dim, num_classes)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes)
            )
            for _ in range(num_experts)
        ])

        # 缓存门控权重，方便后续打印和可视化
        self.last_gate = None
        self.last_expert_gate = None
        self.last_router_aux_loss = None

    def _topk_softmax(self, logits):
        """
        logits: [B, M]
        仅保留 top-k expert，再 softmax。
        """
        if self.topk_experts is None or self.topk_experts >= self.num_experts:
            return torch.softmax(logits, dim=-1)

        topv, topi = torch.topk(logits, k=self.topk_experts, dim=-1)

        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(1, topi, topv)

        return torch.softmax(masked, dim=-1)

    def _router_balance_loss(self, route_prob):
        """
        专家负载均衡损失，避免所有样本都只走一个 expert。
        route_prob: [B, M]
        """
        mean_prob = route_prob.mean(dim=0)
        target = torch.full_like(mean_prob, 1.0 / self.num_experts)
        return F.mse_loss(mean_prob, target)

    def forward(self, x):
        """
        x: [B, 4E]

        返回：
            logits: [B, 3]
        """
        E = self.embed_dim

        zc = x[:, 0:E]
        zd = x[:, E:2 * E]
        diff = x[:, 2 * E:3 * E]
        prod = x[:, 3 * E:4 * E]

        # 四路证据编码
        hc = self.circ_branch(zc)
        hd = self.dis_branch(zd)
        hh = self.diff_branch(diff)
        hp = self.prod_branch(prod)

        # 四路 role-aware gate
        role_alpha = torch.softmax(self.role_gate_net(x), dim=-1)  # [B, 4]

        fused = (
            role_alpha[:, 0:1] * hc +
            role_alpha[:, 1:2] * hd +
            role_alpha[:, 2:3] * hh +
            role_alpha[:, 3:4] * hp
        )

        # 保留显式差异和乘积证据
        expert_in = torch.cat([fused, diff, prod], dim=-1)  # [B, 3E]

        # expert routing
        router_logits = self.router(expert_in)             # [B, M]
        expert_alpha = self._topk_softmax(router_logits)   # [B, M]

        # 每个 expert 输出 [B, 3]
        # stack 后为 [B, M, 3]
        expert_logits = torch.stack(
            [expert(expert_in) for expert in self.experts],
            dim=1
        )

        # 按 expert 权重融合，得到最终三分类 logits
        logits = (expert_alpha.unsqueeze(-1) * expert_logits).sum(dim=1)  # [B, 3]

        self.last_gate = role_alpha.detach()
        self.last_expert_gate = expert_alpha.detach()
        self.last_router_aux_loss = self._router_balance_loss(expert_alpha)

        return logits

class HierarchicalExpertRoutedPairHead(nn.Module):
    """
    层次化专家路由 pair 分类头。

    与普通三分类头不同：
    1) known_logits: [B, 2]
       第 0 类 = unknown
       第 1 类 = directional，即 up/down

    2) direction_logits: [B, 2]
       第 0 类 = up-regulated
       第 1 类 = down-regulated

    3) final_logits: [B, 3]
       第 0 类 = unknown
       第 1 类 = up-regulated
       第 2 类 = down-regulated

    训练时：
       known_logits 对所有样本计算 loss；
       direction_logits 只对 label=1/2 的样本计算 loss；
       final_logits 可作为辅助三分类 loss。
    """

    def __init__(self,
                 embed_dim,
                 hidden_dim=256,
                 gate_hidden=128,
                 dropout_rate=0.5,
                 num_experts=4,
                 topk_experts=2):
        super(HierarchicalExpertRoutedPairHead, self).__init__()

        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.topk_experts = min(topk_experts, num_experts)

        def make_branch():
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )

        # 四路 pair 证据
        self.circ_branch = make_branch()
        self.dis_branch = make_branch()
        self.diff_branch = make_branch()
        self.prod_branch = make_branch()

        # 四路证据门控
        self.role_gate_net = nn.Sequential(
            nn.Linear(embed_dim * 4, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gate_hidden, 4)
        )

        expert_in_dim = embed_dim * 3

        # expert router
        self.router = nn.Sequential(
            nn.Linear(expert_in_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gate_hidden, num_experts)
        )

        # 每个 expert 输出 4 个 logit：
        # 前 2 个用于 known/unknown；
        # 后 2 个用于 up/down。
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 4)
            )
            for _ in range(num_experts)
        ])

        self.last_gate = None
        self.last_expert_gate = None
        self.last_router_aux_loss = None

    def _topk_softmax(self, logits):
        if self.topk_experts is None or self.topk_experts >= self.num_experts:
            return torch.softmax(logits, dim=-1)

        topv, topi = torch.topk(logits, k=self.topk_experts, dim=-1)

        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(1, topi, topv)

        return torch.softmax(masked, dim=-1)

    def _router_balance_loss(self, route_prob):
        mean_prob = route_prob.mean(dim=0)
        target = torch.full_like(mean_prob, 1.0 / self.num_experts)
        return F.mse_loss(mean_prob, target)

    def _compose_three_class_logits(self, known_logits, direction_logits):
        """
        将两个二分类任务组合成三分类 logits。

        known_logits:
            [B, 2]
            col0 = unknown
            col1 = directional

        direction_logits:
            [B, 2]
            col0 = up
            col1 = down

        final_logits:
            [B, 3]
            col0 = unknown
            col1 = up
            col2 = down
        """
        unknown_logit = known_logits[:, 0:1]

        # up/down 必须同时满足 directional 和具体方向
        up_logit = known_logits[:, 1:2] + direction_logits[:, 0:1]
        down_logit = known_logits[:, 1:2] + direction_logits[:, 1:2]

        final_logits = torch.cat(
            [unknown_logit, up_logit, down_logit],
            dim=1
        )

        return final_logits

    def forward(self, x, return_aux=False):
        """
        x: [B, 4E]

        返回：
            默认返回 final_logits: [B, 3]

            return_aux=True 时返回 dict:
                {
                    "final_logits": [B, 3],
                    "known_logits": [B, 2],
                    "direction_logits": [B, 2]
                }
        """
        E = self.embed_dim

        zc = x[:, 0:E]
        zd = x[:, E:2 * E]
        diff = x[:, 2 * E:3 * E]
        prod = x[:, 3 * E:4 * E]

        hc = self.circ_branch(zc)
        hd = self.dis_branch(zd)
        hh = self.diff_branch(diff)
        hp = self.prod_branch(prod)

        role_alpha = torch.softmax(self.role_gate_net(x), dim=-1)

        fused = (
            role_alpha[:, 0:1] * hc +
            role_alpha[:, 1:2] * hd +
            role_alpha[:, 2:3] * hh +
            role_alpha[:, 3:4] * hp
        )

        expert_in = torch.cat([fused, diff, prod], dim=-1)

        router_logits = self.router(expert_in)
        expert_alpha = self._topk_softmax(router_logits)

        expert_outputs = torch.stack(
            [expert(expert_in) for expert in self.experts],
            dim=1
        )  # [B, M, 4]

        out = (expert_alpha.unsqueeze(-1) * expert_outputs).sum(dim=1)  # [B, 4]

        known_logits = out[:, 0:2]
        direction_logits = out[:, 2:4]

        final_logits = self._compose_three_class_logits(
            known_logits=known_logits,
            direction_logits=direction_logits
        )

        self.last_gate = role_alpha.detach()
        self.last_expert_gate = expert_alpha.detach()
        self.last_router_aux_loss = self._router_balance_loss(expert_alpha)

        if return_aux:
            return {
                "final_logits": final_logits,
                "known_logits": known_logits,
                "direction_logits": direction_logits
            }

        return final_logits

class DirectionBranchFusion(nn.Module):
    """
    方向专属动态超图分支融合模块。

    输入:
        z_all:  [N, E]
        z_up:   [N, E]
        z_down: [N, E]

    输出:
        z_fused: [N, E]
        alpha:   [N, 3]
            每个节点对 all / up / down 三个分支的自适应权重。
    """
    def __init__(
            self,
            embed_dim,
            hidden_dim=128,
            dropout_rate=0.1,
            tau=2.0,
            residual_scale=0.1):
        super(DirectionBranchFusion, self).__init__()

        self.embed_dim = embed_dim
        self.tau = tau
        self.residual_scale = residual_scale

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 3)
        )

        self.last_alpha = None

    def forward(self, z_all, z_up, z_down, return_alpha=False):
        assert z_all.shape == z_up.shape == z_down.shape, \
            "DirectionBranchFusion: 三个分支的 embedding shape 必须一致。"

        # 1. 先做 LayerNorm，避免某个分支因为尺度大而主导融合
        z_all_n = F.layer_norm(z_all, z_all.shape[-1:])
        z_up_n = F.layer_norm(z_up, z_up.shape[-1:])
        z_down_n = F.layer_norm(z_down, z_down.shape[-1:])

        # 2. 节点级分支门控
        z_cat = torch.cat(
            [z_all_n, z_up_n, z_down_n],
            dim=-1
        )

        gate_logits = self.gate(z_cat)
        alpha = torch.softmax(gate_logits / self.tau, dim=-1)

        # 3. 三分支加权融合
        z_fused = (
            alpha[:, 0:1] * z_all_n
            + alpha[:, 1:2] * z_up_n
            + alpha[:, 2:3] * z_down_n
        )

        # 4. 加一个弱残差，防止门控早期过度偏向某一分支
        z_res = (z_all_n + z_up_n + z_down_n) / 3.0
        z_fused = z_fused + self.residual_scale * z_res

        self.last_alpha = alpha.detach()

        if return_alpha:
            return z_fused, alpha

        return z_fused

def _inverse_sigmoid(p, eps=1e-4):
    """
    用于把 alpha 初始值转换成可学习 logit。
    """
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class RelationAwareScorer(nn.Module):
    """
    关系感知三分类打分头。

    为 unknown / up-regulated / down-regulated 分别学习一个关系向量，
    用 DistMult 风格计算每个 pair 属于三类关系的分数。

    输入:
        zc_pair: [B, E]
        zd_pair: [B, E]

    输出:
        rel_logits: [B, 3]
            col0 = unknown
            col1 = up-regulated
            col2 = down-regulated
    """
    def __init__(
            self,
            embed_dim,
            num_classes=3,
            dropout_rate=0.1,
            use_layer_norm=True,
            use_scale=True):
        super(RelationAwareScorer, self).__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_scale = use_scale

        self.relation_emb = nn.Parameter(
            torch.empty(num_classes, embed_dim)
        )

        self.bias = nn.Parameter(
            torch.zeros(num_classes)
        )

        self.dropout = nn.Dropout(dropout_rate)

        if use_layer_norm:
            self.norm_circ = nn.LayerNorm(embed_dim)
            self.norm_dis = nn.LayerNorm(embed_dim)
        else:
            self.norm_circ = nn.Identity()
            self.norm_dis = nn.Identity()

        nn.init.xavier_uniform_(self.relation_emb)

        self.last_rel_logits = None

    def forward(self, zc_pair, zd_pair):
        """
        zc_pair: [B, E]
        zd_pair: [B, E]
        """
        zc = self.norm_circ(zc_pair)
        zd = self.norm_dis(zd_pair)

        zc = self.dropout(zc)
        zd = self.dropout(zd)

        # [B, 1, E] * [1, C, E] * [B, 1, E] -> [B, C, E]
        score = (
            zc.unsqueeze(1)
            * self.relation_emb.unsqueeze(0)
            * zd.unsqueeze(1)
        )

        # [B, C]
        rel_logits = score.sum(dim=-1)

        if self.use_scale:
            rel_logits = rel_logits / math.sqrt(self.embed_dim)

        rel_logits = rel_logits + self.bias

        self.last_rel_logits = rel_logits.detach()

        return rel_logits

class DirectionContrastiveProjector(nn.Module):
    """
    方向监督对比学习投影头。

    输入:
        pair_feat: [B, 4E]
            [zc, zd, |zc-zd|, zc*zd]

    输出:
        contrast_z: [B, contrast_dim]
            L2 normalize 后的 pair-level 表示，用于 SupCon loss。
    """
    def __init__(
            self,
            input_dim,
            hidden_dim=256,
            output_dim=128,
            dropout_rate=0.1):
        super(DirectionContrastiveProjector, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, pair_feat):
        z = self.net(pair_feat)
        z = F.normalize(z, p=2, dim=-1)
        return z

class DDHfgCDA(nn.Module):
    """
    面向上调/下调细粒度预测的 DDH-fgCDA 模型。

    与原 DDH-fgCDA 的主要区别：
    1. 不再在模型内部调用 train_features_choose 做二分类负采样。
    2. forward 直接接收 pair_indices。
    3. 输出 [B, 3] logits。
    4. 保留动态超图编码器和 miRNA 桥接超图结构。
    """

    def __init__(self,
                 in_circfeat_size,
                 in_disfeat_size,
                 outfeature_size,
                 drop_rate,
                 negative_times=None,
                 hyper_pca_dim=96,
                 pair_num_experts=4,
                 pair_topk_experts=2,
                 num_classes=3,
                 use_direction_branches=True,
                 branch_fusion_hidden=128,
                 branch_fusion_tau=2.0,
                 branch_residual_scale=0.1,
                 use_relation_scorer=False,
                 rel_alpha_init=0.3,
                 rel_alpha_learnable=True,
                 rel_scorer_dropout=None,
                 use_contrastive_projector=True,
                 contrast_dim=128,
                 contrast_hidden_dim=256,
                 contrast_dropout=None):

        super(DDHfgCDA, self).__init__()

        self.in_circfeat_size = in_circfeat_size
        self.in_disfeat_size = in_disfeat_size
        self.outfeature_size = outfeature_size
        self.drop_rate = drop_rate

        # 这个参数为了兼容你原来的初始化接口保留，但三分类版本不再使用动态负采样
        self.negative_times = negative_times

        self.num_classes = num_classes

        # 超图配置
        self.use_hyper = True
        self.hyper_mode = "both"
        self.hyper_pca_dim = hyper_pca_dim

        self.use_direction_branches = use_direction_branches
        self.last_used_direction_branches = False

        # 单分支动态超图编码器：用于兼容原来的 H_circ/H_dis Tensor 输入
        self.hyper_encoder = HyperGRLEncoder(
            circ_feat_dim=self.hyper_pca_dim,
            dis_feat_dim=self.hyper_pca_dim,
            embed_dim=self.outfeature_size
        )

        # 三个方向专属动态超图分支：
        # all  分支学习普通关联高阶结构；
        # up   分支学习上调方向高阶结构；
        # down 分支学习下调方向高阶结构。
        self.hyper_encoder_all = HyperGRLEncoder(
            circ_feat_dim=self.hyper_pca_dim,
            dis_feat_dim=self.hyper_pca_dim,
            embed_dim=self.outfeature_size
        )

        self.hyper_encoder_up = HyperGRLEncoder(
            circ_feat_dim=self.hyper_pca_dim,
            dis_feat_dim=self.hyper_pca_dim,
            embed_dim=self.outfeature_size
        )

        self.hyper_encoder_down = HyperGRLEncoder(
            circ_feat_dim=self.hyper_pca_dim,
            dis_feat_dim=self.hyper_pca_dim,
            embed_dim=self.outfeature_size
        )

        # circRNA 侧和 disease 侧分别做方向分支融合
        self.circ_branch_fusion = DirectionBranchFusion(
            embed_dim=self.outfeature_size,
            hidden_dim=branch_fusion_hidden,
            dropout_rate=self.drop_rate,
            tau=branch_fusion_tau,
            residual_scale=branch_residual_scale
        )

        self.dis_branch_fusion = DirectionBranchFusion(
            embed_dim=self.outfeature_size,
            hidden_dim=branch_fusion_hidden,
            dropout_rate=self.drop_rate,
            tau=branch_fusion_tau,
            residual_scale=branch_residual_scale
        )
        # 层次化细粒度专家路由 pair head
        # known/unknown 和 up/down 分开学习，
        # 最终仍然组合成 [B, 3] logits。
        self.mlp_prediction = HierarchicalExpertRoutedPairHead(
            embed_dim=self.outfeature_size,
            hidden_dim=256,
            gate_hidden=128,
            dropout_rate=self.drop_rate,
            num_experts=pair_num_experts,
            topk_experts=pair_topk_experts
        )

        # 关系感知打分头：
        # 在原有层次化专家头之外，额外学习 unknown / up / down 三种关系模式。
        self.use_relation_scorer = bool(use_relation_scorer)

        if rel_scorer_dropout is None:
            rel_scorer_dropout = self.drop_rate

        if self.use_relation_scorer:
            self.relation_scorer = RelationAwareScorer(
                embed_dim=self.outfeature_size,
                num_classes=self.num_classes,
                dropout_rate=rel_scorer_dropout,
                use_layer_norm=True,
                use_scale=True
            )

            rel_alpha_init = min(max(float(rel_alpha_init), 1e-4), 1.0 - 1e-4)

            if rel_alpha_learnable:
                self.rel_alpha_logit = nn.Parameter(
                    torch.tensor(
                        _inverse_sigmoid(rel_alpha_init),
                        dtype=torch.float32
                    )
                )
                self.register_buffer(
                    "rel_alpha_fixed",
                    torch.tensor(rel_alpha_init, dtype=torch.float32)
                )
                self.rel_alpha_learnable = True
            else:
                self.register_buffer(
                    "rel_alpha_fixed",
                    torch.tensor(rel_alpha_init, dtype=torch.float32)
                )
                self.rel_alpha_learnable = False
        else:
            self.relation_scorer = None
            self.register_buffer(
                "rel_alpha_fixed",
                torch.tensor(0.0, dtype=torch.float32)
            )
            self.rel_alpha_learnable = False

        self.last_rel_logits = None
        self.last_rel_alpha = None

        # 方向监督对比学习投影头。
        # 它不改变主分类头，只在训练阶段提供 pair-level 表示约束。
        self.use_contrastive_projector = bool(use_contrastive_projector)

        if contrast_dropout is None:
            contrast_dropout = self.drop_rate

        if self.use_contrastive_projector:
            self.contrast_projector = DirectionContrastiveProjector(
                input_dim=self.outfeature_size * 4,
                hidden_dim=contrast_hidden_dim,
                output_dim=contrast_dim,
                dropout_rate=contrast_dropout
            )
        else:
            self.contrast_projector = None

        self.last_contrast_z = None

    def _is_direction_branch_input(self, H_circ, H_dis):
        """
        判断当前输入是否为方向专属三分支超图。
        """
        if not isinstance(H_circ, dict):
            return False
        if not isinstance(H_dis, dict):
            return False

        required_keys = {"all", "up", "down"}
        return required_keys.issubset(set(H_circ.keys())) and required_keys.issubset(set(H_dis.keys()))

    def encode_nodes(self, circ_feat_pca, dis_feat_pca, H_circ, H_dis):
        """
        节点编码。

        支持两种输入:

        1) 原始单分支:
            H_circ: Tensor
            H_dis:  Tensor

        2) 方向专属三分支:
            H_circ = {
                "all":  H_circ_all,
                "up":   H_circ_up,
                "down": H_circ_down
            }

            H_dis = {
                "all":  H_dis_all,
                "up":   H_dis_up,
                "down": H_dis_down
            }
        """
        use_branch = self.use_direction_branches and self._is_direction_branch_input(H_circ, H_dis)
        self.last_used_direction_branches = use_branch

        if not use_branch:
            # ========== 原始单分支逻辑 ==========
            assert circ_feat_pca.size(0) == H_circ.size(0), \
                f"circ_feat_pca rows {circ_feat_pca.size(0)} != H_circ rows {H_circ.size(0)}"

            assert dis_feat_pca.size(0) == H_dis.size(0), \
                f"dis_feat_pca rows {dis_feat_pca.size(0)} != H_dis rows {H_dis.size(0)}"

            Z_circ_hyper, Z_dis_hyper = self.hyper_encoder(
                circ_feat_pca,
                dis_feat_pca,
                H_circ,
                H_dis
            )

            final_embedding = torch.cat(
                (Z_circ_hyper, Z_dis_hyper),
                dim=0
            )

            return final_embedding, Z_circ_hyper, Z_dis_hyper

        # ========== 方向专属三分支逻辑 ==========
        for key in ["all", "up", "down"]:
            assert circ_feat_pca.size(0) == H_circ[key].size(0), \
                f"circ_feat_pca rows {circ_feat_pca.size(0)} != H_circ[{key}] rows {H_circ[key].size(0)}"

            assert dis_feat_pca.size(0) == H_dis[key].size(0), \
                f"dis_feat_pca rows {dis_feat_pca.size(0)} != H_dis[{key}] rows {H_dis[key].size(0)}"

        # 1. all 分支
        Z_circ_all, Z_dis_all = self.hyper_encoder_all(
            circ_feat_pca,
            dis_feat_pca,
            H_circ["all"],
            H_dis["all"]
        )

        # 2. up 分支
        Z_circ_up, Z_dis_up = self.hyper_encoder_up(
            circ_feat_pca,
            dis_feat_pca,
            H_circ["up"],
            H_dis["up"]
        )

        # 3. down 分支
        Z_circ_down, Z_dis_down = self.hyper_encoder_down(
            circ_feat_pca,
            dis_feat_pca,
            H_circ["down"],
            H_dis["down"]
        )

        # 4. 分支融合
        Z_circ_hyper, circ_alpha = self.circ_branch_fusion(
            Z_circ_all,
            Z_circ_up,
            Z_circ_down,
            return_alpha=True
        )

        Z_dis_hyper, dis_alpha = self.dis_branch_fusion(
            Z_dis_all,
            Z_dis_up,
            Z_dis_down,
            return_alpha=True
        )

        # 5. 缓存三分支结果，方便后续调试或可视化
        self.last_circ_branch_alpha = circ_alpha.detach()
        self.last_dis_branch_alpha = dis_alpha.detach()

        self.last_Z_circ_all = Z_circ_all.detach()
        self.last_Z_circ_up = Z_circ_up.detach()
        self.last_Z_circ_down = Z_circ_down.detach()

        self.last_Z_dis_all = Z_dis_all.detach()
        self.last_Z_dis_up = Z_dis_up.detach()
        self.last_Z_dis_down = Z_dis_down.detach()

        final_embedding = torch.cat(
            (Z_circ_hyper, Z_dis_hyper),
            dim=0
        )

        return final_embedding, Z_circ_hyper, Z_dis_hyper

    def get_relation_alpha(self):
        """
        返回当前 relation-aware scorer 的融合权重 alpha。
        """
        if not self.use_relation_scorer:
            return None

        if getattr(self, "rel_alpha_learnable", False):
            return torch.sigmoid(self.rel_alpha_logit)

        return self.rel_alpha_fixed

    def get_relation_alpha_value(self):
        """
        返回 float 类型 alpha，方便训练日志打印。
        """
        alpha = self.get_relation_alpha()

        if alpha is None:
            return None

        return float(alpha.detach().cpu().item())

    def forward(self,
                circ_feat_pca,
                dis_feat_pca,
                H_circ,
                H_dis,
                pair_indices,
                return_embedding=False,
                return_aux=False):
        """
        参数：
            circ_feat_pca: [R, pca_dim]
            dis_feat_pca:  [D, pca_dim]
            H_circ:        [R, E_circ_hyperedge]
            H_dis:         [D, E_dis_hyperedge]
            pair_indices:  [B, 2]
                第 0 列 circ_index
                第 1 列 disease_index

        返回：
            logits: [B, 3]

            如果 return_embedding=True:
                logits, (zc_pair, zd_pair), final_embedding
        """
        final_embedding, Z_circ_hyper, Z_dis_hyper = self.encode_nodes(
            circ_feat_pca,
            dis_feat_pca,
            H_circ,
            H_dis
        )

        num_circ = circ_feat_pca.size(0)

        pair_feat, zc_pair, zd_pair = build_pair_features_from_indices(
            final_embedding=final_embedding,
            pair_indices=pair_indices,
            num_circ=num_circ
        )

        head_out = self.mlp_prediction(
            pair_feat,
            return_aux=return_aux
        )

        # ========== Direction supervised contrastive representation ==========
        contrast_z = None
        if return_aux and self.use_contrastive_projector and (self.contrast_projector is not None):
            contrast_z = self.contrast_projector(pair_feat)
            self.last_contrast_z = contrast_z.detach()

        # base_logits 是原层次化专家头输出
        if return_aux:
            base_logits = head_out["final_logits"]
        else:
            base_logits = head_out

        # ========== Relation-aware Scorer ==========
        rel_logits = None
        rel_alpha = None

        if self.use_relation_scorer and (self.relation_scorer is not None):
            rel_logits = self.relation_scorer(
                zc_pair=zc_pair,
                zd_pair=zd_pair
            )

            rel_alpha = self.get_relation_alpha()

            logits = base_logits + rel_alpha * rel_logits

            self.last_rel_logits = rel_logits.detach()
            self.last_rel_alpha = rel_alpha.detach()
        else:
            logits = base_logits

        # return_aux=True 时，把 relation-aware scorer 的结果也放进 head_out
        if return_aux:
            head_out["base_final_logits"] = base_logits
            head_out["final_logits"] = logits

            if rel_logits is not None:
                head_out["rel_logits"] = rel_logits
                head_out["rel_alpha"] = rel_alpha
            else:
                head_out["rel_logits"] = None
                head_out["rel_alpha"] = None

            # 对比学习表示，只在训练时使用
            head_out["contrast_z"] = contrast_z

        if return_embedding and return_aux:
            return logits, (zc_pair, zd_pair), final_embedding, head_out

        if return_embedding:
            return logits, (zc_pair, zd_pair), final_embedding

        if return_aux:
            return logits, head_out

        return logits

    def _sum_optional_losses(self, losses):
        valid_losses = [x for x in losses if x is not None]

        if len(valid_losses) == 0:
            return None

        out = valid_losses[0]
        for x in valid_losses[1:]:
            out = out + x

        return out

    def get_hyper_consistency_loss(self):
        if (not self.use_hyper) or (self.hyper_mode == "none"):
            return None

        if self.last_used_direction_branches:
            return self._sum_optional_losses([
                self.hyper_encoder_all.get_consistency_loss(),
                self.hyper_encoder_up.get_consistency_loss(),
                self.hyper_encoder_down.get_consistency_loss()
            ])

        if self.hyper_encoder is None:
            return None

        return self.hyper_encoder.get_consistency_loss()

    def get_hyper_structure_loss(self):
        if (not self.use_hyper) or (self.hyper_mode == "none"):
            return None

        if self.last_used_direction_branches:
            return self._sum_optional_losses([
                self.hyper_encoder_all.get_structure_loss(),
                self.hyper_encoder_up.get_structure_loss(),
                self.hyper_encoder_down.get_structure_loss()
            ])

        if self.hyper_encoder is None:
            return None

        return self.hyper_encoder.get_structure_loss()

    def get_hyper_regularization_loss(self):
        if (not self.use_hyper) or (self.hyper_mode == "none"):
            return None

        if self.last_used_direction_branches:
            return self._sum_optional_losses([
                self.hyper_encoder_all.get_regularization_loss(),
                self.hyper_encoder_up.get_regularization_loss(),
                self.hyper_encoder_down.get_regularization_loss()
            ])

        if self.hyper_encoder is None:
            return None

        return self.hyper_encoder.get_regularization_loss()

    def get_last_pair_gate(self):
        if hasattr(self.mlp_prediction, "last_gate"):
            return self.mlp_prediction.last_gate
        return None

    def get_last_expert_gate(self):
        if hasattr(self.mlp_prediction, "last_expert_gate"):
            return self.mlp_prediction.last_expert_gate
        return None

    def get_pair_router_aux_loss(self):
        if hasattr(self.mlp_prediction, "last_router_aux_loss"):
            return self.mlp_prediction.last_router_aux_loss
        return None

    def get_last_direction_branch_alpha(self):
        """
        返回 circRNA 侧和 disease 侧的方向分支门控均值。

        返回:
            None 或 dict:
            {
                "circ": tensor [3],
                "dis":  tensor [3]
            }
        """
        circ_alpha = getattr(self, "last_circ_branch_alpha", None)
        dis_alpha = getattr(self, "last_dis_branch_alpha", None)

        if circ_alpha is None or dis_alpha is None:
            return None

        return {
            "circ": circ_alpha.float().mean(dim=0),
            "dis": dis_alpha.float().mean(dim=0)
        }