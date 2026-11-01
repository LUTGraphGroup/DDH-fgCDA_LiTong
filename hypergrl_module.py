import math
from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def _logit(p: float, eps: float = 1e-4) -> float:

    p = min(max(float(p), eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))

class PairwiseEditGate(nn.Module):


    def __init__(self, feat_dim, hidden_dim=32, init_p=0.5):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.edge_proj = nn.Linear(feat_dim, hidden_dim, bias=False)


        self.sim_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))


        self.bias = nn.Parameter(torch.tensor(_logit(init_p), dtype=torch.float32))

    def forward(self, node_q: torch.Tensor, edge_embed: torch.Tensor,
                sim: Optional[torch.Tensor] = None) -> torch.Tensor:

        q = self.node_proj(node_q)
        k = self.edge_proj(edge_embed)

        logits = (q @ k.t()) / math.sqrt(self.hidden_dim)

        if sim is not None:

            logits = logits + self.sim_scale * (2.0 * sim - 1.0)

        logits = logits + self.bias
        gate = torch.sigmoid(logits)
        gate = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0)
        return gate
class PaperPropagation(nn.Module):


    def __init__(self, input_dim, output_dim):

        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, h):

        dv, de = h.sum(1), h.sum(0)
        iv = torch.where(dv > 0, dv.clamp_min(1e-8).rsqrt(), torch.zeros_like(dv))
        ie = torch.where(de > 0, de.clamp_min(1e-8).reciprocal(), torch.zeros_like(de))
        z = self.linear(x) * iv[:, None]
        z = (h @ ((h.T @ z) * ie[:, None])) * iv[:, None]
        return F.relu(self.norm(z))

class PaperSideEncoder(nn.Module):


    def __init__(self, input_dim, embed_dim, num_edges, gate_class):

        super().__init__()
        self.tau, self.num_coevo_steps = .9, 3
        self.local_extra_k, self.topk_ratio, self.min_topk, self.max_topk = 6, .03, 6, 24
        self.prune_floor, self.lambda_consistency = .2, .01
        self.node_proj = nn.Linear(input_dim, 64, bias=False)
        self.edge_seed_proj = nn.Linear(input_dim, 64, bias=False)
        self.node_target_proj = nn.Linear(embed_dim, 64, bias=False)
        self.edge_target_proj = nn.Linear(embed_dim, 64, bias=False)
        self.coevo_convs = nn.ModuleList([PaperPropagation(input_dim, embed_dim)] +
                                        [PaperPropagation(embed_dim, embed_dim) for _ in range(2)])
        self.keep_gate = gate_class(64, hidden_dim=32, init_p=.60)
        self.prune_gate = gate_class(64, hidden_dim=32, init_p=.05)
        self.recruit_gate = gate_class(64, hidden_dim=32, init_p=.12)
        self.mix_gate = gate_class(64, hidden_dim=32, init_p=.88)

        self.edge_attr_base = nn.Parameter(torch.empty(num_edges, 64))
        nn.init.xavier_uniform_(self.edge_attr_base)
        self.eta_logit = nn.Parameter(torch.tensor(np.log(.35 / .65), dtype=torch.float32))
        self.last_structure_loss = None
        self.last_consistency = None

    def edit(self, h, anchor, q, u, matching):

        support = (h > 0).to(h.dtype)
        candidates = torch.maximum((anchor > 0).to(h.dtype), support)
        local = matching.topk(min(6, matching.shape[1]), dim=1).indices
        candidates = candidates.scatter(1, local, 1.)
        add_mask = (candidates - support).clamp_min(0.)
        keep = self.keep_gate(q, u, matching)
        prune = self.prune_gate(q, u, 1 - matching)
        recruit = self.recruit_gate(q, u, matching)
        mix = self.mix_gate(q, u, matching)
        edited = h * (1 + keep * matching) * (1 - .8 * prune)
        scores = recruit * matching * add_mask
        cap = min(max(6, min(24, round(.03 * h.shape[1]))), h.shape[1])
        retained = scores.topk(cap, dim=1).indices
        added = scores * torch.zeros_like(scores).scatter(1, retained, 1.)
        result = mix * anchor + (1 - mix) * (edited + added)
        return result, added

    def forward(self, x, anchor):

        h, z = anchor, x
        structure, consistency = x.new_zeros(()), x.new_zeros(())
        self.last_step_graphs, self.last_step_proto = [], []
        for step, conv in enumerate(self.coevo_convs):
            node_projection = self.node_proj if step == 0 else self.node_target_proj
            edge_projection = self.edge_seed_proj if step == 0 else self.edge_target_proj
            q = F.normalize(node_projection(z), dim=-1)
            aggregate = (h.T @ z) / (h.sum(0)[:, None] + 1e-8)
            u = F.normalize(edge_projection(aggregate), dim=-1)
            matching = torch.sigmoid((q @ u.T) / self.tau)
            current, added = self.edit(h, anchor, q, u, matching)
            next_z = conv(z, current)
            support = (anchor > 0).to(anchor.dtype)
            edit_loss = (current - h).abs().mean() + .5 * added.abs().mean()
            prior_loss = F.mse_loss(current * support, anchor * support)
            smooth_loss = F.mse_loss(current, h.detach())
            structure = structure + .001 * edit_loss + .005 * prior_loss + .001 * smooth_loss

            eta = self.eta_logit.sigmoid()
            reference = F.normalize((1 - eta) * u + eta * self.edge_attr_base, dim=-1)
            target = F.normalize(self.edge_target_proj(
                (current.T @ next_z) / (current.sum(0)[:, None] + 1e-8)), dim=-1)
            consistency = consistency + F.mse_loss(reference, target.detach())
            h, z = current, next_z
            self.last_step_graphs.append(h.detach())
            self.last_step_proto.append(u.detach())
        self.last_structure_loss = structure / self.num_coevo_steps
        self.last_consistency = consistency / self.num_coevo_steps
        return z

class PaperHyperGRLEncoder(nn.Module):


    def __init__(self, input_dim, embed_dim, circ_edges, disease_edges, gate_class):

        super().__init__()
        self.circ_encoder = PaperSideEncoder(input_dim, embed_dim, circ_edges, gate_class)
        self.dis_encoder = PaperSideEncoder(input_dim, embed_dim, disease_edges, gate_class)

    def forward(self, circ, disease, hc, hd):

        return self.circ_encoder(circ, hc), self.dis_encoder(disease, hd)

    def get_structure_loss(self):

        return self.circ_encoder.last_structure_loss + self.dis_encoder.last_structure_loss

    def get_consistency_loss(self):

        return self.circ_encoder.last_consistency + self.dis_encoder.last_consistency

    def get_regularization_loss(self):

        return self.get_structure_loss() + .01 * self.get_consistency_loss()

class PaperBranchFusion(nn.Module):


    def __init__(self, embed_dim, dropout):

        super().__init__()
        self.tau, self.residual_scale = 3., .1
        self.gate = nn.Sequential(nn.Linear(3 * embed_dim, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, 3))

    def forward(self, z_all, z_up, z_down, return_alpha=False):

        zs = (z_all, z_up, z_down)
        normalized = torch.cat([F.layer_norm(z, z.shape[-1:]) for z in zs], dim=-1)
        alpha = torch.softmax(self.gate(normalized) / self.tau, dim=-1)
        result = sum(alpha[:, i:i+1] * z for i, z in enumerate(zs)) + .1 * sum(zs) / 3
        self.last_alpha = alpha.detach()
        return (result, alpha) if return_alpha else result
