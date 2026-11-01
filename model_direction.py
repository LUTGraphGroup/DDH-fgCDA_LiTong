import torch
from torch import nn
from torch.nn import functional as F
from hypergrl_module import PairwiseEditGate, PaperHyperGRLEncoder, PaperBranchFusion

def build_pair_features_from_indices(final_embedding, pair_indices, num_circ):


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

class HierarchicalExpertRoutedPairHead(nn.Module):


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


        self.router = nn.Linear(expert_in_dim, num_experts)


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


        unknown_logit = known_logits[:, 0:1]


        up_logit = known_logits[:, 1:2] + direction_logits[:, 0:1]
        down_logit = known_logits[:, 1:2] + direction_logits[:, 1:2]

        final_logits = torch.cat(
            [unknown_logit, up_logit, down_logit],
            dim=1
        )

        return final_logits

    def forward(self, x, return_aux=False):


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
        )

        out = (expert_alpha.unsqueeze(-1) * expert_outputs).sum(dim=1)

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

class DirectionContrastiveProjector(nn.Module):


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


    def __init__(self, pca_dim, circ_edges, disease_edges, drop_rate=.1,
                 embed_dim=256, num_experts=4, topk_experts=2):

        super().__init__()
        self.hyper_pca_dim=pca_dim
        self.outfeature_size=embed_dim
        self.drop_rate=drop_rate
        for branch in ('all','up','down'):
            setattr(self,'hyper_encoder_'+branch,PaperHyperGRLEncoder(
                pca_dim,embed_dim,circ_edges,disease_edges,PairwiseEditGate))
        self.circ_branch_fusion=PaperBranchFusion(embed_dim,drop_rate)
        self.dis_branch_fusion=PaperBranchFusion(embed_dim,drop_rate)
        self.mlp_prediction=HierarchicalExpertRoutedPairHead(embed_dim,
            hidden_dim=256,gate_hidden=128,dropout_rate=drop_rate,
            num_experts=num_experts,topk_experts=topk_experts)
        self.contrast_projector=DirectionContrastiveProjector(4*embed_dim,
            hidden_dim=256,output_dim=128,dropout_rate=drop_rate)

    def encode_nodes(self,circ,dis,hc,hd):

        encoded=[getattr(self,'hyper_encoder_'+branch)(circ,dis,hc[branch],hd[branch])
                 for branch in ('all','up','down')]
        zc=self.circ_branch_fusion(*(pair[0] for pair in encoded))
        zd=self.dis_branch_fusion(*(pair[1] for pair in encoded))
        return torch.cat((zc,zd),dim=0),zc,zd

    def forward(self,circ,dis,hc,hd,pair_indices,return_embedding=False,return_aux=False):

        embedding,_,_=self.encode_nodes(circ,dis,hc,hd)
        pair,zc,zd=build_pair_features_from_indices(embedding,pair_indices,circ.shape[0])
        head=self.mlp_prediction(pair,return_aux=return_aux)
        logits=head['final_logits'] if return_aux else head
        if return_aux:
            head['contrast_z']=self.contrast_projector(pair)
        if return_embedding and return_aux:
            return logits,(zc,zd),embedding,head
        if return_embedding:
            return logits,(zc,zd),embedding
        return (logits,head) if return_aux else logits

    def get_hyper_regularization_loss(self):

        return sum(getattr(self,'hyper_encoder_'+b).get_regularization_loss() for b in ('all','up','down'))

    def get_hyper_structure_loss(self):

        return sum(getattr(self,'hyper_encoder_'+b).get_structure_loss() for b in ('all','up','down'))

    def get_hyper_consistency_loss(self):

        return sum(getattr(self,'hyper_encoder_'+b).get_consistency_loss() for b in ('all','up','down'))

    def get_pair_router_aux_loss(self):

        return self.mlp_prediction.last_router_aux_loss
