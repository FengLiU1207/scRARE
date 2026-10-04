import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseInputLinear(nn.Module):
    """nn.Linear equivalent that also accepts a sparse COO/CSR row matrix."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        sparse_csr_layout = getattr(torch, "sparse_csr", None)
        if x.layout == torch.sparse_coo or (
            sparse_csr_layout is not None and x.layout == sparse_csr_layout
        ):
            output = torch.sparse.mm(x, self.weight.t())
            if self.bias is not None:
                output = output + self.bias
            return output
        return F.linear(x, self.weight, self.bias)


class RelationProjector(nn.Module):
    """Lightweight explicit rank-r relation factor U=g_rel(Z)."""

    def __init__(self, input_dim, relation_rank, eps=1e-8):
        super().__init__()
        self.projection = nn.Linear(input_dim, relation_rank)
        self.eps = float(eps)

    def forward(self, z):
        u = self.projection(z)
        return F.normalize(u, p=2, dim=1, eps=self.eps)


class ScRARENetwork(nn.Module):
    """Multi-view encoder with optional clustering and relation modules."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        act,
        n_num,
        cluster_num,
        method="proposed",
        relation_rank=0,
        allocate_legacy_weights=False,
        eps=1e-8,
    ):
        super().__init__()
        self.method = method
        self.eps = float(eps)

        self.AE1 = nn.Linear(input_dim, hidden_dim)
        self.AE2 = nn.Linear(input_dim, hidden_dim)
        self.SE1 = SparseInputLinear(n_num, hidden_dim)
        self.SE2 = SparseInputLinear(n_num, hidden_dim)

        # Allocate the semantic classifier only when the legacy branch uses it.
        self.semantic_classifier = (
            nn.Linear(hidden_dim, cluster_num) if method == "legacy" else None
        )
        self.relation_projector = (
            RelationProjector(hidden_dim, relation_rank, eps=eps)
            if relation_rank > 0 else None
        )

        # Learnable mixture weight for expression- and graph-based similarities.
        self.alpha = nn.Parameter(torch.tensor([0.99999], dtype=torch.float32))

        # Learnable mixture weight for the two structural relation sources.
        # A zero logit initializes the sigmoid output at 0.5.
        self.structure_prior_logit = (
            nn.Parameter(torch.zeros(1, dtype=torch.float32))
            if method == "proposed" else None
        )

        if allocate_legacy_weights:
            self.register_buffer("pos_weight", torch.ones(n_num * 2))
            self.register_buffer(
                "pos_neg_weight", torch.ones(n_num * 2, n_num * 2)
            )
        else:
            self.register_buffer("pos_weight", torch.empty(0), persistent=False)
            self.register_buffer("pos_neg_weight", torch.empty(0), persistent=False)

        if act == "ident":
            self.activate = nn.Identity()
        elif act == "sigmoid":
            self.activate = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {act}")

    def similarity_mix(self, constrained=True):
        return self.alpha.clamp(0.0, 1.0) if constrained else self.alpha

    def structure_prior_mix(self):
        """Learnable eta in (0,1) for A vs. A_knn structural prior fusion."""
        if self.structure_prior_logit is None:
            raise RuntimeError("Learnable structural-prior mixing exists only in proposed mode.")
        return torch.sigmoid(self.structure_prior_logit)

    def forward(self, x, A_dyn, A_knn):
        z1 = F.normalize(self.activate(self.AE1(x)), dim=1, p=2, eps=self.eps)
        z2 = F.normalize(self.activate(self.AE2(x)), dim=1, p=2, eps=self.eps)

        # Encode the fixed auxiliary graph and the current dynamic graph separately.
        h1 = F.normalize(self.SE1(A_knn), dim=1, p=2, eps=self.eps)
        h2 = F.normalize(self.SE2(A_dyn), dim=1, p=2, eps=self.eps)
        return z1, z2, h1, h2

    def semantic_logits(self, z1, z2):
        if self.semantic_classifier is None:
            raise RuntimeError("The semantic classifier exists only in legacy mode.")
        return self.semantic_classifier(z1), self.semantic_classifier(z2)

    def relation_embedding(self, z):
        if self.relation_projector is None:
            raise RuntimeError("Relation projector is disabled for this configuration.")
        return self.relation_projector(z)


# Keep the original class name available for existing training scripts.
hard_sample_aware_network = ScRARENetwork
