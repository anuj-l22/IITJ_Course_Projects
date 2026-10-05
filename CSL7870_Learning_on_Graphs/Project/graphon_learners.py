# # graphon_learners.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class LowRankGraphon(nn.Module):
#     """Class-wise low-rank graphon: W_y = σ(U_y U_y^T) ∈ [0,1]^{N×N}."""
#     def __init__(self, num_classes, N, rank=8):
#         super().__init__()
#         self.N = N
#         self.rank = rank
#         self.U = nn.ParameterDict({
#             str(int(c)): nn.Parameter(torch.randn(N, rank) * 0.1)
#             for c in range(num_classes)
#         })
#     def forward(self, class_idx):
#         Uc = self.U[str(int(class_idx))]
#         return torch.sigmoid(Uc @ Uc.t())

# class NeuralGraphon(nn.Module):
#     """Class-wise neural graphon: W_ij = σ(MLP(|z_i - z_j|))."""
#     def __init__(self, num_classes, N, coord_dim=16, hidden=64):
#         super().__init__()
#         self.N = N
#         self.coords = nn.ParameterDict({
#             str(int(c)): nn.Parameter(torch.randn(N, coord_dim) * 0.1)
#             for c in range(num_classes)
#         })
#         self.mlp = nn.Sequential(
#             nn.Linear(coord_dim, hidden), nn.ReLU(),
#             nn.Linear(hidden, hidden), nn.ReLU(),
#             nn.Linear(hidden, 1)
#         )
#     def forward(self, class_idx):
#         Z = self.coords[str(int(class_idx))]            # [N, d]
#         diff = torch.abs(Z.unsqueeze(1) - Z.unsqueeze(0))  # [N, N, d]
#         W = torch.sigmoid(self.mlp(diff).squeeze(-1))   # [N, N]
#         return W

# graphon_learners.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankGraphon(nn.Module):
    def __init__(self, num_class: int, N: int, rank: int):
        super().__init__()
        self.N = N
        self.rank = rank
        # For binary (num_class==1) create two slots: class 0 and 1
        self.K = max(2, num_class)
        self.U = nn.ParameterDict()
        for c in range(self.K):
            self.U[str(c)] = nn.Parameter(torch.randn(N, rank) * 0.02)

    def forward(self, class_idx: int):
        c = int(class_idx)  # expects 0 or 1 for HIV
        Uc = self.U[str(c)]
        W = torch.sigmoid(Uc @ Uc.t())
        return W


class NeuralGraphon(nn.Module):
    def __init__(self, num_class: int, N: int, coord_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.N = N
        self.coord_dim = coord_dim
        self.hidden = hidden
        # For binary (num_class==1) create two slots: class 0 and 1
        self.K = max(2, num_class)

        # per-class node coordinates Z_y
        self.Z = nn.ParameterDict()
        for c in range(self.K):
            self.Z[str(c)] = nn.Parameter(torch.randn(N, coord_dim) * 0.02)

        # shared MLP g_theta([z_i, z_j]) -> scalar in (0,1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * coord_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, class_idx: int):
        c = int(class_idx)  # expects 0 or 1 for HIV
        Zc = self.Z[str(c)]                # [N, d]
        # build all pairs [z_i, z_j]
        Zi = Zc.unsqueeze(1).expand(-1, self.N, -1)  # [N, N, d]
        Zj = Zc.unsqueeze(0).expand(self.N, -1, -1)  # [N, N, d]
        Pairs = torch.cat([Zi, Zj], dim=-1)          # [N, N, 2d]
        W = torch.sigmoid(self.mlp(Pairs).squeeze(-1))  # [N, N]
        return W
