"""Fallback model when mace-torch is not installed.

Provides a lightweight SchNet-like model for pipeline testing.
Results from this model are NOT scientifically valid for publication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_add(src, index, dim_size):
    """Pure PyTorch scatter_add (replaces torch_geometric)."""
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    index = index.unsqueeze(1).expand(-1, src.shape[1])
    return out.scatter_add_(0, index, src)


def build_radius_graph(pos, batch, r_cut, max_neighbors=32):
    """Build radius graph using pure PyTorch (no torch-cluster dependency).

    For each atom, finds up to max_neighbors neighbors within r_cut.
    """
    n_atoms = pos.shape[0]
    device = pos.device

    # Compute all pairwise distances (only within same batch)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [n, n, 3]
    dist = diff.norm(dim=2)  # [n, n]

    # Mask: must be within cutoff AND in same batch
    same_batch = (batch.unsqueeze(0) == batch.unsqueeze(1))
    within_cutoff = (dist < r_cut) & (dist > 1e-6)

    # For each row, keep only top max_neighbors within cutoff
    dist_masked = dist.clone()
    dist_masked[~within_cutoff] = float("inf")

    # Get top-k smallest distances per atom
    k = min(max_neighbors, n_atoms)
    _, indices = torch.topk(dist_masked, k=k, dim=1, largest=False)

    # Build edge_index
    rows = torch.arange(n_atoms, device=device).unsqueeze(1).expand(-1, k)
    cols = indices

    # Filter invalid edges
    flat_rows = rows.reshape(-1)
    flat_cols = cols.reshape(-1)
    valid = torch.isfinite(dist[flat_rows, flat_cols])
    rows = flat_rows[valid]
    cols = flat_cols[valid]

    edge_index = torch.stack([rows, cols], dim=0)
    return edge_index


class SchNetInteraction(nn.Module):
    """Single continuous-filter convolution layer (pure PyTorch)."""

    def __init__(self, hidden_channels: int = 128, num_gaussians: int = 50):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x, pos, edge_index, edge_attr):
        row, col = edge_index
        msg = self.mlp(edge_attr) * x[col]
        aggr = scatter_add(msg, row, dim_size=x.shape[0])
        return x + self.update(aggr)


class RBFExpansion(nn.Module):
    """Expand interatomic distances in a Gaussian basis."""

    def __init__(self, vmin: float = 0.0, vmax: float = 5.0, bins: int = 50):
        super().__init__()
        self.register_buffer("centers", torch.linspace(vmin, vmax, bins))
        self.register_buffer("width", torch.tensor((vmax - vmin) / bins))

    def forward(self, dist):
        return torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / self.width ** 2)


class FallbackModel(nn.Module):
    """Lightweight SchNet-like model for MACE fallback.

    Architecture:
    - Embedding: atomic number -> hidden channels
    - 3 interaction layers with RBF edge features
    - Output: sum of per-atom contributions -> energy
    """

    def __init__(self, hidden_channels: int = 64, num_interactions: int = 2,
                 num_gaussians: int = 50, cutoff: float = 5.0, max_z: int = 94):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff

        self.embedding = nn.Embedding(max_z + 1, hidden_channels)
        self.rbf = RBFExpansion(vmin=0.0, vmax=cutoff, bins=num_gaussians)

        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_channels, num_gaussians)
            for _ in range(num_interactions)
        ])

        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, z, pos, batch, return_embeddings: bool = True):
        x = self.embedding(z)

        # Build radius graph (pure PyTorch, no torch-cluster needed)
        edge_index = build_radius_graph(pos, batch, r_cut=self.cutoff,
                                        max_neighbors=32)

        # RBF edge features
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=1)
        edge_attr = self.rbf(dist)

        # Interactions
        for interaction in self.interactions:
            x = interaction(x, pos, edge_index, edge_attr)

        # Per-atom energy
        per_atom = self.output(x).squeeze(-1)

        # Pool to per-structure energy
        energy = torch.zeros(batch.max().item() + 1, device=x.device)
        energy.scatter_add_(0, batch, per_atom)

        return energy, x


_fallback_model = None


def fallback_forward(data: dict, training: bool = True):
    """Forward pass using fallback model."""
    global _fallback_model
    if _fallback_model is None:
        _fallback_model = FallbackModel()
    model = _fallback_model
    model.train(training)

    z = data["z"]
    pos = data["pos"]
    batch = data.get("batch", torch.zeros(z.shape[0], dtype=torch.long))

    energy, node_feats = model(z, pos, batch)

    # No forces from this simple model (placeholder zeros)
    forces = torch.zeros_like(pos)

    return energy, forces, node_feats
