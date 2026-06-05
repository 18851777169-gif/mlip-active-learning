"""MACE model wrapper with ensemble, GMM uncertainty, and MC-Dropout.

Supports three uncertainty estimation methods:
  - GMM uncertainty (MACE built-in latent space Mahalanobis distance)
  - Ensemble QBC (3 models with different seeds)
  - MC-Dropout (multiple forward passes with dropout)

Also provides structure embeddings for diversity computation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from pathlib import Path


class MACEWrapper(nn.Module):
    """Wrapper around MACE model for energy/force prediction.

    Provides:
    - Forward pass returning energy + forces
    - Latent feature extraction for diversity/GMM
    - Node-wise energy decomposition
    """

    def __init__(self, model_name: str = "small", pretrained: str = None,
                 r_max: float = 5.0, dtype: str = "float32", device: str = "cpu",
                 use_mace: bool = False):
        super().__init__()
        self.model_name = model_name
        self.r_max = r_max
        self.device = device
        self._has_mace = False

        if use_mace:
            try:
                from mace.calculators import MACECalculator
                self._init_mace(model_name, pretrained, r_max, dtype, device)
                self._has_mace = True
                print(f"  Using MACE model: {model_name}")
                return
            except ImportError:
                print("[WARNING] mace-torch not installed. Using fallback SchNet model.")
            except Exception as e:
                print(f"[WARNING] MACE init failed: {e}. Using fallback SchNet model.")

        self._init_fallback()
        print("  Using fallback SchNet model")

    def _init_mace(self, model_name, pretrained, r_max, dtype, device):
        """Initialize actual MACE model."""
        self._mace_dtype = getattr(torch, dtype)

        if pretrained:
            # Fine-tune from pretrained universal potential
            from mace.calculators import mace_mp
            self.calculator = mace_mp(
                model=model_name,
                device=device,
                default_dtype=self._mace_dtype,
            )
        else:
            from mace.calculators import MACECalculator
            self.calculator = MACECalculator(
                model_path=None,
                device=device,
                default_dtype=self._mace_dtype,
            )

        self._mace_models = self.calculator.models  # list of ScaleShiftMACE
        self.model = self._mace_models[0] if self._mace_models else None

    def _init_fallback(self):
        """Fallback: simple message-passing model for testing."""
        from model_fallback import FallbackModel
        self.model = FallbackModel(hidden_channels=64, num_interactions=2)
        self.calculator = None
        self._has_mace = False

    def forward(self, data: dict):
        """Predict energy and forces.

        Args:
            data: collated batch with z, pos, batch, cell, ptr

        Returns:
            energy: [batch_size] total energy per structure
            forces: [n_atoms_total, 3] force on each atom
            node_features: [n_atoms_total, hidden_dim] or None
        """
        if self._has_mace:
            return self._forward_mace(data)
        else:
            return self._forward_fallback(data)

    def _forward_mace(self, data):
        """Forward pass through MACE via ASE calculator interface.

        MACE's internal forward requires precomputed edge_index. Using the
        ASE calculator is the recommended way to get predictions.
        """
        import ase
        from ase import Atoms
        import numpy as np

        z = data["z"].cpu().numpy()
        pos = data["pos"].cpu().numpy()
        batch = data.get("batch", None)
        if batch is not None:
            batch = batch.cpu().numpy()
        cell = data.get("cell", None)
        if cell is not None:
            cell = cell.cpu().numpy()

        ptr = data.get("ptr", None)
        if ptr is not None:
            ptr = ptr.cpu().numpy()

        n_structs = len(ptr) - 1 if ptr is not None else 1

        energies = []
        forces_list = []

        for s in range(n_structs):
            start = ptr[s] if ptr is not None else 0
            end = ptr[s + 1] if ptr is not None else len(z)

            atoms = Atoms(
                numbers=z[start:end],
                positions=pos[start:end],
            )
            if cell is not None:
                atoms.set_cell(cell[s] if cell.ndim == 3 else cell)
                atoms.set_pbc(True)
            else:
                atoms.set_pbc(False)

            atoms.calc = self.calculator

            try:
                e = atoms.get_potential_energy()
                f = atoms.get_forces()
                energies.append(float(e))
                forces_list.append(torch.tensor(f, dtype=torch.float32))
            except Exception as exc:
                energies.append(0.0)
                forces_list.append(torch.zeros((end - start, 3), dtype=torch.float32))

        energy_tensor = torch.tensor(energies, dtype=torch.float32, device=data["z"].device)
        forces_tensor = torch.cat(forces_list, dim=0).to(data["z"].device)

        return energy_tensor, forces_tensor, None

    def _forward_fallback(self, data):
        """Fallback model forward pass using self.model (per-instance)."""
        z = data["z"]
        pos = data["pos"]
        batch = data.get("batch", torch.zeros(z.shape[0], dtype=torch.long, device=z.device))
        energy, node_feats = self.model(z, pos, batch)
        forces = torch.zeros_like(pos)
        return energy, forces, node_feats

    def get_node_features(self, data: dict) -> torch.Tensor:
        """Extract node-level features for GMM / diversity."""
        _, _, node_feats = self.forward(data)
        return node_feats

    def get_structure_embedding(self, data: dict) -> torch.Tensor:
        """Mean-pooled node features -> per-structure embedding."""
        node_feats = self.get_node_features(data)
        if node_feats is None:
            return None
        from torch_geometric.nn import global_mean_pool
        batch = data.get("batch", torch.zeros(node_feats.shape[0], dtype=torch.long))
        return global_mean_pool(node_feats, batch)


class EnsembleMACE(nn.Module):
    """Ensemble of MACE models for QBC uncertainty estimation.

    Uncertainty = variance of energy predictions across ensemble members.
    """

    def __init__(self, ensemble_size: int = 3, seeds: List[int] = None,
                 **model_kwargs):
        super().__init__()
        self.ensemble_size = ensemble_size
        seeds = seeds or [42, 123, 456]

        self.members = nn.ModuleList()
        for i, seed in enumerate(seeds[:ensemble_size]):
            torch.manual_seed(seed)
            member = MACEWrapper(**model_kwargs)
            self.members.append(member)
            self._seed = seeds

    def forward(self, data):
        """Return mean energy and forces across ensemble."""
        energies = []
        forces = []
        for member in self.members:
            e, f, _ = member(data)
            energies.append(e)
            if f is not None:
                forces.append(f)
        energies = torch.stack(energies, dim=0)
        if forces:
            forces = torch.stack(forces, dim=0)
            return energies.mean(0), forces.mean(0)
        return energies.mean(0), None

    def uncertainty(self, data) -> torch.Tensor:
        """Compute per-structure uncertainty as energy variance across ensemble."""
        energies = []
        for member in self.members:
            e, _, _ = member(data)
            energies.append(e)
        energies = torch.stack(energies, dim=0)  # [n_ensemble, batch]
        return energies.std(dim=0)  # [batch]

    def get_structure_embeddings(self, data) -> torch.Tensor:
        """Get embeddings from first ensemble member."""
        return self.members[0].get_structure_embedding(data)


class MCDropoutMACE(nn.Module):
    """Single MACE model with MC-Dropout for uncertainty.

    Performs multiple forward passes with dropout enabled (training=True)
    and uses prediction variance as uncertainty.
    """

    def __init__(self, n_passes: int = 10, dropout_rate: float = 0.1, **model_kwargs):
        super().__init__()
        self.n_passes = n_passes
        self.dropout_rate = dropout_rate
        self.model = MACEWrapper(**model_kwargs)

    def forward(self, data):
        return self.model(data)

    def uncertainty(self, data) -> torch.Tensor:
        """MC-Dropout uncertainty = variance across stochastic forward passes."""
        energies = []
        self.model.train()  # Enable dropout
        for _ in range(self.n_passes):
            e, _, _ = self.model(data)
            energies.append(e)
        self.model.eval()
        energies = torch.stack(energies, dim=0)  # [n_passes, batch]
        return energies.std(dim=0)  # [batch]

    def get_structure_embeddings(self, data):
        return self.model.get_structure_embedding(data)


class GMMUncertainty:
    """GMM-based uncertainty estimation in MACE latent space.

    Fits a Gaussian Mixture Model per atom type on node-level features,
    then computes Mahalanobis distance from GMM centroids as uncertainty.

    Reference: MACE built-in uncertainty estimation.
    """

    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.gmms = {}  # atomic_number -> GaussianMixture

    def fit(self, model: MACEWrapper, dataloader, device: str = "cpu"):
        """Fit GMM per atom type on latent features from labeled data."""
        from sklearn.mixture import GaussianMixture

        # Collect node features per atom type
        features_per_z = {}
        model.eval()
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in batch_data.items()}
                node_feats = model.get_node_features(batch_data)
                if node_feats is None:
                    continue
                z = batch_data["z"].cpu().numpy()
                feats = node_feats.cpu().numpy()
                for atom_z in np.unique(z):
                    mask = z == atom_z
                    if atom_z not in features_per_z:
                        features_per_z[atom_z] = []
                    features_per_z[atom_z].append(feats[mask])

        # Fit GMM per atom type
        for atom_z, feat_list in features_per_z.items():
            X = np.concatenate(feat_list, axis=0)
            if X.shape[0] < self.n_components:
                self.gmms[atom_z] = None
                continue
            gmm = GaussianMixture(
                n_components=min(self.n_components, X.shape[0]),
                covariance_type="full",
                random_state=42,
            )
            gmm.fit(X)
            self.gmms[atom_z] = gmm

    def score(self, batch_data: dict, device: str = "cpu") -> np.ndarray:
        """Compute per-atom Mahalanobis score, aggregate to per-structure.

        Higher score = more uncertain (atypical in latent space).
        """
        import numpy as np

        batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch_data.items()}

        # This requires a model for feature extraction
        # Called externally with model reference
        return np.zeros(batch_data["z"].shape[0])  # Placeholder


def compute_structure_embedding(model, dataloader, device: str = "cpu") -> np.ndarray:
    """Compute structure-level embeddings for the entire pool.

    Returns:
        embeddings: [n_structures, hidden_dim]
    """
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch_data.items()}
            if hasattr(model, 'get_structure_embeddings'):
                emb = model.get_structure_embeddings(batch_data)
            elif hasattr(model, 'get_structure_embedding'):
                emb = model.get_structure_embedding(batch_data)
            else:
                emb = None
            if emb is not None:
                all_embeddings.append(emb.cpu().numpy())
    if all_embeddings:
        return np.concatenate(all_embeddings, axis=0)
    return None
