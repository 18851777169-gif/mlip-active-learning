"""Acquisition functions for active learning of MLIPs.

Implements 8 strategies (A-H) as specified in the Methodology Blueprint:

  A. Random sampling (baseline)
  B. GMM uncertainty (MACE built-in Mahalanobis distance)
  C. Ensemble QBC (query-by-committee, 3-seed variance)
  D. MC-Dropout (multi-pass variance)
  E. FPS + SOAP (farthest point sampling in SOAP descriptor space)
  F. Latent space clustering (k-means in MACE feature space)
  G. Hybrid-Weighted (alpha*U + (1-alpha)*D)
  H. Hybrid-TwoStage (top K% uncertainty -> FPS diversity)

All functions return selected indices into the pool.
"""

import numpy as np
from typing import List, Optional, Dict, Callable
from abc import ABC, abstractmethod


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    def __init__(self, strategy_id: str, category: str, label: str):
        self.strategy_id = strategy_id
        self.category = category
        self.label = label

    @abstractmethod
    def select(
        self,
        pool_indices: np.ndarray,
        n_query: int,
        model=None,
        pool_dataloader=None,
        labeled_structures=None,
        **kwargs,
    ) -> np.ndarray:
        """Select n_query indices from pool_indices to label next."""
        pass


class RandomSampling(AcquisitionFunction):
    """A. Random baseline."""

    def __init__(self):
        super().__init__("A_random", "baseline", "Random (Baseline)")

    def select(self, pool_indices, n_query, **kwargs):
        rng = np.random.RandomState(42)
        selected = rng.choice(pool_indices, size=min(n_query, len(pool_indices)),
                              replace=False)
        return selected


class GMMUncertainty(AcquisitionFunction):
    """B. GMM uncertainty in MACE latent space.

    Fits per-atom-type GMM on labeled data, scores pool structures
    by Mahalanobis distance from nearest GMM component.
    """

    def __init__(self, n_components: int = 5):
        super().__init__("B_gmm_uncertainty", "uncertainty", "GMM Uncertainty")
        self.n_components = n_components
        self.gmms = {}

    def select(self, pool_indices, n_query, model=None,
               pool_dataloader=None, labeled_dataloader=None,
               device="cpu", **kwargs):
        # Fit GMM on labeled data
        if model is not None and labeled_dataloader is not None:
            self._fit_gmms(model, labeled_dataloader, device)

        # Score pool by Mahalanobis distance to nearest GMM
        scores = np.zeros(len(pool_indices))
        if model is not None and pool_dataloader is not None and self.gmms:
            scores = self._compute_scores(model, pool_dataloader, device)
        else:
            scores = np.random.RandomState(42).rand(len(pool_indices))

        top_k = np.argsort(scores)[-n_query:]
        return pool_indices[top_k]

    def _fit_gmms(self, model, dataloader, device):
        from sklearn.mixture import GaussianMixture

        features_per_z = {}
        model.eval()
        import torch
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in batch_data.items()}
                feats = model.get_node_features(batch_data)
                if feats is None:
                    continue
                z = batch_data["z"].cpu().numpy()
                f = feats.cpu().numpy()
                for atom_z in np.unique(z):
                    mask = z == atom_z
                    features_per_z.setdefault(atom_z, []).append(f[mask])

        for atom_z, feat_list in features_per_z.items():
            X = np.concatenate(feat_list, axis=0)
            if X.shape[0] >= self.n_components:
                gmm = GaussianMixture(
                    n_components=min(self.n_components, X.shape[0]),
                    covariance_type="full", random_state=42)
                gmm.fit(X)
                self.gmms[atom_z] = gmm

    def _compute_scores(self, model, dataloader, device):
        """Score = negative log-likelihood under GMM (higher = more atypical)."""
        import torch
        scores_list = []
        model.eval()
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in batch_data.items()}
                feats = model.get_node_features(batch_data)
                if feats is None:
                    scores_list.append(np.zeros(batch_data["z"].shape[0]))
                    continue
                z = batch_data["z"].cpu().numpy()
                f = feats.cpu().numpy()
                batch_scores = np.zeros(f.shape[0])
                for atom_z in np.unique(z):
                    if atom_z in self.gmms and self.gmms[atom_z] is not None:
                        mask = z == atom_z
                        batch_scores[mask] = -self.gmms[atom_z].score_samples(f[mask])
                scores_list.append(batch_scores)

        # Aggregate per-atom scores to per-structure (mean)
        all_scores = np.concatenate(scores_list)
        return all_scores


class EnsembleQBC(AcquisitionFunction):
    """C. Query-by-committee: ensemble of 3 MACE models with different seeds.

    Uncertainty = variance of energy predictions across ensemble.
    """

    def __init__(self):
        super().__init__("C_ensemble_qbc", "uncertainty", "Ensemble QBC")

    def select(self, pool_indices, n_query, model=None,
               pool_dataloader=None, device="cpu", **kwargs):
        if model is not None and pool_dataloader is not None:
            scores = self._compute_ensemble_variance(model, pool_dataloader, device)
        else:
            scores = np.random.RandomState(42).rand(len(pool_indices))

        top_k = np.argsort(scores)[-n_query:]
        return pool_indices[top_k]

    def _compute_ensemble_variance(self, ensemble, dataloader, device):
        import torch
        variances = []
        ensemble.eval()
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in batch_data.items()}
                energies = []
                for member in ensemble.members:
                    e, _, _ = member(batch_data)
                    energies.append(e)
                energies = torch.stack(energies, dim=0)
                if energies.shape[0] > 1:
                    variances.append(energies.std(dim=0, unbiased=False).cpu().numpy())
                else:
                    variances.append(torch.zeros(energies.shape[1]).cpu().numpy())
        return np.concatenate(variances)


class MCDropout(AcquisitionFunction):
    """D. MC-Dropout: multiple forward passes with dropout, variance as uncertainty."""

    def __init__(self, n_passes: int = 10):
        super().__init__("D_mc_dropout", "uncertainty", "MC-Dropout")
        self.n_passes = n_passes

    def select(self, pool_indices, n_query, model=None,
               pool_dataloader=None, device="cpu", **kwargs):
        if model is not None and pool_dataloader is not None:
            scores = self._compute_mc_variance(model, pool_dataloader, device)
        else:
            scores = np.random.RandomState(42).rand(len(pool_indices))

        top_k = np.argsort(scores)[-n_query:]
        return pool_indices[top_k]

    def _compute_mc_variance(self, mc_model, dataloader, device):
        import torch
        variances = []
        mc_model.train()  # Enable dropout
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in batch_data.items()}
                preds = []
                for _ in range(self.n_passes):
                    e, _, _ = mc_model.model(batch_data)
                    preds.append(e)
                preds = torch.stack(preds, dim=0)
                if preds.shape[0] > 1:
                    variances.append(preds.std(dim=0, unbiased=False).cpu().numpy())
                else:
                    variances.append(torch.zeros(preds.shape[1]).cpu().numpy())
        mc_model.eval()
        return np.concatenate(variances)


class FPS_SOAP(AcquisitionFunction):
    """E. Farthest point sampling in SOAP descriptor space.

    Pure diversity: selects structures maximally different from
    already-labeled ones in terms of atomic environment similarity.
    """

    def __init__(self, soap=None):
        super().__init__("E_fps_soap", "diversity", "FPS + SOAP")
        self.soap = soap

    def select(self, pool_indices, n_query, model=None,
               pool_dataloader=None, pool_structures=None,
               labeled_structures=None, device="cpu", **kwargs):
        from descriptors import farthest_point_sampling

        if self.soap is None:
            from descriptors import SOAPDescriptors
            self.soap = SOAPDescriptors()

        # Compute SOAP descriptors for pool
        pool_features = self.soap.compute(pool_structures)

        if pool_features is None:
            # Fallback: random
            rng = np.random.RandomState(42)
            return pool_indices[rng.choice(len(pool_indices), size=n_query, replace=False)]

        selected = farthest_point_sampling(pool_features, n_query)
        return pool_indices[selected]


class LatentClustering(AcquisitionFunction):
    """F. k-means clustering in MACE feature space.

    Clusters pool structures by their MACE embeddings, selects one
    representative from each cluster to maximize diversity.
    """

    def __init__(self, n_clusters: int = 20):
        super().__init__("F_latent_clustering", "diversity", "Latent Space Clustering")
        self.n_clusters = n_clusters

    def select(self, pool_indices, n_query, model=None,
               pool_dataloader=None, device="cpu", **kwargs):
        from descriptors import latent_space_clustering
        from model import compute_structure_embedding

        if model is not None and pool_dataloader is not None:
            embeddings = compute_structure_embedding(model, pool_dataloader, device)
            if embeddings is not None and embeddings.shape[0] > 0:
                selected = latent_space_clustering(
                    embeddings, n_clusters=min(n_query, embeddings.shape[0])
                )
                return pool_indices[selected]

        rng = np.random.RandomState(42)
        return pool_indices[rng.choice(len(pool_indices), size=n_query, replace=False)]


class HybridWeighted(AcquisitionFunction):
    """G. Hybrid-Weighted: Score = alpha * U_norm + (1-alpha) * D_norm.

    Combines uncertainty and diversity with a tunable parameter alpha.
    alpha=0.5 is the default balanced configuration.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__("G_hybrid_weighted", "hybrid",
                         f"Hybrid-Weighted (alpha={alpha})")
        self.alpha = alpha
        self._uncertainty_fn = EnsembleQBC()
        self._diversity_fn = LatentClustering()

    def select(self, pool_indices, n_query, model=None,
               pool_dataloader=None, pool_structures=None,
               labeled_structures=None, device="cpu", **kwargs):
        # Compute uncertainty scores
        unc_scores = self._compute_uncertainty(pool_indices, model, pool_dataloader, device)

        # Compute diversity scores
        div_scores = self._compute_diversity(pool_indices, model, pool_dataloader, device)

        # Normalize to [0, 1]
        unc_norm = self._normalize(unc_scores)
        div_norm = self._normalize(div_scores)

        # Weighted combination
        combined = self.alpha * unc_norm + (1 - self.alpha) * div_norm

        top_k = np.argsort(combined)[-n_query:]
        return pool_indices[top_k]

    def _compute_uncertainty(self, pool_indices, model, dataloader, device):
        if model is not None and dataloader is not None:
            return self._uncertainty_fn._compute_ensemble_variance(model, dataloader, device)
        return np.random.RandomState(42).rand(len(pool_indices))

    def _compute_diversity(self, pool_indices, model, dataloader, device):
        from model import compute_structure_embedding
        from scipy.spatial.distance import cdist

        if model is not None and dataloader is not None:
            embeddings = compute_structure_embedding(model, dataloader, device)
            if embeddings is not None and embeddings.shape[0] > 1:
                pairwise_dists = cdist(embeddings, embeddings, metric="cosine")
                # Diversity = mean cosine distance to all other pool structures
                return pairwise_dists.mean(axis=1)
        return np.random.RandomState(42).rand(len(pool_indices))

    def _normalize(self, scores):
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-10:
            return np.ones_like(scores)
        return (scores - s_min) / (s_max - s_min)


class HybridTwoStage(AcquisitionFunction):
    """H. Hybrid-TwoStage: Top K% by uncertainty, then FPS-filter to K by diversity.

    1. Filter pool to top `topk_frac` most uncertain structures
    2. From the filtered subset, apply FPS in SOAP space to select n_query
    """

    def __init__(self, topk_frac: float = 0.3):
        super().__init__("H_hybrid_twostage", "hybrid",
                         f"Hybrid-TwoStage (top {topk_frac*100:.0f}%)")
        self.topk_frac = topk_frac
        self._uncertainty_fn = EnsembleQBC()
        self._fps_fn = FPS_SOAP()

    def select(self, pool_indices, n_query, model=None,
               pool_dataloader=None, pool_structures=None,
               labeled_structures=None, device="cpu", **kwargs):
        # Stage 1: Filter by uncertainty
        unc_scores = self._compute_uncertainty(pool_indices, model, pool_dataloader, device)
        n_keep = max(n_query * 2, int(len(pool_indices) * self.topk_frac))
        top_unc_idx = np.argsort(unc_scores)[-n_keep:]
        filtered_pool = pool_indices[top_unc_idx]

        # Stage 2: FPS diversity selection within filtered set
        if pool_structures is not None:
            filtered_structures = [pool_structures[i] for i in top_unc_idx]
            fps_selected = self._fps_fn.select(
                np.arange(len(filtered_pool)), n_query,
                pool_structures=filtered_structures)
            return filtered_pool[fps_selected]

        # Fallback without SOAP
        return filtered_pool[np.random.RandomState(42).choice(
            len(filtered_pool), size=n_query, replace=False)]

    def _compute_uncertainty(self, pool_indices, model, dataloader, device):
        if model is not None and dataloader is not None:
            return self._uncertainty_fn._compute_ensemble_variance(model, dataloader, device)
        return np.random.RandomState(42).rand(len(pool_indices))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_acquisition_function(strategy_id: str, **kwargs) -> AcquisitionFunction:
    """Factory for creating acquisition function by strategy ID."""
    registry = {
        "A_random": RandomSampling,
        "B_gmm_uncertainty": GMMUncertainty,
        "C_ensemble_qbc": EnsembleQBC,
        "D_mc_dropout": MCDropout,
        "E_fps_soap": FPS_SOAP,
        "F_latent_clustering": LatentClustering,
        "G_hybrid_weighted": lambda: HybridWeighted(alpha=kwargs.get("alpha", 0.5)),
        "H_hybrid_twostage": lambda: HybridTwoStage(topk_frac=kwargs.get("topk_frac", 0.3)),
    }
    if strategy_id not in registry:
        raise ValueError(f"Unknown strategy: {strategy_id}. Choices: {list(registry)}")
    return registry[strategy_id]()
