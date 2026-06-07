"""SOAP descriptor computation for diversity-based acquisition.

Uses dscribe for SOAP (Smooth Overlap of Atomic Positions) descriptors,
which form a universal, rotationally-invariant representation of local
atomic environments.
"""

import numpy as np
from typing import List


class SOAPDescriptors:
    """Compute global SOAP descriptors for structures.

    A global SOAP descriptor is the average of atomic SOAP vectors,
    yielding a fixed-length fingerprint for each structure regardless
    of atom count. Used for farthest point sampling (FPS).
    """

    def __init__(self, rcut: float = 5.0, nmax: int = 6, lmax: int = 4,
                 species: List[int] = None):
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.species = species or list(range(1, 84))  # H to Bi
        self._soap = None

    def _get_soap(self):
        if self._soap is None:
            try:
                from dscribe.descriptors import SOAP
                self._soap = SOAP(
                    species=self.species,
                    periodic=True,
                    r_cut=self.rcut,
                    n_max=self.nmax,
                    l_max=self.lmax,
                    average="inner",
                    sparse=False,
                )
            except ImportError:
                print("[WARNING] dscribe not installed. Using simple distance-based diversity.")
                self._soap = None
        return self._soap

    def compute(self, structures) -> np.ndarray:
        """Compute SOAP descriptors for a list of ASE Atoms.

        Args:
            structures: list of ASE Atoms objects

        Returns:
            descriptors: [n_structures, feature_dim] or None
        """
        try:
            soap = self._get_soap()
            if soap is None:
                return None
            return soap.create(structures, n_jobs=1)
        except Exception as e:
            print(f"  [WARNING] SOAP computation failed: {e}")
            return None


def farthest_point_sampling(
    features: np.ndarray,
    n_select: int,
    initial_indices: List[int] = None,
    metric: str = "euclidean",
) -> np.ndarray:
    """Farthest point sampling: iteratively select points maximizing
    minimum distance to already-selected set.

    Args:
        features: [n_samples, feature_dim] descriptor matrix
        n_select: number of points to select
        initial_indices: indices already in the selected set
        metric: "euclidean" or "cosine"

    Returns:
        selected: indices of selected points
    """
    from scipy.spatial.distance import cdist

    n = features.shape[0]
    selected = list(initial_indices or [])

    if len(selected) >= n_select:
        return np.array(selected[:n_select])

    # Initialize distances
    if len(selected) > 0:
        dists = cdist(features, features[selected], metric=metric).min(axis=1)
    else:
        # Pick first point randomly
        first = np.random.RandomState(42).randint(n)
        selected.append(first)
        dists = cdist(features, features[[first]], metric=metric).ravel()

    for _ in range(len(selected), n_select):
        # Select furthest
        next_idx = int(np.argmax(dists))
        selected.append(next_idx)
        # Update distances
        new_dists = cdist(features, features[[next_idx]], metric=metric).ravel()
        dists = np.minimum(dists, new_dists)

    return np.array(selected)


def latent_space_clustering(
    embeddings: np.ndarray,
    n_clusters: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """k-means clustering in latent space, return cluster centroids' nearest
    neighbors as selected indices.

    This ensures selected structures span diverse regions of the feature space.
    """
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=min(n_clusters, embeddings.shape[0]),
                    random_state=seed, n_init=10)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_

    # For each centroid, find nearest actual structure
    from scipy.spatial.distance import cdist
    dists = cdist(centroids, embeddings, metric="euclidean")
    selected = dists.argmin(axis=1)

    return selected
