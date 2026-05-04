"""
Face clustering — groups face embeddings into identity clusters using DBSCAN.

Input:  list of face dicts with 'embedding' and an identifier field
Output: same dicts with 'cluster_id' added (-1 = noise/unclusterable)

Tuned parameters (from photo-intelligence):
  eps=0.4, min_samples=3, metric='cosine'
"""

import logging
import time

import numpy as np
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)


def cluster(
    faces: list[dict],
    id_field: str = "face_id",
    eps: float = 0.4,
    min_samples: int = 3,
) -> list[dict]:
    """
    Run DBSCAN on face embeddings.

    Args:
        faces:       list of dicts, each must have 'embedding' (512-dim list) and id_field
        id_field:    name of the unique identifier field on each dict
        eps:         DBSCAN max cosine distance within a cluster
        min_samples: DBSCAN minimum cluster size

    Returns:
        Same dicts with 'cluster_id' (int) added. -1 = noise.
    """
    if not faces:
        return []

    log.info(f"Clustering {len(faces):,} faces (eps={eps}, min_samples={min_samples})")
    matrix = np.array([f["embedding"] for f in faces], dtype=np.float32)

    t0 = time.time()
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    labels = db.fit_predict(matrix)
    elapsed = time.time() - t0

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    log.info(f"Done in {elapsed:.1f}s — {n_clusters} clusters, {n_noise} noise")

    return [
        {**face, "cluster_id": int(label)}
        for face, label in zip(faces, labels)
    ]


def cluster_noise(faces: list[dict], eps: float = 0.45, min_samples: int = 2) -> list[dict]:
    """
    Re-cluster noise faces (cluster_id == -1) with looser parameters.
    Returns the noise faces with updated cluster_ids (still negative to distinguish
    from main clusters — offset by -2 so they don't collide with -1).
    """
    noise = [f for f in faces if f.get("cluster_id") == -1]
    if not noise:
        return faces

    log.info(f"Re-clustering {len(noise):,} noise faces")
    reclustered = cluster(noise, eps=eps, min_samples=min_samples)

    # Offset noise sub-cluster IDs so they don't collide with main cluster IDs
    max_noise_id = max((f["cluster_id"] for f in reclustered if f["cluster_id"] != -1), default=-1)
    noise_id_map = {}
    next_id = -(max_noise_id + 2) if max_noise_id >= 0 else -2

    for f in reclustered:
        if f["cluster_id"] != -1:
            if f["cluster_id"] not in noise_id_map:
                noise_id_map[f["cluster_id"]] = next_id
                next_id -= 1
            f["cluster_id"] = noise_id_map[f["cluster_id"]]

    # Merge reclustered noise back into the full list (preserve original order)
    noise_iter = iter(reclustered)
    result = []
    for f in faces:
        if f.get("cluster_id") == -1:
            result.append(next(noise_iter))
        else:
            result.append(f)
    return result
