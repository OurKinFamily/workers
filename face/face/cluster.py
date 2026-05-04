"""
Face clustering — groups face embeddings into identity clusters using DBSCAN.

Input:  embeddings from DB
Output: cluster assignments written back to DB
"""


def cluster_faces(embeddings: list[dict], eps: float = 0.4, min_samples: int = 3) -> dict:
    """
    Run DBSCAN on embeddings.
    Returns {face_id: cluster_id} mapping.
    """
    raise NotImplementedError
