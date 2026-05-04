"""
Face extraction — detects faces in a photo and writes crops to disk.

Input:  absolute path to a photo file
Output: list of detected faces with embeddings and crop paths
"""

from pathlib import Path


def extract_faces(photo_path: str, crops_dir: Path) -> list[dict]:
    """
    Detect faces in photo_path, save crops under crops_dir.
    Returns list of dicts: {face_index, crop_path, embedding, confidence, age, gender}
    """
    raise NotImplementedError
