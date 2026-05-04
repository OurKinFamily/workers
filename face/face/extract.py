"""
Face extraction — detects faces in a photo, saves crops to disk.

Returns structured dicts; no sidecars written. The caller decides
what to do with the results (write to DB, queue for clustering, etc).
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

PHOTOS_ROOT = Path(os.environ.get("PHOTOS_ROOT", "/photos"))
CROPS_DIR   = PHOTOS_ROOT / "__faces" / "crops"
ARCHIVE_ROOT = PHOTOS_ROOT / "archive"

log = logging.getLogger(__name__)


class FaceExtractor:
    def __init__(self, model_name: str = "buffalo_l"):
        self.model_name = model_name
        self._app = None

    def load(self):
        if self._app is not None:
            return
        log.info(f"Loading InsightFace model: {self.model_name}")
        self._app = FaceAnalysis(name=self.model_name)
        try:
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            log.info("Model loaded on GPU")
        except Exception:
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            log.info("Model loaded on CPU")

    def extract(self, photo_path: str | Path) -> list[dict]:
        """
        Detect faces in photo_path, save crops, return face dicts.

        Each dict contains:
          photo_path, face_index, bbox, confidence, embedding,
          crop_path, age (optional), gender (optional)
        """
        if self._app is None:
            self.load()

        photo_path = Path(photo_path)
        img = cv2.imread(str(photo_path))
        if img is None:
            raise ValueError(f"Could not read image: {photo_path}")

        faces = self._app.get(img)
        results = []

        for i, face in enumerate(faces):
            bbox = face.bbox.tolist()
            crop_path = self._save_crop(img, bbox, photo_path, i)

            result = {
                "photo_path":  str(photo_path.relative_to(PHOTOS_ROOT)),
                "face_index":  i,
                "bbox":        bbox,
                "confidence":  float(face.det_score),
                "embedding":   face.normed_embedding.tolist(),
                "crop_path":   str(crop_path.relative_to(PHOTOS_ROOT)) if crop_path else None,
            }
            if hasattr(face, "age"):
                result["age"] = int(face.age)
            if hasattr(face, "gender"):
                result["gender"] = int(face.gender)

            results.append(result)

        return results

    def _save_crop(self, img: np.ndarray, bbox: list, photo_path: Path, face_index: int) -> Path | None:
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            try:
                rel = photo_path.relative_to(ARCHIVE_ROOT)
            except ValueError:
                rel = Path(photo_path.name)

            crop_dir = CROPS_DIR / rel.parent
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / f"{photo_path.name}_face{face_index}.jpg"
            cv2.imwrite(str(crop_path), crop)
            return crop_path
        except Exception as e:
            log.warning(f"Failed to save crop: {e}")
            return None
