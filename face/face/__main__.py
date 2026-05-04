"""
Face worker CLI.

Commands:
  serve                                      Keep container alive (default CMD)
  extract <path> [<path> ...]               Detect faces in photos/dirs, write to DB
  cluster                                    Run DBSCAN on all DB embeddings
"""

import argparse
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.webp', '.tiff', '.tif', '.bmp'}
SKIP_DIRS  = {'__faces', '__people', '__places', '__data'}


def _collect_photos(paths: list[str], force: bool) -> list[Path]:
    """Expand file/dir paths into a flat list of image files to process."""
    from face import db

    already_processed = set()
    if not force:
        already_processed = db.all_processed_paths()

    collected = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            if p.suffix.lower() in IMAGE_EXTS and str(p) not in already_processed:
                collected.append(p)
        elif p.is_dir():
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                for fname in sorted(files):
                    fp = Path(root) / fname
                    if fp.suffix.lower() in IMAGE_EXTS and str(fp) not in already_processed:
                        collected.append(fp)
        else:
            log.warning(f"Skipping — not a file or directory: {p}")
    return collected


def cmd_extract(paths: list[str], force: bool = False):
    from face.extract import FaceExtractor
    from face import db

    photos = _collect_photos(paths, force)
    total  = len(photos)
    if not total:
        log.info("No unprocessed photos found")
        return

    log.info(f"Processing {total:,} photos")
    extractor = FaceExtractor()

    for i, photo_path in enumerate(photos, 1):
        log.info(f"[{i}/{total}] {photo_path}")
        try:
            faces = extractor.extract(photo_path)
            for face in faces:
                db.upsert_face(face)
            log.info(f"  → {len(faces)} face(s)")
        except Exception as e:
            log.error(f"  failed: {e}")

    log.info(f"Done — processed {total:,} photos")


def cmd_cluster(eps: float, min_samples: int):
    from face import db
    from face.cluster import cluster, cluster_noise

    faces = db.all_faces_with_embeddings()
    if not faces:
        log.info("No faces in DB — nothing to cluster")
        return

    run_id = db.start_cluster_run(eps, min_samples, len(faces))
    log.info(f"Started cluster run {run_id} on {len(faces):,} faces")

    clustered = cluster(faces, id_field="face_id", eps=eps, min_samples=min_samples)
    clustered = cluster_noise(clustered)

    n_clusters = len({f["cluster_id"] for f in clustered if f["cluster_id"] >= 0})
    n_noise    = sum(1 for f in clustered if f["cluster_id"] == -1)

    db.save_cluster_assignments(run_id, clustered)
    db.finish_cluster_run(run_id, n_clusters, n_noise)
    log.info(f"Run {run_id} complete — {n_clusters} clusters, {n_noise} noise")


def cmd_serve():
    log.info("Face worker ready — awaiting docker exec commands")
    while True:
        time.sleep(3600)


def main():
    parser = argparse.ArgumentParser(prog="face")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="Keep container alive (used as default CMD)")

    p_extract = sub.add_parser("extract", help="Detect faces in photos/dirs and write to DB")
    p_extract.add_argument("paths", nargs="+", metavar="path")
    p_extract.add_argument("--force", action="store_true", help="Re-process already-seen photos")

    p_cluster = sub.add_parser("cluster", help="Run DBSCAN on all embeddings in DB")
    p_cluster.add_argument("--eps", type=float, default=0.4)
    p_cluster.add_argument("--min-samples", type=int, default=3)

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve()
    elif args.command == "extract":
        cmd_extract(args.paths, force=args.force)
    elif args.command == "cluster":
        cmd_cluster(args.eps, args.min_samples)


if __name__ == "__main__":
    main()
