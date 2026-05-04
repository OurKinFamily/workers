"""
Face worker CLI.

Commands:
  extract <photo_path> [<photo_path> ...]   Detect faces and write to DB
  cluster                                    Run DBSCAN on all DB embeddings
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def cmd_extract(paths: list[str]):
    from face.extract import FaceExtractor
    from face import db

    extractor = FaceExtractor()
    for path in paths:
        log.info(f"Extracting: {path}")
        try:
            faces = extractor.extract(path)
            for face in faces:
                face_id = db.upsert_face(face)
                log.info(f"  face {face['face_index']} → face_id={face_id}")
        except Exception as e:
            log.error(f"  failed: {e}")


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


def main():
    parser = argparse.ArgumentParser(prog="face")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Detect faces in photos and write to DB")
    p_extract.add_argument("paths", nargs="+", metavar="photo_path")

    p_cluster = sub.add_parser("cluster", help="Run DBSCAN on all embeddings in DB")
    p_cluster.add_argument("--eps", type=float, default=0.4)
    p_cluster.add_argument("--min-samples", type=int, default=3)

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args.paths)
    elif args.command == "cluster":
        cmd_cluster(args.eps, args.min_samples)


if __name__ == "__main__":
    main()
