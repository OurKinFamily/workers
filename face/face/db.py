"""
MariaDB access layer for the face worker.

All paths stored relative to PHOTOS_ROOT (no leading slash).
Embeddings stored as raw float32 bytes (512 × 4 = 2048 bytes per face).
"""

import json
import logging
import os
import struct

import pymysql
import pymysql.cursors

log = logging.getLogger(__name__)

_pool = None


def _connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 3306)),
        user=os.environ.get("DB_USER", "workers"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ.get("DB_NAME", "ourkin_workers"),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def get_conn() -> pymysql.connections.Connection:
    global _pool
    try:
        if _pool is not None:
            _pool.ping(reconnect=True)
            return _pool
    except Exception:
        pass
    _pool = _connect()
    return _pool


# ---------------------------------------------------------------------------
# Faces
# ---------------------------------------------------------------------------

def upsert_face(face: dict) -> int:
    """
    Insert or update a face row. Returns face_id.

    face dict must have: photo_path, face_index, embedding (list[float]),
    bbox, confidence. Optional: crop_path, age, gender.
    """
    embedding_bytes = struct.pack(f"{len(face['embedding'])}f", *face["embedding"])
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO faces (photo_path, face_index, crop_path, bbox, confidence, age, gender, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                crop_path  = VALUES(crop_path),
                bbox       = VALUES(bbox),
                confidence = VALUES(confidence),
                age        = VALUES(age),
                gender     = VALUES(gender),
                embedding  = VALUES(embedding)
            """,
            (
                face["photo_path"],
                face["face_index"],
                face.get("crop_path"),
                json.dumps(face["bbox"]),
                face["confidence"],
                face.get("age"),
                face.get("gender"),
                embedding_bytes,
            ),
        )
        if cur.lastrowid:
            return cur.lastrowid
        cur.execute(
            "SELECT face_id FROM faces WHERE photo_path=%s AND face_index=%s",
            (face["photo_path"], face["face_index"]),
        )
        return cur.fetchone()["face_id"]


def faces_without_cluster(run_id: int) -> list[dict]:
    """Return all faces not yet assigned in run_id, with embeddings decoded."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.face_id, f.photo_path, f.face_index, f.crop_path, f.embedding
            FROM faces f
            LEFT JOIN face_clusters fc ON fc.face_id = f.face_id AND fc.run_id = %s
            WHERE fc.face_id IS NULL
            """,
            (run_id,),
        )
        rows = cur.fetchall()

    result = []
    for row in rows:
        n = len(row["embedding"]) // 4
        row["embedding"] = list(struct.unpack(f"{n}f", row["embedding"]))
        result.append(row)
    return result


def all_faces_with_embeddings() -> list[dict]:
    """Return every face with decoded embedding — for a full cluster run."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT face_id, photo_path, face_index, crop_path, embedding FROM faces")
        rows = cur.fetchall()

    for row in rows:
        n = len(row["embedding"]) // 4
        row["embedding"] = list(struct.unpack(f"{n}f", row["embedding"]))
    return rows


# ---------------------------------------------------------------------------
# Cluster runs
# ---------------------------------------------------------------------------

def start_cluster_run(eps: float, min_samples: int, face_count: int) -> int:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO cluster_runs (eps, min_samples, face_count) VALUES (%s, %s, %s)",
            (eps, min_samples, face_count),
        )
        return cur.lastrowid


def finish_cluster_run(run_id: int, cluster_count: int, noise_count: int):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE cluster_runs SET cluster_count=%s, noise_count=%s WHERE run_id=%s",
            (cluster_count, noise_count, run_id),
        )


def save_cluster_assignments(run_id: int, faces: list[dict]):
    """Bulk-insert face→cluster assignments for a completed run."""
    if not faces:
        return
    conn = get_conn()
    rows = [(run_id, f["face_id"], f["cluster_id"]) for f in faces]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT IGNORE INTO face_clusters (run_id, face_id, cluster_id) VALUES (%s, %s, %s)",
            rows,
        )
    log.info(f"Saved {len(rows)} cluster assignments for run {run_id}")
