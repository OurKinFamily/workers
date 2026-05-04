# Workers

Background workers for the ourkin platform — ML processing, face detection, clustering.

**Note:** Read the root ourkin CLAUDE.md at `/home/stephen/Documents/ourkin/CLAUDE.md` first for project-wide conventions.

## Purpose

Workers handle compute-heavy tasks that don't belong in the API. Each worker is a separate Python package with its own Docker container. They run independently and can be deployed/scaled without touching the API or app.

Workers are the eventual replacement for the face pipeline in `photo-intelligence/`. Logic (DBSCAN params, InsightFace model choice, crop extraction) is worth carrying over; file-based I/O (JSON sidecars) is not.

## Structure

```
workers/
  docker-compose.yml    ← runs all workers together
  face/                 ← face detection + clustering worker
    Dockerfile
    requirements.txt
    face/
      extract.py        ← InsightFace detection, crop saving
      cluster.py        ← DBSCAN clustering on embeddings
      __main__.py       ← entry point
```

## Current State

Face worker extract/cluster logic is complete. MariaDB schema and DB access layer written.
`__main__.py` has CLI skeleton for `extract` and `cluster` commands.
Dispatch mechanism (how to trigger workers) is still TBD.

**Face worker** (`face/`) is the first priority:
- `extract.py` — detect faces in a photo, save crops to `/photos/__faces/crops/`, return embeddings
- `cluster.py` — read embeddings from DB, run DBSCAN, write cluster assignments back
- `db.py` — MariaDB access: upsert_face, all_faces_with_embeddings, cluster run lifecycle

## Decisions Made

- **One container per worker** — face extraction and clustering are separate concerns with different resource profiles (GPU vs CPU)
- **Dispatch mechanism TBD** — options: DB jobs table, Redis queue, file watcher. Not decided yet.
- **MariaDB for ML data** — faces table (embeddings as BLOB), cluster_runs, face_clusters. Person↔face links stay in Neo4j (APPEARS_IN).
- **Crops still written to disk** — `/photos/__faces/crops/` stays as the crop image store; DB holds paths + embeddings
- **PHOTOS_ROOT env var** — all paths relative to this, default `/photos`
- **Embeddings as raw BLOB** — 512 × float32 = 2048 bytes; pack/unpack with `struct`

## MariaDB Schema

```
faces          — one row per detected face; embedding as BLOB
cluster_runs   — one row per DBSCAN run (eps, min_samples, stats)
face_clusters  — face_id → cluster_id mapping per run
```

Schema file: `ourkin/db/maria/schema.sql` (auto-loaded by MariaDB container on first start).
Data files: `ourkin/db/maria/data/` (bind-mounted, gitignored in the graph repo).
Credentials via env vars: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`.
Copy `.env.example` → `.env` and fill in passwords before `docker compose up`.

## Key Source Material (photo-intelligence)

| File | What to port |
|------|-------------|
| `services/face-recognition/face_recognition/extractors/face_sidecar.py` | InsightFace setup, detection loop, crop saving |
| `services/face-recognition/cluster_faces.py` | DBSCAN parameters (eps=0.4, min_samples=3), embedding matrix construction |
| `services/face-recognition/cluster_noise.py` | Noise re-clustering pass |
| `services/face-recognition/auto_assign.py` | Gallery-based auto-assignment logic |

## Docker

```bash
# Build and run all workers
docker compose up -d

# Face worker only
docker compose up -d face
```

GPU access for the face container will need `deploy.resources.reservations.devices` in docker-compose when wiring up the RTX 5060 Ti.
