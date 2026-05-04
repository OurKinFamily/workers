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

Scaffolded — stubs only. No dispatch mechanism yet.

**Face worker** (`face/`) is the first priority:
- `extract.py` — detect faces in a photo, save crops to `/photos/__faces/crops/`, return embeddings
- `cluster.py` — read embeddings from DB, run DBSCAN, write cluster assignments back
- Port logic from `photo-intelligence/services/face-recognition/` — keep the ML parameters, replace the file I/O

## Decisions Made

- **One container per worker** — face extraction and clustering are separate concerns with different resource profiles (GPU vs CPU)
- **Dispatch mechanism TBD** — options: DB jobs table, Redis queue, file watcher. Not decided yet.
- **Writes to DB, not sidecars** — face detections and embeddings go into a database (MariaDB planned), not `.faces.json` files alongside photos
- **Crops still written to disk** — `/photos/__faces/crops/` stays as the crop image store; DB holds paths + embeddings
- **PHOTOS_ROOT env var** — all paths relative to this, default `/photos`

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
