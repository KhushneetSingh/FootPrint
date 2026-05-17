# `src/` — Module Documentation

This directory contains the core pipeline modules. Each module has a single responsibility and communicates through plain Python dicts and lists — no module directly depends on another's internals.

---

## Module Overview

### [`config.py`](config.py)
**Centralised constants and hyperparameters.**

All tuneable values — model paths, thresholds, file paths, and display settings — live here. No other module contains hard-coded magic numbers. Modify this file to adjust pipeline behaviour without touching any logic.

### [`detector.py`](detector.py)
**YOLOv8 detection + ByteTrack tracking wrapper.**

The only module that imports `ultralytics`. Exposes a `PlayerDetector` class with a single `.track(frame)` method that returns a list of detection dicts:

```python
[{'bbox': [x1, y1, x2, y2], 'track_id': int, 'conf': float, 'cls': int}, ...]
```

### [`tracker.py`](tracker.py)
**Motion trail and position history bookkeeping.**

Maintains two data structures per track ID:
- **`trail_history`** — a bounded `deque` of recent `(cx, cy)` positions for visual trail rendering
- **`full_history`** — an unbounded list of `(cx, cy, frame_no)` tuples for post-processing metrics

Also provides `filter_short_tracks(min_frames)` to remove ghost/noise tracks.

### [`annotator.py`](annotator.py)
**Frame-level drawing utilities.**

Pure rendering module — draws bounding boxes, track-ID labels, and motion trails onto each frame. Uses a 12-colour palette that cycles by track ID. Has no state and no side effects beyond the returned frame.

### [`metrics.py`](metrics.py)
**Speed estimation and heatmap generation.**

Two public functions:
- `compute_speeds(track_history, fps)` → per-player average speed in px/s
- `generate_heatmap(track_id, positions, frame_h, frame_w, out_dir)` → saves a PNG heatmap

### [`pipeline.py`](pipeline.py)
**Orchestrator — wires all modules into a video-processing loop.**

The only module that imports from multiple siblings. Handles:
1. Opening the input video and creating the output writer
2. Frame-by-frame detection → tracking → annotation loop
3. Post-processing: ghost track filtering, speed computation, heatmap generation, CSV export

---

## Data Flow

```
Video Frame
    │
    ▼
PlayerDetector.track(frame)
    │  returns: list[dict] with bbox, track_id, conf, cls
    ▼
Tracker.update(tracked_dets, frame_no)
    │  updates: trail_history (bounded) + full_history (unbounded)
    │  returns: trail_history for annotation
    ▼
Annotator.draw_frame(frame, tracked, trail_history)
    │  returns: annotated frame → written to output video
    ▼
[After all frames processed]
    │
Tracker.filter_short_tracks(min_frames)
    │
compute_speeds(full_history, fps) → CSV
generate_heatmap(positions, ...) → PNG per player
```

---

## Adding a New Module

1. Create `src/your_module.py` with a clear single responsibility
2. Import shared constants from `src.config`
3. Accept and return plain Python data structures (dicts, lists, tuples)
4. Wire it into `pipeline.py` — this is the only place cross-module imports should happen
5. Update `src/__init__.py` docstring to list the new module
