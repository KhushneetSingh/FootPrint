# Football Video Analysis Pipeline

A modular Python pipeline that ingests a football match video, detects and tracks players across frames using YOLOv8 and ByteTrack, computes per-player movement metrics, and produces an annotated output video with bounding-box overlays, track-ID labels, motion trails, speed reports, and individual heatmap visualisations.

---

## Project Structure

```
football_analysis/
├── data/
│   ├── input/                # Place raw match video here
│   └── output/               # Annotated video, CSVs, and heatmaps
├── src/
│   ├── __init__.py           # Package docstring
│   ├── config.py             # All tuneable constants in one place
│   ├── detector.py           # YOLOv8 detection + ByteTrack wrapper
│   ├── tracker.py            # Motion trail history bookkeeping
│   ├── annotator.py          # Frame drawing utilities (boxes, IDs, trails)
│   ├── metrics.py            # Speed estimation + heatmap generation
│   └── pipeline.py           # Orchestrator — wires all modules together
├── main.py                   # CLI entry point (argparse)
├── requirements.txt          # Pinned Python dependencies
├── .gitignore                # Git exclusions (model weights, data, caches)
└── README.md                 # This file
```

---

## Setup Instructions

1. **UnZip the File:**

   ```bash
   unzip football_analysis.zip
   cd football_analysis
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv && source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   > For GPU support, install PyTorch separately first from [pytorch.org](https://pytorch.org) with the correct CUDA version.

4. **Place your input video:**
   Copy your match clip into `data/input/` (default expected name: `match.mp4`).

---

## How to Run

```bash
python main.py --input data/input/match.mp4
```

**Optional arguments:**

| Flag       | Default                     | Description                   |
| ---------- | --------------------------- | ----------------------------- |
| `--input`  | `data/input/match.mp4`      | Path to the raw video         |
| `--output` | `data/output/annotated.mp4` | Path for the annotated output |
| `--model`  | `yolov8n.pt`                | YOLOv8 weights file           |

**Outputs generated in `data/output/`:**

- `annotated.mp4` — video with bounding boxes, track IDs, and motion trails.
- `speed_report.csv` — per-player average speed and frames tracked.
- `heatmaps/player_<ID>_heatmap.png` — spatial density map per player.

---

## Pipeline Design & Design Choices

### Why YOLOv8 for detection?

YOLOv8 (Ultralytics) provides state-of-the-art real-time object detection with a simple Python API. The `yolov8n.pt` nano variant strikes an effective balance between speed and accuracy for this use case, running at interactive frame rates even on CPU.

### Why ByteTrack for tracking?

ByteTrack is bundled directly inside Ultralytics, requiring zero additional installation. It uses a simple yet effective association strategy that matches high- and low-confidence detections separately, providing stable track IDs even through brief occlusions.

### Why are detection and tracking separate concerns?

- **Detection** (`model.predict`) answers: _"Who is in this frame?"_ — no memory of previous frames.
- **Tracking** (`model.track`) answers: _"Which detection in frame N corresponds to which in frame N-1?"_ — maintains identity across time.

In this pipeline, `detector.py` handles both via `model.track()` with `persist=True`, while `tracker.py` is responsible for _recording_ centroid trails and `metrics.py` for _analysing_ them. This separation means any module can be swapped independently (e.g. replacing ByteTrack with DeepSORT) without touching the metrics or annotation code.

---

## Speed Estimation Approach

```
avg_speed (px/s) = (total_pixel_displacement / num_frames) × fps
```

For each player, the total Euclidean distance between consecutive centroid positions is summed and divided by the number of frames the player was observed, then scaled by the video's frames-per-second.

> **Important limitation:** This speed is measured in **pixel space**, not real-world m/s. Converting to physical units would require a homography transform mapping pixel coordinates to known pitch dimensions (e.g. using corner-flag reference points).

---

## Assumptions Made

- **Static camera:** The input video is assumed to be from a fixed camera. Pan/tilt/zoom cameras would require stabilisation before processing.
- **COCO "person" class:** Detection uses the generic COCO class 0 (person). A football-specific fine-tuned model would improve recall on partially occluded or distant players.
- **Fixed FPS:** Frame rate is read once from the video metadata and assumed constant throughout.
- **No team differentiation:** All detected players are treated identically — no jersey-colour-based team assignment.
- **Track IDs reset per run:** ByteTrack IDs are not persistent across separate video files or pipeline executions.
- **Empirical heatmap sigma:** The Gaussian smoothing sigma (default 20) is chosen empirically and should be tuned for different video resolutions.

---

## Limitations & Possible Improvements

| Area               | Limitation                          | Possible Improvement                                                                      |
| ------------------ | ----------------------------------- | ----------------------------------------------------------------------------------------- |
| Speed accuracy     | Pixel-space only                    | **Homography transform** to map pixel coordinates to real pitch metres using corner flags |
| Team awareness     | All players treated equally         | **Colour clustering** (K-Means on HSV crops) to assign team labels and per-team heatmaps  |
| Ball tracking      | Not tracked                         | Separate YOLO head or custom-trained detector for the football                            |
| Occlusion handling | IDs can swap during long occlusions | **Re-identification** (ReID) with appearance embeddings (e.g. OSNet + DeepSORT)           |
| Detection quality  | Generic COCO model                  | **Fine-tuned YOLO** on SoccerNet or football-specific datasets                            |
| Tactical analysis  | Not implemented                     | Compute team centroids, formations, off-ball spacing from track positions                 |
| User interface     | CLI only                            | **Streamlit dashboard** with interactive video player and heatmap overlays                |
