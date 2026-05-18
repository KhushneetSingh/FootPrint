<div align="center">

# ⚽ FootPrint Analytics

**AI-Powered Football Match Video Analysis Pipeline**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=white)](https://docs.ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A modular Python pipeline that ingests football match video, detects and tracks players using **YOLOv8 + ByteTrack**, computes per-player movement metrics, and produces an annotated output video with bounding-box overlays, track-ID labels, motion trails, speed reports, and individual heatmap visualisations.

</div>

---

## ✨ Features

| Feature                      | Description                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| 🎯 **Player Detection**      | Real-time detection using YOLOv8 medium model with configurable confidence thresholds |
| 🔗 **Multi-Object Tracking** | ByteTrack-based identity association across frames with ghost track filtering         |
| 🏃 **Speed Estimation**      | Per-player average speed computed from centroid displacement (pixel-space)            |
| 🗺️ **Spatial Heatmaps**      | Gaussian-smoothed density maps showing each player's movement coverage                |
| 🎬 **Annotated Video**       | Output video with bounding boxes, track IDs, and motion trail overlays                |
| 📊 **CSV Reports**           | Structured speed data export for further analysis                                     |

---

## 📁 Project Structure

```
SmartPitch-Analytics/
├── data/
│   ├── input/                # Place raw match video here
│   └── output/               # Annotated video, CSVs, and heatmaps
├── src/
│   ├── __init__.py           # Package docstring
│   ├── config.py             # All tuneable constants in one place
│   ├── detector.py           # YOLOv8 detection + ByteTrack wrapper
│   ├── tracker.py            # Motion trail & full position history
│   ├── annotator.py          # Frame drawing utilities (boxes, IDs, trails)
│   ├── metrics.py            # Speed estimation + heatmap generation
│   └── pipeline.py           # Orchestrator — wires all modules together
├── main.py                   # CLI entry point (argparse)
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── .gitignore                # Git exclusions
└── README.md                 # You are here
```

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              main.py (CLI)                  │
                    │   Parses --input, --output, --model args    │
                    └──────────────────┬──────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │           pipeline.py (Orchestrator)        │
                    │   Opens video → loops frames → post-process │
                    └──┬──────────┬──────────┬──────────┬─────────┘
                       │          │          │          │
                       ▼          ▼          ▼          ▼
                  ┌─────────┐ ┌────────┐ ┌─────────┐ ┌────────┐
                  │detector │ │tracker │ │annotator│ │metrics │
                  │  .py    │ │  .py   │ │   .py   │ │  .py   │
                  ├─────────┤ ├────────┤ ├─────────┤ ├────────┤
                  │ YOLOv8  │ │Centroid│ │Bounding │ │Speed   │
                  │ detect  │ │trail & │ │box draw │ │compute │
                  │ + track │ │history │ │+ trails │ │+heatmap│
                  └─────────┘ └────────┘ └─────────┘ └────────┘
                       │                                  │
                       ▼                                  ▼
                  config.py ◄────────────────────── Shared constants
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **pip** (Python package manager)
- A football match video file (`.mp4`, `.avi`, etc.)

### 1. Clone the Repository

```bash
git clone https://github.com/KhushneetSingh/SmartPitch-Analytics.git
cd SmartPitch-Analytics
```

### 2. Create a Virtual Environment

```bash
python -m venv venv && source venv/bin/activate
```

> **Windows:** Use `venv\Scripts\activate` instead.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU Support:** For CUDA acceleration, install PyTorch separately first from [pytorch.org](https://pytorch.org) with the correct CUDA version. Apple Silicon users get MPS acceleration automatically.

### 4. Add Your Input Video

```bash
cp /path/to/your/match.mp4 data/input/match.mp4
```

### 5. Run the Pipeline

```bash
python main.py
```

The YOLOv8 model weights (`yolov8m.pt`) will be **automatically downloaded** by Ultralytics on the first run (~50 MB).

---

## ⚙️ CLI Options

```bash
python main.py [--input PATH] [--output PATH] [--model PATH]
```

| Flag       | Default                     | Description                            |
| ---------- | --------------------------- | -------------------------------------- |
| `--input`  | `data/input/match.mp4`      | Path to the raw match video            |
| `--output` | `data/output/annotated.mp4` | Path for the annotated output video    |
| `--model`  | `yolov8m.pt`                | YOLOv8 model weights (auto-downloaded) |

### Example

```bash
# Custom input and output paths
python main.py --input videos/premier_league.mp4 --output results/annotated.mp4

# Use a lighter model for faster processing
python main.py --model yolov8n.pt
```

---

## 📤 Outputs

All outputs are saved to `data/output/` by default:

| Output             | File                               | Description                                             |
| ------------------ | ---------------------------------- | ------------------------------------------------------- |
| 🎬 Annotated Video | `annotated.mp4`                    | Video with bounding boxes, track IDs, and motion trails |
| 📊 Speed Report    | `speed_report.csv`                 | Per-player average speed (px/s) and frames tracked      |
| 🗺️ Heatmaps        | `heatmaps/player_<ID>_heatmap.png` | Gaussian-smoothed spatial density map per player        |

---

## 🔧 Configuration

All tuneable parameters are centralized in [`src/config.py`](src/config.py):

| Parameter          | Default | Description                                   |
| ------------------ | ------- | --------------------------------------------- |
| `CONF_THRESHOLD`   | `0.35`  | Minimum detection confidence                  |
| `IOU_THRESHOLD`    | `0.5`   | IoU threshold for NMS                         |
| `MIN_TRACK_FRAMES` | `90`    | Minimum frames to keep a track (ghost filter) |
| `TRAIL_MAX_LEN`    | `30`    | Visual trail length (frames)                  |
| `HEATMAP_SIGMA`    | `20`    | Gaussian smoothing sigma for heatmaps         |
| `LOG_INTERVAL`     | `30`    | Print progress every N frames                 |

---

## 🧠 Design Decisions

### Why YOLOv8?

YOLOv8 (Ultralytics) provides state-of-the-art real-time object detection with a clean Python API. The `yolov8m.pt` medium variant balances speed and accuracy, running at interactive frame rates even on CPU.

### Why ByteTrack?

ByteTrack is bundled inside Ultralytics with zero additional setup. Its dual-threshold association strategy matches high- and low-confidence detections separately, providing stable track IDs even through brief occlusions.

### Separation of Concerns

- **Detection** (`detector.py`) — _"Who is in this frame?"_ (no temporal memory)
- **Tracking** (`tracker.py`) — _"Which detection corresponds to which across frames?"_ (maintains identity)
- **Annotation** (`annotator.py`) — Pure visual rendering, no logic
- **Metrics** (`metrics.py`) — Post-processing analysis, decoupled from frame loop

This modular design allows swapping any component independently (e.g., replacing ByteTrack with DeepSORT) without touching the others.

### Speed Estimation

```
avg_speed (px/s) = (total_pixel_displacement / num_frames) × fps
```

> **Note:** Speed is measured in **pixel space**, not real-world m/s. Converting to physical units requires a homography transform mapping pixel coordinates to known pitch dimensions.

---

## ⚠️ Assumptions

- **Static camera** — Pan/tilt/zoom cameras would need stabilisation first
- **COCO "person" class** — Uses generic class 0; a football-specific model would improve recall
- **Constant FPS** — Frame rate is read once from metadata and assumed uniform
- **No team differentiation** — All players are treated identically
- **Track IDs reset per run** — ByteTrack IDs are not persistent across executions
- **Empirical heatmap sigma** — The default value of 20 should be tuned for different resolutions

---

## 🚧 Limitations & Future Improvements

| Area               | Current Limitation                  | Planned Improvement                                             |
| ------------------ | ----------------------------------- | --------------------------------------------------------------- |
| Speed Accuracy     | Pixel-space only                    | Homography transform using corner flag reference points         |
| Team Awareness     | All players treated equally         | K-Means colour clustering on HSV jersey crops                   |
| Ball Tracking      | Not tracked                         | Custom-trained YOLO head for football detection                 |
| Occlusion Handling | IDs may swap during long occlusions | ReID with appearance embeddings (OSNet + DeepSORT)              |
| Detection Quality  | Generic COCO model                  | Fine-tuned on SoccerNet or football-specific datasets           |
| Tactical Analysis  | Not implemented                     | Team centroids, formations, off-ball spacing metrics            |
| User Interface     | CLI only                            | Streamlit dashboard with interactive player and heatmap overlay |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection framework
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — Multi-object tracking algorithm
- [OpenCV](https://opencv.org/) — Computer vision library
- [SciPy](https://scipy.org/) — Gaussian filtering for heatmaps
- [Matplotlib](https://matplotlib.org/) — Heatmap visualisation

---

<div align="center">

**Built with ❤️ by [Khushneet Singh](https://github.com/KhushneetSingh)**

</div>
