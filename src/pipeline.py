"""
Pipeline — orchestrator that ties all modules into a single video-processing loop.

This is the only module that imports from multiple sibling modules. It opens the
input video, runs frame-by-frame detection and tracking, generates an annotated
output video, and computes post-processing metrics (speed reports and heatmaps).
"""

import csv
import os
from collections import defaultdict, deque

import cv2

from src import config
from src.annotator import Annotator
from src.detector import PlayerDetector
from src.metrics import compute_speeds, generate_heatmap


def save_speed_csv(speed_results, filepath=None):
    """Write per-player speed data to a CSV file.

    Args:
        speed_results (dict): {track_id: {'speed': float, 'frames': int}}
        filepath (str | None): Output path. Defaults to config.SPEED_REPORT_PATH.
    """
    if filepath is None:
        filepath = config.SPEED_REPORT_PATH

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            'track_id',
            'avg_speed_px_per_sec',
            'frames_tracked',
        ])
        for tid, data in sorted(speed_results.items()):
            writer.writerow([tid, data['speed'], data['frames']])


def run(video_in, video_out, model_path):
    """Execute the full analysis pipeline on a single video file.

    Args:
        video_in (str): Path to the input video.
        video_out (str): Path to write the annotated output video.
        model_path (str): Path to the YOLOv8 model weights.

    Returns:
        dict | None: Summary with total_frames, unique_players, output_path.
    """
    try:
        cap = cv2.VideoCapture(video_in)
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Could not open video: '{video_in}'. "
                "Check that the file exists and is a valid video format."
            )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)

    os.makedirs(os.path.dirname(video_out), exist_ok=True)
    writer = cv2.VideoWriter(
        video_out, fourcc, fps, (frame_width, frame_height),
    )

    detector = PlayerDetector(
        model_path=model_path,
        conf=config.CONF_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        classes=config.CLASSES_TO_DETECT,
        tracker_config=config.TRACKER_CONFIG,
    )
    annotator = Annotator()

    track_history = defaultdict(list)
    trail_history = defaultdict(
        lambda: deque(maxlen=config.TRAIL_MAX_LEN),
    )

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tracked = detector.track(frame)

        for det in tracked:
            tid = det['track_id']
            x1, y1, x2, y2 = det['bbox']
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            track_history[tid].append((cx, cy, frame_count))
            trail_history[tid].append((cx, cy))

        annotated = annotator.draw_frame(frame, tracked, trail_history)
        writer.write(annotated)

        frame_count += 1
        if frame_count % config.LOG_INTERVAL == 0:
            print(f"  Processed {frame_count} frames ...")

    cap.release()
    writer.release()
    print(f"  Video processing complete — {frame_count} frames.")

    print("  Computing speed metrics and generating heatmaps ...")

    os.makedirs(config.HEATMAP_DIR, exist_ok=True)

    speed_results = compute_speeds(track_history, fps)

    for tid, positions in track_history.items():
        if len(positions) >= 2:
            generate_heatmap(
                tid, positions, frame_height, frame_width,
                config.HEATMAP_DIR,
            )

    save_speed_csv(speed_results)
    print("  Metrics saved.")

    return {
        'total_frames': frame_count,
        'unique_players': len(track_history),
        'output_path': video_out,
    }
