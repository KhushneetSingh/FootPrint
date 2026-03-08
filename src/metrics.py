"""
Metrics — speed estimation and heatmap generation for tracked players.

Provides compute_speeds() for average pixel-space speed per player and
generate_heatmap() for Gaussian-smoothed spatial density maps.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from src import config


def compute_speeds(track_history, fps):
    """Compute average pixel-space speed for every tracked player.

    Args:
        track_history (dict): {track_id: [(cx, cy, frame_no), ...]}
        fps (float): Frames per second of the source video.

    Returns:
        dict: {track_id: {'speed': float, 'frames': int}}
    """
    results = {}

    for tid, positions in track_history.items():
        num_frames = len(positions)

        if num_frames < 2:
            results[tid] = {'speed': 0.0, 'frames': num_frames}
            continue

        total_displacement = 0.0
        for i in range(1, num_frames):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            total_displacement += math.sqrt(dx ** 2 + dy ** 2)

        avg_speed = (total_displacement / num_frames) * fps
        results[tid] = {
            'speed': round(avg_speed, 2),
            'frames': num_frames,
        }

    return results


def generate_heatmap(track_id, positions, frame_h, frame_w, out_dir):
    """Generate and save a Gaussian-smoothed density heatmap for one player.

    Args:
        track_id (int): Player track ID.
        positions (list[tuple]): [(cx, cy, frame_no), ...] centroid history.
        frame_h (int): Video frame height in pixels.
        frame_w (int): Video frame width in pixels.
        out_dir (str): Directory to write the output PNG into.
    """
    heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)

    for cx, cy, _frame_no in positions:
        if 0 <= cy < frame_h and 0 <= cx < frame_w:
            heatmap[cy, cx] += 1

    heatmap = gaussian_filter(heatmap, sigma=config.HEATMAP_SIGMA)

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, cmap='hot', interpolation='bilinear')
    plt.colorbar(label='Dwell density')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title(f'Player {track_id} — Spatial Heatmap')
    plt.savefig(
        os.path.join(out_dir, f'player_{track_id}_heatmap.png'),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close()
