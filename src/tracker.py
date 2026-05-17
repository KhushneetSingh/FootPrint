"""
Tracker — motion trail history for tracked objects.

Records the centroid trail of each tracked player across frames.
Maintains two views of the data:
  - trail_history : sliding-window deque used by the annotator for drawing.
  - full_history  : cumulative list of (cx, cy, frame_no) used by metrics.
"""

from collections import deque


class Tracker:
    """Maintains centroid position history per track ID.

    Provides a bounded trail (for annotation) and an unbounded full history
    (for post-processing metrics like speed estimation and heatmaps).
    """

    def __init__(self, max_trail_len=30):
        """Initialise the tracker memory.

        Args:
            max_trail_len (int): Max recent positions to retain per track
                for the visual trail overlay.
        """
        self.trail_history = {}
        self.full_history = {}
        self.max_trail_len = max_trail_len

    def update(self, tracked_dets, frame_no):
        """Append current-frame centroids to each player's trail and history.

        Args:
            tracked_dets (list[dict]): Detection dicts with track_id and bbox.
            frame_no (int): Current frame number (0-indexed).

        Returns:
            dict: {track_id: deque((cx, cy), ...)} updated trail history
                suitable for passing to the annotator.
        """
        for det in tracked_dets:
            tid = det['track_id']
            x1, y1, x2, y2 = det['bbox']
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Sliding window trail for annotation
            if tid not in self.trail_history:
                self.trail_history[tid] = deque(
                    maxlen=self.max_trail_len,
                )
            self.trail_history[tid].append((cx, cy))

            # Cumulative history for metrics
            if tid not in self.full_history:
                self.full_history[tid] = []
            self.full_history[tid].append((cx, cy, frame_no))

        return self.trail_history

    def get_full_history(self):
        """Return the full (unbounded) position history for all tracks.

        Returns:
            dict: {track_id: [(cx, cy, frame_no), ...]}
        """
        return self.full_history

    def get_trail_history(self):
        """Return the sliding-window trail history for all tracks.

        Returns:
            dict: {track_id: deque((cx, cy), ...)}
        """
        return self.trail_history

    def filter_short_tracks(self, min_frames):
        """Remove tracks shorter than min_frames from both histories.

        Args:
            min_frames (int): Minimum number of frames a track must span.

        Returns:
            int: Number of tracks remaining after filtering.
        """
        keep_ids = {
            tid for tid, positions in self.full_history.items()
            if len(positions) >= min_frames
        }

        self.full_history = {
            tid: pos for tid, pos in self.full_history.items()
            if tid in keep_ids
        }
        self.trail_history = {
            tid: trail for tid, trail in self.trail_history.items()
            if tid in keep_ids
        }

        return len(self.full_history)
