"""
Tracker — motion trail history for tracked objects.

Records the centroid trail of each tracked player across frames.
Trails are used by the annotator for drawing and by the pipeline
for feeding positional data into the metrics module.
"""

from collections import deque


class Tracker:
    """Maintains a sliding-window trail of centroid positions per track ID."""

    def __init__(self, max_trail_len=30):
        """Initialise the tracker memory.

        Args:
            max_trail_len (int): Max recent positions to retain per track.
        """
        self.trail_history = {}
        self.max_trail_len = max_trail_len

    def update_trails(self, tracked_dets):
        """Append current-frame centroids to each player's trail.

        Args:
            tracked_dets (list[dict]): Detection dicts with track_id and bbox.

        Returns:
            dict: {track_id: deque((cx, cy), ...)} updated trail history.
        """
        for det in tracked_dets:
            tid = det['track_id']
            x1, y1, x2, y2 = det['bbox']
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if tid not in self.trail_history:
                self.trail_history[tid] = deque(
                    maxlen=self.max_trail_len,
                )

            self.trail_history[tid].append((cx, cy))

        return self.trail_history
