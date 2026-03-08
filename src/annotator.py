"""
Annotator — frame-level drawing utilities for visualising tracked players.

Draws bounding boxes, track-ID labels, and motion trails on each video frame.
"""

import cv2

PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
]


class Annotator:
    """Draws detection overlays (boxes, IDs, trails) on video frames."""

    def draw_frame(self, frame, tracked, trail_history):
        """Annotate a single frame with bounding boxes, IDs, and trails.

        Args:
            frame (numpy.ndarray): Raw BGR image.
            tracked (list[dict]): Detection dicts with bbox, track_id keys.
            trail_history (dict): {track_id: deque((cx, cy), ...)} for trails.

        Returns:
            numpy.ndarray: Annotated copy of the input frame.
        """
        out = frame.copy()

        for det in tracked:
            tid = det['track_id']
            x1, y1, x2, y2 = map(int, det['bbox'])
            colour = PALETTE[tid % len(PALETTE)]

            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

            cv2.putText(
                out, f'ID:{tid}', (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
            )

            pts = trail_history.get(tid, [])
            for i in range(1, len(pts)):
                cv2.line(out, pts[i - 1], pts[i], colour, 2)

        return out
