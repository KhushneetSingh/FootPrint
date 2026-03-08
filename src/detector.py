"""
Detector — YOLOv8 wrapper for object detection and multi-object tracking.

This module is the only place that imports the Ultralytics library.  The rest
of the pipeline interacts with detections through plain Python dicts.
"""

from ultralytics import YOLO


class PlayerDetector:
    """Wraps Ultralytics YOLOv8 for player detection and ByteTrack tracking."""

    def __init__(self, model_path, conf, iou, classes, tracker_config):
        """Load the YOLO model and store inference parameters.

        Args:
            model_path (str): Path to YOLOv8 weights file.
            conf (float): Minimum confidence threshold.
            iou (float): IoU threshold for NMS.
            classes (list[int]): COCO class IDs to detect.
            tracker_config (str): Tracker config file name.
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.tracker_config = tracker_config

    def track(self, frame):
        """Run detection and tracking on a single BGR frame.

        Args:
            frame (numpy.ndarray): Input BGR image.

        Returns:
            list[dict]: Dicts with bbox, track_id, conf, cls keys.
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
        )

        tracked = []
        if results[0].boxes.id is None:
            return tracked

        for box, tid in zip(results[0].boxes, results[0].boxes.id):
            tracked.append({
                'bbox': box.xyxy[0].tolist(),
                'track_id': int(tid),
                'conf': float(box.conf),
                'cls': int(box.cls),
            })
        return tracked
