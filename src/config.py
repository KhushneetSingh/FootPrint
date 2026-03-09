"""
Configuration — centralised constants for the Football Video Analysis Pipeline.

All tuneable hyperparameters, file paths, and model settings live here so that
no other module contains hard-coded magic numbers or paths.
"""

MODEL_PATH = 'yolov8n.pt'
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
CLASSES_TO_DETECT = [0]

TRACKER_CONFIG = 'bytetrack.yaml'
# Tracks shorter than this many frames are discarded as noise
MIN_TRACK_FRAMES = 25
TRAIL_MAX_LEN = 30

VIDEO_INPUT = 'data/input/match.mp4'
VIDEO_OUTPUT = 'data/output/annotated.mp4'
HEATMAP_DIR = 'data/output/heatmaps/'
SPEED_REPORT_PATH = 'data/output/speed_report.csv'

VIDEO_CODEC = 'mp4v'

HEATMAP_SIGMA = 20
LOG_INTERVAL = 30