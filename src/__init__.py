"""
src — Football Video Analysis Pipeline package.

Modules:
    config     – centralised constants and hyperparameters.
    detector   – YOLOv8 detection and ByteTrack tracking wrapper.
    tracker    – motion trail history bookkeeping.
    annotator  – frame-level drawing utilities.
    metrics    – speed estimation and heatmap generation.
    pipeline   – orchestrator that wires all modules together.
"""
