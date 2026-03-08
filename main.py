"""
Main — command-line entry point for the Football Video Analysis Pipeline.

Parses optional --input, --output, and --model arguments (falling back to
the defaults in config.py) and delegates to pipeline.run().
"""

import argparse

import torch

from src import config as cfg
from src import pipeline


def _detect_device():
    """Return a human-readable string describing the compute device."""
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "MPS (Apple Silicon GPU)"
    return "CPU"


def main():
    """Parse CLI arguments and run the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Football Video Analysis Pipeline",
    )
    parser.add_argument(
        '--input', default=cfg.VIDEO_INPUT,
        help="Path to input video (default: %(default)s)",
    )
    parser.add_argument(
        '--output', default=cfg.VIDEO_OUTPUT,
        help="Path to output annotated video (default: %(default)s)",
    )
    parser.add_argument(
        '--model', default=cfg.MODEL_PATH,
        help="Path to YOLOv8 model weights (default: %(default)s)",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("  Football Video Analysis Pipeline")
    print("=" * 50)
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Model  : {args.model}")
    print(f"  Device : {_detect_device()}")
    print("=" * 50)

    summary = pipeline.run(args.input, args.output, args.model)

    if summary is not None:
        print()
        print("-" * 50)
        print("  Run complete!")
        print(f"  Total frames processed : {summary['total_frames']}")
        print(f"  Unique players tracked : {summary['unique_players']}")
        print(f"  Annotated video        : {summary['output_path']}")
        print(f"  Speed report           : {cfg.SPEED_REPORT_PATH}")
        print(f"  Heatmap directory      : {cfg.HEATMAP_DIR}")
        print("-" * 50)
    else:
        print("\n  Pipeline did not complete. Check error above.")


if __name__ == '__main__':
    main()
