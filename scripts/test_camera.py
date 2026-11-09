#!/usr/bin/env python3
"""
Script de prueba y calibración del tracker de bolita con webcam.

Uso:
    python scripts/test_camera.py
    python scripts/test_camera.py --camera 1
    python scripts/test_camera.py --calibrate
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ball_tracker import BallTracker, BallTrackerConfig


def main():
    parser = argparse.ArgumentParser(description="Prueba del tracker de bolita")
    parser.add_argument("--camera", type=int, default=0, help="ID de la cámara (0, 1, ...)")
    parser.add_argument("--calibrate", action="store_true", help="Abrir calibración HSV interactiva")
    parser.add_argument("--roi", action="store_true", help="Seleccionar región de la placa")
    parser.add_argument("--duration", type=float, default=60.0, help="Segundos de preview")
    args = parser.parse_args()

    cfg = BallTrackerConfig(camera_id=args.camera)

    print("=" * 55)
    print("  Ball Tracker – Prueba de cámara")
    print(f"  Cámara ID : {args.camera}")
    print("=" * 55)

    with BallTracker(cfg) as tracker:
        if args.roi:
            tracker.set_roi_interactive()

        if args.calibrate:
            tracker.calibrate_color()

        print("\nIniciando preview. Pulsa 'q' en la ventana para salir.")
        tracker.preview(duration_sec=args.duration)

    print("Listo.")


if __name__ == "__main__":
    main()
