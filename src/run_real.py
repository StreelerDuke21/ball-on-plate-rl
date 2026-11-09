#!/usr/bin/env python3
"""
Control real Ball-on-Plate:
  - Carga el modelo SAC entrenado
  - Lee la posición de la bolita con la webcam
  - Calcula la acción del agente
  - Envía ángulos a los servomotores

Uso básico (modo prueba sin hardware):
    python scripts/run_real.py

Con cámara y servos por serie (Arduino):
    python scripts/run_real.py --servo serial --port COM3

Solo cámara + modelo (servos en mock):
    python scripts/run_real.py --servo mock --camera 0

Opciones útiles:
    --model models/best/best_model
    --calibrate          (ajusta color HSV antes de empezar)
    --roi                (selecciona zona de la placa)
    --max-angle 12       (grados máximos de inclinación)
    --fps 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
src_dir = ROOT / "src"
for candidate in [src_dir, ROOT]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from stable_baselines3 import SAC

import importlib.util


def _load_module_from_path(module_name: str, file_path: Path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise ImportError(f"Missing module file: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_ball_tracker():
    candidates = [
        ROOT / "src" / "ball_tracker.py",
        ROOT / "src" / "ball_on_plate" / "ball_tracker.py",
        ROOT / "ball_tracker.py",
        ROOT / "ball_on_plate" / "ball_tracker.py",
    ]
    for path in candidates:
        try:
            module = _load_module_from_path("ball_tracker_runtime", path)
            return module.BallTracker, module.BallTrackerConfig
        except (ImportError, AttributeError, FileNotFoundError):
            continue

    # Fallback to standard package import if the project is installed in the environment.
    try:
        module = __import__("ball_tracker", fromlist=["BallTracker", "BallTrackerConfig"])
        return module.BallTracker, module.BallTrackerConfig
    except ImportError:
        pass

    try:
        module = __import__("ball_on_plate.ball_tracker", fromlist=["BallTracker", "BallTrackerConfig"])
        return module.BallTracker, module.BallTrackerConfig
    except ImportError as exc:
        raise ImportError("No se encontró ball_tracker.py ni ball_on_plate.ball_tracker") from exc


def _load_servo_controller():
    candidates = [
        ROOT / "src" / "servo_controller.py",
        ROOT / "src" / "ball_on_plate" / "servo_controller.py",
        ROOT / "servo_controller.py",
        ROOT / "ball_on_plate" / "servo_controller.py",
    ]
    for path in candidates:
        try:
            module = _load_module_from_path("servo_controller_runtime", path)
            return module.make_servo_controller
        except (ImportError, AttributeError, FileNotFoundError):
            continue

    try:
        module = __import__("servo_controller", fromlist=["make_servo_controller"])
        return module.make_servo_controller
    except ImportError:
        pass

    try:
        module = __import__("ball_on_plate.servo_controller", fromlist=["make_servo_controller"])
        return module.make_servo_controller
    except ImportError as exc:
        raise ImportError("No se encontró servo_controller.py ni ball_on_plate.servo_controller") from exc


BallTracker, BallTrackerConfig = _load_ball_tracker()
make_servo_controller = _load_servo_controller()


def build_observation(
    x: float,
    y: float,
    vx: float,
    vy: float,
    alpha_norm: float,
    beta_norm: float,
    last_action: np.ndarray,
) -> np.ndarray:
    """
    Construye el vector de observación igual que en el entorno de simulación:
    [x_norm, y_norm, vx_norm, vy_norm, alpha_norm, beta_norm, last_ax, last_ay]
    """
    obs = np.array(
        [
            np.clip(x, -1.5, 1.5),
            np.clip(y, -1.5, 1.5),
            np.clip(vx, -3.0, 3.0),
            np.clip(vy, -3.0, 3.0),
            np.clip(alpha_norm, -1.0, 1.0),
            np.clip(beta_norm, -1.0, 1.0),
            float(last_action[0]),
            float(last_action[1]),
        ],
        dtype=np.float32,
    )
    return obs


def main():
    parser = argparse.ArgumentParser(description="Control real Ball-on-Plate con SAC + cámara")
    parser.add_argument("--model", type=str, default="models/best/best_model",
                        help="Ruta al modelo (sin .zip)")
    parser.add_argument("--camera", type=int, default=0, help="ID de la webcam")
    parser.add_argument("--servo", type=str, default="mock", choices=["mock", "serial"],
                        help="Backend de servos")
    parser.add_argument("--port", type=str, default="COM3", help="Puerto serie (si --servo serial)")
    parser.add_argument("--max-angle", type=float, default=12.0, help="Ángulo máximo de la placa (grados)")
    parser.add_argument("--fps", type=float, default=20.0, help="Frecuencia de control aproximada")
    parser.add_argument("--calibrate", action="store_true", help="Calibrar color HSV al inicio")
    parser.add_argument("--roi", action="store_true", help="Seleccionar ROI de la placa al inicio")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Segundos de ejecución (0 = hasta Ctrl+C)")
    args = parser.parse_args()

    dt = 1.0 / max(args.fps, 1.0)
    max_angle_rad = np.deg2rad(args.max_angle)

    # ---------- Modelo ----------
    model_path = ROOT / args.model
    if not model_path.with_suffix(".zip").exists() and not model_path.exists():
        alt = ROOT / "models" / "sac_ball_final"
        if alt.with_suffix(".zip").exists() or alt.exists():
            model_path = alt
            print(f"Modelo por defecto no encontrado. Usando: {model_path}")
        else:
            print(f"ERROR: No se encontró el modelo en {model_path}")
            print("Entrena primero con:  python scripts/train_sac.py")
            sys.exit(1)

    print(f"Cargando modelo: {model_path}")
    model = SAC.load(str(model_path))
    print("Modelo cargado.")

    # ---------- Servos ----------
    servos = make_servo_controller(
        backend=args.servo,
        port=args.port,
        max_angle_deg=args.max_angle,
    )
    if not servos.connect():
        print("No se pudo conectar con los servos. Abortando.")
        sys.exit(1)

    # ---------- Cámara ----------
    cam_cfg = BallTrackerConfig(camera_id=args.camera)
    tracker = BallTracker(cam_cfg)

    if not tracker.open():
        print(f"No se pudo abrir la cámara {args.camera}")
        servos.disconnect()
        sys.exit(1)

    try:
        if args.roi:
            tracker.set_roi_interactive()
        if args.calibrate:
            tracker.calibrate_color()

        print("\n" + "=" * 60)
        print("  CONTROL REAL – Ball-on-Plate")
        print(f"  Modelo     : {model_path.name}")
        print(f"  Servos     : {args.servo}" + (f" ({args.port})" if args.servo == "serial" else ""))
        print(f"  Cámara     : {args.camera}")
        print(f"  Frecuencia  : {args.fps:.1f} Hz")
        print("  Ctrl+C para detener")
        print("=" * 60 + "\n")

        last_action = np.zeros(2, dtype=np.float32)
        alpha_deg = 0.0
        beta_deg = 0.0
        t0 = time.time()
        step = 0

        while True:
            loop_start = time.time()

            if args.duration > 0 and (loop_start - t0) >= args.duration:
                print("Tiempo de ejecución cumplido.")
                break

            # 1) Leer posición / velocidad de la bolita
            x, y, vx, vy = tracker.get_observation_components()

            # 2) Estado de los servos (normalizado)
            alpha_norm = np.deg2rad(alpha_deg) / max_angle_rad
            beta_norm = np.deg2rad(beta_deg) / max_angle_rad

            # 3) Observación completa
            obs = build_observation(x, y, vx, vy, alpha_norm, beta_norm, last_action)

            # 4) Acción del agente
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).flatten()
            last_action = action.copy()

            # 5) Convertir acción → incremento de ángulo (igual que en el env)
            #    En el entorno: d_angle ≈ action * max_angle * 0.35
            scale = 0.35
            d_alpha = float(action[0]) * args.max_angle * scale
            d_beta = float(action[1]) * args.max_angle * scale

            # Limitar velocidad angular aproximada de los servos
            max_delta = 80.0 * dt  # ~80 °/s
            d_alpha = float(np.clip(d_alpha, -max_delta, max_delta))
            d_beta = float(np.clip(d_beta, -max_delta, max_delta))

            alpha_deg = float(np.clip(alpha_deg + d_alpha, -args.max_angle, args.max_angle))
            beta_deg = float(np.clip(beta_deg + d_beta, -args.max_angle, args.max_angle))

            # 6) Enviar a los servos
            servos.set_angles(alpha_deg, beta_deg)

            step += 1
            if step % int(max(args.fps, 1)) == 0:
                print(
                    f"t={loop_start - t0:6.1f}s | "
                    f"ball=({x:+.3f}, {y:+.3f}) | "
                    f"α={alpha_deg:+5.1f}° β={beta_deg:+5.1f}° | "
                    f"act=({action[0]:+.2f}, {action[1]:+.2f})"
                )

            # Mantener frecuencia aproximada
            elapsed = time.time() - loop_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\nDetenido por el usuario.")
    finally:
        print("Centrando placa y cerrando...")
        servos.center()
        time.sleep(0.3)
        servos.disconnect()
        tracker.close()
        print("Listo.")


if __name__ == "__main__":
    main()
