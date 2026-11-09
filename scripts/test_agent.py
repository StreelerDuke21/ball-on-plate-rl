#!/usr/bin/env python3
"""
Prueba un agente entrenado (evaluación + visualización simple).
Uso:
    python scripts/test_agent.py
    python scripts/test_agent.py --model models/best/best_model
    python scripts/test_agent.py --episodes 20
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from stable_baselines3 import SAC
from ball_on_plate_env import BallOnPlateEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/best/best_model",
        help="Ruta al modelo (sin .zip)",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    model_path = ROOT / args.model
    if not model_path.with_suffix(".zip").exists() and not model_path.exists():
        model_path = ROOT / "models" / "sac_ball_final"
        print(f"Modelo no encontrado. Usando: {model_path}")

    print(f"Cargando modelo: {model_path}")
    model = SAC.load(str(model_path))

    env = BallOnPlateEnv(max_episode_steps=300)

    distances = []
    successes = 0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        max_dist = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            max_dist = max(max_dist, info["dist"])
            done = terminated or truncated

        final_dist = info["dist"]
        distances.append(final_dist)
        if final_dist < 0.02:
            successes += 1

        print(
            f"Episodio {ep+1:2d} | "
            f"Reward: {total_reward:7.1f} | "
            f"Dist final: {final_dist*1000:5.1f} mm | "
            f"Max dist: {max_dist*1000:5.1f} mm"
        )

    print("\n" + "=" * 50)
    print(f"Episodios exitosos (< 20 mm): {successes}/{args.episodes}")
    print(f"Distancia media final       : {np.mean(distances)*1000:.1f} mm")
    print(f"Distancia mediana final     : {np.median(distances)*1000:.1f} mm")
    print("=" * 50)


if __name__ == "__main__":
    main()
