#!/usr/bin/env python3
"""
Entrenamiento del agente SAC para Ball-on-Plate.
Uso:
    python scripts/train_sac.py
    python scripts/train_sac.py --timesteps 400000
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ball_on_plate_env import BallOnPlateEnv


def make_env():
    env = BallOnPlateEnv(max_episode_steps=250)
    env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser(description="Entrena SAC para Ball-on-Plate")
    parser.add_argument("--timesteps", type=int, default=300_000, help="Total de timesteps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    models_dir = ROOT / "models"
    logs_dir = ROOT / "logs"
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    (models_dir / "best").mkdir(exist_ok=True)
    (models_dir / "checkpoints").mkdir(exist_ok=True)
    (logs_dir / "tb").mkdir(exist_ok=True)

    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    check_env(BallOnPlateEnv())

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "best"),
        log_path=str(logs_dir),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=30_000,
        save_path=str(models_dir / "checkpoints"),
        name_prefix="sac_ball",
    )

    tb_log = str(logs_dir / "tb")
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        tb_log = None
        print("Aviso: tensorboard no instalado. Continuando sin logs de TensorBoard.")

    import torch
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detectada: {gpu_name}")
    else:
        device = "cpu"
        print("No se detecto GPU CUDA. Usando CPU.")

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=200_000,
        learning_starts=2_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        tensorboard_log=tb_log,
        device=device,
    )

    print("=" * 60)
    print("  Entrenando SAC - Ball-on-Plate")
    print(f"  Timesteps totales : {args.timesteps:,}")
    print(f"  Dispositivo       : {device}")
    print(f"  Seed              : {args.seed}")
    print("=" * 60)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False,
    )

    final_path = models_dir / "sac_ball_final"
    model.save(str(final_path))
    print(f"\nModelo final guardado en: {final_path}.zip")
    print("Mejor modelo (evaluacion) en: models/best/")


if __name__ == "__main__":
    main()
