#!/usr/bin/env python3
"""Sakhalin Colony — PPO training on GPU (RTX 5080).

Usage:
  python train.py --steps 1000000 --envs 8
  python train.py --steps 10000000 --envs 32 --n-epochs 5 --batch-size 16384
  python train.py --async --queue-size 2
  python train.py --compile --amp bfloat16
"""
import argparse
import sys
import time
import json
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from rl.config import Config, RewardConfig
from rl.env_manager import EnvManager
from rl.async_trainer import AsyncTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Sakhalin Colony PPO Training")
    p.add_argument("--steps", type=int, default=1_000_000, help="Total timesteps")
    p.add_argument("--envs", type=int, default=8, help="Number of parallel envs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--map-size", type=int, default=280)
    p.add_argument("--n-steps", type=int, default=4096, help="Rollout length per env")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.98)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="bfloat16", choices=["bfloat16", "float16", "off"])
    p.add_argument("--compile", action="store_true", help="Enable torch.compile")
    p.add_argument("--async", action="store_true", dest="async_train", help="Async training")
    p.add_argument("--queue-size", type=int, default=2)
    p.add_argument("--cpp-threads", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=0)
    p.add_argument("--save-freq", type=int, default=500_000)
    p.add_argument("--eval-freq", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--log-dir", type=str, default="")
    p.add_argument("--model-dir", type=str, default="")
    p.add_argument("--name", type=str, default="colony_run")
    p.add_argument("--reward-config", type=str, default="", help="Path to reward JSON")
    p.add_argument("--disable-net-worth", action="store_true")
    p.add_argument("--disable-daily-income", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    use_amp = args.amp != "off"
    amp_dtype = args.amp if use_amp else "bfloat16"

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    if device.type == "cuda":
        print(f"[Config] CUDA: {torch.cuda.get_device_name(0)}")
        print(f"[Config] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"[Config] bf16: {torch.cuda.is_bf16_supported()}")

    reward = RewardConfig(
        disable_net_worth=args.disable_net_worth,
        disable_daily_income=args.disable_daily_income,
    )
    if args.reward_config:
        with open(args.reward_config) as f:
            rd = json.load(f)
        reward = RewardConfig.from_dict(rd)

    cfg = Config(
        map_size=args.map_size,
        n_envs=args.envs,
        seed=args.seed,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        net_arch=args.net_arch,
        total_timesteps=args.steps,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        device=str(device),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        torch_compile=args.compile,
        cpp_threads=args.cpp_threads,
        torch_threads=args.torch_threads,
        async_train=args.async_train,
        queue_size=args.queue_size,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        reward=reward,
    )

    run_dir = Path(cfg.model_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir = str(run_dir)

    print(f"[Config] steps={cfg.total_timesteps:,} envs={cfg.n_envs} "
          f"n_steps={cfg.n_steps} batch={cfg.batch_size} epochs={cfg.n_epochs}")
    print(f"[Config] lr={cfg.learning_rate} gamma={cfg.gamma} gae={cfg.gae_lambda}")
    print(f"[Config] AMP={cfg.use_amp} ({cfg.amp_dtype}) compile={cfg.torch_compile}")
    print(f"[Config] device={device}")
    print(f"[Config] model_dir={cfg.model_dir}")

    t0 = time.time()
    em = EnvManager(cfg, device)
    t_env = time.time() - t0
    print(f"[Env] Created in {t_env:.1f}s (obs={em.obs_size}, actions={em.n_actions})")

    norm_path = Path(cfg.model_dir) / "normalization.json"
    em.env.venv.save_normalization(str(norm_path))
    print(f"[Save] Normalization: {norm_path}")

    log_dir = Path(cfg.log_dir) / args.name
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(log_dir))
    except ImportError:
        pass

    def progress_cb(metrics):
        if writer:
            writer.add_scalar("train/fps", metrics.fps, metrics.total_timesteps)
            writer.add_scalar("train/policy_loss", metrics.policy_loss, metrics.total_timesteps)
            writer.add_scalar("train/value_loss", metrics.value_loss, metrics.total_timesteps)
            writer.add_scalar("train/entropy", metrics.entropy, metrics.total_timesteps)
            writer.add_scalar("train/approx_kl", metrics.approx_kl, metrics.total_timesteps)
            writer.add_scalar("train/learning_rate", metrics.learning_rate, metrics.total_timesteps)
            writer.add_scalar("train/gpu_mem_mb", metrics.gpu_mem_mb, metrics.total_timesteps)
            writer.add_scalar("train/best_reward", metrics.best_reward, metrics.total_timesteps)
            # Eval metrics
            if metrics.eval_days > 0:
                writer.add_scalar("eval/days", metrics.eval_days, metrics.total_timesteps)
                writer.add_scalar("eval/people", metrics.eval_people, metrics.total_timesteps)
                writer.add_scalar("eval/bases", metrics.eval_bases, metrics.total_timesteps)
                writer.add_scalar("eval/score", metrics.eval_score, metrics.total_timesteps)
                writer.add_scalar("eval/best_score", metrics.best_score, metrics.total_timesteps)

    trainer = AsyncTrainer(
        cfg=cfg,
        env_manager=em,
        logger=None,
        progress_callback=progress_cb,
        stop_check=None,
    )

    t_train = time.time()
    metrics = trainer.train()
    t_total = time.time() - t_train

    if writer:
        writer.close()
    em.close()

    print(f"\n{'=' * 60}")
    print(f"Training complete: {metrics.total_timesteps:,} steps")
    print(f"Total time: {t_total:.1f}s")
    print(f"Average FPS: {metrics.total_timesteps / t_total:,.0f}")
    print(f"Episodes completed: {metrics.n_episodes}")
    print(f"Best reward: {metrics.best_reward:.2f}")
    print(f"Final policy_loss: {metrics.policy_loss:.4f}")
    print(f"Final value_loss: {metrics.value_loss:.4f}")
    print(f"Final entropy: {metrics.entropy:.4f}")
    print(f"Final approx_kl: {metrics.approx_kl:.5f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
