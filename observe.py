#!/usr/bin/env python3
"""Observe model behavior: load best checkpoint, run episodes, log all steps + observations."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from rl.config import Config
from rl.actor_critic import ActorCritic


def find_model(explicit: str = "") -> str:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        print(f"ERROR: {explicit} not found")
        sys.exit(1)
    for d in [PROJECT_ROOT / "models", Path.home() / "colony_runs" / "models"]:
        if not d.exists():
            continue
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                best = sub / "best.pt"
                if best.exists():
                    return str(best)
                final = sub / "final_model.pt"
                if final.exists():
                    return str(final)
                ckpts = sorted(sub.glob("checkpoint_*_steps.pt"))
                if ckpts:
                    return str(ckpts[-1])
        best = d / "best.pt"
        if best.exists():
            return str(best)
        ckpts = sorted(d.glob("checkpoint_step_*.pt"))
        if ckpts:
            return str(ckpts[-1])
    print("ERROR: No model found. Use --model <path>.")
    sys.exit(1)


BUILD_NAMES = [
    "WaterChannel", "Farm", "Garden", "House", "SmallHouse",
    "Sawmill", "Coalmine", "Ironmine", "Refinery", "Goldmine",
    "PowerStation", "HydroStation", "Road", "Fish", "CoalCut",
    "HuntingLand", "CowFarm", "Mushroom", "BigHouse", "BigFarm",
    "Apiary", "Torchlight", "Hothouse", "SuperHouse", "BigSawmill",
    "WaterMill", "BigRefinary", "Puerperal", "BigIronmine",
    "AirStation", "SmallAtomStation", "AtomStation",
]

def action_name(action: int, n_build: int) -> str:
    if action == 0:
        return "DAY"
    if action == 1:
        return "WEEK"
    if action >= 2:
        idx = action - 2
        if idx < n_build:
            name = BUILD_NAMES[idx] if idx < len(BUILD_NAMES) else f"BUILD{idx}"
            return f"BUILD:{name}"
        if idx < n_build + 11:
            return f"MANAGER:{idx - n_build}"
    return f"ACTION:{action}"


def get_build_debug(env, env_idx: int, action: int, n_build: int) -> str:
    """Get debug info about why a build failed."""
    if action < 2 or action >= 2 + n_build:
        return ""
    idx = action - 2
    name = BUILD_NAMES[idx] if idx < len(BUILD_NAMES) else f"BUILD{idx}"
    # Try to get state info to understand why build failed
    try:
        obs = env.dump_obs(env_idx)
        # Parse key fields from obs string
        return f"BUILD:{name}"
    except:
        return f"BUILD:{name}"

def parse_args():
    p = argparse.ArgumentParser(description="Observe model behavior")
    p.add_argument("--model", type=str, default="")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", type=str, default="")
    p.add_argument("--every", type=int, default=10, help="Dump observation every N steps")
    p.add_argument("--all", action="store_true", help="Log every single step")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = find_model(args.model)
    print(f"Model: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = Config.from_dict(cfg_dict)
    cfg.n_envs = 1
    cfg.seed = args.seed
    cfg.obs_mode = "flat"

    device = torch.device("cpu")

    from cpp_vecenv import make_cpp_vec_env
    env = make_cpp_vec_env(
        n_envs=1,
        map_size=cfg.map_size,
        curriculum_stage=cfg.curriculum_stage,
        unlock_ids=cfg.unlock_ids or None,
        disable_net_worth=cfg.reward.disable_net_worth,
        disable_daily_income=cfg.reward.disable_daily_income,
        reward_config=cfg.reward.to_dict(),
        seed=cfg.seed,
        n_threads=0,
    )
    obs_size = env.cpp_vec.obs_size()
    n_actions = env.cpp_vec.n_actions()
    n_build = n_actions - 2 - 11
    print(f"obs_size={obs_size}, n_actions={n_actions}, n_build={n_build}")
    print(f"reward: disable_net_worth={cfg.reward.disable_net_worth}, "
          f"disable_daily_income={cfg.reward.disable_daily_income}")

    # Read hidden_sizes from checkpoint (handles empty config dict case)
    hidden_sizes = ckpt.get("hidden_sizes") or None
    if hidden_sizes is None:
        model_state = ckpt.get("model_state", {})
        hidden_keys = sorted(
            (k for k in model_state if k.startswith("trunk.") and k.endswith(".weight")),
            key=lambda k: int(k.split(".")[1]),
        )
        hidden_sizes = [model_state[k].shape[0] for k in hidden_keys] if hidden_keys else cfg.net_arch
    print(f"net_arch (from checkpoint): {hidden_sizes}")

    policy = ActorCritic(obs_size, n_actions, hidden_sizes, device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    log_dir = Path(args.log_dir) if args.log_dir else PROJECT_ROOT / "obs_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"observe_{ts}.log"
    step_log_file = log_dir / f"observe_{ts}_steps.log"

    env.cpp_vec.set_step_log(0, str(step_log_file))

    print(f"Running {args.episodes} episodes, {args.steps} steps each")
    print(f"Log: {log_file}")
    print(f"Step log (reward breakdown): {step_log_file}")
    print()

    ep_rewards = []
    total_steps = 0
    action_counts = {}

    with open(log_file, "w") as log:
        for ep in range(args.episodes):
            obs = env.reset()
            obs_flat = np.asarray(obs[0], dtype=np.float32)
            # Get action masks from C++ env
            masks_np = np.asarray(env.action_masks, dtype=np.float32)
            masks_t = torch.from_numpy(masks_np).unsqueeze(0)
            ep_reward = 0.0
            ep_steps = 0

            log.write(f"{'='*70}\n")
            log.write(f"EPISODE {ep+1}\n")
            log.write(f"{'='*70}\n")
            log.write(f"init: {env.dump_obs(0)}\n")
            log.flush()

            for step in range(args.steps):
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_flat).unsqueeze(0)
                    logits, values = policy(obs_t)
                    # Apply action mask
                    logits = logits.masked_fill(masks_t == 0, float("-inf"))
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()
                    val = values.item()
                    probs = dist.probs.cpu().numpy().ravel()
                    top5_idx = np.argsort(probs)[-5:][::-1]
                    top5 = ", ".join(f"a{int(i)}={probs[int(i)]:.3f}" for i in top5_idx)

                # Step env (async pattern)
                env.step_async(np.array([action], dtype=np.int32))
                obs, rewards, dones, infos = env.step_wait()
                obs_flat = np.asarray(obs[0], dtype=np.float32)
                # Update masks after step
                masks_np = np.asarray(env.action_masks, dtype=np.float32)
                masks_t = torch.from_numpy(masks_np).unsqueeze(0)
                reward = float(rewards[0])
                done = bool(dones[0])

                ep_reward += reward
                ep_steps += 1

                act_name = action_name(action, n_build)
                action_counts[act_name] = action_counts.get(act_name, 0) + 1

                # Log EVERY step with detailed info
                # Build debug: if this was a build action, check if it failed
                debug_info = ""
                if action >= 2 and action < 2 + n_build:
                    idx = action - 2
                    name = BUILD_NAMES[idx] if idx < len(BUILD_NAMES) else f"BUILD{idx}"
                    if reward < -1.0:
                        # Build FAILED - log why (money, free people)
                        obs_str = env.dump_obs(0)
                        money = 0
                        free = 0
                        for part in obs_str.split():
                            if part.startswith("money="):
                                money = int(part.split("=")[1])
                            elif part.startswith("free="):
                                free = int(part.split("=")[1])
                        debug_info = f" [BUILD FAILED: {name} money={money} free={free}]"
                    elif reward > 10.0:
                        debug_info = f" [BUILD SUCCESS: {name}]"
                    else:
                        debug_info = f" [BUILD: {name}]"
                
                log.write(f"  s={step:5d} a={action:3d}({act_name}) r={reward:+8.4f} "
                          f"cum={ep_reward:+10.2f} v={val:+8.4f} "
                          f"top5=[{top5}]{debug_info}\n")

                # Log full observation state EVERY step
                log.write(f"    OBS: {env.dump_obs(0)}\n")
                log.flush()

                if done:
                    break

            ep_rewards.append(ep_reward)
            total_steps += ep_steps
            log.write(f"  >>> Ep {ep+1}: reward={ep_reward:.2f}, steps={ep_steps}\n")
            log.write(f"  >>> final: {env.dump_obs(0)}\n")
            log.flush()
            print(f"  Ep {ep+1}: reward={ep_reward:+.2f}, steps={ep_steps}")

        log.write(f"\n{'='*70}\nSUMMARY\n{'='*70}\n")
        log.write(f"Episodes: {len(ep_rewards)}\n")
        log.write(f"Total steps: {total_steps}\n")
        log.write(f"Mean reward: {np.mean(ep_rewards):.2f}\n")
        log.write(f"Max reward:  {max(ep_rewards):.2f}\n")
        log.write(f"Min reward:  {min(ep_rewards):.2f}\n")
        for i, r in enumerate(ep_rewards):
            log.write(f"  Ep {i+1}: {r:+.2f}\n")

        # Action distribution summary
        log.write(f"\nACTION DISTRIBUTION (all {total_steps} steps)\n")
        for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            log.write(f"  {name:20s}: {count:5d} ({100*count/total_steps:.1f}%)\n")
        log.flush()

    env.close()
    print(f"\nDone. Log: {log_file}")
    print(f"Mean: {np.mean(ep_rewards):.2f}  Max: {max(ep_rewards):.2f}  Min: {min(ep_rewards):.2f}")


if __name__ == "__main__":
    main()
