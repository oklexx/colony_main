from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "python"))


def _probe_env_dims(map_size: int = 280):
    """Get (obs_size, n_actions) from a fresh colony env."""
    from cpp_env import CppColonyEnv

    env = CppColonyEnv(map_size=map_size)
    try:
        return int(env.observation_space.shape[0]), int(env.action_space.n)
    finally:
        env.close()


def _load_policy(model_path: Path, device, mode: str = "auto", minimap_radius: int = 14):
    """Load a trained policy from a checkpoint.

    mode: "auto" (detect MLP vs CNN from the state_dict), "flat", or "minimap".
    """
    import torch
    from rl.actor_critic import ActorCritic
    from rl.actor_critic_cnn import ActorCriticCNN

    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint must be a dict")
    model_state = ckpt.get("model_state")
    if model_state is None:
        raise ValueError("checkpoint missing 'model_state'")

    clean = {}
    for k, v in model_state.items():
        ck = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        clean[ck] = v
    model_state = clean

    has_cnn = any(k.startswith("cnn.") for k in model_state)

    if mode == "auto":
        mode = "minimap" if has_cnn else "flat"
    if mode == "minimap" and not has_cnn:
        raise ValueError("checkpoint is not a CNN (minimap) policy; use mode='flat'")
    if mode == "flat" and has_cnn:
        raise ValueError("checkpoint is a CNN (minimap) policy; use mode='minimap'")

    hidden = ckpt.get("hidden_sizes") or None
    if hidden is None:
        hidden_keys = sorted(
            (k for k in model_state if k.startswith("trunk.") and k.endswith(".weight")),
            key=lambda k: int(k.split(".")[1]),
        )
        if not hidden_keys:
            raise ValueError("cannot infer hidden sizes from checkpoint")
        hidden = [model_state[k].shape[0] for k in hidden_keys]

    n_actions = model_state["actor_head.weight"].shape[0]

    if mode == "minimap":
        first_conv = None
        for k in sorted(model_state):
            if k.startswith("cnn.") and k.endswith(".weight") and model_state[k].dim() == 4:
                first_conv = model_state[k]
                break
        if first_conv is None:
            raise ValueError("cannot infer CNN input channels from checkpoint")
        n_channels = int(first_conv.shape[1])
        grid = int(ckpt.get("grid_size", 2 * minimap_radius + 1))
        model = ActorCriticCNN(
            n_channels=n_channels,
            grid_size=grid,
            n_actions=int(n_actions),
            hidden_sizes=[int(h) for h in hidden],
            device=device,
        )
    else:
        obs_size = int(ckpt.get("obs_size", 0)) or int(model_state["trunk.0.weight"].shape[1])
        model = ActorCritic(
            obs_size=obs_size,
            n_actions=int(n_actions),
            hidden_sizes=[int(h) for h in hidden],
            device=device,
        )

    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        raise ValueError(
            "checkpoint state_dict does not match the inferred architecture"
        )
    model.eval()
    return model


def run_eval(
    model_path: str | Path,
    episodes: int = 5,
    max_days: int = 1000,
    seed: int = 7,
    device: str = "cpu",
    normalization_path: str | Path | None = None,
    map_size: int = 280,
    mode: str = "auto",
    minimap_radius: int = 14,
) -> Dict[str, float]:
    """Run the trained policy in the colony env and return mean stats.

    mode: "auto" (detect from checkpoint), "flat" (209-dim MLP), "minimap" (CNN).
    Returns dict: days, people, bases, episodes, avg_return (all non-negative).
    """
    import torch
    from cpp_env import CppColonyEnv
    from minimap import MinimapSingleEnvWrapper

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    policy = _load_policy(model_path, dev, mode=mode, minimap_radius=minimap_radius)
    from rl.actor_critic_cnn import ActorCriticCNN
    use_minimap = isinstance(policy, ActorCriticCNN)

    env = CppColonyEnv(map_size=map_size)
    if normalization_path is not None and not use_minimap:
        norm_path = Path(normalization_path)
        if not norm_path.exists():
            raise FileNotFoundError(
                f"normalization file not found: {norm_path}. "
                "Refusing to run eval on un-normalized observations."
            )
        env.normalizer.load(str(norm_path))
        env.normalizer.set_update(False)
    mm_wrap = MinimapSingleEnvWrapper(env) if use_minimap else None
    days_list: list[int] = []
    people_list: list[int] = []
    bases_list: list[int] = []
    returns: list[float] = []

    try:
        for ep in range(episodes):
            obs, _info = env.reset(seed=seed + ep)
            total_reward = 0.0
            for _ in range(max_days):
                with torch.no_grad():
                    if use_minimap:
                        mm = mm_wrap.minimap_obs()
                        obs_t = torch.from_numpy(np.ascontiguousarray(mm, dtype=np.float32)).to(dev).reshape(1, *mm.shape)
                    else:
                        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(dev)
                        obs_t = obs_t.reshape(1, -1)
                    logits, _values = policy(obs_t)
                    action = int(logits.argmax(dim=-1).item())
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            days_list.append(int(info.get("days", 0)))
            people_list.append(int(info.get("people", 0)))
            bases_list.append(int(info.get("bases", 0)))
            returns.append(total_reward)
    finally:
        env.close()

    n = max(len(days_list), 1)
    return {
        "days": float(sum(days_list) / n),
        "days_std": float(np.std(days_list)),
        "people": float(sum(people_list) / n),
        "bases": float(sum(bases_list) / n),
        "episodes": float(len(days_list)),
        "avg_return": float(sum(returns) / n),
        "episode_days": days_list,
        "episode_bases": bases_list,
        "episode_people": people_list,
        "episode_returns": returns,
    }
