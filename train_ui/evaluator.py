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


def _probe_env_dims():
    """Get (obs_size, n_actions) from a fresh colony env."""
    from cpp_env import CppColonyEnv

    env = CppColonyEnv(map_size=200)
    try:
        return int(env.observation_space.shape[0]), int(env.action_space.n)
    finally:
        env.close()


def _load_policy(model_path: Path, device):
    """Load a trained ActorCritic from a checkpoint.

    Requires checkpoint key: model_state. Optional keys: obs_size, n_actions,
    hidden_sizes. If absent, dimensions are inferred from the state_dict shapes.
    """
    import torch
    from rl.actor_critic import ActorCritic

    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint must be a dict")
    model_state = ckpt.get("model_state")
    if model_state is None:
        raise ValueError("checkpoint missing 'model_state'")

    obs_size = int(ckpt.get("obs_size", 0))
    n_actions = int(ckpt.get("n_actions", 0))
    hidden = ckpt.get("hidden_sizes") or None

    if obs_size <= 0 or n_actions <= 0 or hidden is None:
        clean = {}
        for k, v in model_state.items():
            ck = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
            clean[ck] = v
        model_state = clean
        try:
            obs_size = model_state["trunk.0.weight"].shape[1]
            n_actions = model_state["actor_head.weight"].shape[0]
            hidden_keys = sorted(
                (k for k in model_state if k.startswith("trunk.") and k.endswith(".weight")),
                key=lambda k: int(k.split(".")[1]),
            )
            hidden = [model_state[k].shape[0] for k in hidden_keys]
        except (KeyError, IndexError):
            raise ValueError(
                "checkpoint is not a valid ActorCritic state_dict; "
                "retrain with train_ui/worker.py to save a compatible checkpoint"
            )

    model = ActorCritic(
        obs_size=obs_size,
        n_actions=n_actions,
        hidden_sizes=[int(h) for h in hidden],
        device=device,
    )
    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        raise ValueError(
            "checkpoint state_dict does not match the inferred ActorCritic architecture"
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
) -> Dict[str, float]:
    """Run the trained policy in the colony env and return mean stats.

    Returns dict: days, people, bases, episodes, avg_return (all non-negative).
    """
    import torch
    from cpp_env import CppColonyEnv

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    policy = _load_policy(model_path, dev)

    env = CppColonyEnv(map_size=200)
    if normalization_path is not None:
        norm_path = Path(normalization_path)
        if not norm_path.exists():
            raise FileNotFoundError(
                f"normalization file not found: {norm_path}. "
                "Refusing to run eval on un-normalized observations."
            )
        env.normalizer.load(str(norm_path))
        env.normalizer.set_update(False)
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
        "people": float(sum(people_list) / n),
        "bases": float(sum(bases_list) / n),
        "episodes": float(len(days_list)),
        "avg_return": float(sum(returns) / n),
    }
