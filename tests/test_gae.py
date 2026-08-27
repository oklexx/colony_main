"""Test GAE computation against reference implementation."""
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.rollout_buffer import RolloutBuffer


def reference_gae(
    rewards: list[float],
    values: list[float],
    last_value: float,
    dones: list[bool],
    gamma: float = 0.99,
    lam: float = 0.98,
) -> tuple[list[float], list[float]]:
    """Reference GAE implementation (sequential, for verification)."""
    T = len(rewards)
    advantages = [0.0] * T
    last_gae = 0.0
    for t in range(T - 1, -1, -1):
        next_value = values[t + 1] if t < T - 1 else last_value
        next_done = dones[t + 1] if t < T - 1 else False
        delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
        last_gae = delta + gamma * lam * (1 - next_done) * last_gae
        advantages[t] = last_gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def test_gae_single_env():
    """Test GAE with a single env (n_envs=1) against reference."""
    T = 100
    gamma = 0.99
    lam = 0.95
    np.random.seed(42)

    rewards = [float(np.random.randn()) for _ in range(T)]
    values = [float(np.random.randn()) for _ in range(T)]
    dones = [False] * T
    dones[30] = True
    dones[70] = True
    last_value = 1.5
    last_done = False

    ref_adv, ref_ret = reference_gae(rewards, values, last_value, dones, gamma, lam)

    device = torch.device("cpu")
    buf = RolloutBuffer(
        n_steps=T,
        n_envs=1,
        obs_size=10,
        n_actions=5,
        gamma=gamma,
        gae_lambda=lam,
        device=device,
    )

    for t in range(T):
        obs = torch.randn(1, 10)
        action = torch.tensor([t % 5])
        reward = torch.tensor([rewards[t]])
        log_prob = torch.randn(1)
        value = torch.tensor([values[t]])
        done = torch.tensor([dones[t]])
        buf.add(obs, action, reward, log_prob, value, done)

    buf.compute_gae(
        last_value=torch.tensor([last_value]),
        last_done=torch.tensor([last_done]),
    )

    gpu_ret = buf.returns[:T].squeeze().cpu().numpy()

    ref_ret_arr = np.array(ref_ret)
    ret_diff = np.abs(gpu_ret - ref_ret_arr).max()

    print(f"GAE single env: max_ret_diff={ret_diff:.2e}")
    assert ret_diff < 1e-4, f"GAE return mismatch: {ret_diff}"
    print("PASS: GAE single env matches reference")


def test_gae_multi_env():
    """Test GAE with multiple envs."""
    T = 50
    n_envs = 4
    gamma = 0.99
    lam = 0.95
    np.random.seed(42)

    device = torch.device("cpu")
    buf = RolloutBuffer(
        n_steps=T,
        n_envs=n_envs,
        obs_size=10,
        n_actions=5,
        gamma=gamma,
        gae_lambda=lam,
        device=device,
    )

    rewards_all = np.random.randn(T, n_envs)
    values_all = np.random.randn(T, n_envs)
    dones_all = np.zeros((T, n_envs), dtype=bool)
    dones_all[20, 1] = True
    dones_all[40, 3] = True
    last_values = np.random.randn(n_envs)
    last_dones = np.array([False, True, False, True])

    for t in range(T):
        obs = torch.randn(n_envs, 10)
        action = torch.randint(0, 5, (n_envs,))
        reward = torch.tensor(rewards_all[t], dtype=torch.float32)
        log_prob = torch.randn(n_envs)
        value = torch.tensor(values_all[t], dtype=torch.float32)
        done = torch.tensor(dones_all[t])
        buf.add(obs, action, reward, log_prob, value, done)

    buf.compute_gae(
        last_value=torch.tensor(last_values, dtype=torch.float32),
        last_done=torch.tensor(last_dones),
    )

    n = T * n_envs
    assert buf.advantages[:n].shape == (n,)
    assert buf.returns[:n].shape == (n,)
    assert not torch.isnan(buf.advantages[:n]).any(), "NaN in advantages"
    assert not torch.isnan(buf.returns[:n]).any(), "NaN in returns"

    print(f"GAE multi env: advantages shape={buf.advantages[:n].shape}, "
          f"mean={buf.advantages[:n].mean():.4f}, std={buf.advantages[:n].std():.4f}")
    print("PASS: GAE multi env no NaNs")


def test_gae_normalization():
    """Test that advantages are normalized (mean~0, std~1)."""
    T = 200
    n_envs = 2
    gamma = 0.99
    lam = 0.95
    np.random.seed(42)

    device = torch.device("cpu")
    buf = RolloutBuffer(
        n_steps=T,
        n_envs=n_envs,
        obs_size=10,
        n_actions=5,
        gamma=gamma,
        gae_lambda=lam,
        device=device,
    )

    for t in range(T):
        obs = torch.randn(n_envs, 10)
        action = torch.randint(0, 5, (n_envs,))
        reward = torch.randn(n_envs)
        log_prob = torch.randn(n_envs)
        value = torch.randn(n_envs)
        done = torch.zeros(n_envs, dtype=torch.bool)
        buf.add(obs, action, reward, log_prob, value, done)

    buf.compute_gae(
        last_value=torch.randn(n_envs),
        last_done=torch.zeros(n_envs, dtype=torch.bool),
    )

    n = T * n_envs
    adv = buf.advantages[:n]
    mean = adv.mean().item()
    std = adv.std().item()

    print(f"GAE normalization: mean={mean:.6f}, std={std:.6f}")
    assert abs(mean) < 0.01, f"Advantage mean not ~0: {mean}"
    assert 0.9 < std < 1.1, f"Advantage std not ~1: {std}"
    print("PASS: GAE normalization correct")


if __name__ == "__main__":
    test_gae_single_env()
    test_gae_multi_env()
    test_gae_normalization()
    print("\nAll GAE tests passed!")
