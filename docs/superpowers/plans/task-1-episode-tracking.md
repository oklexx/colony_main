### Task 1: Per-env Episode Tracking

**Problem:** `async_trainer.py:58-59` uses a single `_current_ep_return` float for all 8 envs. When multiple envs finish episodes in the same step, their returns are summed together, making `best_reward` an artifact of vectorization rather than a measure of policy quality.

**Solution:** The C++ vec env already tracks per-env `episode_return_[i]` and `episode_length_[i]` (`env.cpp:779-780`) and reports them in `info["episode"]["r"]` / `info["episode"]["l"]` on done (`env.cpp:883-886`). Read from `infos` instead of maintaining a Python accumulator.

**Files:**
- Modify: `rl/env_manager.py` — expose `infos` from `collect_step`
- Modify: `rl/async_trainer.py` — read `info["episode"]`, remove broken accumulator
- Test: `tests/test_async_trainer.py` (new)

**Interfaces:**
- Consumes: `CppVecEnv.step_wait()` → `infos: list[dict]` with `info["episode"]["r"]` and `info["episode"]["l"]` on done
- Produces: `AsyncTrainer._ep_returns: list[float]` (correct per-episode values), `AsyncTrainer.best_reward: float` (max over all episodes)

- [ ] **Step 1: Write the failing test**

Create `tests/test_async_trainer.py`:

```python
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.config import Config
from rl.async_trainer import AsyncTrainer


class FakeEnvManager:
    """Minimal env manager that simulates 2 envs with known episode returns."""

    def __init__(self, n_envs=2, obs_size=10, n_actions=5):
        self.n_envs = n_envs
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.device = torch.device("cpu")
        self._step_count = 0
        self._episodes_done = []
        self._episode_returns = {0: 100.0, 1: 200.0}

    def reset(self):
        return torch.randn(self.n_envs, self.obs_size)

    def collect_step(self, obs):
        self._step_count += 1
        new_obs = torch.randn(self.n_envs, self.obs_size)
        return new_obs

    def get_infos(self):
        """Return infos for the last step. Simulate both envs ending an episode."""
        if self._step_count >= 3:
            return [
                {"episode": {"r": self._episode_returns[0], "l": 50}},
                {"episode": {"r": self._episode_returns[1], "l": 80}},
            ]
        return [{}, {}]

    def close(self):
        pass


def test_per_env_episode_tracking():
    """Verify that episodes from different envs are tracked separately."""
    cfg = Config(
        n_envs=2,
        n_steps=3,
        total_timesteps=6,
        save_freq=0,
        eval_freq=0,
    )
    em = FakeEnvManager(n_envs=2)
    trainer = AsyncTrainer(cfg=cfg, env_manager=em)

    trainer.train(total_timesteps=6)

    # Both episodes should be tracked separately
    assert len(trainer._ep_returns) == 2, f"Expected 2 episodes, got {len(trainer._ep_returns)}"
    # best_reward should be the max of individual episodes, not their sum
    assert trainer.best_reward == 200.0, f"Expected best_reward=200.0, got {trainer.best_reward}"
    assert 100.0 in trainer._ep_returns
    assert 200.0 in trainer._ep_returns


if __name__ == "__main__":
    test_per_env_episode_tracking()
    print("PASS: per-env episode tracking")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_async_trainer.py`
Expected: FAIL — `best_reward` will be 300.0 (sum) instead of 200.0 (max), or `_ep_returns` will have 1 element instead of 2.

- [ ] **Step 3: Modify `env_manager.py` to expose infos**

In `rl/env_manager.py`, modify `collect_step` to return infos:

```python
# Current (line 117-139):
def collect_step(self, obs: torch.Tensor) -> torch.Tensor:
    """One full step: policy → env step → buffer add. Returns new obs."""
    policy_out = self.ppo.collect_step(obs)
    action_gpu = policy_out["action"]
    action_np = action_gpu.cpu().numpy().astype(np.int32)

    env_out = self.step(action_np)
    new_obs = env_out["obs"]
    rewards = env_out["rewards"]
    dones = env_out["dones"]
    terminated = env_out["terminated"]

    self.buffer.add(
        obs=obs,
        action=action_gpu,
        reward=rewards,
        log_prob=policy_out["log_prob"],
        value=policy_out["value"],
        done=dones,
        terminated=terminated,
    )

    return new_obs

# Change to:
def collect_step(self, obs: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
    """One full step: policy → env step → buffer add. Returns (new_obs, infos)."""
    policy_out = self.ppo.collect_step(obs)
    action_gpu = policy_out["action"]
    action_np = action_gpu.cpu().numpy().astype(np.int32)

    env_out = self.step(action_np)
    new_obs = env_out["obs"]
    rewards = env_out["rewards"]
    dones = env_out["dones"]
    terminated = env_out["terminated"]
    infos = env_out["infos"]

    self.buffer.add(
        obs=obs,
        action=action_gpu,
        reward=rewards,
        log_prob=policy_out["log_prob"],
        value=policy_out["value"],
        done=dones,
        terminated=terminated,
    )

    return new_obs, infos
```

- [ ] **Step 4: Modify `async_trainer.py` to use infos**

In `rl/async_trainer.py`:

**4a. Remove the broken accumulator (lines 56-59):**

```python
# Remove these lines:
self.best_reward = float("-inf")
self._ep_returns: list[float] = []
self._ep_lengths: list[int] = []
self._current_ep_return = 0.0
self._current_ep_len = 0

# Replace with:
self.best_reward = float("-inf")
self._ep_returns: list[float] = []
self._ep_lengths: list[int] = []
```

**4b. Update `_collect_rollout` to read from infos (lines 78-124):**

```python
def _collect_rollout(self, obs: torch.Tensor) -> Dict[str, Any]:
    """Collect n_steps of transitions."""
    n_envs = self.em.n_envs

    for _ in range(self.cfg.n_steps):
        if self._check_stop():
            break

        new_obs, infos = self.em.collect_step(obs)

        # Track per-episode returns from C++ info
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                r = ep["r"]
                l = ep["l"]
                self._ep_returns.append(r)
                self._ep_lengths.append(l)
                if r > self.best_reward:
                    self.best_reward = r

        obs = new_obs

    with torch.no_grad():
        last_value = self.em.ppo.model.get_value(obs)
        last_done = torch.tensor(
            np.zeros(n_envs, dtype=bool), dtype=torch.bool, device=self.device
        )

    return {
        "last_value": last_value.cpu().numpy(),
        "last_done": last_done,
        "final_obs": obs,
    }
```

Note: The old code read `rewards`, `dones`, `terminated` from the buffer to track episodes. Now we read from `infos` directly. The `last_done` for GAE bootstrap is computed separately (see Step 5).

**4c. Fix `last_done` computation**

The old code used `terminated` from the buffer for the last step. We need to preserve this. Add a helper method:

```python
def _get_last_terminated(self, n_envs: int) -> torch.Tensor:
    """Get the terminated flag for the last collected step."""
    buf = self.em.buffer
    pos = buf.pos - 1
    if pos < 0:
        return torch.zeros(n_envs, dtype=torch.bool, device=self.device)
    start = pos * n_envs
    end = start + n_envs
    return buf.terminated[start:end]
```

Update `_collect_rollout` to use it:

```python
# In _collect_rollout, replace:
last_done = torch.tensor(terminated, dtype=torch.bool, device=self.device)
# With:
last_done = self._get_last_terminated(n_envs)
```

- [ ] **Step 5: Update `train()` to use new `collect_step` return**

In `rl/async_trainer.py`, the `train()` method calls `_collect_rollout(obs)` which now handles the new return type internally. No changes needed in `train()` itself.

However, update the `metrics.n_episodes` and `metrics.best_reward` lines (190-191) — they already work correctly with the new `_ep_returns` list.

- [ ] **Step 6: Update `FakeEnvManager` in test to match new interface**

The test's `FakeEnvManager.collect_step` must return `(new_obs, infos)`:

```python
def collect_step(self, obs):
    self._step_count += 1
    new_obs = torch.randn(self.n_envs, self.obs_size)
    return new_obs, self.get_infos()
```

Also, the `AsyncTrainer` constructor calls `self.em.reset()` which returns obs. The `FakeEnvManager` already handles this.

The `AsyncTrainer` also accesses `self.em.ppo.model.get_value(obs)` and `self.em.buffer`. Add these to `FakeEnvManager`:

```python
class FakeEnvManager:
    def __init__(self, n_envs=2, obs_size=10, n_actions=5):
        self.n_envs = n_envs
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.device = torch.device("cpu")
        self._step_count = 0
        self._episode_returns = {0: 100.0, 1: 200.0}

        # Minimal PPO model for get_value
        from rl.actor_critic import ActorCritic
        self.ppo = type("FakePPO", (), {})()
        self.ppo.model = ActorCritic(obs_size, n_actions, [16], torch.device("cpu"))

        # Minimal buffer for terminated access
        from rl.rollout_buffer import RolloutBuffer
        self.buffer = RolloutBuffer(
            n_steps=10, n_envs=n_envs, obs_size=obs_size,
            n_actions=n_actions, gamma=0.99, gae_lambda=0.95,
            device=torch.device("cpu"),
        )

    def reset(self):
        return torch.randn(self.n_envs, self.obs_size)

    def collect_step(self, obs):
        self._step_count += 1
        new_obs = torch.randn(self.n_envs, self.obs_size)
        return new_obs, self.get_infos()

    def get_infos(self):
        if self._step_count >= 3:
            return [
                {"episode": {"r": self._episode_returns[0], "l": 50}},
                {"episode": {"r": self._episode_returns[1], "l": 80}},
            ]
        return [{}, {}]

    def close(self):
        pass
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python tests/test_async_trainer.py`
Expected: PASS

- [ ] **Step 8: Run existing tests to verify no regressions**

Run: `python tests/test_ppo_smoke.py`
Expected: PASS

Run: `python tests/test_evaluator.py`
Expected: PASS (or skip if env not available)

- [ ] **Step 9: Commit**

```bash
git add rl/async_trainer.py rl/env_manager.py tests/test_async_trainer.py
git commit -m "fix: per-env episode tracking from C++ info instead of broken accumulator"
```
