# Colony RL - Minimap Research + System Improvements

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Close the minimap CNN research loop (configurable radius, hybrid obs, multi-seed eval), add minimap support to watch_champion, and extend Optuna to search obs_mode.

**Architecture:** All changes extend the existing C++/Python/PyTorch pipeline. C++ side adds set_minimap_radius(int). Python side adds a hybrid CNN variant, --minimap-radius CLI flag, --eval-seeds CLI flag, and hybrid support in evaluator/watch.

**Tech Stack:** C++ (pybind11), Python 3.11, PyTorch 2.13 CUDA bf16, PySide6, Optuna, CMake + VS 2022 BuildTools.

**Spec:** docs/minimap_training_report.md (500k comparison + Next steps / open questions).

## Global Constraints

- C++ must build with CMake Release + pybind11, output to python/colony_cpp.pyd, copy to python/colony_cpp.pyd.
- All new Python code must pass python -m pytest tests/ -q (102+ tests).
- No new dependencies beyond what is already in the repo.
- Minimap channels fixed at 8 (LT_NORMAL, LT_WATER, LT_WOOD, LT_COAL, LT_IRON, LT_OIL, LT_GOLD, occupied).
- Eval scoring: score = 0.4*days + 3.0*bases + 0.2*people + 0.0001*max(0, return), threshold bases >= 5 AND return >= 0.

---

### Task 1: C++ set_minimap_radius() + --minimap-radius CLI

Make the minimap radius configurable at runtime (no recompile for R=28, R=40, etc.).

**Files:**
- Modify: include/colony/env.h (add setter to ColonyEnvCpp + ColonyVecEnvCpp)
- Modify: src/bindings.cpp (expose as Python method)
- Modify: python/minimap.py (add refresh() method)
- Modify: rl/env_manager.py (apply radius after env creation)
- Modify: train.py (add --minimap-radius flag)
- Test: tests/test_minimap_radius.py (new)

**Produces:** env.cpp_env.set_minimap_radius(R), venv.venv.set_minimap_radius(R). minimap() returns [8, 2R+1, 2R+1] for the new R.

- [ ] **Step 1: Add set_minimap_radius to ColonyEnvCpp in include/colony/env.h**

After line 69 (std::vector<float> minimap() const;), add:

```cpp
    void set_minimap_radius(int r) { minimap_radius_ = r; }
```

- [ ] **Step 2: Add set_minimap_radius to ColonyVecEnvCpp in include/colony/env.h**

After line 205 (std::vector<float> minimap_batch() const;), add:

```cpp
    void set_minimap_radius(int r) { minimap_radius_ = r; for (auto& e : envs_) e.set_minimap_radius(r); }
```

- [ ] **Step 3: Expose set_minimap_radius in src/bindings.cpp**

In the ColonyEnvCpp binding block (after .def("minimap_radius", ...) line 342), add:

```cpp
        .def("set_minimap_radius", [](ColonyEnvCpp& env, int r) { env.set_minimap_radius(r); })
```

In the ColonyVecEnvCpp binding block (after .def("minimap_radius", ...) line 456), add:

```cpp
        .def("set_minimap_radius", [](ColonyVecEnvCpp& v, int r) { v.set_minimap_radius(r); })
```

- [ ] **Step 4: Rebuild the C++ extension**

```bash
cmake --build build --config Release
copy build\Release\colony_cpp.pyd python\colony_cpp.pyd
```

- [ ] **Step 5: Write the failing test**

Create tests/test_minimap_radius.py:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import numpy as np
from cpp_env import CppColonyEnv


def test_set_minimap_radius_single():
    env = CppColonyEnv(map_size=100)
    try:
        r0 = int(env.cpp_env.minimap_radius())
        assert r0 == 14
        mm0 = env.cpp_env.minimap()
        assert mm0.shape == (8, 29, 29)

        env.cpp_env.set_minimap_radius(5)
        assert int(env.cpp_env.minimap_radius()) == 5
        mm5 = env.cpp_env.minimap()
        assert mm5.shape == (8, 11, 11)
    finally:
        env.close()


def test_set_minimap_radius_vec():
    from cpp_env import CppVecEnv
    venv = CppVecEnv(num_envs=2, map_size=100)
    try:
        venv.venv.set_minimap_radius(3)
        assert int(venv.venv.minimap_radius()) == 3
        mm = venv.venv.minimap_batch()
        assert mm.shape == (2, 8, 7, 7)
    finally:
        venv.close()
```

- [ ] **Step 6: Run test to verify it passes**

```bash
python -m pytest tests/test_minimap_radius.py -v
```

Expected: PASS (both tests). If FAIL, the C++ rebuild or bindings are wrong.

- [ ] **Step 7: Update python/minimap.py - add refresh() method**

In MinimapVecEnvWrapper, add:

```python
    def refresh(self):
        self.radius = int(self.venv.venv.minimap_radius())
        self.grid = 2 * self.radius + 1
        self.minimap_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.channels, self.grid, self.grid),
            dtype=np.float32,
        )
```

In MinimapSingleEnvWrapper, add:

```python
    def refresh(self):
        self.radius = int(self.env.cpp_env.minimap_radius())
        self.grid = 2 * self.radius + 1
        self.minimap_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.channels, self.grid, self.grid),
            dtype=np.float32,
        )
```

- [ ] **Step 8: Update rl/env_manager.py to apply radius after env creation**

In EnvManager.__init__, after self.env = make_cpp_vec_env(...) (line 44) and before the minimap wrapper creation, add:

```python
        radius = cfg.minimap_radius
        if radius != 14:
            self.env.venv.set_minimap_radius(radius)
```

After creating MinimapVecEnvWrapper(self.env), add self.mm_env.refresh().

- [ ] **Step 9: Add --minimap-radius flag to train.py**

After the --obs-mode argument (line 45-46), add:

```python
    p.add_argument("--minimap-radius", type=int, default=14,
                   help="Minimap radius R (grid = 2R+1). Default 14 -> 29x29")
```

Wire to config in main() after cfg = Config(...):

```python
    cfg.minimap_radius = args.minimap_radius
```

- [ ] **Step 10: Run full test suite**

```bash
python -m pytest tests/ -q
```

Expected: 104+ passed (102 original + 2 new).

- [ ] **Step 11: Commit**

```bash
git add include/colony/env.h src/bindings.cpp python/minimap.py rl/env_manager.py train.py tests/test_minimap_radius.py
git commit -m "feat: configurable minimap radius via set_minimap_radius() + --minimap-radius CLI"
```

---

### Task 2: Hybrid CNN (flat + minimap) observation

Flat 203-dim has global stats (money, taxes, credits) the minimap lacks. Minimap has spatial layout the flat obs lacks. A hybrid model uses both.

**Files:**
- Create: rl/actor_critic_hybrid.py
- Modify: rl/env_manager.py (add "hybrid" obs_mode)
- Modify: rl/config.py (add "hybrid" to valid obs_modes)
- Modify: rl/ppo.py (save hybrid architecture info)
- Modify: train_ui/evaluator.py (_load_policy detects hybrid)
- Modify: train.py (add "hybrid" to --obs-mode choices)
- Modify: rl/rollout_buffer.py (_TensorRolloutBuffer stores flat obs alongside minimap)
- Test: tests/test_hybrid.py (new)

**Produces:** ActorCriticHybrid(obs_size=203, n_channels=8, grid_size=29, n_actions=45, hidden_sizes=[256,256]) - forward takes (B, 203) + (B, 8, 29, 29), returns (B, 45) logits + (B, 1) values.

- [ ] **Step 1: Create rl/actor_critic_hybrid.py**

```python
"""Hybrid actor-critic: flat MLP branch + CNN branch, merged trunk.

The flat branch handles global stats (money, taxes, credits, day count).
The CNN branch handles spatial layout (land types, buildings, resources).
Both branches feed into a shared trunk MLP, then actor/critic heads.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()


class ActorCriticHybrid(nn.Module):
    def __init__(
        self,
        obs_size: int,
        n_channels: int,
        grid_size: int,
        n_actions: int,
        hidden_sizes: Sequence[int] | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.obs_size = obs_size
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.n_actions = n_actions
        hidden = list(hidden_sizes) if hidden_sizes else [256, 256]
        self.hidden_sizes = hidden

        # Flat branch: obs_size -> hidden[0]
        self.flat_proj = nn.Linear(obs_size, hidden[0])

        # CNN branch: n_channels x grid x grid -> hidden[0]
        conv1 = nn.Conv2d(n_channels, 32, 4, 2, 1)   # 32 x (grid//2) x (grid//2)
        conv2 = nn.Conv2d(32, 64, 2, 2)              # 64 x (grid//4) x (grid//4)
        self.cnn = nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU())
        g1 = (grid_size + 2) // 2 - 1
        g2 = (g1 + 2) // 2 - 1
        self.cnn_proj = nn.Linear(64 * g2 * g2, hidden[0])

        # Shared trunk
        trunk: list[nn.Module] = []
        prev = hidden[0]
        for h in hidden[1:]:
            trunk.append(nn.Linear(prev, h))
            trunk.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*trunk)

        self.actor = nn.Linear(prev, n_actions)
        self.critic = nn.Linear(prev, 1)

        # Init
        _orthogonal_init(self.flat_proj, gain=1.0)
        for m in self.cnn:
            _orthogonal_init(m, gain=math.sqrt(2))
        _orthogonal_init(self.cnn_proj, gain=1.0)
        for m in self.trunk:
            _orthogonal_init(m, gain=1.0)
        _orthogonal_init(self.actor, gain=1.0)
        _orthogonal_init(self.critic, gain=1.0)

        self.to(device)

    def forward(self, flat_obs: torch.Tensor, minimap: torch.Tensor):
        flat_feat = self.flat_proj(flat_obs)
        cnn_feat = self.cnn_proj(self.cnn(minimap).flatten(1))
        x = torch.relu(flat_feat + cnn_feat)
        x = self.trunk(x)
        logits = self.actor(x)
        values = self.critic(x)
        return logits, values

    def act(
        self,
        flat_obs: torch.Tensor,
        minimap: torch.Tensor,
        deterministic: bool = False,
    ):
        with torch.no_grad():
            logits, values = self.forward(flat_obs, minimap)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() if not deterministic else logits.argmax(-1)
        log_prob = dist.log_prob(action)
        return action, log_prob, values.squeeze(-1)
```

- [ ] **Step 2: Add "hybrid" to valid obs_modes in rl/config.py**

In Config.__post_init__ (line 125), change:

```python
        if self.obs_mode not in ("flat", "minimap"):
            raise ValueError("obs_mode must be 'flat' or 'minimap'")
```

to:

```python
        if self.obs_mode not in ("flat", "minimap", "hybrid"):
            raise ValueError("obs_mode must be 'flat', 'minimap', or 'hybrid'")
```

- [ ] **Step 3: Add hybrid model creation in rl/env_manager.py**

In the `if cfg.obs_mode == "minimap":` block (lines 103-124), change the condition to also handle `"hybrid"`. Replace the entire if-block with:

```python
        if cfg.obs_mode in ("minimap", "hybrid"):
            from python.minimap import MinimapVecEnvWrapper
            self.mm_env = MinimapVecEnvWrapper(self.env)
            if cfg.minimap_radius != 14:
                self.mm_env.refresh()
            grid = self.mm_env.grid
            n_channels = self.mm_env.channels
            if cfg.obs_mode == "hybrid":
                from rl.actor_critic_hybrid import ActorCriticHybrid
                self.model = ActorCriticHybrid(
                    obs_size=obs_size, n_channels=n_channels, grid_size=grid,
                    n_actions=n_actions, hidden_sizes=cfg.net_arch, device=device,
                )
            else:
                from rl.actor_critic_cnn import ActorCriticCNN
                self.model = ActorCriticCNN(
                    obs_size=obs_size, n_channels=n_channels, grid_size=grid,
                    n_actions=n_actions, hidden_sizes=cfg.net_arch, device=device,
                )
        else:
            from rl.actor_critic import ActorCritic
            self.model = ActorCritic(
                obs_size=obs_size, n_actions=n_actions,
                hidden_sizes=cfg.net_arch, device=device,
            )
```

- [ ] **Step 4: Update rl/rollout_buffer.py for hybrid mode**

In `_TensorRolloutBuffer.__init__`, the buffer currently stores `minimap` as `(B, C, H, W)`. For hybrid mode, we also need flat observations. Add a `flat_dim` parameter:

```python
    def __init__(self, n, n_envs, n_steps, obs_size, n_channels, grid_size,
                 n_actions, dtype, device, flat_dim: int = 0):
```

If `flat_dim > 0`, allocate `self.flat_obs = torch.empty(n_envs, n_steps, flat_dim, dtype=dtype, device=device)` and store it in `store()`. In `get()`, return it as part of the batch dict as `"flat"`.

In `RolloutBuffer` (the NumPy version), for hybrid mode the flat obs is already the standard `obs` array, so no change needed there.

- [ ] **Step 4a: Create _TensorRolloutBuffer for hybrid mode in EnvManager**

In `rl/env_manager.py`, after model creation, when `cfg.obs_mode == "hybrid"`, create the tensor buffer with `flat_dim`:

```python
        if cfg.obs_mode == "hybrid":
            self.buffer = _TensorRolloutBuffer(
                n_envs=n_envs, n_steps=cfg.n_steps,
                obs_size=obs_size, n_channels=n_channels, grid_size=grid,
                n_actions=n_actions, dtype=torch.float32, device=device,
                flat_dim=obs_size,
            )
```

- [ ] **Step 5: Update rl/ppo.py to handle hybrid forward pass**

In `PPO.__init__`, set the hybrid flag explicitly:

```python
        from rl.actor_critic_hybrid import ActorCriticHybrid
        self.is_hybrid = isinstance(model, ActorCriticHybrid)
```

In `PPO._compute_loss()`, branch the forward call:

```python
        if self.is_hybrid:
            logits, values = self.model(
                flat_obs=batch["obs"],        # (B, 203)
                minimap=batch["minimap"],     # (B, 8, H, W)
            )
        elif self.is_cnn:
            logits, values = self.model(batch["minimap"])
        else:
            logits, values = self.model(batch["obs"])
```

In the rollout collection (`collect_step` / `_collect_rollout`), ensure the batch dict contains both `"obs"` and `"minimap"` keys when hybrid.

In `PPO._save()`, add `n_channels`, `grid_size`, and `obs_size` for hybrid so the flat branch can be reconstructed.

- [ ] **Step 6: Update train_ui/evaluator.py _load_policy for hybrid**

In `_load_policy`, add hybrid detection before the CNN check:

```python
    has_flat_net = any(k.startswith("flat_proj") for k in model_state)
    if has_flat_net:
        from rl.actor_critic_hybrid import ActorCriticHybrid
        model = ActorCriticHybrid(
            obs_size=int(checkpoint.get("obs_size", 203)),
            n_channels=int(checkpoint.get("n_channels", 8)),
            grid_size=int(checkpoint.get("grid_size", 29)),
            n_actions=n_actions,
            hidden_sizes=hidden,
            device=device,
        )
        model.load_state_dict(model_state)
        model.eval()
        return model
```

- [ ] **Step 7: Add "hybrid" to train.py --obs-mode choices**

Change line 45-46:

```python
    p.add_argument("--obs-mode", choices=["flat", "minimap", "hybrid"],
                   default="flat", help="flat=203-dim, minimap=8x29x29 CNN, hybrid=both")
```

- [ ] **Step 8: Update train_ui/evaluator.py run_evaluation for hybrid stepping**

In `run_evaluation`, the step loop needs to pass both flat obs and minimap when the model is hybrid. Add a check:

```python
    is_hybrid = hasattr(model, "flat_proj") and hasattr(model, "cnn")
```

In the step loop, when `is_hybrid`:
```python
        obs_t = torch.tensor(observations[0], dtype=torch.float32, device=device).unsqueeze(0)
        mm_t = torch.tensor(mm, dtype=torch.float32, device=device).unsqueeze(0)
        action, _, _ = model.act(obs_t, mm_t, deterministic=True)
```

- [ ] **Step 9: Write the test**

Create tests/test_hybrid.py:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

@pytest.fixture
def hybrid_model():
    from rl.actor_critic_hybrid import ActorCriticHybrid
    return ActorCriticHybrid(obs_size=203, n_channels=8, grid_size=29,
                             n_actions=45, hidden_sizes=[256, 256], device="cpu")


def test_hybrid_forward_shapes(hybrid_model):
    flat = torch.randn(4, 203)
    mm = torch.randn(4, 8, 29, 29)
    logits, values = hybrid_model(flat, mm)
    assert logits.shape == (4, 45)
    assert values.shape == (4, 1)


def test_hybrid_act(hybrid_model):
    flat = torch.randn(203)
    mm = torch.randn(8, 29, 29)
    action, log_prob, value = hybrid_model.act(flat.unsqueeze(0), mm.unsqueeze(0))
    assert action.shape == (1,)
    assert 0 <= int(action.item()) < 45
    assert log_prob.shape == (1,)


def test_hybrid_state_dict_roundtrip(hybrid_model):
    sd = hybrid_model.state_dict()
    from rl.actor_critic_hybrid import ActorCriticHybrid
    m2 = ActorCriticHybrid(obs_size=203, n_channels=8, grid_size=29,
                           n_actions=45, hidden_sizes=[256, 256], device="cpu")
    m2.load_state_dict(sd)
    flat = torch.randn(203)
    mm = torch.randn(8, 29, 29)
    with torch.no_grad():
        a1, _, _ = hybrid_model.act(flat.unsqueeze(0), mm.unsqueeze(0))
        a2, _, _ = m2.act(flat.unsqueeze(0), mm.unsqueeze(0))
    assert a1.item() == a2.item()
```

- [ ] **Step 10: Run tests**

```bash
python -m pytest tests/test_hybrid.py -v
python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 11: Commit**

```bash
git add rl/actor_critic_hybrid.py rl/config.py rl/env_manager.py rl/ppo.py rl/rollout_buffer.py train_ui/evaluator.py train.py tests/test_hybrid.py
git commit -m "feat: hybrid CNN (flat+minimap) observation mode"
```

---

### Task 3: Multi-seed eval CLI flags

Expose `--eval-seeds` and `--eval-use-mean` in train.py so eval is properly reproducible.

**Files:**
- Modify: train.py (add CLI flags)
- Test: tests/test_multiseed_eval.py (new)

- [ ] **Step 1: Add --eval-seeds and --eval-use-mean to train.py**

After the --eval-samples argument, add:

```python
    p.add_argument("--eval-seeds", type=int, nargs="+", default=None,
                   help="Seeds for eval rollouts (e.g. --eval-seeds 42 43 44)")
    p.add_argument("--eval-use-mean", action="store_true",
                   help="Use mean instead of median for multi-seed eval")
```

Wire in main():

```python
    if args.eval_seeds is not None:
        cfg.eval_seeds = args.eval_seeds
    cfg.eval_use_median = not args.eval_use_mean
```

- [ ] **Step 2: Write the test**

Create tests/test_multiseed_eval.py:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.config import Config


def test_eval_seeds_default():
    cfg = Config(num_envs=2, n_steps=5, map_size=100)
    assert cfg.eval_seeds == [42]
    assert cfg.eval_use_median is True


def test_eval_seeds_custom():
    cfg = Config(num_envs=2, n_steps=5, map_size=100)
    cfg.eval_seeds = [1, 2, 3]
    cfg.eval_use_median = False
    assert cfg.eval_seeds == [1, 2, 3]
    assert cfg.eval_use_median is False


def test_eval_seeds_parsing():
    """Verify argparse-style parsing works."""
    import shlex
    args = shlex.split("--eval-seeds 42 43 44")
    seeds = []
    i = 1
    while i < len(args):
        seeds.append(int(args[i]))
        i += 1
    assert seeds == [42, 43, 44]
```

- [ ] **Step 3: Run tests + commit**

```bash
python -m pytest tests/test_multiseed_eval.py -v
python -m pytest tests/ -q
git add train.py tests/test_multiseed_eval.py
git commit -m "feat: --eval-seeds and --eval-use-mean CLI flags for reproducible eval"
```

---

### Task 4: Minimap support in watch_champion.py

watch_champion.py currently only handles flat MLP models. Extend it to support minimap CNN and hybrid models.

**Files:**
- Modify: watch_champion.py (add minimap/hybrid stepping)
- Test: tests/test_watch_champion_minimap.py (new)

**Prerequisite:** --curriculum-stage flag must work in watch_champion.py. Verify it exists and functions correctly with minimap/hybrid models (stage affects C++ env behavior, not model architecture, so it should work transparently).

**Note on normalization:** For hybrid models, flat observations should be normalized (via normalizer from normalization.json), but the minimap should NOT be normalized (it is already binary 0/1). Ensure the step loop applies normalization only to the flat branch.

- [ ] **Step 1: Update watch_champion.py to detect and handle CNN/hybrid models**

In the step loop (around lines 176-210), add minimap support. The key changes:

1. After loading the model via `load_policy()`, detect the model type:

```python
    is_cnn = hasattr(model, "cnn")
    is_hybrid = hasattr(model, "flat_proj") and hasattr(model, "cnn")
```

2. In the step loop, when `is_cnn or is_hybrid`:
   - Get minimap from the single env: `mm = env.cpp_env.minimap()`
   - For CNN-only: `mm_t = torch.tensor(mm, dtype=torch.float32, device=device).unsqueeze(0); action, _, _ = model.act(mm_t, deterministic=True)`
   - For hybrid: `obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0); mm_t = torch.tensor(mm, dtype=torch.float32, device=device).unsqueeze(0); action, _, _ = model.act(obs_t, mm_t, deterministic=True)`

3. For flat models, keep the existing code path.

- [ ] **Step 2: Write the test**

Create tests/test_watch_champion_minimap.py:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rl.config import Config


def test_hybrid_model_act_with_minimap():
    """Verify hybrid model can act with both flat obs and minimap."""
    from rl.actor_critic_hybrid import ActorCriticHybrid
    model = ActorCriticHybrid(
        obs_size=203, n_channels=8, grid_size=29,
        n_actions=45, hidden_sizes=[128, 128], device="cpu",
    )
    model.eval()
    flat = torch.randn(203)
    mm = torch.randn(8, 29, 29)
    action, log_prob, value = model.act(
        flat.unsqueeze(0), mm.unsqueeze(0), deterministic=True
    )
    assert 0 <= int(action.item()) < 45


def test_cnn_model_act_with_minimap():
    """Verify CNN model can act with minimap."""
    from rl.actor_critic_cnn import ActorCriticCNN
    model = ActorCriticCNN(
        obs_size=203, n_channels=8, grid_size=29,
        n_actions=45, hidden_sizes=[128, 128], device="cpu",
    )
    model.eval()
    mm = torch.randn(8, 29, 29)
    action, log_prob, value = model.act(mm.unsqueeze(0), deterministic=True)
    assert 0 <= int(action.item()) < 45
```

- [ ] **Step 3: Run tests + commit**

```bash
python -m pytest tests/test_watch_champion_minimap.py -v
python -m pytest tests/ -q
git add watch_champion.py tests/test_watch_champion_minimap.py
git commit -m "feat: minimap + hybrid support in watch_champion.py"
```

---

### Task 5: Extend Optuna search space to include obs_mode and minimap_radius

Let auto-trainer search over observation modes and minimap radius.

**Files:**
- Modify: auto_trainer.py (add obs_mode + minimap_radius to objective)
- Test: tests/test_auto_trainer.py (new)

- [ ] **Step 1: Add obs_mode and minimap_radius to auto_trainer.py objective()**

In the objective function, add:

```python
    obs_mode = trial.suggest_categorical("obs_mode", ["flat", "minimap", "hybrid"])
    minimap_radius = trial.suggest_int("minimap_radius", 10, 28, step=4) \
        if obs_mode in ("minimap", "hybrid") else 14
```

Pass these to the Config:

```python
    cfg.eval_obs_mode = obs_mode
    cfg.minimap_radius = minimap_radius
```

- [ ] **Step 2: Write the test**

Create tests/test_auto_trainer.py:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.config import Config


def test_config_accepts_minimap_radius():
    cfg = Config(num_envs=2, n_steps=5, map_size=100)
    cfg.minimap_radius = 28
    assert cfg.minimap_radius == 28


def test_config_accepts_hybrid_obs_mode():
    cfg = Config(num_envs=2, n_steps=5, map_size=100)
    cfg.obs_mode = "hybrid"
    assert cfg.obs_mode == "hybrid"
```

- [ ] **Step 3: Run tests + commit**

```bash
python -m pytest tests/test_auto_trainer.py -v
python -m pytest tests/ -q
git add auto_trainer.py tests/test_auto_trainer.py
git commit -m "feat: Optuna search space extended with obs_mode and minimap_radius"
```

---

### Task 6: Curriculum + CNN comparison experiment

Run a controlled experiment: 500k steps, stage 1->2->3 curriculum, CNN minimap vs flat MLP.

**Files:**
- Create: docs/curriculum_cnn_comparison.md (report)

**Produces:** A report comparing CNN vs flat MLP under curriculum training, with the same eval protocol as the 500k baseline.

**Prerequisite:** `--curriculum-schedule` flag must exist in `train.py`. Verify:

```bash
python train.py --help
```

If missing, add:

```python
    p.add_argument("--curriculum-schedule", type=str, default=None,
                   help="Stage schedule as 'timesteps:stage,timesteps:stage,...' e.g. '200000:1,400000:2,500000:3'")
```

And wire to `cfg.curriculum_schedule` in `main()`.

- [ ] **Step 1: Train flat MLP with curriculum, 500k steps**

```bash
python train.py --run-id flat_cur_500k --timesteps 500000 --num-envs 32 \
  --curriculum-schedule "200000:1,400000:2,500000:3" \
  --eval-interval 50000 --eval-seeds 42 43 44 \
  --model-dir C:\Users\oklex\colony_runs\models\flat_cur_500k
```

- [ ] **Step 2: Train minimap CNN with curriculum, 500k steps**

```bash
python train.py --run-id minimap_cur_500k --timesteps 500000 --num-envs 32 \
  --obs-mode minimap --curriculum-schedule "200000:1,400000:2,500000:3" \
  --eval-interval 50000 --eval-seeds 42 43 44 \
  --model-dir C:\Users\oklex\colony_runs\models\minimap_cur_500k
```

- [ ] **Step 3: Train hybrid with curriculum, 500k steps**

```bash
python train.py --run-id hybrid_cur_500k --timesteps 500000 --num-envs 32 \
  --obs-mode hybrid --curriculum-schedule "200000:1,400000:2,500000:3" \
  --eval-interval 50000 --eval-seeds 42 43 44 \
  --model-dir C:\Users\oklex\colony_runs\models\hybrid_cur_500k
```

- [ ] **Step 4: Compare and write the report**

Create docs/curriculum_cnn_comparison.md with:
- Table: mode x (best_score, best_days, best_bases, best_people, best_return, entropy_trajectory, wall_time)
- Comparison vs the 500k non-curriculum baseline (flat=484, minimap=202)
- Conclusion: does curriculum help CNN? does hybrid beat both?

- [ ] **Step 5: Commit**

```bash
git add docs/curriculum_cnn_comparison.md
git commit -m "docs: curriculum + CNN comparison experiment results"
```

