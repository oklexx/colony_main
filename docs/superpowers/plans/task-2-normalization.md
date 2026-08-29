### Task 2: Normalization Save/Load

**Problem:** VecNormalize (RunningMeanStd) is maintained in C++ during training but never saved. `save_normalization`/`load_normalization` exist in C++ (`env.cpp:915-937`) and are exported to Python (`bindings.cpp:426-427`), but `train.py` never calls them. The evaluator (`train_ui/evaluator.py`) uses `CppColonyEnv` (single env) which does NOT normalize observations, while training uses `CppVecEnv` which DOES. This means eval runs on raw obs while training used normalized obs — the policy sees different input distributions.

**Solution:**
1. Add a `Normalizer` class in `python/cpp_env.py` that wraps `colony_cpp.RunningMeanStd` for single-env normalization.
2. Integrate `save_normalization`/`load_normalization` into the training and eval flow.
3. Update `evaluator.py` to load normalization stats and apply them to observations.

**Files:**
- Modify: `python/cpp_env.py` — add `Normalizer` class, integrate into `CppColonyEnv`
- Modify: `train_ui/evaluator.py` — load normalization, apply to obs
- Modify: `train.py` — save normalization alongside model checkpoints
- Test: `tests/test_normalizer.py` (new)
- Test: `tests/test_evaluator.py` (update)

**Interfaces:**
- Consumes: `colony_cpp.RunningMeanStd` (mean, var, count, normalize, update)
- Produces: `Normalizer.normalize(obs) -> np.ndarray`, `Normalizer.update(obs)`, `Normalizer.save(path)`, `Normalizer.load(path)`, `Normalizer.to_dict() -> dict`, `Normalizer.from_dict(d) -> Normalizer`

- [ ] **Step 1: Write the failing test for Normalizer**

Create `tests/test_normalizer.py`:

```python
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

try:
    import colony_cpp
    ENV_OK = True
except Exception:
    ENV_OK = False


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_normalizer_normalize():
    from cpp_env import Normalizer

    norm = Normalizer(obs_size=5)
    # Set known stats
    norm._rms.set_mean([0.0, 0.0, 0.0, 0.0, 0.0])
    norm._rms.set_var([1.0, 1.0, 1.0, 1.0, 1.0])

    obs = np.array([1.0, -1.0, 2.0, 0.0, 0.5], dtype=np.float32)
    result = norm.normalize(obs)

    # With mean=0, var=1: normalize = (v - 0) / sqrt(1 + eps) ≈ v
    np.testing.assert_allclose(result, [1.0, -1.0, 2.0, 0.0, 0.5], atol=0.01)


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_normalizer_save_load(tmp_path):
    from cpp_env import Normalizer

    norm = Normalizer(obs_size=3)
    norm._rms.set_mean([1.0, 2.0, 3.0])
    norm._rms.set_var([0.5, 0.5, 0.5])
    norm._rms.set_count(100.0)

    path = tmp_path / "norm.json"
    norm.save(str(path))

    norm2 = Normalizer(obs_size=3)
    norm2.load(str(path))

    np.testing.assert_allclose(norm2._rms.mean(), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(norm2._rms.var(), [0.5, 0.5, 0.5])
    assert norm2._rms.count() == 100


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_normalizer_to_from_dict():
    from cpp_env import Normalizer

    norm = Normalizer(obs_size=4)
    norm._rms.set_mean([1.0, 2.0, 3.0, 4.0])
    norm._rms.set_var([1.0, 1.0, 1.0, 1.0])
    norm._rms.set_count(50.0)

    d = norm.to_dict()
    assert d["mean"] == [1.0, 2.0, 3.0, 4.0]
    assert d["var"] == [1.0, 1.0, 1.0, 1.0]
    assert d["count"] == 50.0

    norm2 = Normalizer.from_dict(d)
    np.testing.assert_allclose(norm2._rms.mean(), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(norm2._rms.var(), [1.0, 1.0, 1.0, 1.0])
    assert norm2._rms.count() == 50


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_normalizer_set_update():
    """Verify set_update(False) prevents statistics updates."""
    from cpp_env import Normalizer

    norm = Normalizer(obs_size=3)
    norm._rms.set_mean([0.0, 0.0, 0.0])
    norm._rms.set_var([1.0, 1.0, 1.0])

    # Enable updates, update stats
    norm.set_update(True)
    norm.update(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    mean_after_update = list(norm._rms.mean())

    # Disable updates, try to update again
    norm.set_update(False)
    norm.update(np.array([100.0, 200.0, 300.0], dtype=np.float32))
    mean_after_disabled = list(norm._rms.mean())

    # Mean should not have changed after disabling updates
    np.testing.assert_allclose(mean_after_update, mean_after_disabled)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_normalizer.py -v`
Expected: FAIL — `ImportError: cannot import name 'Normalizer' from 'cpp_env'`

- [ ] **Step 3: Implement Normalizer in `cpp_env.py`**

Add to `python/cpp_env.py` (after the imports, before `CppColonyEnv`):

```python
class Normalizer:
    """Single-env observation normalizer using C++ RunningMeanStd.

    Wraps colony_cpp.RunningMeanStd for use with CppColonyEnv (single env),
    matching the normalization applied by CppVecEnv during training.
    """

    def __init__(self, obs_size: int):
        self._rms = colony_cpp.RunningMeanStd(obs_size)
        self._obs_size = obs_size
        self._clip = 10.0
        self._update_enabled = True

    def set_update(self, enable: bool):
        """Enable/disable running statistics updates.

        Set to False during eval to prevent statistics drift.
        """
        self._update_enabled = enable

    def update(self, obs: np.ndarray):
        """Update running statistics with a single observation (if enabled)."""
        if not self._update_enabled:
            return
        obs_flat = np.asarray(obs, dtype=np.float32).flatten()
        self._rms.update(obs_flat, 1, self._obs_size)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize a single observation using current statistics."""
        obs_flat = np.asarray(obs, dtype=np.float32).flatten().copy()
        self._rms.normalize(obs_flat, 1, self._obs_size, self._clip)
        return obs_flat

    def save(self, path: str):
        """Save normalization stats to JSON file."""
        import json
        d = {
            "mean": list(self._rms.mean()),
            "var": list(self._rms.var()),
            "count": self._rms.count(),
            "obs_size": self._obs_size,
            "clip": self._clip,
        }
        with open(path, "w") as f:
            json.dump(d, f)

    def load(self, path: str):
        """Load normalization stats from JSON file."""
        import json
        with open(path) as f:
            d = json.load(f)
        self._rms.set_mean(d["mean"])
        self._rms.set_var(d["var"])
        self._rms.set_count(d["count"])
        self._obs_size = d.get("obs_size", self._obs_size)
        self._clip = d.get("clip", 10.0)

    def to_dict(self) -> dict:
        return {
            "mean": list(self._rms.mean()),
            "var": list(self._rms.var()),
            "count": self._rms.count(),
            "obs_size": self._obs_size,
            "clip": self._clip,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        n = cls(obs_size=d.get("obs_size", 0))
        n._rms.set_mean(d["mean"])
        n._rms.set_var(d["var"])
        n._rms.set_count(d["count"])
        n._clip = d.get("clip", 10.0)
        return n
```

- [ ] **Step 4: Integrate Normalizer into CppColonyEnv**

In `python/cpp_env.py`, modify `CppColonyEnv`:

**4a. Add `normalizer` attribute in `__init__`:**

```python
# After self.action_space = gym.spaces.Discrete(self.cpp_env.n_actions())
# Add:
self.normalizer = Normalizer(obs_size=self.cpp_env.obs_size())
```

**4b. Apply normalization in `reset()`:**

```python
# Current:
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    s = seed if seed is not None else np.random.randint(1 << 31)
    self.cpp_env.reset(s)
    obs = np.array(self.cpp_env.obs(), dtype=np.float32)
    return obs, {"seed": s, "days": 0}

# Change to:
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    s = seed if seed is not None else np.random.randint(1 << 31)
    self.cpp_env.reset(s)
    raw_obs = np.array(self.cpp_env.obs(), dtype=np.float32)
    self.normalizer.update(raw_obs)
    obs = self.normalizer.normalize(raw_obs)
    return obs, {"seed": s, "days": 0}
```

**4c. Apply normalization in `step()`:**

```python
# Current:
def step(self, action):
    result = self.cpp_env.step(int(action))
    obs = np.array(result["obs"], dtype=np.float32)
    obs = np.clip(obs, -self.clip_obs, self.clip_obs).astype(np.float32)
    reward = float(result["reward"])
    # ...

# Change to:
def step(self, action):
    result = self.cpp_env.step(int(action))
    raw_obs = np.array(result["obs"], dtype=np.float32)
    self.normalizer.update(raw_obs)
    obs = self.normalizer.normalize(raw_obs)
    reward = float(result["reward"])
    # ... (rest unchanged)
```

- [ ] **Step 5: Update `evaluator.py` to load normalization**

In `train_ui/evaluator.py`:

**5a. Add `normalization_path` parameter to `run_eval`:**

```python
def run_eval(
    model_path: str | Path,
    episodes: int = 5,
    max_days: int = 1000,
    seed: int = 7,
    device: str = "cpu",
    normalization_path: str | Path | None = None,
) -> Dict[str, float]:
```

**5b. Load normalization and disable updates before running episodes:**

```python
# After creating env:
env = CppColonyEnv(map_size=200)

# Load normalization if provided and disable statistics updates
if normalization_path is not None:
    norm_path = Path(normalization_path)
    if norm_path.exists():
        env.normalizer.load(str(norm_path))
        env.normalizer.set_update(False)
```

**5c. Add `normalization_path` passthrough in `worker.py`:**

In `train_ui/worker.py`, find the `run_eval` function (line 262) and add:

```python
def run_eval(model_path: str, episodes: int, max_days: int, seed: int,
             device: str = "cpu", normalization_path: str = ""):
    from train_ui.evaluator import run_eval as _run_eval
    norm = Path(normalization_path) if normalization_path else None
    result = _run_eval(model_path, episodes=episodes, max_days=max_days,
                       seed=seed, device=device, normalization_path=norm)
```

- [ ] **Step 6: Save normalization in `train.py`**

In `train.py`, save the normalization **immediately after creating `EnvManager`** (before training starts), so the file exists for the first eval:

```python
# After: em = EnvManager(cfg, device)
# Add:
norm_path = Path(cfg.model_dir) / "normalization.json"
em.env.venv.save_normalization(str(norm_path))
print(f"[Save] Normalization: {norm_path}")
```

Also save alongside each checkpoint. In `rl/async_trainer.py`, in the `train()` method where checkpoints are saved (around line 214-217):

In `rl/async_trainer.py`, in the `train()` method where checkpoints are saved (around line 214-217):

```python
if save_every > 0 and rollout_idx % save_every == 0:
    ckpt_path = save_dir / f"checkpoint_{total_done}_steps.pt"
    self.em.ppo.save(str(ckpt_path))
    # Save normalization alongside
    norm_path = str(ckpt_path).replace(".pt", ".norm.json")
    self.em.env.venv.save_normalization(norm_path)
    self._log(f"[Save] {ckpt_path}")
    self._log(f"[Save] {norm_path}")
```

And for the final model (around line 224-225):

```python
final_path = save_dir / "final_model.pt"
self.em.ppo.save(str(final_path))
norm_path = str(final_path).replace(".pt", ".norm.json")
self.em.env.venv.save_normalization(norm_path)
self._log(f"[Save] Final model: {final_path}")
self._log(f"[Save] Normalization: {norm_path}")
```

- [ ] **Step 7: Update `test_evaluator.py` to test normalization loading**

Add to `tests/test_evaluator.py`:

```python
@pytest.mark.skipif(not ENV_OK, reason="env not available")
def test_run_eval_with_normalization(tmp_path, monkeypatch):
    """Test that run_eval loads normalization when provided."""
    import train_ui.evaluator as ev
    import numpy as np

    ckpt = _make_actor_critic_checkpoint(tmp_path)

    # Create a fake normalization file
    import json
    norm_data = {
        "mean": [0.0] * 203,
        "var": [1.0] * 203,
        "count": 100,
        "obs_size": 203,
        "clip": 10.0,
    }
    norm_path = tmp_path / "norm.json"
    with open(norm_path, "w") as f:
        json.dump(norm_data, f)

    fake_env = _fake_env_class()
    monkeypatch.setattr(ev, "CppColonyEnv", fake_env, raising=False)

    class FakePolicy:
        def __call__(self, obs):
            return torch.zeros(1, 45), torch.zeros(1, 1)

    monkeypatch.setattr(ev, "_load_policy", lambda *a, **kw: FakePolicy())
    result = ev.run_eval(ckpt, episodes=1, max_days=10, seed=1, device="cpu",
                         normalization_path=str(norm_path))

    assert result["episodes"] == 1.0
```

- [ ] **Step 8: Run tests**

Run: `python -m pytest tests/test_normalizer.py -v`
Expected: PASS

Run: `python -m pytest tests/test_evaluator.py -v`
Expected: PASS

Run: `python tests/test_ppo_smoke.py`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add python/cpp_env.py train_ui/evaluator.py train_ui/worker.py train.py rl/async_trainer.py tests/test_normalizer.py tests/test_evaluator.py
git commit -m "feat: save/load VecNormalize stats, integrate into eval"
```
