### Task 3: In-Training Eval + Best Model Saving

**Problem:** `config.py:83-84` declares `eval_freq: int = 100_000` and `eval_episodes: int = 10`, but these are never used in `async_trainer.py` or `train.py`. There is no in-training evaluation — the only eval is the manual `train_ui/evaluator.py` run post-training. There is no best-model saving: the trainer saves periodic checkpoints and a final model, but never saves a "best" model based on eval performance.

**Solution:**
1. Add an `_eval()` method to `AsyncTrainer` that calls `run_eval` with the current policy.
2. Call `_eval()` every `eval_freq` steps in `train()`.
3. Log `eval/days`, `eval/people`, `eval/bases` to TensorBoard.
4. Save `best_model.pt` + `best_model.meta.json` when `eval/days` improves.
5. Pass the TensorBoard writer to the trainer via `progress_callback`.

**Files:**
- Modify: `rl/async_trainer.py` — add `_eval()`, call in `train()`, save best model
- Modify: `train.py` — pass writer to trainer, log eval metrics
- Test: `tests/test_async_trainer.py` (extend)

**Interfaces:**
- Consumes: `train_ui.evaluator.run_eval(model_path, episodes, max_days, seed, device, normalization_path) -> dict`
- Produces: `AsyncTrainer._eval() -> dict`, `AsyncTrainer.best_eval_days: float`, `best_model.pt` + `best_model.meta.json` in `cfg.model_dir`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_async_trainer.py`:

```python
def test_eval_and_best_model_saving(tmp_path):
    """Verify that eval runs and best model is saved when eval improves."""
    import json
    from pathlib import Path

    cfg = Config(
        n_envs=2,
        n_steps=3,
        total_timesteps=6,
        save_freq=0,
        eval_freq=3,
        eval_episodes=2,
        model_dir=str(tmp_path),
    )
    em = FakeEnvManager(n_envs=2)
    trainer = AsyncTrainer(cfg=cfg, env_manager=em)

    # Mock run_eval to return improving results
    eval_results = [
        {"days": 10.0, "people": 5.0, "bases": 2.0, "episodes": 2.0, "avg_return": 100.0},
        {"days": 20.0, "people": 10.0, "bases": 4.0, "episodes": 2.0, "avg_return": 200.0},
    ]
    eval_call_count = [0]

    def mock_run_eval(*args, **kwargs):
        idx = min(eval_call_count[0], len(eval_results) - 1)
        eval_call_count[0] += 1
        return eval_results[idx]

    import train_ui.evaluator as ev
    original_run_eval = ev.run_eval
    ev.run_eval = mock_run_eval

    try:
        trainer.train(total_timesteps=6)
    finally:
        ev.run_eval = original_run_eval

    # Verify best model was saved
    best_model = tmp_path / "best_model.pt"
    best_meta = tmp_path / "best_model.meta.json"
    assert best_model.exists(), "best_model.pt should exist"
    assert best_meta.exists(), "best_model.meta.json should exist"

    with open(best_meta) as f:
        meta = json.load(f)
    assert meta["best_eval_days"] == 20.0
    assert meta["eval_people"] == 10.0
    assert meta["eval_bases"] == 4.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_async_trainer.py::test_eval_and_best_model_saving -v`
Expected: FAIL — `best_model.pt` does not exist.

- [ ] **Step 3: Add `_eval()` method to `AsyncTrainer`**

In `rl/async_trainer.py`, add after `_update_ppo`:

```python
def _eval(self, total_done: int) -> Dict[str, float]:
    """Run evaluation episodes with the current policy.

    Returns dict with days, people, bases, avg_return.
    Saves best_model.pt if eval/days improves.
    """
    import json
    from train_ui.evaluator import run_eval

    save_dir = Path(self.cfg.model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save current model to a temp path for eval
    eval_model_path = save_dir / "_eval_temp.pt"
    self.em.ppo.save(str(eval_model_path))

    # Load normalization if available
    norm_path = save_dir / "normalization.json"
    norm_str = str(norm_path) if norm_path.exists() else None

    try:
        result = run_eval(
            model_path=str(eval_model_path),
            episodes=self.cfg.eval_episodes,
            max_days=1000,
            seed=42,
            device=str(self.device),
            normalization_path=norm_str,
        )
    finally:
        # Clean up temp model
        if eval_model_path.exists():
            eval_model_path.unlink()

    self._log(
        f"[Eval @ {total_done:,}] days={result['days']:.1f} "
        f"people={result['people']:.1f} bases={result['bases']:.1f} "
        f"return={result['avg_return']:.1f}"
    )

    # Save best model if eval/days improved
    if self.best_eval_days is None or result["days"] > self.best_eval_days:
        self.best_eval_days = result["days"]
        best_path = save_dir / "best_model.pt"
        self.em.ppo.save(str(best_path))

        norm_path = save_dir / "normalization.json"
        if norm_path.exists():
            import shutil
            shutil.copy2(norm_path, str(best_path).replace(".pt", ".norm.json"))

        meta = {
            "best_eval_days": result["days"],
            "eval_people": result["people"],
            "eval_bases": result["bases"],
            "eval_return": result["avg_return"],
            "total_timesteps": total_done,
            "episodes": self.cfg.eval_episodes,
        }
        meta_path = save_dir / "best_model.meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._log(f"[Best] Saved best_model.pt (days={result['days']:.1f})")

    return result
```

- [ ] **Step 4: Initialize `best_eval_days` in `__init__`**

In `rl/async_trainer.py`, in `__init__` (after line 59):

```python
self.best_eval_days: Optional[float] = None
```

- [ ] **Step 5: Call `_eval()` in `train()`**

In `rl/async_trainer.py`, in the `train()` method, after the checkpoint save block (around line 214-217), add:

```python
# Run eval every eval_freq steps
if self.cfg.eval_freq > 0 and total_done % self.cfg.eval_freq < steps_per_rollout:
    eval_result = self._eval(total_done)
    # Log to progress callback (TensorBoard)
    if self.progress_callback:
        try:
            self.metrics.eval_days = eval_result["days"]
            self.metrics.eval_people = eval_result["people"]
            self.metrics.eval_bases = eval_result["bases"]
            self.progress_callback(self.metrics)
        except Exception:
            pass
```

- [ ] **Step 6: Add eval fields to `TrainMetrics`**

In `rl/async_trainer.py`, add to the `TrainMetrics` dataclass:

```python
@dataclass
class TrainMetrics:
    # ... existing fields ...
    eval_days: float = 0.0
    eval_people: float = 0.0
    eval_bases: float = 0.0
    best_eval_days: float = 0.0
```

- [ ] **Step 7: Update `train.py` to log eval metrics to TensorBoard**

In `train.py`, update `progress_cb`:

```python
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
```

- [ ] **Step 8: Update `FakeEnvManager` for eval test**

The `FakeEnvManager` needs a `ppo.save()` method. Update the test's `FakeEnvManager`:

```python
class FakeEnvManager:
    def __init__(self, n_envs=2, obs_size=10, n_actions=5):
        # ... existing init ...

        # Add save method
        self.ppo = type("FakePPO", (), {
            "save": staticmethod(lambda path: None),
        })()

    def save_normalization(self, path):
        """No-op for testing."""
        pass
```

Also, the test needs to mock `train_ui.evaluator.run_eval`. The test already does this.

- [ ] **Step 9: Run tests**

Run: `python -m pytest tests/test_async_trainer.py -v`
Expected: PASS (both test_per_env_episode_tracking and test_eval_and_best_model_saving)

Run: `python tests/test_ppo_smoke.py`
Expected: PASS

Run: `python -m pytest tests/test_evaluator.py -v`
Expected: PASS

- [ ] **Step 10: Add integration smoke test**

Add to `tests/test_integration.py` (or create if not exists):

```python
def test_training_with_eval_integration(tmp_path):
    """End-to-end: train for a few steps with eval, verify best_model and normalization exist."""
    import sys
    from pathlib import Path
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

    from rl.config import Config
    from rl.env_manager import EnvManager
    from rl.async_trainer import AsyncTrainer

    cfg = Config(
        n_envs=2,
        n_steps=32,
        total_timesteps=256,
        save_freq=128,
        eval_freq=128,
        eval_episodes=2,
        map_size=100,
        model_dir=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
    )

    device = torch.device("cpu")
    em = EnvManager(cfg, device)

    # Save initial normalization
    norm_path = Path(cfg.model_dir) / "normalization.json"
    em.env.venv.save_normalization(str(norm_path))
    assert norm_path.exists(), "normalization.json should exist after initial save"

    trainer = AsyncTrainer(cfg=cfg, env_manager=em)
    trainer.train(total_timesteps=256)

    # Verify best model was saved (eval should have run at step 128 and 256)
    best_model = Path(cfg.model_dir) / "best_model.pt"
    best_meta = Path(cfg.model_dir) / "best_model.meta.json"
    assert best_model.exists(), "best_model.pt should exist after training with eval"
    assert best_meta.exists(), "best_model.meta.json should exist"

    # Verify normalization exists
    assert norm_path.exists(), "normalization.json should still exist"

    em.close()
```

- [ ] **Step 11: Commit**

```bash
git add rl/async_trainer.py train.py tests/test_async_trainer.py tests/test_integration.py
git commit -m "feat: in-training eval with TensorBoard logging and best-model saving"
```
