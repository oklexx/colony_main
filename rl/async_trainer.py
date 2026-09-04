from __future__ import annotations

import time
import threading
import queue
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field

from rl.config import Config
from rl.env_manager import EnvManager
from rl.loop_detector import LoopDetector


@dataclass
class TrainMetrics:
    total_timesteps: int = 0
    fps: float = 0.0
    episode_return: float = 0.0
    episode_length: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    learning_rate: float = 0.0
    gpu_mem_mb: float = 0.0
    wall_time_s: float = 0.0
    n_episodes: int = 0
    best_reward: float = float("-inf")
    eval_days: float = 0.0
    eval_people: float = 0.0
    eval_bases: float = 0.0
    best_eval_days: float = 0.0
    eval_score: float = 0.0
    best_score: float = 0.0


class AsyncTrainer:
    """Async PPO trainer: producer (env steps) + consumer (GPU PPO update)."""

    def __init__(
        self,
        cfg: Config,
        env_manager: EnvManager,
        logger: Optional[Any] = None,
        progress_callback: Optional[Callable[[TrainMetrics], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ):
        self.cfg = cfg
        self.em = env_manager
        self.device = env_manager.device
        self.logger = logger
        self.progress_callback = progress_callback
        self.stop_check = stop_check

        self._stop = False
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=cfg.queue_size)

        self.metrics = TrainMetrics()
        self.best_reward = float("-inf")
        self.best_eval_days: Optional[float] = None
        self.best_score: Optional[float] = None
        self._ep_returns: list[float] = []
        self._ep_lengths: list[int] = []
        self._curriculum_stage = getattr(cfg, "curriculum_stage", 0)

        # Loop detection
        loop_config = getattr(cfg, "loop_detection", {})
        self.loop_detector = LoopDetector(threshold_config=loop_config) if loop_config else None

        # Action name mapping (for displaying loop alerts)
        self._action_names: List[str] = [
            "INITIALIZE", "MOVE_RIGHT", "MOVE_LEFT", "MOVE_UP", "MOVE_DOWN",
            "BUILD_HOUSE", "BUILD_FURNITURE", "GATHER_WOOD", "GATHER_STONE",
            "GATHER_IRON", "PRESERVE", "WAIT"
        ]

        # Boost tracking
        self._boost_history: Dict[int, float] = {}  # step -> boost_factor

    def _request_stop(self):
        self._stop = True

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg, flush=True)

    def _check_stop(self) -> bool:
        if self._stop:
            return True
        if self.stop_check and self.stop_check():
            self._stop = True
            return True
        return False

    def _collect_rollout(self, obs: torch.Tensor) -> Dict[str, Any]:
        """Collect n_steps of transitions."""
        n_envs = self.em.n_envs
        action_names = self._action_names

        # Count actions for top_actions tracking
        action_counts: List[int] = [0] * len(action_names)

        for _ in range(self.cfg.n_steps):
            if self._check_stop():
                break

            new_obs, infos = self.em.collect_step(obs)

            # True termination flag (excludes truncation) for the bootstrap mask.
            terminated = self.em.buffer.terminated[
                (self.em.buffer.pos - 1) * n_envs : self.em.buffer.pos * n_envs
            ].cpu().numpy()

            # Track per-episode returns from C++ info (per-env, no cross-env
            # summing artifacts).
            for info in infos:
                ep = info.get("episode")
                if ep is not None:
                    r = ep.get("r")
                    l = ep.get("l")
                    if r is not None:
                        self._ep_returns.append(r)
                        self._ep_lengths.append(l or 0)
                        if r > self.best_reward:
                            self.best_reward = r

            obs = new_obs

        with torch.no_grad():
            if getattr(self.em, "obs_mode", "flat") == "hybrid":
                last_flat, last_minimap = obs
                last_value = self.em.ppo.model.get_value(last_flat, last_minimap)
            else:
                last_value = self.em.ppo.model.get_value(obs)
            # last_done must be the TRUE termination flag of the LAST collected
            # step (== buffer.terminated[T-1] == terminal(s_T)), NOT an OR over
            # the whole rollout and NOT including time-limit truncation. This is
            # what the GAE bootstrap mask needs.
            last_done = torch.tensor(terminated, dtype=torch.bool, device=self.device)

        # Calculate top actions
        action_counts = self._calculate_action_distribution()

        return {
            "last_value": last_value.cpu().numpy(),
            "last_done": last_done,
            "final_obs": obs,
            "action_counts": action_counts,
        }

    def _calculate_action_distribution(self) -> List[int]:
        """Calculate action frequency counts from current rollout.
        
        Returns list of (count, action_name) tuples sorted by count descending.
        """
        if self.loop_detector is None:
            return [0] * len(self._action_names)
        
        # Get recent action data from buffer (simplified - in practice you'd track this)
        n_envs = self.em.n_envs
        total_steps = self.cfg.n_steps * n_envs
        
        # Sample last 100 actions per env for distribution estimate
        sample_size = min(100, total_steps // max(n_envs, 1))
        
        action_counts = [0] * len(self._action_names)
        
        # Simplified: estimate distribution from recent episodes
        # In a full implementation, you'd track action sequences during rollout
        if self.loop_detector and hasattr(self.loop_detector, '_state'):
            for state in self.loop_detector._state.values():
                for action, _ in state.action_sequence[-sample_size:]:
                    try:
                        idx = self._action_names.index(action)
                        if 0 <= idx < len(action_counts):
                            action_counts[idx] += 1
                    except ValueError:
                        pass
        
        # Normalize to percentages
        total = sum(action_counts)
        if total > 0:
            action_counts = [count / total * 100 for count in action_counts]
        
        return action_counts[:5]  # Return top 5

    def _update_ppo(self, rollout: Dict[str, Any]) -> Dict[str, float]:
        """Run PPO update on GPU."""
        last_value = torch.tensor(rollout["last_value"], dtype=torch.float32, device=self.device)
        last_done = torch.tensor(rollout["last_done"], dtype=torch.bool, device=self.device)

        t0 = time.perf_counter()
        stats = self.em.ppo.update(last_value=last_value, last_done=last_done)
        update_time = time.perf_counter() - t0

        stats["update_time_s"] = update_time
        if torch.cuda.is_available() and self.device.type == "cuda":
            stats["gpu_mem_mb"] = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            stats["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(self.device) / (1024 * 1024)

        return stats

    def _get_current_loop_action(self, stats: Dict[str, Any]) -> Optional[str]:
        """Get the action name currently in loop if any."""
        if self.loop_detector is None:
            return None
        
        # Simple heuristic: check last 10 alerts for most frequent action
        if not hasattr(self.loop_detector, '_history') or not self.loop_detector._history:
            return None
            
        recent_alerts = []
        for entry in self.loop_detector._history[-10:]:
            if isinstance(entry, dict) and "alerts" in entry:
                for env_idx, action in entry["alerts"].items():
                    if action:
                        recent_alerts.append(action)
        
        if recent_alerts:
            from collections import Counter
            counts = Counter(recent_alerts)
            return counts.most_common(1)[0][0]
        
        return None

    def _eval(self, total_done: int) -> Dict[str, float]:
        """Run evaluation episodes with the current policy.

        Returns dict with days, people, bases, avg_return, score.
        Saves best_model.pt if composite score improves AND thresholds are met.
        """
        import json
        from train_ui.evaluator import run_eval

        save_dir = Path(self.cfg.model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        eval_model_path = save_dir / "_eval_temp.pt"
        self.em.ppo.save(str(eval_model_path))

        norm_path = save_dir / "normalization.json"
        norm_str = str(norm_path) if norm_path.exists() else None

        seeds = getattr(self.cfg, "eval_seeds", [42]) or [42]
        all_days: list[float] = []
        all_bases: list[float] = []
        all_people: list[float] = []
        all_returns: list[float] = []

        try:
            for seed in seeds:
                result = run_eval(
                    model_path=str(eval_model_path),
                    episodes=self.cfg.eval_episodes,
                    max_days=1000,
                    seed=seed,
                    device=str(self.device),
                    normalization_path=norm_str,
                    map_size=self.em.cfg.map_size,
                    mode=getattr(self.cfg, "obs_mode", "flat"),
                    minimap_radius=getattr(self.cfg, "minimap_radius", 14),
                )
                all_days.extend(result.get("episode_days", [result["days"]]))
                all_bases.extend(result.get("episode_bases", [result["bases"]]))
                all_people.extend(result.get("episode_people", [result["people"]]))
                all_returns.extend(result.get("episode_returns", [result["avg_return"]]))
        finally:
            if eval_model_path.exists():
                eval_model_path.unlink()

        use_median = getattr(self.cfg, "eval_use_median", True)
        if use_median:
            days_agg = float(np.median(all_days))
            bases_agg = float(np.median(all_bases))
            people_agg = float(np.median(all_people))
            return_agg = float(np.median(all_returns))
        else:
            days_agg = float(np.mean(all_days))
            bases_agg = float(np.mean(all_bases))
            people_agg = float(np.mean(all_people))
            return_agg = float(np.mean(all_returns))

        w1, w2, w3, w4 = getattr(self.cfg, "eval_score_weights", (0.4, 3.0, 0.2, 0.0001))
        score = days_agg * w1 + bases_agg * w2 + people_agg * w3 + max(0.0, return_agg) * w4

        min_bases = getattr(self.cfg, "eval_min_bases", 5)
        min_return = getattr(self.cfg, "eval_min_return", 0.0)
        thresholds_met = (bases_agg >= min_bases) and (return_agg >= min_return)

        self._log(
            f"[Eval @ {total_done:,}] days={days_agg:.1f} "
            f"people={people_agg:.1f} bases={bases_agg:.1f} "
            f"return={return_agg:.1f} score={score:.2f} "
            f"thresholds={'PASS' if thresholds_met else 'FAIL'}"
        )

        if thresholds_met and (self.best_score is None or score > self.best_score):
            self.best_score = score
            self.best_eval_days = days_agg
            best_path = save_dir / "best_model.pt"
            self.em.ppo.save(str(best_path))

            norm_path = save_dir / "normalization.json"
            if norm_path.exists():
                import shutil
                shutil.copy2(norm_path, str(best_path).replace(".pt", ".norm.json"))

            meta = {
                "best_score": score,
                "best_days": days_agg,
                "best_bases": bases_agg,
                "best_people": people_agg,
                "best_return": return_agg,
                "score_weights": list(getattr(self.cfg, "eval_score_weights", (0.4, 3.0, 0.2, 0.0001))),
                "min_bases": min_bases,
                "min_return": min_return,
                "total_timesteps": total_done,
                "episodes": len(all_days),
                "curriculum_stage_at_best": self._curriculum_stage,
            }
            meta_path = save_dir / "best_model.meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            self._log(f"[Best] Saved best_model.pt (score={score:.2f}, days={days_agg:.1f}, bases={bases_agg:.1f})")

        return {
            "days": days_agg,
            "bases": bases_agg,
            "people": people_agg,
            "avg_return": return_agg,
            "score": score,
            "saved": thresholds_met and (self.best_score == score),
        }

    def train(self, total_timesteps: Optional[int] = None) -> TrainMetrics:
        total = total_timesteps or self.cfg.total_timesteps
        n_envs = self.em.n_envs
        steps_per_rollout = self.cfg.n_steps * n_envs

        self._log(f"[Trainer] Starting: total={total:,} steps, n_envs={n_envs}, "
                  f"n_steps={self.cfg.n_steps}, batch={self.cfg.batch_size}, "
                  f"epochs={self.cfg.n_epochs}, device={self.device}")
        self._log(f"[Trainer] AMP={self.cfg.use_amp} ({self.cfg.amp_dtype}), "
                  f"compile={self.cfg.torch_compile}")
        self._log(f"[Trainer] curriculum_stage={self._curriculum_stage}, "
                  f"schedule={self.cfg.curriculum_schedule}")

        obs = self.em.reset()
        t_start = time.perf_counter()
        total_done = 0
        rollout_idx = 0
        # Save every `save_every` rollouts (robust to any save_freq/steps_per_rollout
        # combination; the old `total_done % save_freq < steps_per_rollout` condition
        # almost never fired for the default hyperparameters).
        save_every = (
            max(1, int(round(self.cfg.save_freq / steps_per_rollout)))
            if self.cfg.save_freq > 0
            else 0
        )
        eval_every = (
            max(1, int(round(self.cfg.eval_freq / steps_per_rollout)))
            if self.cfg.eval_freq > 0
            else 0
        )

        while total_done < total and not self._stop:
            t_rollout_start = time.perf_counter()
            rollout = self._collect_rollout(obs)
            rollout_time = time.perf_counter() - t_rollout_start

            # Update loop detector after each step
            if self.loop_detector:
                n_envs = self.em.n_envs
                action_data = []
                
                for env_idx in range(n_envs):
                    # Get action from buffer (simplified tracking)
                    pos = self.em.buffer.pos - 1
                    start = pos * n_envs + env_idx
                    end = min(start + self.cfg.n_steps, self.em.buffer.pos * n_envs)
                    
                    if self.em.buffer.actions.dim(0) > 0:
                        actions = self.em.buffer.actions.cpu().numpy()[:, start:end]
                        for t in range(actions.shape[1]):
                            action_name = self._action_names.get(int(actions[t]), "UNKNOWN")
                            action_data.append({
                                "env_idx": env_idx,
                                "action": action_name,
                                "step": total_done + t
                            })
                
                # Update loop detector with batch of actions
                if action_data:
                    alerts = self.loop_detector.update_batch(action_data)
                    
                    # Get curriculum progress
                    curriculum_stage_active = self._curriculum_stage
                    
                    # Detect if any alert is active (current step loops)
                    loop_detected = len(alerts) > 0
                    loop_action_name = None
                    envs_with_loops = len(alerts)
                    
                    if alerts:
                        # Find the most recent loop action
                        loop_action_names = set()
                        for env_idx, action_name in alerts.items():
                            if action_name:
                                loop_action_names.add(action_name)
                        
                        if loop_action_names:
                            loop_action_name = list(loop_action_names)[0]
                    
                    self._log(
                        f"[LoopDetector] {envs_with_loops}/{n_envs} envs in loops, "
                        f"action={loop_action_name}"
                    )

            if self._stop:
                break

            stats = self._update_ppo(rollout)

            total_done += steps_per_rollout
            rollout_idx += 1
            elapsed = time.perf_counter() - t_start
            fps = total_done / max(elapsed, 1e-10)

            self.metrics.total_timesteps = total_done
            self.metrics.fps = fps
            self.metrics.policy_loss = stats.get("policy_loss", 0.0)
            self.metrics.value_loss = stats.get("value_loss", 0.0)
            self.metrics.entropy = stats.get("entropy", 0.0)
            self.metrics.approx_kl = stats.get("approx_kl", 0.0)
            self.metrics.learning_rate = stats.get("learning_rate", 0.0)
            self.metrics.gpu_mem_mb = stats.get("gpu_mem_mb", 0.0)
            self.metrics.wall_time_s = elapsed
            self.metrics.n_episodes = len(self._ep_returns)
            self.metrics.best_reward = self.best_reward

            # Calculate top actions from rollout
            top_actions = {}
            if rollout and "action_counts" in rollout:
                action_names = self._action_names[:5]
                action_counts = rollout["action_counts"]
                total = sum(action_counts)
                for i, (name, count) in enumerate(zip(action_names, action_counts)):
                    pct = round(count / total * 100, 2) if total > 0 else 0.0
                    top_actions[name] = pct

            # Loop detection stats
            loop_detected = False
            loop_action_name = None
            envs_with_loops = 0
            curriculum_stage_active = self._curriculum_stage
            curriculum_next_at_step = None
            
            if self.loop_detector:
                stats = self.loop_detector.get_stats()
                envs_with_loops = stats["envs_with_loops"]
                
                # Get loop action name from alerts (need to store it in detector)
                loop_action_name = self._get_current_loop_action(stats)
                
                if envs_with_loops > 0:
                    loop_detected = True

            self._log(
                f"[Step {total_done:,}/{total:,}] "
                f"FPS={fps:,.0f} | "
                f"episodes={self.metrics.n_episodes} "
                f"best_score={self.best_score or 0:.1f} "
                f"best_reward={self.best_reward:.2f} | "
                f"p_loss={stats.get('policy_loss', 0):.4f} "
                f"v_loss={stats.get('value_loss', 0):.4f} "
                f"ent={stats.get('entropy', 0):.4f} "
                f"KL={stats.get('approx_kl', 0):.5f} | "
                f"GPU_mem={stats.get('gpu_mem_mb', 0):.0f}MB "
                f"loops={envs_with_loops}"
            )

            if self.progress_callback:
                try:
                    # Prepare curriculum progress for UI
                    curriculum_stage_active = self._curriculum_stage
                    
                    # Get next step threshold from schedule
                    schedule = getattr(self.cfg, "curriculum_schedule", None)
                    curriculum_next_at_step = None
                    if schedule:
                        for threshold, stage in schedule:
                            if total_done >= threshold and stage > curriculum_stage_active:
                                curriculum_next_at_step = threshold
                    
                    # Calculate progress to next curriculum transition
                    progress_percent = 0.0
                    if curriculum_next_at_step:
                        remaining_steps = curriculum_next_at_step - total_done
                        steps_in_current_stage = self._get_steps_in_curriculum_stage(
                            total_done, curriculum_next_at_step
                        )
                        if steps_in_current_stage > 0:
                            progress_percent = 1.0 - (remaining_steps / steps_in_current_stage)
                    
                    # Format available actions
                    available_actions = ""
                    if schedule and self._curriculum_stage < len(schedule):
                        threshold, _ = schedule[self._curriculum_stage]
                        next_threshold = schedule[min(self._curriculum_stage + 1, len(schedule) - 1)][0]
                        available_actions = f"{self.em.get_allowed_buildings_for_stage(self._curriculum_stage)} " \
                                         f"(→ {self.em.get_allowed_buildings_for_stage(min(self._curriculum_stage + 1, len(schedule) - 1))})"
                    
                    # Prepare loop detection stats for UI
                    loop_stats = {}
                    if self.loop_detector:
                        loop_stats = self.loop_detector.get_stats()
                        
                        # Add action name to loop stats
                        if loop_action_name := self._get_current_loop_action(loop_stats):
                            loop_stats["current_loop_action"] = loop_action_name
                    
                    # Create extended metrics for UI
                    ui_metrics = {
                        "done": total_done,
                        "total": total,
                        "fps": fps,
                        "best_reward": self.best_reward,
                        "episodes": len(self._ep_returns),
                        "policy_loss": stats.get("policy_loss", 0.0),
                        "value_loss": stats.get("value_loss", 0.0),
                        "entropy": stats.get("entropy", 0.0),
                        "kl": stats.get("approx_kl", 0.0),
                        "top_actions": top_actions,
                        "loop_detected": loop_detected,
                        "loop_action_name": loop_action_name,
                        "envs_with_loops": envs_with_loops,
                        "curriculum_stage_active": curriculum_stage_active,
                        "curriculum_next_at_step": curriculum_next_at_step,
                        "curriculum_progress_percent": progress_percent,
                        "available_actions": available_actions,
                    }
                    
                    # Add loop stats if available
                    if loop_stats:
                        ui_metrics.update(loop_stats)

                    # Call original callback with base metrics
                    self.progress_callback(self.metrics)
                    
                    # Optionally send extended data (if callback can handle it)
                except Exception:
                    pass

            save_dir = Path(self.cfg.model_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            if save_every > 0 and rollout_idx % save_every == 0:
                ckpt_path = save_dir / f"checkpoint_{total_done}_steps.pt"
                self.em.ppo.save(str(ckpt_path))
                norm_path = str(ckpt_path).replace(".pt", ".norm.json")
                self.em.env.venv.save_normalization(norm_path)
                self._log(f"[Save] {ckpt_path}")
                self._log(f"[Save] {norm_path}")

            # Curriculum stage switching
            schedule = getattr(self.cfg, "curriculum_schedule", None)
            if schedule:
                for step_threshold, stage in schedule:
                    if total_done >= step_threshold and self._curriculum_stage < stage:
                        self.em.set_curriculum_stage(stage)
                        self._curriculum_stage = stage
                        self._log(f"[Curriculum] Stage -> {stage} at step {total_done:,}")
                        break

            # Run eval every eval_every rollouts (same pattern as save_every)
            if eval_every > 0 and rollout_idx % eval_every == 0:
                eval_result = self._eval(total_done)
                if self.progress_callback:
                    try:
                        self.metrics.eval_days = eval_result["days"]
                        self.metrics.eval_people = eval_result["people"]
                        self.metrics.eval_bases = eval_result["bases"]
                        self.metrics.eval_score = eval_result.get("score", 0.0)
                        self.metrics.best_eval_days = self.best_eval_days or 0.0
                        self.metrics.best_score = self.best_score or 0.0
                        self.progress_callback(self.metrics)
                    except Exception:
                        pass

        elapsed = time.perf_counter() - t_start
        final_fps = total_done / max(elapsed, 1e-10)

        save_dir = Path(self.cfg.model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        final_path = save_dir / "final_model.pt"
        self.em.ppo.save(str(final_path))
        norm_path = str(final_path).replace(".pt", ".norm.json")
        self.em.env.venv.save_normalization(norm_path)
        self._log(f"[Save] Final model: {final_path}")
        self._log(f"[Save] Normalization: {norm_path}")

        self.metrics.total_timesteps = total_done
        self.metrics.fps = final_fps
        self.metrics.wall_time_s = elapsed
        self.metrics.best_reward = self.best_reward

        self._log(f"[Done] {total_done:,} steps in {elapsed:.1f}s "
                  f"({final_fps:,.0f} FPS), best_reward={self.best_reward:.2f}, "
                  f"episodes={self.metrics.n_episodes}")

        return self.metrics

    def _get_steps_in_curriculum_stage(self, current_step: int, next_threshold: int) -> int:
        """Calculate steps available in current curriculum stage."""
        if next_threshold <= current_step:
            return 0
        schedule = getattr(self.cfg, "curriculum_schedule", None)
        if not schedule or len(schedule) < 2:
            return max(next_threshold - current_step, 10000)
        
        # Find stage boundary before current step
        prev_threshold = 0
        for threshold, _ in schedule:
            if threshold > current_step:
                break
            prev_threshold = threshold
        
        stage_steps = next_threshold - prev_threshold
        return min(stage_steps, 100000)  # Cap at 100k

    def close(self):
        self.em.close()
