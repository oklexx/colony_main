from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RewardConfig:
    build_bonus: float = 5.0
    chain_bonus: float = 0.5
    chain_daily: float = 2.0
    novelty: float = 20.0
    daily_income: float = 0.1
    sale_bonus: float = 0.1
    tax_daily_bonus: float = 0.77
    survival_bonus: float = 0.0
    game_over_penalty: float = 20.0
    disable_net_worth: bool = False
    disable_daily_income: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "build_bonus": self.build_bonus,
            "chain_bonus": self.chain_bonus,
            "chain_daily": self.chain_daily,
            "novelty": self.novelty,
            "daily_income": self.daily_income,
            "sale_bonus": self.sale_bonus,
            "tax_daily_bonus": self.tax_daily_bonus,
            "survival_bonus": self.survival_bonus,
            "game_over_penalty": self.game_over_penalty,
            "disable_net_worth": self.disable_net_worth,
            "disable_daily_income": self.disable_daily_income,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardConfig":
        return cls(
            build_bonus=d.get("build_bonus", 5.0),
            chain_bonus=d.get("chain_bonus", 0.5),
            chain_daily=d.get("chain_daily", 2.0),
            novelty=d.get("novelty", 20.0),
            daily_income=d.get("daily_income", 0.1),
            sale_bonus=d.get("sale_bonus", 0.1),
            tax_daily_bonus=d.get("tax_daily_bonus", 0.77),
            survival_bonus=d.get("survival_bonus", 0.0),
            game_over_penalty=d.get("game_over_penalty", 20.0),
            disable_net_worth=d.get("disable_net_worth", False),
            disable_daily_income=d.get("disable_daily_income", False),
        )


@dataclass
class Config:
    # Environment
    map_size: int = 280
    n_envs: int = 8
    seed: int = 42
    curriculum_stage: int = 0
    unlock_ids: str = ""
    reward: RewardConfig = field(default_factory=RewardConfig)

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 4096
    batch_size: int = 8192
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.98
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    # Observation mode: "flat" = 209-dim vector + MLP (legacy),
    # "minimap" = 2D spatial tensor (8, 2R+1, 2R+1) + CNN trunk
    obs_mode: str = "flat"
    minimap_radius: int = 14

    # Training
    total_timesteps: int = 1_000_000
    save_freq: int = 500_000
    eval_freq: int = 100_000
    eval_episodes: int = 20

    # Eval scoring (composite champion selection)
    eval_score_weights: tuple = (0.4, 3.0, 0.2, 0.0001)
    eval_min_bases: int = 5
    eval_min_return: float = 0.0
    eval_use_median: bool = True
    eval_seeds: list = field(default_factory=lambda: [42])

    # Curriculum schedule: list of (step_threshold, stage) pairs
    curriculum_schedule: list = field(default_factory=list)

    # GPU / performance
    device: str = "cuda"
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    torch_compile: bool = False
    cpp_threads: int = 0
    torch_threads: int = 0

    # Paths
    log_dir: str = ""
    model_dir: str = ""

    # Async
    async_train: bool = False
    queue_size: int = 2

    def __post_init__(self):
        if not self.log_dir:
            home = Path.home()
            self.log_dir = str(home / "colony_runs" / "logs")
        if not self.model_dir:
            home = Path.home()
            self.model_dir = str(home / "colony_runs" / "models")
        if self.amp_dtype not in ("bfloat16", "float16"):
            raise ValueError(f"amp_dtype must be bfloat16 or float16, got {self.amp_dtype}")
        if self.obs_mode not in ("flat", "minimap"):
            raise ValueError(f"obs_mode must be 'flat' or 'minimap', got {self.obs_mode}")

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k in (
            "map_size", "n_envs", "seed", "curriculum_stage", "unlock_ids",
            "learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
            "net_arch", "obs_mode", "minimap_radius",
            "total_timesteps", "save_freq", "eval_freq", "eval_episodes",
            "eval_score_weights", "eval_min_bases", "eval_min_return",
            "eval_use_median", "eval_seeds", "curriculum_schedule",
            "device", "use_amp", "amp_dtype", "torch_compile",
            "cpp_threads", "torch_threads", "async_train", "queue_size",
            "log_dir", "model_dir",
        ):
            d[k] = getattr(self, k)
        d["reward"] = self.reward.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        reward = d.pop("reward", None)
        cfg = cls(
            map_size=d.get("map_size", 280),
            n_envs=d.get("n_envs", 8),
            seed=d.get("seed", 42),
            curriculum_stage=d.get("curriculum_stage", 0),
            unlock_ids=d.get("unlock_ids", ""),
            learning_rate=d.get("learning_rate", 3e-4),
            n_steps=d.get("n_steps", 4096),
            batch_size=d.get("batch_size", 8192),
            n_epochs=d.get("n_epochs", 10),
            gamma=d.get("gamma", 0.995),
            gae_lambda=d.get("gae_lambda", 0.98),
            clip_range=d.get("clip_range", 0.2),
            ent_coef=d.get("ent_coef", 0.01),
            vf_coef=d.get("vf_coef", 0.5),
            max_grad_norm=d.get("max_grad_norm", 0.5),
            net_arch=d.get("net_arch", [256, 256]),
            obs_mode=d.get("obs_mode", "flat"),
            minimap_radius=d.get("minimap_radius", 14),
            total_timesteps=d.get("total_timesteps", 1_000_000),
            save_freq=d.get("save_freq", 500_000),
            eval_freq=d.get("eval_freq", 100_000),
            eval_episodes=d.get("eval_episodes", 20),
            eval_score_weights=tuple(d.get("eval_score_weights", (0.4, 3.0, 0.2, 0.0001))),
            eval_min_bases=d.get("eval_min_bases", 5),
            eval_min_return=d.get("eval_min_return", 0.0),
            eval_use_median=d.get("eval_use_median", True),
            eval_seeds=d.get("eval_seeds", [42]),
            curriculum_schedule=d.get("curriculum_schedule", []),
            device=d.get("device", "cuda"),
            use_amp=d.get("use_amp", True),
            amp_dtype=d.get("amp_dtype", "bfloat16"),
            torch_compile=d.get("torch_compile", False),
            cpp_threads=d.get("cpp_threads", 0),
            torch_threads=d.get("torch_threads", 0),
            async_train=d.get("async_train", False),
            queue_size=d.get("queue_size", 2),
            log_dir=d.get("log_dir", ""),
            model_dir=d.get("model_dir", ""),
        )
        if reward is not None:
            cfg.reward = RewardConfig.from_dict(reward)
        return cfg

    def load_from_file(self, path: str) -> "Config":
        p = Path(path)
        if not p.exists():
            return self
        with open(p) as f:
            data = json.load(f)
        self.__dict__.update({k: v for k, v in data.items() if k != "reward"})
        if "reward" in data:
            self.reward = RewardConfig.from_dict(data["reward"])
        return self
