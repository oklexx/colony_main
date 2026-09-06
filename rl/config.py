from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RewardConfig:
    # --- base bonuses ---
    build_bonus: float = 1.0
    chain_bonus: float = 1.0
    chain_daily: float = 0.5
    novelty: float = 15.0
    daily_income: float = 0.3
    sale_bonus: float = 0.2
    tax_daily_bonus: float = 0.3
    survival_bonus: float = 0.1
    game_over_penalty: float = 10.0
    diversity_bonus: float = 8.0
    # --- penalties for errors / special actions ---
    error_penalty: float = -1.0
    preserve_penalty: float = 0.0
    demolish_penalty: float = -3.0
    manual_tax_penalty: float = -0.5
    build_cost_penalty: float = 0.0001
    idle_build_penalty: float = -10.0
    idle_build_threshold_days: int = 3
    survival_coeff: float = 0.001
    # --- milestone bonuses ---
    milestone_base_bonus: float = 10.0
    milestone_people_bonus: float = 2.0
    milestone_day_bonus: float = 2.0
    milestone_year_bonus: float = 5.0
    # --- spatial bonuses ---
    proximity_bonus: float = 0.5
    # --- clip raw reward ---
    clip_reward_min: float = -10.0
    clip_reward_max: float = 10.0
    # --- flags ---
    disable_net_worth: bool = False
    disable_daily_income: bool = False
    disable_provider_bonus: bool = False

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
            "diversity_bonus": self.diversity_bonus,
            "error_penalty": self.error_penalty,
            "preserve_penalty": self.preserve_penalty,
            "demolish_penalty": self.demolish_penalty,
            "manual_tax_penalty": self.manual_tax_penalty,
            "build_cost_penalty": self.build_cost_penalty,
            "idle_build_penalty": self.idle_build_penalty,
            "idle_build_threshold_days": self.idle_build_threshold_days,
            "survival_coeff": self.survival_coeff,
            "milestone_base_bonus": self.milestone_base_bonus,
            "milestone_people_bonus": self.milestone_people_bonus,
            "milestone_day_bonus": self.milestone_day_bonus,
            "milestone_year_bonus": self.milestone_year_bonus,
            "proximity_bonus": self.proximity_bonus,
            "clip_reward_min": self.clip_reward_min,
            "clip_reward_max": self.clip_reward_max,
            "disable_net_worth": self.disable_net_worth,
            "disable_daily_income": self.disable_daily_income,
            "disable_provider_bonus": self.disable_provider_bonus,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardConfig":
        return cls(
            build_bonus=d.get("build_bonus", 1.0),
            chain_bonus=d.get("chain_bonus", 1.0),
            chain_daily=d.get("chain_daily", 0.5),
            novelty=d.get("novelty", 15.0),
            daily_income=d.get("daily_income", 0.3),
            sale_bonus=d.get("sale_bonus", 0.2),
            tax_daily_bonus=d.get("tax_daily_bonus", 0.3),
            survival_bonus=d.get("survival_bonus", 0.1),
            game_over_penalty=d.get("game_over_penalty", 10.0),
            diversity_bonus=d.get("diversity_bonus", 8.0),
            error_penalty=d.get("error_penalty", -1.0),
            preserve_penalty=d.get("preserve_penalty", 0.0),
            demolish_penalty=d.get("demolish_penalty", -3.0),
            manual_tax_penalty=d.get("manual_tax_penalty", -0.5),
            build_cost_penalty=d.get("build_cost_penalty", 0.0001),
            idle_build_penalty=d.get("idle_build_penalty", -10.0),
            idle_build_threshold_days=int(d.get("idle_build_threshold_days", 3)),
            survival_coeff=d.get("survival_coeff", 0.001),
            milestone_base_bonus=d.get("milestone_base_bonus", 10.0),
            milestone_people_bonus=d.get("milestone_people_bonus", 2.0),
            milestone_day_bonus=d.get("milestone_day_bonus", 2.0),
            milestone_year_bonus=d.get("milestone_year_bonus", 5.0),
            proximity_bonus=d.get("proximity_bonus", 0.5),
            clip_reward_min=d.get("clip_reward_min", -10.0),
            clip_reward_max=d.get("clip_reward_max", 10.0),
            disable_net_worth=d.get("disable_net_worth", False),
            disable_daily_income=d.get("disable_daily_income", False),
            disable_provider_bonus=d.get("disable_provider_bonus", False),
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
    ent_coef: float = 0.05  # Increased to prevent policy collapse / action loops
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
    early_stopping_patience: int = 0

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
        if self.obs_mode not in ("flat", "minimap", "hybrid"):
            raise ValueError(f"obs_mode must be 'flat', 'minimap', or 'hybrid', got {self.obs_mode}")

        import torch
        if self.torch_compile and not torch.cuda.is_available():
            import warnings
            warnings.warn(
                "torch_compile=True but CUDA is not available. "
                "torch.compile will be disabled at runtime.",
                UserWarning,
                stacklevel=2,
            )

        if self.use_amp and self.amp_dtype == "bfloat16" and torch.cuda.is_available():
            if not torch.cuda.is_bf16_supported():
                import warnings
                warnings.warn(
                    "amp_dtype='bfloat16' but GPU does not support BF16. "
                    "AMP will silently fall back to float32.",
                    UserWarning,
                    stacklevel=2,
                )

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k in (
            "map_size", "n_envs", "seed", "curriculum_stage", "unlock_ids",
            "learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
            "net_arch", "obs_mode", "minimap_radius",
            "total_timesteps", "save_freq", "eval_freq", "eval_episodes",
            "eval_score_weights", "eval_min_bases", "eval_min_return",
            "eval_use_median", "eval_seeds", "early_stopping_patience", "curriculum_schedule",
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
            ent_coef=d.get("ent_coef", 0.05),
            vf_coef=d.get("vf_coef", 0.5),
            max_grad_norm=d.get("max_grad_norm", 0.5),
            net_arch=d.get("net_arch", [256, 256]),
            obs_mode=d.get("obs_mode", "flat"),
            minimap_radius=d.get("minimap_radius", 14),
            total_timesteps=d.get("total_timesteps", 1_000_000),
            save_freq=d.get("save_freq", 500_000),
            eval_freq=d.get("eval_freq", 100_000),
            eval_episodes=d.get("eval_episodes", 20),
            eval_score_weights=tuple(float(x) for x in d.get("eval_score_weights", (0.4, 3.0, 0.2, 0.0001))),
            eval_min_bases=d.get("eval_min_bases", 5),
            eval_min_return=d.get("eval_min_return", 0.0),
            eval_use_median=d.get("eval_use_median", True),
            eval_seeds=d.get("eval_seeds", [42]),
            early_stopping_patience=d.get("early_stopping_patience", 0),
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
