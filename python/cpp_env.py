import os
import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple
import colony_cpp
from pathlib import Path

# Project root is parent of this file's directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class CppColonyEnv(gym.Env):
    """
    Gymnasium wrapper for C++ ColonyEnvCpp.
    Observation space: 207-dim float32 vector
    Action space: Discrete(45) - 0=DAY, 1=WEEK, 2-33=BUILD, 34-44=MANAGER
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        map_size: int = 280,
        curriculum_stage: int = 0,
        unlock_ids: Optional[str] = None,
        disable_net_worth: bool = False,
        disable_daily_income: bool = False,
        reward_config: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        
        # Load static data
        self.base_data = colony_cpp.load_base_data(str(PROJECT_ROOT / "configs" / "bases.json"))
        self.events_data = colony_cpp.load_events(str(PROJECT_ROOT / "configs" / "events.json"))
        
        # Reward configuration
        rc = colony_cpp.RewardConfig()
        _REWARD_KEYS = ("build_bonus", "chain_bonus", "chain_daily",
                        "novelty", "daily_income", "sale_bonus", "tax_bonus",
                        "survival_bonus", "game_over_penalty")
        if reward_config:
            for k in _REWARD_KEYS:
                if k in reward_config:
                    setattr(rc, k, reward_config[k])
        rc.disable_net_worth = disable_net_worth
        rc.disable_daily_income = disable_daily_income
        
        if os.environ.get("COLONY_DEBUG", ""):
            print("=" * 60, flush=True)
            print("[DEBUG CppColonyEnv] C++ RewardConfig after setup:", flush=True)
            for k in _REWARD_KEYS:
                print(f"  rc.{k:25s} = {getattr(rc, k)}", flush=True)
            print(f"  rc.disable_net_worth      = {rc.disable_net_worth}", flush=True)
            print(f"  rc.disable_daily_income   = {rc.disable_daily_income}", flush=True)
            print(f"  reward_config dict keys   = {list(reward_config.keys()) if reward_config else 'None'}", flush=True)
            print("=" * 60, flush=True)
        
        # Parse unlock_ids
        unlock_list = []
        if unlock_ids:
            unlock_list = [s.strip() for s in unlock_ids.split(",") if s.strip()]
        
        # Create C++ environment
        self.cpp_env = colony_cpp.ColonyEnvCpp(
            self.base_data,
            self.events_data,
            seed=0,  # will be set in reset
            map_size=map_size,
            curriculum_stage=curriculum_stage,
            unlock_ids=unlock_list,
            reward=rc,
        )
        
        # Spaces (use clipped observation space as SB3 expects)
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.cpp_env.obs_size(),),
            dtype=np.float32
        )
        # Fix: VecNormalize expects clip_obs to be set, default None causes TypeError
        # Set clip_obs=10.0 which matches the VecNormalize clip_obs parameter in train.py
        self.clip_obs = 10.0
        self.action_space = gym.spaces.Discrete(self.cpp_env.n_actions())
        
        # Action name mapping (for debugging)
        self._action_names = ["DAY", "WEEK"] + self.cpp_env.build_ids() + [
            "IMPROVE_LAND", "REPAIR", "REPAIR_ALL", "DEMOLISH", "PRESERVE",
            "UNPRESERVE", "SELL_SURPLUS", "BUY_FOOD", "TAKE_LOAN", "REPAY_LOAN", "PAY_TAX"
        ]
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        s = seed if seed is not None else np.random.randint(1 << 31)
        self.cpp_env.reset(s)
        obs = np.array(self.cpp_env.obs(), dtype=np.float32)
        return obs, {"seed": s, "days": 0}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        result = self.cpp_env.step(int(action))
        obs = np.array(result["obs"], dtype=np.float32)
        # Clip observations as VecNormalize expects; prevents TypeError when clip_obs=None
        obs = np.clip(obs, -self.clip_obs, self.clip_obs).astype(np.float32)
        reward = float(result["reward"])
        terminated = bool(result["terminated"])
        truncated = bool(result["truncated"])
        info = {
            "days": int(result["days"]),
            "people": int(result["people"]),
            "money": int(result["money"]),
            "bases": int(result["bases"]),
            "tax_due_days": int(result["tax_due_days"]),
            "ep_return": float(result["ep_return"]),
            "steps": int(result["steps"]),
            "action_name": self._action_names[action] if action < len(self._action_names) else str(action),
        }
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            # Return a simple representation - could be extended to show actual map
            return np.zeros((400, 400, 3), dtype=np.uint8)
        return None
    
    def close(self):
        pass
    
    @property
    def unwrapped(self):
        return self

def make_env(
    map_size: int = 280,
    curriculum_stage: int = 0,
    unlock_ids: Optional[str] = None,
    disable_net_worth: bool = False,
    disable_daily_income: bool = False,
    reward_config: Optional[Dict[str, float]] = None,
    seed: int = 0,
) -> gym.Env:
    """Factory function for SubprocVecEnv compatibility."""
    def _init():
        env = CppColonyEnv(
            map_size=map_size,
            curriculum_stage=curriculum_stage,
            unlock_ids=unlock_ids,
            disable_net_worth=disable_net_worth,
            disable_daily_income=disable_daily_income,
            reward_config=reward_config,
        )
        env.reset(seed=seed)
        return env
    return _init