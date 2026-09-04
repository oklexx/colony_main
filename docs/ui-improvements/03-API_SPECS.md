# 📐 03-API_SPECS.md — Спецификации API и протоколы

## 🔄 Protocol Extensions (train_ui/protocol.py)

### ProgressMsg — Расширение существующего класса

```python
@dataclass
class ProgressMsg:
    # Existing fields...
    done: int
    total: int
    fps: float = 0.0
    best_reward: float = 0.0
    episodes: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl: float = 0.0
    
    # NEW FIELDS (P1 - Critical)
    top_actions: Dict[str, float] = field(default_factory=dict)
        # Example: {"PRESERVE": 0.48, "BUILD_HOUSE": 0.12, "DAY": 0.30}
        # Only top-5 actions with non-zero count, sorted by frequency
    
    loop_detected: bool = False
        # True if any environment is in action loop
    
    loop_action_name: Optional[str] = None
        # Name of the action that's being repeated (e.g., "PRESERVE")
    
    envs_with_loops: int = 0
        # Count of environments currently in loops out of total n_envs
    
    curriculum_stage_active: int = 0
        # Current curriculum stage ID
        
    curriculum_next_at_step: Optional[int] = None
        # Step number when next curriculum transition will occur (None if unlimited)
    
    # NEW FIELDS (P2 - Metrics Display)
    return_normalization_enabled: bool = False
        # Status of return normalization (ON/OFF indicator)
    
    avg_return: float = 0.0
        # Average return over last N episodes (for ReturnStatisticsWidget)
    
    median_return: float = 0.0
        # Median return over last N episodes
        
    max_return: float = 0.0
        # Maximum return observed
        
    min_return: float = 0.0
        # Minimum return observed
```

### CommandMsg — Новый класс для команд UI → worker

```python
@dataclass
class CommandMsg:
    cmd: str
    type: MsgType = field(default=MsgType.COMMAND, init=False)
    payload: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "cmd": self.cmd,
            **(self.payload or {}),
        }


# Supported commands (P1 - Critical)

CMD_BOOST_ENTROPY = "boost_entropy"
    payload: {"factor": float}  # e.g., {"factor": 2.0}
    
    Example: boost current ent_coef by ×2.0 (max cap at 0.05)


CMD_RESET_CURRICULUM = "reset_curriculum"
    payload: {}  # Empty
    Example: Reset curriculum_stage to 0 immediately


CMD_PAUSE_TRAINING = "pause_training"
    payload: {"save_checkpoint": bool}  # Optional, default True
    
    Example: Pause training and optionally save checkpoint


CMD_RESUME_TRAINING = "resume_training"
    payload: {}  # Empty


CMD_STOP_TRAINING = "stop_training"
    payload: {"final_save": bool}  # Always True by default
```

### Message Encoding/Decoding

```python
def encode(msg: Msg) -> str:
    """Serialize message to single JSON line."""
    d = msg.to_dict()
    return json.dumps(d, ensure_ascii=False, allow_nan=False)


def decode(line: str) -> Msg:
    """Parse JSON line into typed message."""
    d = json.loads(line.strip())
    msg_type = MsgType(d["type"])
    
    # Union/Dispatch logic...
    if msg_type == MsgType.PROGRESS:
        return ProgressMsg(...)
    elif msg_type == MsgType.COMMAND:
        return CommandMsg(cmd=d["cmd"], payload=d.get("payload"))
```

## 🔍 LoopDetector API (rl/loop_detector.py)

### Класс LoopDetector

**Конструктор:**
```python
class LoopDetector:
    def __init__(self, threshold_config: Optional[Dict[str, int]] = None):
        """
        Args:
            threshold_config: Dict mapping action_name -> max_consecutive_count
                
            Default thresholds:
            - PRESERVE: 3
            - BUILD_HOUSE/BUILD_SAWMILL/etc.: 5
            - DEMOLISH: 4
            - DAY/WEEK: 15, 20 (lenient - strategic waiting)
        """
```

**Метод update_batch:**
```python
def update_batch(
    self,
    env_indices: List[int],           # [0, 1, ..., n_envs-1]
    actions: torch.Tensor,            # Tensor shape: [n_envs], dtype=int64
    action_names: List[str]           # ["PRESERVE", "BUILD_HOUSE", ...]
) -> Dict[int, Optional[str]]:
    """
    Update state for multiple environments at once.
    
    Args:
        env_indices: List of environment indices to process
        actions: Action tensor (one action per env)
        action_names: Mapping from action_idx -> action_name
    
    Returns:
        Dict mapping env_idx -> action_name if loop detected, else None
    
    Example usage in AsyncTrainer._collect_rollout():
        alerts = self.loop_detector.update_batch(
            env_indices=list(range(self.em.n_envs)),
            actions=actions_tensor,
            action_names=self.em._action_names
        )
        
        # Check for any detected loops
        if any(a is not None for a in alerts.values()):
            loop_action = next(a for a in alerts.values() if a)
            # Pass to progress_callback
            
        # Calculate return statistics for UI
        avg_return = np.mean(self._ep_returns[-50:]) if len(self._ep_returns) > 0 else 0.0
        median_return = np.median(self._ep_returns[-50:]) if len(self._ep_returns) > 0 else 0.0
        max_return = np.max(self._ep_returns[-50:]) if len(self._ep_returns) > 0 else 0.0
        min_return = np.min(self._ep_returns[-50:]) if len(self._ep_returns) > 0 else 0.0
        
        # Pass return stats to progress_callback (aggregated, not raw returns)
```

**Метод get_stats:**
```python
def get_stats(self) -> Dict[str, Any]:
    """
    Get aggregated statistics across all environments.
    
    Returns:
        {
            "total_envs": int,              # Total number of tracked envs
            "envs_with_loops": int,         # Count currently in loops
            "max_loop_count": int,          # Maximum consecutive count seen
            "avg_consecutive_actions": float,  # Average over all envs
            "loop_rate_percent": float      # (envs_with_loops / total_envs) * 100
        }
    """
```

**Метод detect_and_boost_entropy:**
```python
def detect_and_boost_entropy(
    self,
    em: EnvManager,     # For getting n_envs
    ppo: PPO,           # To modify ent_coef attribute
    current_step: int,  # Global training step
    threshold_ratio: float = 0.3  # Default: boost if >30% envs in loops
) -> bool:
    """
    Auto-correct strategy by boosting entropy when loops are detected.
    
    Algorithm:
    1. Calculate loop_rate = envs_with_loops / total_envs
    2. If loop_rate > threshold_ratio AND avg_entropy < 2.5:
       - ent_coef = min(ent_coef * 1.3, 0.05)
       - Log the change with reason
    
    Returns:
        True if boost was applied, False otherwise
    """
```

## 🎓 EnvManager Extensions (rl/env_manager.py)

### Метод get_curriculum_progress

```python
def get_curriculum_progress(self, current_step: int = None) -> Dict[str, Any]:
    """
    Get curriculum progress information for UI display.
    
    Args:
        current_step: Optional manual step override (defaults to actual step)
    
    Returns:
        {
            "stage": int,                    # Current stage ID
            "total_stages": int,             # Total number of stages in schedule
            "progress_percent": float,       # 0-100% progress to next stage transition
            "next_stage_step": Optional[int], # Step number of next transition (None if last stage)
            "available_actions": str         # Comma-separated list of first 5 allowed buildings
        }
    
    Example output:
        {
            "stage": 0,
            "total_stages": 3,
            "progress_percent": 45.5,
            "next_stage_step": 200_000,
            "available_actions": "House, Farm, Road"
        }
    """
```

### Метод get_allowed_buildings_for_stage

```python
def get_allowed_buildings_for_stage(self, stage_id: int) -> List[str]:
    """
    Get list of building IDs allowed in specified curriculum stage.
    
    Args:
        stage_id: Curriculum stage identifier
    
    Returns:
        List of building names (strings). Default fallback if stage not found.
    
    Example schedule:
        0 → ["House", "Farm", "Road"]
        1 → ["SmallHouse", "Garden", "Sawmill", "WaterChannel", "Coalmine"]
        2 → ["Ironmine", "Brewery", "Bakery", "GeneralStore", "Church"]
    """
```

## 🎨 UI Widget API (train_ui/*_widget.py)

### KLStatusWidget

```python
class KLStatusWidget(QWidget):
    def __init__(self, parent=None):
        # Internal state:
        # - current_kl: float
        # - ent_coef_display: str
        
    def update(self, kl: float, ent_coef: float):
        """
        Update status indicator based on KL divergence value.
        
        Status levels:
        - LOW ⚠️ (orange):      kl < 0.01     → Agent is too deterministic
        - OK ✓ (lime):          0.01 <= kl < 0.025  → Acceptable
        - OPTIMAL ⭐ (turquoise): 0.025 <= kl < 0.08   → Perfect range
        - HIGH ⚠️ (red):        kl >= 0.08    → Too much variance
        
        Updates:
        - Progress bar shows normalized position in 0.01-0.08 range
        - Text label shows current status and target info
        """
```

### CurriculumProgressWidget

```python
class CurriculumProgressWidget(QWidget):
    def __init__(self, parent=None):
        # Internal state:
        # - current_stage: int
        # - progress_percent: float (0-100)
        # - next_stages: List[Dict]
        # - available_actions: str
        
    def update(self, 
               current_stage: int,
               total_stages: int,
               progress_percent: float,
               next_stages: List[Dict[str, Any]],  # [{"stage": 1, "step": 200000}, ...]
               available_actions: str):
        """
        Update curriculum progress display.
        
        Args:
            current_stage: Current curriculum stage ID
            total_stages: Total number of stages in schedule
            progress_percent: 0-100% progress to next transition
            next_stages: List of upcoming transitions (max 3 shown)
            available_actions: Comma-separated string of allowed buildings
        
        Visual updates:
        - Stage badge shows "Stage: 0/3" with color coding
        - Progress bar fills from 0-100%
        - Upcoming stages listed below progress bar
        - Available actions shown at bottom (first 5)
        """
```

### ActionLoopWidget

```python
class ActionLoopWidget(QWidget):
    def __init__(self, parent=None):
        # Internal state:
        # - active_loop: bool
        # - loop_action_name: Optional[str]
        # - history: List[Dict]  # Last N updates for statistics
        
    def set_loop_status(self, 
                        active: bool,
                        action_name: Optional[str] = None,
                        count: int = 0,
                        threshold: int = 5,
                        env_idx: int = -1):
        """
        Update active loop detection status.
        
        If active=True and action_name is provided:
        - Show warning alert with red/orange color
        - Display environment index (if not -1)
        - Show consecutive count vs threshold
        - Mark as CRITICAL if count > 2*threshold
        
        If active=False:
        - Hide alert
        - Reset action name display
        """
    
    def update_history_stats(self, stats: Dict[str, Any]):
        """
        Update aggregated history statistics.
        
        Stats format from UI aggregation:
        {
            "total_loops": int,              # Count of loop detections in window
            "max_consecutive": int,          # Maximum count seen
            "avg_consecutive": float,        # Average across all envs
            "loop_rate_percent": float       # Percentage of time in loops
        }
        
        Updates grid display with 4 rows:
        - Envs with loops: X
        - Max consecutive: X
        - Avg consecutive: X.X
        - Loop rate: X.X%
        """
```

## 📊 Data Flow Contract

### Progress Callback Signature

```python
def progress_callback(
    metrics: Dict[str, Any]
) -> None:
    """
    Called periodically during training (every 1000 steps).
    
    Expected keys in metrics dict:
    - total_timesteps: int
    - fps: float
    - best_reward: float (or -inf)
    - episodes: int
    - policy_loss: float
    - value_loss: float
    - entropy: float
    - approx_kl: float
    
    Optional keys (P1 features):
    - top_actions: Dict[str, float]  # Top-5 action distribution
    - loop_detected: bool
    - loop_action_name: str
    - envs_with_loops: int
    - curriculum_stage_active: int
    - curriculum_next_at_step: Optional[int]
    """
```

### Command Handler Signature

```python
def handle_command(
    cmd: str,
    payload: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Process command from UI thread. Should not block.
    
    Supported commands (P1):
    - "boost_entropy": {"factor": 2.0} → Returns {"success": True, "new_coef": X.XX}
    - "reset_curriculum": {} → Returns {"success": True, "stage": 0}
    - "pause_training": {"save_checkpoint": bool} → Returns {"paused": True}
    - "resume_training": {} → Returns {"resumed": True}
    - "stop_training": {"final_save": bool} → Returns {"stopped": True}
    
    Returns:
        Dict with success status and relevant info
    """
```

## 🔧 Thread Safety Notes

### Critical sections requiring synchronization:
1. **LoopDetector._state** - Use threading.Lock when accessing from multiple threads
2. **EnvManager.current_stage** - Protect during transitions
3. **UI widget state updates** - All UI modifications happen in main thread only

### Recommended patterns:
```python
# In LoopDetector (if accessed from worker thread):
self._lock = threading.Lock()

def update(self, ...):
    with self._lock:
        # Critical section
        pass

# In command handler (UI thread calls worker via queue):
# No locks needed - communication is process-bound, not thread-bound
```
