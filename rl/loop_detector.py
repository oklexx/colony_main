from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any


@dataclass
class EnvState:
    """State tracking for a single environment."""
    env_idx: int
    last_action: str = ""
    action_count: int = 0
    action_sequence: List[Tuple[str, float]] = field(default_factory=list)


class LoopDetector:
    """
    Detects action loops in training environments.
    
    Tracks action sequences per environment and alerts when an action
    repeats consecutively beyond a configurable threshold.
    """

    def __init__(self, threshold_config: Optional[Dict[str, Any]] = None):
        """
        Initialize LoopDetector.
        
        Args:
            threshold_config: Dict with thresholds:
                - consecutive_threshold: Max consecutive actions before alert (default: 3)
                - history_size: Number of states to keep in history (default: 1000)
        """
        self.consecutive_threshold = 3
        
        if threshold_config:
            self.consecutive_threshold = threshold_config.get(
                "consecutive_threshold", 3
            )
        
        self.history_size = threshold_config.get("history_size", 1000) if threshold_config else 1000
        
        self._state: Dict[int, EnvState] = {}
        self._history: List[Dict[str, Any]] = []
        
        # Statistics
        self._total_loops_detected = 0
        self._total_actions_sampled = 0
        self._loop_start_times: Dict[int, float] = {}

    def _get_or_create_env_state(self, env_idx: int) -> EnvState:
        """Get existing state or create new one for environment."""
        if env_idx not in self._state:
            self._state[env_idx] = EnvState(env_idx=env_idx)
        return self._state[env_idx]

    def update_batch(self, action_data: List[Dict[str, Any]]) -> Dict[int, Optional[str]]:
        """
        Process multiple environments at once.
        
        Args:
            action_data: List of dicts with keys:
                - env_idx: Environment index
                - action: Action name or identifier
                - step: Current step number (optional)
                
        Returns:
            Dict mapping env_idx -> action_name if loop detected, else None
        """
        alerts: Dict[int, Optional[str]] = {}
        
        for data in action_data:
            env_idx = data["env_idx"]

            action = data.get("action", "")
            
            state = self._get_or_create_env_state(env_idx)
            
            # Update action count and sequence
            state.action_count += 1
            state.last_action = action
            
            # Keep last 100 actions for detection
            if len(state.action_sequence) >= 100:
                state.action_sequence.pop(0)
            
            state.action_sequence.append((action, time.time()))
            
            self._total_actions_sampled += 1
            
            # Check for loop (consecutive same action)
            if len(state.action_sequence) >= self.consecutive_threshold:
                recent_actions = state.action_sequence[-self.consecutive_threshold:]
                
                if all(a == action for a, _ in recent_actions):
                    if env_idx not in alerts:  # Only alert once per env
                        alerts[env_idx] = action
                        self._total_loops_detected += 1
                        
                        # Track loop start time
                        self._loop_start_times[env_idx] = time.time()
            
            # Add to global history (sample every 10 updates to save memory)
            if self._total_actions_sampled % 10 == 0:
                self._history.append({
                    "timestamp": time.time(),
                    "envs_with_loops": len(alerts),
                    "alerts": alerts.copy()
                })
                
                # Trim history
                while len(self._history) > self.history_size:
                    self._history.pop(0)

        return alerts

    def get_stats(self) -> Dict[str, Any]:
        """
        Calculate loop detection statistics.
        
        Returns:
            Dict with statistics including:
            - total_envs: Number of tracked environments
            - envs_with_loops: Currently detected loops
            - max_consecutive: Maximum consecutive actions seen
            - avg_consecutive: Average consecutive action count
            - loop_rate_percent: Percentage of samples in loops
        """
        if not self._state:
            return {
                "total_envs": 0,
                "envs_with_loops": 0,
                "max_consecutive": 0,
                "avg_consecutive": 0.0,
                "loop_rate_percent": 0.0,
                "consecutive_threshold": self.consecutive_threshold,
            }

        max_consec = 0
        total_consec = 0
        
        for state in self._state.values():
            max_consec = max(max_consec, state.action_count)
            total_consec += state.action_count
        
        avg_consec = total_consec / len(self._state) if self._state else 0.0
        
        # Calculate loop rate from history
        total_samples = sum(len(d.get("alerts", {})) for d in self._history)
        loop_rate = (total_samples / (len(self._history) * max(1, self.consecutive_threshold))) * 100 if self._history else 0.0
        
        return {
            "total_envs": len(self._state),
            "envs_with_loops": sum(
                1 for state in self._state.values() 
                if state.action_count >= self.consecutive_threshold
            ),
            "max_consecutive": max_consec,
            "avg_consecutive": avg_consec,
            "loop_rate_percent": loop_rate,
            "consecutive_threshold": self.consecutive_threshold,
            "total_loops_detected": self._total_loops_detected,
        }

    def clear(self):
        """Clear all state and reset statistics."""
        self._state.clear()
        self._history.clear()
        self._loop_start_times.clear()
        self._total_loops_detected = 0
        self._total_actions_sampled = 0

    @property
    def consecutive_threshold(self) -> int:
        """Get current threshold value."""
        return self._consecutive_threshold

    @consecutive_threshold.setter
    def consecutive_threshold(self, value: int):
        """Update threshold and re-evaluate existing states."""
        self._consecutive_threshold = max(1, value)
        
        # Update all existing states if available
        if hasattr(self, '_state') and self._state:
            for state in self._state.values():
                if state.action_count >= value:
                    state.action_count = value
