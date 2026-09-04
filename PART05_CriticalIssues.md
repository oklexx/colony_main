
## 7. CRITICAL ISSUES & CODE PROBLEMS

### 7.1 BOOTSTRAP MASK INCORRECTNESS (CRITICAL) ⚠️⚠️⚠️

**Location**: l/async_trainer.py:LINE 96-98

`python
terminated = self.em.buffer.terminated[
    (self.em.buffer.pos - 1) * n_envs : self.em.buffer.pos * n_envs
].cpu().numpy()
`

**PROBLEM**: This grabs terminated flags from PREVIOUS rollout step, not from CURRENT terminal state!

**CORRECT LOGIC SHOULD BE**:

`python
# Bootstrap mask: WHERE NOT TO BOOTSTRAP VALUE at episode end
bootstrap_mask = ~terminated                    # [n_envs] boolean
last_value = ...                                # Value of terminal state ONLY if terminated=True
`

**IMPACT**: 
- Incorrect GAE bootstrapping
- Value estimates biased by non-terminal states
- Learning signal corrupted → suboptimal policies

---

### 7.2 ENTROPY COEFFICIENT TOO LOW ⚠️⚠️

**Location**: l/config.py:LINE 130

`python
ent_coef: float = 0.005  # Comment: "Reduced from 0.01 to discourage action loops"
`

**PROBLEM**: 
- Current KL divergence is very low (0.007) indicating overly deterministic policy
- Low entropy means model explores insufficiently
- Explains PRESERVE action dominance (~48% probability)

**RECOMMENDATION**: Increase ent_coef to 0.01-0.02 for better exploration

---

### 7.3 ACTION CLAMP DEFENSIVE CODING (MEDIUM PRIORITY) ⚠️

**Location**: l/ppo.py:LINE 84

`python
action = action.clamp(0, n_actions - 1)      # Prevent CUDA assert from NaN!
`

**PROBLEM**: 
- Indicates potential numerical instability in Categorical distribution
- NaN logits can occur if action_masks applied incorrectly
- Should investigate why NaN prevention is needed at all

---

### 7.4 VALUE FUNCTION LOSS IMBALANCE ⚠️⚠️

**Location**: l/ppo.py:LINE 196

`python
value_loss = 0.5 * ((values - returns) ** 2).mean()
loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
`

**PROBLEM**:
- f_coef=0.5 gives equal weight to policy and value losses
- Policy loss ~0 while value_loss ~0.22 → potential imbalance
- May need dynamic vf_coef adjustment or gradient balancing

---

### 7.5 OBSERVATION MODE DEFAULT ⚠️

**Location**: l/config.py:LINE 139

`python
obs_mode: str = "flat"     # legacy mode, no spatial awareness
`

**PROBLEM**: 
- Flat observations discard spatial relationships
- Hybrid or minimap modes would give model better context
- Explains inability to understand building placement strategy

---

## 8. RECOMMENDATIONS SUMMARY

### Immediate Fixes (Priority 1):

1. **Fix bootstrap mask logic** in async_trainer.py
   - Use correct termination flags from current rollout step
   
2. **Increase entropy coefficient**
   - Change ent_coef from 0.005 to 0.01-0.02
   
3. **Improve reward shaping**
   - Increase build_bonus: 1.0 → 2.0
   - Decrease preserve_penalty: -0.5 → -0.3
   - Increase novelty bonus: 5.0 → 8.0

### Medium Priority (Priority 2):

4. **Switch to hybrid/minimap observations**
   - Enable spatial awareness for building placement
   
5. **Tune vf_coef dynamically**
   - Start with higher value loss weight, decrease over training

### Long-term Improvements (Priority 3):

6. **Enable torch.compile** after training stabilizes (~20% speedup)
7. **Add curriculum learning schedule** for gradual complexity increase
8. **Multi-seed evaluation** for robustness checks

---

## 9. FILE STRUCTURE SUMMARY

`
sakhalin_colony_main/
├── rl/
│   ├── config.py              # RewardConfig + Config dataclasses
│   ├── actor_critic.py        # MLP network (~131K params)
│   ├── actor_critic_cnn.py    # CNN network (~152K params)  
│   ├── actor_critic_hybrid.py # Hybrid network (~1.49M params)
│   ├── ppo.py                 # PPO implementation
│   ├── async_trainer.py       # Async training loop ⚠️ BUG HERE
│   └── env_manager.py         # Environment management
├── train.py                   # Training CLI entry point
├── auto_trainer.py            # Optuna-based hyperparameter tuning
├── configs/
│   └── reward.json            # Reward configuration used by best model
└── saved/models/
    └── best_model.meta.json   # Best model metadata with weights
`

---

## 10. PERFORMANCE METRICS

| Metric | Value | Assessment |
|--------|-------|------------|
| FPS (training) | ~24,900 steps/sec | ✅ Excellent |
| VRAM usage | 2.4 GB / 17 GB | ✅ Efficient (14%) |
| bf16 support | Yes | ✅ Utilized |
| Episodes/sec | ~74 | ✅ High throughput |
| Training time (1M steps) | ~425 sec | ✅ Fast (~7 min) |

---

*Отчёт сгенерирован: 04 сентября 2026*
*Дата анализа: sakhalin_colony_main v1.x*