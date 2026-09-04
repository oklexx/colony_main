
# ОТЧЁТ: ГЛУБОКИЙ АНАЛИЗ SAKHALIN COLONY RL
## Deep Code Review Report

**Дата**: 04 сентября 2026  
**Проект**: Sakhalin Colony Reinforcement Learning  
**Версия**: v1.x (подтверждённая)

---

## СОДЕРЖАНИЕ

### Файлы отчёта:

1. **PART01_Architecture.md** - Архитектура модели
2. **PART02_TrainingLogic.md** - Логика обучения PPO  
3. **PART03_WeightsRewards.md** - Система вознаграждений и веса
4. **PART04_LogsAnalysis.md** - Анализ логов обучения
5. **PART05_CriticalIssues.md** - Критические проблемы и баги

---

## КРАТКОЕ СООБЩЕНИЕ (EXECUTIVE SUMMARY)

### Основные выводы:

✅ **Что работает хорошо:**
- Производительность: ~25k FPS на RTX 5080 с bf16
- Стабильное обучение: KL divergence в норме (~0.007)
- Сходимость value функции: loss ~0.22
- Высокая пропускная способность: ~74 эпизодов/сек

⚠️ **Критические проблемы:**

1. **BOOTSTRAP MASK BUG** (CRITICAL) - неверное использование termination flags в async_trainer.py:96-98
   - Приводит к смещённым оценкам value функции
   - Корумпирует signal обучения
   
2. **ENTROPY TOO LOW** (HIGH) - ent_coef=0.005 слишком низкий
   - KL divergence очень низкий (0.007), политика чрезмерно детерминирована
   - Модель застревает на PRESERVE action (48% prob)
   
3. **REWARD SHAPING ISSUE** (HIGH) - текущие веса не мотивируют строительство
   - build_bonus=1.0 недостаточно vs idle_build_penalty=-10.0
   - preserve_penalty=-0.5 подавляет exploration
   
4. **OBSERVATION MODE** (MEDIUM) - используется flat mode без spatial awareness
   - Модель не видит пространственные паттерны зданий

### Рекомендуемые исправления:

`python
# Priority 1 - Immediate fixes
ent_coef: float = 0.02    # ↑ от 0.005 к 0.02 (4x increase!)
build_bonus: float = 2.0  # ↑ от 1.0 к 2.0 (2x increase)  
preserve_penalty: float = -0.3  # ↓ от -0.5 к -0.3 (60% decrease)

# Priority 2 - Architecture changes
obs_mode: str = "hybrid"  # ↑ от flat к hybrid для spatial awareness
`

---

## ДЕТАЛЬНЫЙ АНАЛИЗ

### Раздел 1: Архитектура модели (PART01_Architecture.md)

#### Доступные архитектуры:

| Тип | Файл | Параметры | Memory (float32) | Описание |
|-----|------|-----------|------------------|----------|
| MLP | actor_critic.py | ~131K | 0.5 MB | Flat observation + MLP |
| CNN | actor_critic_cnn.py | ~152K | 0.6 MB | Spatial CNN trunk |
| Hybrid | actor_critic_hybrid.py | ~1.49M | 6.0 MB | Flat + CNN fused |

#### Key equations:

**MLP Forward:**
`python
h = MLP(obs)                          # [B, 256]
logits = Linear(h, 45)                # [B, actions]
values = Linear(h, 1)                 # [B, value]
`

**Hybrid Fusion:**
`python
flat_feat = Linear(flat_obs, 256)     # [B, 256]  
cnn_feat = CNN(minimap).flatten()     # [B, 3136] → [B, 256]
x = ReLU(flat_feat + cnn_feat)       # FUSED!
`

---

### Раздел 2: Логика обучения PPO (PART02_TrainingLogic.md)

#### Hyperparameters:

| Param | Default | Range to test | Impact |
|-------|---------|---------------|--------|
| lr | 3e-4 | [1e-4, 1e-3] | Speed of convergence |
| gamma | 0.995 | [0.99, 0.999] | Long-term planning |
| ent_coef | 0.005 | [0.001, 0.02] | Exploration level ⚠️ |
| clip_range | 0.2 | [0.1, 0.3] | Update stability |

#### PPO Loss Function:

`python
ratio = exp(log_prob_new - log_prob_old)
surrogate1 = ratio * advantages
surrogate2 = clamp(ratio, 1-clip, 1+clip) * advantages

# Standard PPO: minimize negative of min(surrogate1, surrogate2)
policy_loss = -min(surrogate1, surrogate2).mean()
value_loss = 0.5 * (values - returns)^2

total_loss = policy_loss + vf_coef*value_loss - ent_coef*entropy
`

---

### Раздел 3: Система вознаграждений (PART03_WeightsRewards.md)

#### Reward composition:

**Positive contributions:**
`
reward = build_bonus × is_build + chain_bonus × has_chain + 
         novelty_bonus × novel_action + daily_income + tax_income + ...
`

**Negative contributions (penalties):**
`
reward -= error_penalty × error + preserve_penalty × preserve + 
           demolish_penalty × demolish + idle_penalty × invalid_build
`

#### Current weights analysis:

| Type | Weight | Relative impact |
|------|--------|-----------------|
| build_bonus | 1.0 | Base positive |
| novelty | 5.0 | High incentive for exploration |
| preserve_penalty | -0.5 | ⚠️ Moderate negative bias |
| idle_build_penalty | -10.0 | Strong deterrent for invalid actions |

---

### Раздел 4: Анализ логов (PART04_LogsAnalysis.md)

#### Training run_012 metrics:

**Summary:**
`
Total timesteps: 10,485,760 (~10M steps)
Training time: 425 seconds (~7 минут)
Average FPS: 24,912
Episodes completed: 30,902
Best reward achieved: 4.41 (low!)
`

**Loss convergence:**
`
policy_loss: -0.0021   ✅ converged to near-zero
value_loss:  0.2240    ✅ stable low value
entropy:     2.6059    ✅ healthy diversity
KL divergence: 0.00740 ✅ within normal range (< 0.01)
`

#### Problematic pattern detected:

**Model behavior (from logs):**
`
Steps 0-50+: 
  - Action PRESERVE selected 48% of time (highest prob)
  - BigHouse second at 27%
  - Road third at 14%
  - Reward per step: -0.6 (constant penalty!)
  
Result after 50 steps:
  - Episode return: -29.4
  - Status: FAILED (thresholds not met)
`

**Root cause:**
- Model overly conservative, avoids building actions
- PRESERVE action dominates despite being penalized
- No successful exploration of construction strategies

---

### Раздел 5: Критические проблемы (PART05_CriticalIssues.md)

#### PROBLEM 1: Bootstrap Mask Bug ⚠️⚠️⚠️

**Location:** l/async_trainer.py:96-98

`python
# CURRENT (BUGGY):
terminated = self.em.buffer.terminated[
    (self.em.buffer.pos - 1) * n_envs : self.em.buffer.pos * n_envs
].cpu().numpy()  # ← Grabs from PREVIOUS rollout step!
`

**Should be:**
`python
# CORRECT:
bootstrap_mask = ~buffer.terminated[-n_envs:]  # Current terminal states only
last_value = value_buffer[buffer.pos*n_envs:(buffer.pos+1)*n_envs]
`

**Impact:**
- GAE bootstrap uses wrong termination flags
- Value estimates corrupted for non-terminal states
- Learning signal fundamentally broken → suboptimal policies

---

#### PROBLEM 2: Entropy Too Low ⚠️⚠️

**Location:** l/config.py:130

`python
ent_coef: float = 0.005  # Too conservative!
`

**Evidence from logs:**
`
KL divergence: 0.00740   ← Very low, indicates overly deterministic policy
Policy entropy: 2.6059   ← Should be higher for exploration
`

**Impact:**
- Policy converges prematurely to PRESERVE action
- Insufficient exploration of building strategies  
- Model fails to learn optimal construction sequence

**Fix:** Increase ent_coef to 0.01-0.02 (2-4x current value)

---

#### PROBLEM 3: Action Clamp Defensive Coding ⚠️

**Location:** l/ppo.py:84

`python
action = action.clamp(0, n_actions - 1)  # Why needed?!
`

**Implication:**
- Indicates potential NaN in logits from Categorical distribution
- Could be caused by incorrect action masking or numerical instability
- Should investigate root cause rather than just patching symptom

---

#### PROBLEM 4: Value Loss Imbalance ⚠️⚠️

**Location:** l/ppo.py:196

`python
value_loss = 0.5 * ((values - returns) ** 2).mean()
loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
# With vf_coef=0.5 and current metrics:
#   policy_loss ≈ 0 (negligible)
#   value_loss ≈ 0.22 (stable but non-zero)
`

**Issue:**
- Policy loss has converged to near-zero while value still learning
- May need dynamic adjustment of vf_coef over training
- Or gradient balancing between policy and value heads

---

#### PROBLEM 5: Flat Observation Mode ⚠️

**Location:** l/config.py:139

`python
obs_mode: str = "flat"  # No spatial information!
`

**Limitation:**
- Discards ALL spatial relationships between buildings
- Model cannot learn optimal placement strategies
- Explains poor building behavior (random PRESERVE instead of strategic construction)

**Fix:** Switch to obs_mode="hybrid" for dual flat+spatial input

---

## 10. РЕКОМЕНДАЦИИ ПО ПУТЬ ДАЛЬШЕ

### Phase 1: Immediate Fixes (24-48 hours)

1. **Fix bootstrap mask bug** in async_trainer.py
   - Patch the terminated flag extraction logic
   - Expected improvement: stable value estimates
   
2. **Increase entropy coefficient**  
   - Update ent_coef from 0.005 to 0.015 (3x)
   - Monitor KL divergence, keep < 0.1
   
3. **Improve reward shaping**
   - build_bonus: 1.0 → 2.0 (+100%)
   - preserve_penalty: -0.5 → -0.2 (-60%)
   - novelty: 5.0 → 8.0 (+60%)

### Phase 2: Architecture Improvements (1 week)

4. **Enable hybrid observations**
   - Switch obs_mode from "flat" to "hybrid"
   - Provide spatial + global context
   - Expected: better building placement learning
   
5. **Hyperparameter sweep** via auto_trainer.py
   - Grid search over [ent_coef, lr, gamma] ranges
   - Target KL ∈ [0.01, 0.1], stable value loss

### Phase 3: Validation & Testing (2 weeks)

6. **Multi-seed evaluation**
   - Add seeds=[42, 43, 44, 45] for robustness
   - Check consistency across random initializations
   
7. **Curriculum learning schedule**
   - Implement staged unlocking of building types
   - Prevent overwhelming early training

8. **Extended training run**
   - Target: 5M steps with fixes applied
   - Expectation: improved composite score from current ~413

---

## 11. FILE REFERENCES

### Source Code Files Analyzed:

| File | Lines | Key Components | Issues Found |
|------|-------|----------------|--------------|
| rl/config.py | ~259 | Config + RewardConfig dataclasses | Low ent_coef, flat obs_mode |
| rl/actor_critic.py | 78 | MLP ActorCritic | None (baseline) |
| rl/actor_critic_cnn.py | 91 | CNN ActorCritic | None (alternative) |
| rl/actor_critic_hybrid.py | 136 | Hybrid architecture | None (recommended) |
| rl/ppo.py | 234 | PPO implementation | Value loss imbalance, action clamp |
| rl/async_trainer.py | 396 | Async training loop | ⚠️ Bootstrap mask bug |
| rl/env_manager.py | - | Environment management | - |
| train.py | 253 | Training CLI | None (entry point) |
| auto_trainer.py | 179 | Optuna tuner | None (tuning framework) |

### Configuration Files:

| File | Purpose | Current Values |
|------|---------|----------------|
| configs/reward.json | Reward weights | build=1.0, novelty=5.0, preserve_pen=-0.5 |
| best_model.meta.json | Best model metadata | score=413, reward config snapshot |

### Log Files Analyzed:

| File | Content | Key Findings |
|------|---------|--------------|
| run_012.log | Training metrics | 10M steps, ~7 min train time |
| reward_debug.log | Reward breakdown | PRESERVE dominance pattern |
| eval_*.log | Evaluation results | score=413, thresholds FAIL |

---

## ПРИЛОЖЕНИЕ: КЛЮЧЕВЫЕ ФОРМУЛЫ

### Composite Score Formula (evaluation):

`python
score = days_agg × 0.4 + bases_agg × 3.0 + people_agg × 0.2 + return_agg × 0.0001
# Weights: (0.4, 3.0, 0.2, 0.0001) ← bases dominates!
`

### PPO Loss Components:

`python
# Policy loss (standard clipped surrogate)
ratio = exp(log_prob_new - log_prob_old)
surrogate1 = ratio × advantages
surrogate2 = clamp(ratio, 1-clip, 1+clip) × advantages  
policy_loss = -min(surrogate1, surrogate2).mean()

# Value loss (MSE against returns)
value_loss = 0.5 × mean((values - returns)^2)

# Total loss with entropy regularization
total_loss = policy_loss + vf_coef × value_loss - ent_coef × entropy
`

### Model Parameter Counts:

`python
MLP [256×256]:    ~131K params  →  0.5 MB (float32)
CNN 8ch×29²:      ~152K params  →  0.6 MB (float32)  
Hybrid [256+512]: ~1.49M params →  6.0 MB (float32)
`

---

*Конец отчёта*
*Создан: 04 сентября 2026*
*Для вопросов и уточнений обратитесь к файлам PART*.md