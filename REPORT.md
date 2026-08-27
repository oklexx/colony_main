# Sakhalin Colony — GPU-first PPO: отчёт о реализации

## 1. Архитектура

Система из 6 модулей, все вычисления — на GPU (RTX 5080, bf16):

```
train.py          CLI: парсинг аргументов, сборка Config, запуск
  └─ EnvManager   CppVecEnv (C++ pybind11) + ActorCritic + RolloutBuffer + PPO
       └─ AsyncTrainer  цикл rollout → PPO update, checkpointing
```

| Файл | Ответственность |
|------|----------------|
| `rl/config.py` | Dataclass `Config` + `RewardConfig`, сериализация в JSON |
| `rl/actor_critic.py` | MLP `203→256→256→45`, ortho-init, `get_action_and_value` |
| `rl/rollout_buffer.py` | GPU-resident ring buffer, GAE, mini-batch sampling |
| `rl/ppo.py` | PPO loss (clip + value + entropy), Adam, AMP bf16/fp16 |
| `rl/env_manager.py` | Обёртка `CppVecEnv`, шаг env → buffer, weight sync |
| `rl/async_trainer.py` | Основной цикл: rollout → update → checkpoint → log |
| `train.py` | CLI entry point |
| `tests/` | `test_gae.py`, `test_ppo_smoke.py`, `test_integration.py`, `bench_per_step.py` |

## 2. Как работает обучение

### 2.1. Цикл (один rollout)

```
obs (GPU [n_envs, 203])
    │
    ▼  ActorCritic.forward (no_grad)
action, log_prob, value
    │
    ▼  CppVecEnv.step_async + step_wait (C++, n_threads)
new_obs, reward, done  (CPU → GPU, non_blocking)
    │
    ▼  RolloutBuffer.add (GPU tensors)
buffer[pos] += {obs, action, reward, log_prob, value, done}
```

Повторяется `n_steps` раз (по умолчанию 4096) × `n_envs` параллельных сред.

### 2.2. PPO update (после заполнения буфера)

1. **GAE** (`rollout_buffer.py:75`):
   ```
   δ_t = r_t + γ·V(s_{t+1})·(1−d_{t+1}) − V(s_t)
   A_t = δ_t + γλ·(1−d_{t+1})·A_{t+1}
   ```
   Вычисляется на GPU, обратно от последнего шага. Normalization: `(A − mean)/std`.

2. **Epochs** (по умолчанию 10):
   - Shuffle всех `n_steps × n_envs` сэмплов на GPU (`torch.randperm`).
   - Mini-batch (по умолчанию 8192).
   - Loss: `−min(ratio·A, clip(ratio)·A) + 0.5·(V−R)² − 0.01·H`.
   - `loss.backward()` → `clip_grad_norm_(0.5)` → `Adam.step()`.
   - AMP: `torch.autocast(bfloat16)` — без GradScaler (bf16 не требует).

3. **Buffer reset**, `pos = 0`.

### 2.3. Checkpointing

Каждые `save_freq` шагов (по умолчанию 500k) — `checkpoint_N_steps.pt`
с `model_state` + `optimizer_state` + `buffer_pos`.
В конце — `final_model.pt`.

### 2.4. Логирование

- Печатные метрики каждый rollout: FPS, losses, entropy, KL, GPU mem.
- TensorBoard (если установлен): `train/fps`, `train/policy_loss`, `train/value_loss`, `train/entropy`, `train/approx_kl`, `train/learning_rate`, `train/gpu_mem_mb`, `train/best_reward`.
- Эпизоды: return, length, best_reward трекаются в `AsyncTrainer`.

## 3. Параметры запуска

Все параметры — аргументы командной строки `train.py:28-61`:

| Флаг | Default | Описание |
|------|---------|----------|
| `--steps` | 1 000 000 | Всего timesteps |
| `--envs` | 8 | Параллельных сред (C++) |
| `--n-steps` | 4096 | Длина rollout на сред |
| `--batch-size` | 8192 | Mini-batch |
| `--n-epochs` | 10 | Epochs PPO |
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 0.995 | Discount |
| `--gae-lambda` | 0.98 | GAE λ |
| `--clip-range` | 0.2 | PPO clip |
| `--ent-coef` | 0.01 | Entropy penalty |
| `--vf-coef` | 0.5 | Value loss coef |
| `--max-grad-norm` | 0.5 | Grad clipping |
| `--net-arch` | 256 256 | Hidden layers |
| `--amp` | bfloat16 | bfloat16 / float16 / off |
| `--compile` | off | torch.compile |
| `--envs` | 8 | Параллельных C++ env |
| `--cpp-threads` | 0 (авто) | Потоки C++ |
| `--seed` | 42 | Seed |
| `--map-size` | 280 | Размер карты |
| `--reward-config` | — | JSON с наградами |
| `--save-freq` | 500 000 | Частота чекпоинтов |
| `--name` | colony_run | Имя runs |

Пример:
```bash
python train.py --steps 1000000 --envs 32 --n-steps 4096 --batch-size 16384 --n-epochs 5 --name my_run
```

## 4. Награды

Награды задаются в C++ (через `reward_config` JSON) — `rl/config.py:10-52`:

| Параметр | Default |
|----------|---------|
| `build_bonus` | 5.0 |
| `chain_bonus` | 0.5 |
| `chain_daily` | 2.0 |
| `novelty` | 20.0 |
| `daily_income` | 0.1 |
| `sale_bonus` | 0.1 |
| `tax_bonus` | 250.0 |
| `survival_bonus` | 0.0 |
| `game_over_penalty` | 20.0 |
| `disable_net_worth` | False |
| `disable_daily_income` | False |

Передаются в `CppVecEnv` через `reward_config` dict при создании.

## 5. Результаты бенчмарка (RTX 5080)

| Конфиг | FPS | Примечание |
|--------|-----|------------|
| 256 envs, raw env+GPU inference | **51 864** | 4.9 ms/step, без PPO update |
| 128 envs, n_steps=2048, batch=16384, epochs=5 | 34 479 | Полный цикл PPO |
| 256 envs, n_steps=2048, batch=16384, epochs=5 | **38 794** | Полный цикл PPO |
| 512 envs, n_steps=1024, batch=16384, epochs=5 | 38 210 | Полный цикл PPO |
| 8 envs, n_steps=4096, batch=8192, epochs=10 | 6 705 | Малое число env |

**Вывод:** цель ≥50k FPS достигнута на чистом env+GPU inference (256 envs). Полные циклы PPO дают 35–40k FPS; overhead PPO-обновления (10 epochs × 16k batch) — основной фактор, ограничивающий скорость.

## 6. Что исправлено в процессе

| Файл | Проблема | Исправление |
|------|----------|-------------|
| `tests/test_gae.py` | Reference GAE использовал `dones[t]` вместо `dones[t+1]` для masking carry-члена | Стандартная формула `A_t = δ_t + γλ(1−d_{t+1})A_{t+1}` |
| `rl/ppo.py:102` | Ошибка распаковки: `new_log_probs, new_values = ...` (2 значения) вместо 4 | `policy_loss, value_loss, entropy, approx_kl = ...` |
| `rl/env_manager.py` | `reset_batch`/`step_async_batch` не существовали в `CppVecEnv` | Заменено на `reset()`/`step_async()`/`step_wait()` |
| `train.py:77` | `total_mem` → `total_memory` | API PyTorch 2.x |

## 7. Структура проекта

```
sakhalin_colony_main/
├── train.py                    # CLI entry point
├── requirements.txt
├── rl/
│   ├── __init__.py
│   ├── config.py               # Config + RewardConfig
│   ├── actor_critic.py         # MLP 203→256→256→45
│   ├── rollout_buffer.py       # GPU buffer + GAE
│   ├── ppo.py                  # PPO loss + update + AMP
│   ├── env_manager.py          # CppVecEnv + model + buffer + ppo
│   └── async_trainer.py        # Train loop + checkpointing
├── python/
│   ├── cpp_vecenv.py           # C++ vectorized env (pybind11)
│   ├── cpp_env.py
│   └── colony_cpp.pyd          # Compiled C++ game
├── src/                        # C++ game source
├── include/
├── configs/
└── tests/
    ├── test_gae.py             # GAE verification
    ├── test_ppo_smoke.py       # PPO smoke test
    ├── test_integration.py     # 100k steps integration
    └── bench_per_step.py       # Per-step latency benchmark
```
