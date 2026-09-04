# РЕВЬЮ 03 — Обучение моделей (PPO / RL pipeline)

**Объём:** `rl/ppo.py`, `rl/actor_critic.py`, `rl/rollout_buffer.py`, `rl/async_trainer.py`,
`rl/config.py`, `rl/env_manager.py`, `python/cpp_env.py`, `python/cpp_vecenv.py`, `train.py`.
**Метод:** полное чтение + ручная верификация GAE-маски в `rollout_buffer.py`.
**Игнорируемые папки:** `extracted*`, `resources_out`, `promer`.

> Это ключевая часть задания. Ниже — настоящий глубокий review обучения.

---

## Резюме намерения

Синхронный on-policy PPO (Categorical, дискретные действия) с GPU-resident буфером, GAE,
AMP (bf16/fp16) и C++ векторизованным окружением. Код «почти работает», но содержит
**два критических бага в GAE/терминальных масках**, которые систематически портят
преимущества (advantages) на границах эпизодов, плюс несколько major-проблем (падение на
CPU, мёртвый чекпоинтинг, «асинхронность» не реализована).

---

## CRITICAL

### [C-G1] Off-by-one в маске терминальности в GAE (`dones[t+1]` вместо `dones[t]`)

**Место:** `rl/rollout_buffer.py:86-103`
```python
86:  for t in range(self.n_steps - 1, -1, -1):
87:      start = t * self.n_envs
88:      end = start + self.n_envs
89:      if t == self.n_steps - 1:
90:          next_values = last_value
91:          next_dones = last_done
92:      else:
93:          next_start = (t + 1) * self.n_envs
94:          next_values = self.values[next_start:next_start + self.n_envs]   # V(s_{t+1})  — верно
95:          next_dones = self.dones[next_start:next_start + self.n_envs]      # dones[t+1]  — НЕВЕРНО
...
97:      delta = (
98:          self.rewards[start:end]
99:          + self.gamma * next_values * (~next_dones)      # маска сдвинута на шаг
100:         - self.values[start:end]
101:     )
102:     next_adv = delta + self.gamma * self.gae_lambda * (~next_dones) * next_adv
```

**Почему ломает.** Gym/CppVecEnv возвращают `dones` в **transition-based** семантике:
`buffer.dones[t] = terminal(s_{t+1})` (флаг относится к состоянию, В которое перешли на шаге `t`).
Корректная GAE-формула для шага `t`:
```
δ_t = r_t + γ · (1 - terminal(s_{t+1})) · V(s_{t+1}) - V(s_t)
A_t = δ_t + γλ · (1 - terminal(s_{t+1})) · A_{t+1}
```
Т.е. маска бустрапа для шага `t` должна быть `terminal(s_{t+1}) = dones[t]`. Код же берёт
`dones[t+1] = terminal(s_{t+2})` — маска сдвинута на один шаг вперёд. Следствие: на границе
эпизода бустрап режется не там, где надо (или, наоборот, НЕ режется, когда `s_{t+1}`
терминально, а `s_{t+2}` уже из новой игры — и тогда в бустрап идёт `V(s_{t+1})`
терминального/сброшенного состояния, что является мусорным значением). Advantages и
value-таргеты смещены практически во всех траекториях с хотя бы одним завершением.

**Почему юнит-тест этого не ловит.** `tests/test_gae.py` сам написан в той же неверной
конвенции: его `reference_gae` тоже использует `next_done = dones[t+1]` (строка 26), а в
буфер кладёт `done = dones[t]` (строка 69). То есть тест проверяет *внутреннюю
согласованность* кода с самим собой, а не корректность относительно gym-конвенции. Ложная
уверенность.

**Правильный вариант.** В `compute_gae` использовать `dones[t]` как маску (данные
transition-based), а `values[t+1]` как `next_values`:
```python
else:
    next_start = (t + 1) * self.n_envs
    next_values = self.values[next_start:next_start + self.n_envs]
    next_dones  = self.dones[start:end]          # dones[t] = terminal(s_{t+1})
```
и `last_done` (из C-G2) = `dones[T-1] = terminal(s_T)` — уже корректно. Тест
`reference_gae` тогда нужно привести к `next_done = dones[t]` и передавать
`last_done = dones[T-1]`.

> Вместе C-G1 и C-G2 — одна корневая причина: код предполагает `dones[t] = terminal(s_t)`
> (state-based), а окружение даёт `dones[t] = terminal(s_{t+1})` (transition-based). Это
> основной источник порчи обучения.

**Серьёзность: critical.**

---

### [C-G2] `last_done` для GAE считается как OR по всему роллауту (async_trainer.py)

**Место:** `rl/async_trainer.py:96-112`
```python
96:  for i, (r, d) in enumerate(zip(rewards, dones)):
97:      self._current_ep_return += r
98:      self._current_ep_len += 1
99:      if d:
100:         self._ep_returns.append(self._current_ep_return)
...
106:         done_mask[i] = True          # НИКОГДА не сбрасывается в False
...
110: with torch.no_grad():
111:     last_value = self.em.ppo.model.get_value(obs)
112:     last_done = done_mask                # <-- БАГ
```

**Почему ломает.** `done_mask[i]` выставляется в `True`, если env `i` ХОТЬ РАЗ завершился за
роллаут, и никогда не сбрасывается. Но `last_done` нужен GAE только для **последнего** шага
`t = n_steps-1` — это маска бустрапа из состояния `s_T` (после последнего шага). Если env
завершился на шаге 10 и C++ его сбросил, то к шагу 4095 его финальное состояние `s_T` НЕ
терминально, однако `last_done[i]=True`. В `rollout_buffer.compute_gae` это даёт для
последнего шага:
```
δ_{T-1} = r_{T-1} + γ · last_value · (1 - True) - V(s_{T-1})
        = r_{T-1} - V(s_{T-1})
```
т.е. бустрап из НЕ-терминального `s_T` незаконно обрезается → advantage последнего шага
каждого env, в котором был хоть один game-over за роллаут, систематически занижен. При
`n_steps=4096` и частых game-over это бо́льшая часть траекторий.

**Правильный вариант.** Использовать done флаг именно последнего добавленного шага (он уже
читается в строках 92-94 из буфера):
```python
last_dones = self.em.buffer.dones[
    (self.em.buffer.pos - 1) * n_envs : self.em.buffer.pos * n_envs
].cpu().numpy()
...
last_done = torch.tensor(last_dones, dtype=torch.bool, device=self.device)
```
Локальная переменная `dones` в конце цикла (строки 92-94 перезаписываются каждую итерацию)
уже содержит done последнего шага — можно просто `last_done = torch.tensor(dones, ...)`.
**Серьёзность: critical.**

---

### [C-G3] Жёстко зашит `device_type="cuda"` в AMP → падение на CPU (ppo.py)

**Места:** `rl/ppo.py:48` и `rl/ppo.py:101`
```python
48:  self.scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == "float16"))
...
101: with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
```
`train.py:73` ставит `device="cpu"`, если CUDA недоступна, но `use_amp` по умолчанию `True`
(`--amp` ≠ `"off"`). Тогда `torch.autocast(device_type="cuda")` и `GradScaler("cuda")` бросают
`RuntimeError` на CPU. Пайплайн не запускается без GPU и без явного `--amp off`.

**Фикс:** `device_type=self.device.type` в обоих местах (и `GradScaler` только если
`self.device.type=="cuda"`).
**Серьёзность: critical** (блокирует запуск на CPU).

---

## MAJOR

### [M-RL1] Периодическое сохранение чекпоинтов никогда не срабатывает (async_trainer.py)

**Места:** `rl/async_trainer.py:161,198-201`
```python
161: total_done += steps_per_rollout
...
198: if total_done % self.cfg.save_freq < steps_per_rollout:
199:     ckpt_path = save_dir / f"checkpoint_{total_done}_steps.pt"
200:     self.em.ppo.save(str(ckpt_path))
```
`total_done` — всегда кратно `steps_per_rollout = n_steps*n_envs` (по умолчанию
`4096*8 = 32768`). Условие `total_done % save_freq < steps_per_rollout` истинно тогда и
только тогда, когда `total_done % save_freq == 0` (т.к. остаток сам кратен 32768 и < 32768 ⇒ 0).
При `save_freq=500000` и шаге 32768: НОД(32768,500000)=32, требуется `32768·k % 500000 == 0`
⇒ `k` кратно `5^6=15625` ⇒ `total_done = 511 000 000`, что за пределами
`total_timesteps=1_000_000`. **Результат:** при дефолтных гиперпараметрах ни один
промежуточный чекпоинт не пишется, только `final_model.pt`. При падении процесса весь
прогресс теряется.

**Фикс:** сравнивать номер итерации роллаута, а не таймстепы, либо
`if (total_done // steps_per_rollout) % (save_freq // steps_per_rollout) == 0`. Или хранить
`rollout_idx` и сохранять по `rollout_idx % save_every == 0`.
**Серьёзность: major.**

---

### [M-RL2] «AsyncTrainer» на самом деле синхронный — очередь не используется (async_trainer.py)

**Места:** `rl/async_trainer.py:52,151-159`
```python
52:  self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=cfg.queue_size)
...
151: while total_done < total and not self._stop:
152:     t_rollout_start = time.perf_counter()
153:     rollout = self._collect_rollout(obs)      # блокирующий сбор n_steps на главном потоке
154:     rollout_time = time.perf_counter() - t_rollout_start
155:     if self._stop: break
156:     stats = self._update_ppo(rollout)         # блокирующий GPU-апдейт
```
`cfg.async_train` и `_queue` нигде не используются; сбор роллаута и PPO-апдейт выполняются
последовательно в одном потоке. Нет producer/consumer, нет отдельных процессов/потоков,
значит **нет data races (это плюс), но и нет параллелизма** между C++ симуляцией и
GPU-обучением. Имя класса и `queue_size` вводят в заблуждение; заявленного ускорения нет.

**Следствие:** `state_dict_for_env`/`load_state_dict_from_env` в `actor_critic.py:74-78` —
мёртвый код (веса не пушатся в воркеры). Это корректно для синхронной схемы, но противоречит
названию и ТЗ. Если планировался реальный async — он не реализован вообще.
**Серьёзность: major** (вводит в заблуждение + нет заявленного ускорения; функционально не ломает).

---

### [M-RL3] Truncation слит с termination → бустрап незаконно режется на time-limit (cpp_vecenv.py)

**Место:** `python/cpp_vecenv.py:119-121`
```python
119: terminateds = np.array(result.terminateds, dtype=bool)
120: trunceds = np.array(result.trunceds, dtype=bool)
121: dones = terminateds | trunceds
```
В GAE (C-G1/C-G2) `dones` используется как маска бустрапа. При **truncated** (достигнут
лимит времени, игра продолжается) состояние `s_{t+1}` валидно и `V(s_{t+1})` — корректная
оценка, бустрап резать НЕ надо. Слитие `|` трактует truncation как termination и режет бустрап
→ смещённые advantages (занижение). Если в C++ есть time-limit (truncated), это баг; если
truncation не используется (бесконечный горизонт / только game_over), — нейтрально.
**Нужно проверить C++ (`ColonyVecEnvCpp`):** должны ли `truncated` приходить, и различать их
(маска бустрапа только по `terminated`).
**Серьёзность: major** (зависит от C++).

---

### [M-RL4] Масштаб наград несогласован (dominates `tax_bonus=250`) (config.py)

**Место:** `rl/config.py:11-19`
```python
11:  build_bonus: float = 5.0
12:  chain_bonus: float = 0.5
13:  chain_daily: float = 2.0
14:  novelty: float = 20.0
15:  daily_income: float = 0.1
16:  sale_bonus: float = 0.1
17:  tax_bonus: float = 250.0      # на 2-3 порядка больше остальных
18:  survival_bonus: float = 0.0
19:  game_over_penalty: float = 20.0
```
`tax_bonus=250` при `daily_income=0.1`, `build_bonus=5` создаёт колоссальный разброс сигнала.
Value-функция вынуждена «растягиваться» под редкие ±250, а градиент по обычным шагам
(≈0.1–5) тонет. Это не баг кода, но сильно ухудшает сходимость (эффективно — шум в returns).
Рекомендация: привести награды к сопоставимому масштабу (нормировать `tax_bonus`, либо
reward scaling, либо `reward_clip`), и задокументировать единицы. См. также `04_data_config_inconsistencies.md`
(значения в `reward.json` ещё и отличаются от дефолтов).
**Серьёзность: major** (качество обучения).

---

## MINOR

### [m-RL1] Value loss без клиппинга (ppo.py)
`rl/ppo.py:161-162`:
```python
161: values = values.squeeze(-1)
162: value_loss = 0.5 * ((values - returns) ** 2).mean()
```
Стандартный PPO (SB3) клиппует value loss: `max((V - R)^2, (V_clipped - R)^2)`. Без клипа
редкие большие returns (см. M-RL4) могут вызвать всплеск градиента критика. Рекомендуется
для стабильности.

### [m-RL2] `old_values` вычисляется и передаётся, но не используется (ppo.py / rollout_buffer.py)
`old_values` приходит в `_compute_loss_components` и в `get_batches`, но внутри не участвует
(таргет — `returns`). Только dead data; если планируется value-clip (m-RL1), он понадобится.

### [m-RL3] `loss.backward()` вне `torch.autocast` при bf16 (ppo.py)
При `amp_dtype=="bfloat16"` forward идёт в bf16, а `backward()` — уже вне контекста autocast
(fp32). Численно корректно, но теряется выгода bf16 в backward. Рекомендуется оборачивать и
backward в тот же `autocast`.

### [m-RL4] `approx_kl` считается, но не используется (ppo.py)
`rl/ppo.py:164-165` — логируется, но нет ни `target_kl` early-stopping, ни адаптивного LR. Для
PPO это стандартный монитор. Также `approx_kl` — грубая оценка (`|.|` от средней разности),
лучше `0.5*((old-new)^2).mean()` или `exp(new-old)-1-(new-old)`.

### [m-RL5] Нет LR-расписания (constant LR)
LR фиксирован (`cfg.learning_rate`). Линейный аннейлинг или косинусный спад обычно улучшают
финальное качество PPO. По ТЗ («learning rate schedule») — отсутствует.

### [m-RL6] Нет детерминизма инициализации модели (env_manager.py / train.py)
`ActorCritic` инициализируется с дефолтным `torch` RNG, `cfg.seed` прокидывается только в C++
окружение (`reset`/`base_seed`), но `torch.manual_seed(cfg.seed)` не вызывается. Воспроизводимость
прогона моделью не обеспечена. Добавить `torch.manual_seed(cfg.seed)` до создания модели.

### [m-RL7] Observation только клиппуется, не стандартизируется (cpp_env.py / cpp_vecenv.py)
`python/cpp_env.py:106`: `obs = np.clip(obs, -self.clip_obs, self.clip_obs).astype(np.float32)`.
Observation space задан `Box(low=-10, high=10)`. В `env_manager` obs берётся «как есть» без
нормализации. Docstring утверждает «VecNormalize handled in C++», но в питоне наблюдение лишь
клиппуется. Если C++ не делает mean/std нормализацию, признаки с большими масштабами (деньги,
people) окажутся обрезанными на ±10 и сжатыми — плохо для MLP. **Требуется верификация в
`colony_cpp`, что нормализация реально есть**; иначе добавить `VecNormalize`/`RunningMeanStd`.

### [m-RL8] Мёртвый/вводящий в заблуждение код
- `rl/actor_critic.py:74-78` `state_dict_for_env`/`load_state_dict_from_env` — не вызываются (см. M-RL2).
- `rl/env_manager.py:128-132` `finish_episode` — не используется.
- `rl/env_manager.py:134-141` `get_stats` — всегда возвращает `{}` (пустой цикл).
- `rl/rollout_buffer.py:83,85` `last_gae`, `next_adv` инициализируются, но `last_gae` не используется.
- `rl/async_trainer.py:52` `queue.Queue` не используется (M-RL2).

### [m-RL9] Тип `rewards` float64 → float32 (cpp_vecenv.py)
`rewards = np.array(result.rewards, dtype=np.float64)` затем конвертируется в float32. Потеря
точности некритична, но желательно сразу float32.

### [m-RL10] `save()` сохраняет `buffer_pos`, `load()` его не восстанавливает (ppo.py)
После `load` буфер не сбрасывается в сохранённую позицию; на практике `update` в конце делает
`buffer.reset()`, так что поломки нет, но сохранённый `buffer_pos` бессмыслен.

---

## ЧТО СДЕЛАНО ПРАВИЛЬНО (позитив)

- **Clipped surrogate** (`ppo.py:156-159`): `ratio = exp(new - old)`,
  `min(ratio·A, clip(ratio)·A)`, знак `policy_loss = -min(...)` — корректно.
- **Entropy / value веса**: `loss = policy_loss + vf_coef*value_loss - ent_coef*entropy` —
  верный знак и веса (0.5 / 0.01).
- **GAE-формула в целом** (кроме маски dones, C-G1) — `δ`, `next_adv` рекурсия, нормализация
  advantages (mean 0 / std 1) верны.
- **RolloutBuffer**: размерности `[n_steps*n_envs, ...]` корректны, GPU-resident, `flatten`
  через `randperm` + индексация — верно; shuffle каждую эпоху.
- **ActorCritic**: ортогональная инициализация весов (`gain=1.0`), отдельные policy/value
  головы, `params` только trainable — стандартно и хорошо.
- **AMP GradScaler** включён ровно для fp16, для bf16 — нет (верно).
- **grad_clip** по `max_grad_norm` применяется — хорошо.
- **C++ observation/action space согласованы** с `ActorCritic` (`obs_size=203`, `Discrete(45)`).
- Нет data races (синхронно), `get_action_and_value` под `no_grad` и `eval()` — корректно.

---

## VERDICT: REQUEST CHANGES

Обучение **не будет корректным** до фикса C-G1 и C-G2 — они систематически портят advantages
на каждой границе эпизода, что для игры с частыми game-over даёт смещённый policy-градиент.
C-G3 (падение на CPU), M-RL1 (потеря чекпоинтов) и M-RL2 (ложная async-архитектура)
блокируют надёжную эксплуатацию.

**Приоритет исправлений:**
1. C-G1 — маска `dones[t]` вместо `dones[t+1]` в `compute_gae` (+ обновить `test_gae.py`).
2. C-G2 — `last_done` = done последнего шага, а не OR по роллауту.
3. C-G3 — `device_type=self.device.type`.
4. M-RL1 — корректное условие периодического сохранения.
5. M-RL3 — разделить `terminated`/`truncated` (проверить C++).
6. M-RL4 — привести масштаб наград.

После C-G1/C-G2 рекомендую прогнать `tests/test_gae.py` с *исправленной* reference-реализацией
(state-based dones), чтобы тест реально валидировал корректность, а не внутреннюю
согласованность бага.
