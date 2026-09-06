# 04 — LOGIC REVIEW

## Проверка корректности логики взаимодействия

---

## 1. Двухканальная коммуникация

### Спецификация
```
metrics_queue: worker → UI (ProgressMsg)
command_queue: UI → worker (CommandMsg)
```

### Фактическая реализация
```
metrics: JSONL-файл, worker пишет → UI читает через QTimer (250ms poll)
commands: stdin, UI пишет → worker читает через thread
```

**Проблема:** Command processing в worker полностью отключён:
1. `args.command_queue` не добавлен в argparse → `hasattr(args, 'command_queue')` всегда False
2. `command_queue` создаётся в main() но **не передаётся** в `run_train()` (строка 374 создаёт новый `_queue.Queue`)
3. `_watch_stdin()` получает команды, кладёт в queue, но этот queue **никто не читает**

**Результат:** Boost Entropy, Reset Curriculum, Pause/Resume команды **не доходят** до training loop.

---

## 2. Сериализация/десериализация Protocol

### Encode cycle
```python
ProgressMsg → .to_dict() → json.dumps → строка → в JSONL-файл
```

### Decode cycle
```python
JSONL-файл → readline → json.loads → MsgType dispatch → ProgressMsg
```

**Корректно:** `_safe_float()` обрабатывает NaN/Inf. `decode()` проверяет обязательные поля.

**Проблема:** `CommandMsg.type = MsgType.LOG` (protocol.py:135). При `encode()` создаётся `{"type": "command", ...}`, но `decode()` обрабатывает `"command"` отдельно (строка 222). Если кому-то нужно будет `encode` → `decode` цикл для CommandMsg, он сломается — `MsgType("command")` не существует.

---

## 3. LoopDetector — корректность алгоритма

### Алгоритм (`loop_detector.py:57-115`)
```python
for each action_data:
    state.action_sequence.append((action, time.time()))
    if len(sequence) >= threshold:
        recent = sequence[-threshold:]
        if all(a == action for a, _ in recent):
            alert!
```

**Проблемы:**
1. **Временные метки не используются** — `time.time()` сохраняется, но никогда не проверяется. Детектор реагирует на последовательные действия независимо от времени между ними.

2. **action_sequence.pop(0)** (строка 85) — O(n) операция. При 100 элементах и 8 envs — приемлемо, но не масштабируется.

3. **Метод `get_stats()`** (строка 154):
   ```python
   "envs_with_loops": sum(
       1 for state in self._state.values()
       if state.action_count >= self.consecutive_threshold
   )
   ```
   `action_count` — это **общее** количество действий, а не количество последовательных. Любой env с >3 действиями будет помечен как "in loop".

4. **В async_trainer.py:173-187** — попытка получения данных из buffer:
   ```python
   actions = self.em.buffer.actions.cpu().numpy()[:, start:end]
   ```
   Но `self.em.buffer` может быть `_TensorRolloutBuffer` (для minimap/hybrid) — формат данных другой.

---

## 4. Curriculum — логика переключения этапов

### В AsyncTrainer (`async_trainer.py:322-327`)
```python
schedule = getattr(self.cfg, "curriculum_schedule", None)
if schedule:
    for step_threshold, stage in schedule:
        if total_done >= step_threshold and self._curriculum_stage < stage:
            self.em.set_curriculum_stage(stage)
            self._curriculum_stage = stage
```

**Корректно:** Пороги проверяются последовательно, stage монотонно возрастает.

**Проблема:** `curriculum_schedule` в `Config` — это `list` (не `List[tuple]`). Формат JSON: `[[0, 1], [500000, 2], ...]`. При `for step_threshold, stage in schedule` — это распаковка подсписков, работает корректно.

### В EnvManager (`env_manager.py:263-286`)
```python
def get_allowed_buildings_for_stage(self, stage_id: int) -> List[str]:
    curriculum_schedule = {
        0: ["INITIALIZE", "MOVE_RIGHT", ...],  # Все действия
        1: ["INITIALIZE", "MOVE_RIGHT", ...],  # Только базовые
        ...
    }
```

**Проблема:** Списки действий **не совпадают** с реальным action space:
- Реальные действия: DAY, WEEK, BUILD_WATER_CHANNEL, BUILD_FARM, ..., IMPROVE_LAND, REPAIR, PRESERVE, UNPRESERVE, SELL_SURPLUS, BUY_FOOD, TAKE_LOAN, REPAY_LOAN, PAY_TAX
- В `get_allowed_buildings_for_stage`: INITIALIZE, MOVE_RIGHT, MOVE_LEFT, BUILD_HOUSE, BUILD_FURNITURE, GATHER_WOOD

Эти имена **не существуют** в реальном C++ окружении. Метод всегда возвращает захардкоженный список, не связанный с curriculum stage в C++.

---

## 5. Adapтивная энтропия

### Спецификация
```python
def detect_and_boost_entropy(self, em, ppo, current_step, threshold_ratio=0.3):
    if loop_rate > threshold_ratio AND avg_entropy < 2.5:
        ent_coef = min(ent_coef * 1.3, 0.05)
```

### Реализация
**Отсутствует.** `LoopDetector`提供ляет `get_stats()`, но автоматического буста нет. `QuickActionsWidget`提供ляет ручную кнопку "Boost Entropy ×2", но она не подключена к worker.

---

## 6. GAE и bootstrapping

### В `async_trainer.py:78-92`
```python
terminated = self.em.buffer.terminated[
    (self.em.buffer.pos - 1) * n_envs : self.em.buffer.pos * n_envs
].cpu().numpy()
```

Корректно извлекает terminated для последнего шага rollout.

### В `ppo.py:99`
```python
self.buffer.compute_gae(last_value, last_done)
```

`last_done` — это `terminated` (без truncation), что правильно для GAE bootstrap.

**Оценка: Корректно** — правильное разделение terminated/truncated.

---

## 7. Action masking

### В `ppo.py:76-77`
```python
if action_masks is not None:
    logits = logits.masked_fill(action_masks == 0, float("-inf"))
```

### В `env_manager.py:195-198`
```python
action_masks_np = getattr(self.env, "action_masks", None)
if action_masks_np is not None:
    action_masks = torch.from_numpy(action_masks_np).to(self.device)
```

**Корректно:** Маски обновляются после каждого step (cpp_vecenv.py:144-146), передаются в policy. Blocked actions получают logit = -inf → probability = 0.

---

## 8. Weight sync (C++ ↔ PyTorch)

### EnvManager не синхронизирует веса!
`EnvManager.__init__()` создаёт `ActorCritic` и `PPO`, но **не загружает веса** из C++ окружения. C++ окружение не хранит модель — оно только выполняет `step()`. Веса хранятся в PyTorch модели и синхронизируются только при `save()`/`load()`.

**Это корректно** — C++ окружение является только simulation engine, policy lives in PyTorch.

---

## 9. Observation flow при обучении

```
1. EnvManager.reset() → CppVecEnv.reset() → obs_np [n_envs, 207]
2. _policy_obs(obs_np) → torch.Tensor [n_envs, 207] (flat) или [n_envs, 8, 29, 29] (minimap)
3. PPO.collect_step(obs) → action [n_envs]
4. EnvManager.step(action_np) → CppVecEnv.step_async/wait → new_obs, rewards, dones
5. buffer.add(obs, action, reward, log_prob, value, done, terminated)
6. Repeat n_steps times
7. PPO.update(last_value, last_done) → GAE → clipped loss → Adam step
```

**Корректно.** Единственная тонкость: `terminated` (истинный terminal) передаётся отдельно от `dones` (terminated | truncated) для правильного GAE bootstrap.

---

## 10. Observation при наблюдении (watch)

`observe.py` загружает модель, создаёт env с n_envs=1, и для каждого step:
1. `policy(obs_t)` → logits
2. `logits.masked_fill(masks_t == 0, -inf)` — masking
3. `Categorical(logits=logits).sample()` → action
4. `env.step(action)` → obs, reward, terminated

**Ключевой момент:** Модель видит **только нормализованный observation** (207 float32 или 8x29x29 minimap). Она НЕ видит:
- Текстовые описания зданий
- Стратегические цели
- Историю действий
- Сравнение с другими игроками

Её "понимание" — это статистические паттерны в数值数据, выученные через policy gradient.

---

## Выводы

| Логика | Оценка | Комментарий |
|--------|--------|-------------|
| Communication | 2/10 | Command queue полностью отключён |
| Protocol serialization | 7/10 | CommandMsg.type mismatch |
| LoopDetector | 4/10 | action_count считает общее, не consecutive |
| Curriculum switching | 6/10 | C++ stage переключается, но get_allowed_buildings возвращает мусор |
| Adaptive entropy | 1/10 | Не реализована |
| GAE/bootstrapping | 9/10 | Корректно |
| Action masking | 9/10 | Корректно |
| Weight sync | 8/10 | Корректно (PyTorch owns weights) |
