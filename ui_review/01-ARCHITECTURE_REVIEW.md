# 01 — ARCHITECTURE REVIEW

## Анализ архитектурных решений

---

## 1. Общая структура

```
sakhalin_colony_main/
├── train_ui/           # UI-слой (PySide6)
│   ├── app.py          # Точка входа QApplication
│   ├── main_window.py  # Главное окно (~1505 строк)
│   ├── worker.py       # Worker-процесс (398 строк)
│   ├── protocol.py     # Протокол коммуникации (269 строк)
│   ├── *widget.py      # 5 виджетов + dashboard
│   ├── models.py       # Реестр моделей
│   └── parameter_widget.py  # Панель параметров
├── rl/                 # RL-логика
│   ├── async_trainer.py    # Основной цикл обучения
│   ├── env_manager.py      # Управление окружениями
│   ├── loop_detector.py    # Детектор циклов
│   ├── ppo.py              # PPO алгоритм
│   ├── config.py           # Конфигурация
│   └── actor_critic*.py    # Модели (MLP/CNN/Hybrid)
└── python/             # C++ bindings
    ├── cpp_env.py      # Single env wrapper
    ├── cpp_vecenv.py   # Vectorized env wrapper
    └── minimap.py      # Minimap wrapper
```

**Оценка: 7/10** — Структура логичная, разделение ответственности в целом соблюдено. Проблема в том, что `main_window.py` (1505 строк) слишком монолитный и содержит смешение UI-логики, бизнес-логики и коммуникации.

---

## 2. Паттерн коммуникации

### Архитектура по спецификации (02-ARCHITECTURE.md)
```
Worker Process ─── metrics_queue ───▶ UI Thread
Worker Process ◀── command_queue ──── UI Thread
```

### Фактическая реализация
```
Worker Process ─── JSONL файл ───▶ UI QTimer (poll every 250ms)
Worker Process ◀── stdin JSON ──── UI (write to stdin)
```

**Разрыв с ТЗ:** Спецификация описывала `multiprocessing.Queue`, но реализация использует JSONL-файл + `tasklist`-based PID polling. Это **правильное** решение для Windows (Queue в Python через multiprocessing на Windows ненадёжный), но оно не документировано.

**Потенциальные проблемы:**
- Файловая коммуникация добавляет задержку (polling 250ms)
- `tasklist` вызов для проверки PID — это Windows-only (`_poll_pid_alive` в `main_window.py:992-1008`)
- Нет cleanup при аварийном завершении UI (JSONL-файл может остаться)

---

## 3. Изоляция worker-процесса

Worker запускается как отдельный `subprocess.Popen` (`main_window.py:1104-1111`):
- **stdout/stderr** перенаправляются в PIPE
- **Связь:** JSONL-файл для сообщений, stdin для команд
- **Остановка:** `taskkill /F /PID` (`main_window.py:1240-1257`)

**Оценка: 8/10** — Хорошая изоляция. Падение UI не крашит training (worker продолжает работать). Но `taskkill /F` не даёт worker-у сохранить модель при остановке.

---

## 4. Типизация protocol.py

```python
@dataclass
class ProgressMsg:
    done: int
    total: int
    fps: float = 0.0
    # ... 11 полей
```

**Оценка: 8/10** — Чистый protocol с dataclass, encode/decode, NaN-safe `_safe_float()`. Единственная проблема: `CommandMsg.type` переиспользует `MsgType.LOG` (строка 135) для обратной совместимости, что вводит в заблуждение.

---

## 5. Observations — что модель видит

### Flat режим (207 измерений)
Из `python/cpp_env.py:162-166` и `observe.py`:
```
obs[0..N] — нормализованные float32 значения:
  - Количество людей, денег, ресурсов
  - Типы и количество зданий каждого типа
  - День, неделя, состояние налога
  - Кредиты, занятость, производство
```

### Minimap режим (8 каналов, 29x29)
Из `python/minimap.py:1-13`:
```
Channel 0-6: one-hot типы земли (вода, лес, горы, etc.)
Channel 7:   occupied (где стоят здания)
Размер: 2 * R + 1 = 29 при R=14
Центр: позиция колонии (init_sel)
```

### Hybrid режим
MLP-ветка обрабатывает flat (глобальную статистику), CNN-ветка обрабатывает minimap (пространственную информацию). Затем merged trunk.

**Ключевой момент:** Модель видит **текущее состояние** колонии. Она НЕ планирует на будущее через observation — стратегическое поведение формируется через reward shaping и policy gradient.

---

## 6. PPO-пайплайн

```
EnvManager.reset() → [_collect_rollout] → PPO.update() → loop
                              ↓
                    Policy(obs) → action → env.step() → reward, next_obs
                              ↓
                    buffer.add(obs, action, reward, log_prob, value)
                              ↓
                    GAE → PPO clipped loss → AdamW step
```

**Оценка: 8/10** — Стандартная PPO реализация. AMP, gradient clipping, action masking корректно реализованы. Единственная проблема: `collect_step` в `ppo.py:63-89` передаёт `flat` как первый аргумент, но для MLP/CNN модели `forward(obs)` принимает один аргумент — это работает только для hybrid.

---

## 7. Адаптивная энтропия

Спецификация (03-API_SPECS.md) описывала `detect_and_boost_entropy()` — метод, который автоматически увеличивает `ent_coef` при обнаружении loops. **Фактически не реализован в production-коде.** `LoopDetector` предоставляет `get_stats()`, но автоматического буста нет. QuickActionsWidget предлагает ручной буст кнопкой.

---

## Выводы

| Аспект | Оценка | Комментарий |
|--------|--------|-------------|
| Разделение ответственности | 7/10 | main_window слишком монолитен |
| Коммуникация worker↔UI | 8/10 | JSONL файл — хорошее решение для Windows |
| Изоляция процессов | 8/10 | Корректная, но taskkill не даёт graceful shutdown |
| Protocol | 8/10 | Чистый, типизированный |
| Observations | 9/10 | Три режима (flat/minimap/hybrid) грамотно реализованы |
| PPO-пайплайн | 8/10 | Стандартная реализация, AMP корректен |
| Соответствие ТЗ | 5/10 | Ключевые фичи (auto-boost entropy, dual-channel queue) не реализованы как в спецификации |
