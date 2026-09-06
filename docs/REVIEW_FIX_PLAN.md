# ПЛАН ИСПРАВЛЕНИЙ ПО РЕВЬЮ ОБУЧЕНИЯ

**Дата:** 2026-09-06
**Проект:** Sakhalin Colony — PPO Training

---

## СВОДКА НАЙДЕННЫХ ОШИБОК

| # | Серьёзность | Файл | Статус |
|---|---|---|---|
| 1 | КРИТИЧЕСКАЯ | `rl/env_manager.py:218-241` | Буфер хранит `s'` вместо `s` |
| 2 | КРИТИЧЕСКАЯ | `train_ui/main_window.py:45,1126` | `use_amp` default=False, `amp_dtype` не передаётся |
| 3 | СРЕДНЯЯ | `rl/ppo.py` | Нет LR scheduler, нет AMP inference, compile без guard |
| 4 | СРЕДНЯЯ | `rl/async_trainer.py:225` | `ent_coef` может вырасти в 10× без ограничения |
| 5 | СРЕДНЯЯ | `rl/config.py:160` | `use_amp` default=True не соответствует UI |
| 6 | НИЗКАЯ | `rl/async_trainer.py:445` | save/eval freq в rollout-ах — неочевидно |
| 7 | НИЗКАЯ | `rl/ppo.py:160` | `model.train()` после update — избыточно |

---

## ЧЕКЛИСТ РЕАЛИЗАЦИИ

### Этап 1: Критические (-blocking обучение)
- [ ] **fix_01**: Исправить порядок obs в буфере (`env_manager.py`)
- [ ] **fix_02**: Передача `amp_dtype` из UI + default `use_amp` в UI
- [ ] **fix_05**: Выровнять default `use_amp` между Config и UI

### Этап 2: Средние (улучшение качества обучения)
- [ ] **fix_03**: LR scheduler + AMP inference + torch.compile guard
- [ ] **fix_04**: Ограничение `ent_coef` boost + fix freq calculation

### Этап 3: Низкие (чистота кода)
- [ ] **fix_06**: Убрать избыточный `model.train()` после update

---

## ФАЙЛЫ-ИСПРАВЛЕНИЯ

| Файл | Что исправляет |
|---|---|
| `docs/fix_01_buffer_obs_order.py` | Порядок obs в буфере (env_manager.py) |
| `docs/fix_02_amp_ui_flow.py` | Передача AMP/compile из UI (main_window.py + worker.py) |
| `docs/fix_03_ppo_improvements.py` | LR scheduler, AMP inference, compile guard (ppo.py) |
| `docs/fix_04_trainer_improvements.py` | ent_coef safety, freq calc (async_trainer.py) |
| `docs/fix_05_config_defaults.py` | Default значения (config.py + main_window.py) |
| `docs/fix_06_cleanup.py` | Мелкие исправления (ppo.py, loop_detector) |

---

## ПОРЯДОК ПРИМЕНЕНИЯ

```
fix_05 → fix_01 → fix_02 → fix_03 → fix_04 → fix_06
```

**Причина:** fix_05 выравнивает default'ы, затем fix_01 исправляет критический баг,
затем fix_02 делает UI-интеграцию AMP рабочей, затем fix_03-06 — улучшения.

---

## ТЕСТИРОВАНИЕ

После каждого fix запускать:
```bash
python -m pytest tests/test_ppo_smoke.py tests/test_gae.py -v
```

После fix_01 additionally:
```bash
python -m pytest tests/test_integration.py -v
```

После fix_02 — ручная проверка UI:
1. Запустить `python run_train_ui.py`
2. Включить чекбокс "AMP (bfloat16)"
3. Запустить обучение — в логе должно быть `amp=True(bfloat16)`
4. Выключить → в логе `amp=False`
