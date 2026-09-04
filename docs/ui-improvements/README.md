# UI Улучшения: Техническое задание

## Обзор

Это техническое задание для добавления продвинутого мониторинга и управления в UI обучения моделей.

Версия: 1.0
Статус: Готово к реализации
Дата: 2026-09-04

---

## Цели

Добавить в UI:
- Распределение действий в реальном времени (top-5 с процентами)
- KL divergence статус с цветной индикацией (LOW/OPTIMAL/HIGH)  
- Curriculum progress (stage, доступные здания, next transition)
- Action loop detection автоматическая детекция застрявших агентов
- Quick Actions панель (boost entropy, reset curriculum, pause/resume)

---

## Структура документов

| Файл | Описание |
|------|----------|
| 01-EXECUTIVE_SUMMARY.md | Исполнительное резюме и основные цели |
| 02-ARCHITECTURE.md | Архитектура, диаграммы потоков данных |
| 03-API_SPECS.md | Детальные спецификации API и протоколы |
| 04-UI_COMPONENTS.md | Спецификации UI виджетов и layout |
| 05-IMPLEMENTATION_PLAN.md | План реализации по фазам с оценкой времени |

---

## Быстрый старт (для разработчиков)

### Предварительные требования

pip install pyqtgraph numpy  # Для графиков (опционально)

### План реализации

1. Фаза 1 (2 дня): Core Infrastructure
   - Расширение Protocol с новыми полями
   - Dual-channel коммуникация (metrics + commands queues)
   - LoopDetector модуль в rl/loop_detector.py
   - Интеграция в AsyncTrainer

2. Фаза 2 (2 дня): Metrics Collection and Display  
   - EnvManager extensions для curriculum progress
   - Top actions collection в training loop
   - Создание 5 UI виджетов (KL, Curriculum, Loops, ReturnStats, CurriculumMetrics)

3. Фаза 3 (1 день): Advanced Interactivity
   - Quick Actions Panel с командами
   - Interactive Training Dashboard (pyqtgraph)
   - Headless mode (--headless flag)

4. Фаза 4 (1 день): Polish and Optimization
   - Error handling и logging
   - Performance optimization
   - Documentation

Итого: ~6 дней работы

---

## Ключевые метрики успеха

- UI показывает top-5 действий с процентами
- KL Status Widget цветной индикатор (LOW/OPTIMAL/HIGH)
- CurriculumProgressWidget stage и next transition
- ActionLoopWidget предупреждает на >30% envs в loops
- Quick Actions работает без пересоздания процесса

---

## Критичные решения

| Решение | Обоснование |
|---------|-------------|
| Отдельная command_queue для команд | Нет конфликтов приоритетов с metrics |
| UI загружает curriculum из конфига | Экономия трафика между процессами |
| Hybrid history (worker to current, UI to history) | UI строит long-term statistics гибче |
| Headless режим необходим | Для серверов/Docker/CI pipelines |
| PyQtGraph для графиков | Real-time performance лучше matplotlib |

---

## Архитектура

```
+-------------+     +-------------+
| EnvManager  |<-->| AsyncTrainer |
+------+------+     +-------------+
                            |
                 metrics_queue (worker to UI)
                             
+----------------------------------------------------+
|                 Main Window (UI)                   |
|  - Parameters Panel                                |
|  - KLStatusWidget                                  |
|  - CurriculumProgressWidget                       |
|  - ActionLoopWidget                               |
|  - ReturnStatisticsWidget                         |
|  - QuickActionsPanel                              |
+----------------------------------------------------+
                            ^
                 command_queue (UI to worker)
```

---

## Детали см. в документах

- 02-ARCHITECTURE.md — Диаграммы потоков данных и модульная структура  
- 03-API_SPECS.md — Полные спецификации API и протоколы сообщений  
- 04-UI_COMPONENTS.md — Детальные спецификации виджетов с кодом  
- 05-IMPLEMENTATION_PLAN.md — Пошаговый план с оценкой времени  

---

## Checkpoint: Готово к реализации

[ ] Все вопросы согласованы
[ ] Архитектура финализирована
[ ] API спецификации детализированы
[ ] План реализации оценён
[ ] Документация структурирована

---

Версия: 1.0 | Дата: 2026-09-04
