# 🎯 ТЕХНИЧЕСКОЕ ЗАДАНИЕ: УЛУЧШЕНИЯ UI ОБУЧЕНИЯ

## 📌 Исполнительное резюме

**Версия ТЗ:** 1.0  
**Дата создания:** 2026-09-04  
**Статус:** Готово к реализации  
**Приоритет:** P1 (Критично важно для отладки нового кода)

---

## 🎯 ЦЕЛИ И ЗАДАЧИ

### Основная цель
Добавить в UI обучения **полную прозрачность процесса**, позволяющую:
- В реальном времени видеть распределение действий и обнаруживать паттерны PRESERVE loops
- Мониторить KL divergence status с цветовой индикацией (LOW/OPTIMAL/HIGH)
- Отслеживать прогресс curriculum learning (stage, доступные здания, next transition)
- Детектировать action loops автоматически и визуализировать статистику
- Получать быстрые команды управления (boost entropy, reset curriculum)

### Ключевые метрики успеха
1. ✅ UI показывает top-5 действий с процентом в реальном времени (обновление каждые 2048 шагов)
2. ✅ KL Status Widget показывает цветной индикатор статуса + прогресс бар
3. ✅ Curriculum Progress Widget отображает текущий stage, % прогресса до следующего, список доступных зданий
4. ✅ Action Loop Detector показывает предупреждения когда >30% envs застряли в loops
5. ✅ Quick Actions Panel позволяет управлять обучением без пересоздания процесса

### Scope (Включено)
- Расширение Protocol с новыми полями ProgressMsg и CommandMsg
- Создание модуля LoopDetector в `rl/loop_detector.py`
- Интеграция detector в AsyncTrainer
- Добавление методов curriculum progress в EnvManager
- Создание 5 новых виджетов UI (KL, Curriculum, Loops, ReturnStats, CurriculumMetrics)
- Реализация dual-channel коммуникации (metrics + commands queues)
- Headless режим (--headless flag)

### Out of Scope (Отложено до P3/P4)
- Экспорт/импорт профилей настроек (JSON presets)
- Сравнение multiple runs side-by-side
- GPU/memory usage monitoring
- Advanced analytics (trend analysis, anomaly detection)
- Interactive plot comparisons (zoom/pan overlays между запусками)

---

## 📋 ДОКУМЕНТЫ-СОПРУТАГИ

- `02-ARCHITECTURE.md` — Общая архитектура и диаграммы потоков данных
- `03-API_SPECS.md` — Детальные спецификации API (Protocol, LoopDetector, EnvManager)
- `04-UI_COMPONENTS.md` — Спецификации UI виджетов и layout
- `05-IMPLEMENTATION_PLAN.md` — План реализации по фазам с оценкой времени
