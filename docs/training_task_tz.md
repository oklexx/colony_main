## План задач по улучшению системы наград

Я подготовил детальные спецификации для всех задач (F, A, B, C, D, E, G). Каждый файл самодостаточен: содержит описание проблемы, решение, список изменяемых файлов, пошаговые инструкции с кодом, тесты и критерии готовности.

Порядок выполнения:
1. **Task F** – Балансировка коэффициентов (меняем дефолты)
2. **Task A** – Клиппинг сырой награды
3. **Task B** – Удаление разовых налоговых бонусов
4. **Task C** – Единый штраф за ошибки + параметры для консервации/налога
5. **Task D** – Штраф за стоимость строительства
6. **Task E** – Milestone-бонусы
7. **Task G** – Пространственные бонусы (опционально)

---

# Task F: Балансировка коэффициентов наград

**Проблема:**  
Текущие дефолтные значения в `RewardConfig` приводят к слишком большому разбросу наград (от −50 до +90 000). Это делает обучение нестабильным, так как PPO чувствителен к масштабу вознаграждений. Основные виновники: `build_bonus=5.0`, `chain_bonus=0.5`, `chain_daily=2.0`.

**Решение:**  
Уменьшить коэффициенты так, чтобы все компоненты наград были в сопоставимом диапазоне (примерно от −100 до +100). Новые дефолты:

| Параметр | Старое значение | Новое значение |
|----------|----------------|----------------|
| `build_bonus` | 5.0 | 0.5 |
| `chain_bonus` | 0.5 | 0.1 |
| `chain_daily` | 2.0 | 0.5 |
| `daily_income` | 0.1 | 0.05 |
| `sale_bonus` | 0.1 | 0.05 |
| `game_over_penalty` | 20.0 | 100.0 |
| `error_penalty` (новый) | – | −5.0 |
| `preserve_penalty` (новый) | – | 0.5 |
| `manual_tax_penalty` (новый) | – | 0.5 |
| `build_cost_penalty` (новый) | – | 0.01 |
| Milestone параметры (новые) | – | как указано в Task E |

**Файлы:**  
- `include/colony/env.h` – изменить дефолтные значения в `RewardConfig`
- `rl/config.py` – синхронизировать дефолты
- `configs/reward.json` (если существует) – обновить значения

**Шаги:**

1. Открыть `include/colony/env.h`, найти `struct RewardConfig`.
2. Изменить дефолты:
   ```cpp
   double build_bonus = 0.5;
   double chain_bonus = 0.1;
   double chain_daily = 0.5;
   double daily_income = 0.05;
   double sale_bonus = 0.05;
   double game_over_penalty = 100.0;
   ```
3. Добавить новые поля (для Task C, D, E они понадобятся позже, но можно добавить сейчас):
   ```cpp
   double error_penalty = -5.0;
   double preserve_penalty = 0.5;
   double manual_tax_penalty = 0.5;
   double build_cost_penalty = 0.01;
   double milestone_base_bonus = 50.0;
   double milestone_people_bonus = 20.0;
   double milestone_day_bonus = 30.0;
   double milestone_year_bonus = 100.0;
   double proximity_bonus = 0.0; // опционально, по умолчанию 0
   ```
4. В `rl/config.py` синхронизировать `RewardConfig`:
   ```python
   build_bonus: float = 0.5
   chain_bonus: float = 0.1
   chain_daily: float = 0.5
   daily_income: float = 0.05
   sale_bonus: float = 0.05
   game_over_penalty: float = 100.0
   error_penalty: float = -5.0
   preserve_penalty: float = 0.5
   manual_tax_penalty: float = 0.5
   build_cost_penalty: float = 0.01
   milestone_base_bonus: float = 50.0
   milestone_people_bonus: float = 20.0
   milestone_day_bonus: float = 30.0
   milestone_year_bonus: float = 100.0
   proximity_bonus: float = 0.0
   ```
   Добавить эти поля в `to_dict()` и `from_dict()`.
5. Обновить `python/cpp_env.py` и `python/cpp_vecenv.py` – добавить новые ключи в `_REWARD_KEYS`.
6. Обновить `src/bindings.cpp` – добавить привязки для новых полей в `py::class_<RewardConfig>`.
7. Если есть `configs/reward.json` – обновить значения.
8. Написать тест `tests/test_balance_defaults.py`, проверяющий, что дефолтные значения соответствуют новым.
9. Пересобрать C++ расширение и запустить тесты.

**Критерий готовности:**  
- Все дефолты изменены.  
- При создании `Config()` и `RewardConfig()` значения равны новым.  
- Тесты проходят.

**Коммит:**  
`git add include/colony/env.h rl/config.py python/cpp_env.py python/cpp_vecenv.py src/bindings.cpp configs/reward.json tests/test_balance_defaults.py`  
`git commit -m "balance: reduce build_bonus, chain_* and adjust other reward defaults"`

---

# Task A: Клиппинг сырой награды

**Проблема:**  
Даже после балансировки отдельных компонентов экстремальные значения (например, очень удачный ход) могут давать выбросы, которые нарушают нормализацию и стабильность обучения. В `VecEnv` уже есть клиппинг нормализованной награды, но сырая награда не ограничена.

**Решение:**  
Добавить клиппинг сырой награды в `ColonyEnvCpp::step()` перед передачей в VecEnv. Использовать два новых параметра: `clip_reward_min` и `clip_reward_max` (дефолт −100 и +100). Это позволит отсечь аномалии до нормализации.

**Файлы:**  
- `include/colony/env.h` – добавить `clip_reward_min`, `clip_reward_max`
- `src/env.cpp` – в конце `step()` применить `std::clamp`
- `rl/config.py` – добавить поля
- `python/cpp_env.py`, `cpp_vecenv.py` – ключи
- `src/bindings.cpp` – привязки
- `tests/test_reward_clip.py` – новый тест

**Шаги:**

1. В `RewardConfig` добавить:
   ```cpp
   double clip_reward_min = -100.0;
   double clip_reward_max = 100.0;
   ```
2. В `ColonyEnvCpp::step()` после суммирования всех компонент (перед `return`) добавить:
   ```cpp
   rew = std::clamp(rew, cfg_.clip_reward_min, cfg_.clip_reward_max);
   ```
3. Синхронизировать Python.
4. Написать тест, проверяющий, что награда за постройку BigFarm (которая давала >1000) обрезается до 100.
5. Собрать и запустить тесты.

**Критерий готовности:**  
- При превышении `clip_reward_max` награда обрезается.  
- При достижении `clip_reward_min` – аналогично.  
- Существующий клиппинг нормализованной награды остаётся без изменений.

**Коммит:**  
`git add ...` (аналогично)  
`git commit -m "feat: clip raw reward before normalization"`

---

# Task B: Удаление разовых налоговых бонусов

**Проблема:**  
За уплату annual tax даётся `+15`, за main tax – `+30`. Это создаёт спайки и мотивирует агента «ловить момент» уплаты, а не поддерживать стабильную экономику. Ежедневный `tax_daily_bonus` уже существует и должен быть основным стимулом.

**Решение:**  
Убрать хардкодные `rew += 15.0` и `rew += 30.0` из блоков уплаты налогов. Оставить только штраф `-5.0` при недостатке средств и ежедневный бонус.

**Файлы:**  
- `src/env.cpp` – удалить строки с `rew += 15.0` и `rew += 30.0`
- `tests/test_tax_reward.py` – обновить/добавить тесты

**Шаги:**

1. В `ColonyEnvCpp::step()` найти блоки:
   ```cpp
   if (!g.tax_postponed_ && g.annual_tax_due()) {
       if (g.money >= g.annual_tax_amount()) {
           g.pay_annual_tax();
           rew += 15.0;  // <-- удалить эту строку
       } else {
           rew -= 5.0;
       }
   }
   if (!g.tax_postponed_ && g.main_tax_due()) {
       if (g.money >= g.main_tax_amount()) {
           g.pay_main_tax();
           rew += 30.0;  // <-- удалить эту строку
       } else {
           rew -= 5.0;
       }
   }
   ```
   Заменить на:
   ```cpp
   if (!g.tax_postponed_ && g.annual_tax_due()) {
       if (g.money >= g.annual_tax_amount()) {
           g.pay_annual_tax();
           // бонус убран
       } else {
           rew -= 5.0;
       }
   }
   if (!g.tax_postponed_ && g.main_tax_due()) {
       if (g.money >= g.main_tax_amount()) {
           g.pay_main_tax();
           // бонус убран
       } else {
           rew -= 5.0;
       }
   }
   ```
2. Убедиться, что `tax_daily_bonus` начисляется в отдельном блоке (уже есть).
3. Написать тест, проверяющий, что при уплате налога не добавляется дополнительных бонусов, только штрафы при недостатке.
4. Запустить тесты.

**Критерий готовности:**  
- При уплате любого налога с достаточным балансом награда не изменяется (только `tax_daily_bonus` был начислен ранее).  
- При недостатке средств штраф `-5.0` остаётся.

**Коммит:**  
`git commit -m "fix: remove one-shot tax bonuses, keep daily tax_compliance bonus"`

---

# Task C: Единый штраф за ошибки и параметры для консервации/налога

**Проблема:**  
Разные ошибки имеют разные штрафы (`-0.1`, `-1.0`), а успешные действия (консервация, ручная уплата налога) тоже наказываются хардкодом. Это негибко и не позволяет легко настраивать баланс.

**Решение:**  
- Ввести новый параметр `error_penalty` (дефолт −5.0) и заменить им все штрафы за неудачные действия (кроме особых случаев, таких как смерть/потеря базы, которые остаются отдельными параметрами).  
- Ввести `preserve_penalty` (дефолт 0.5) для успешной консервации/разконсервации.  
- Ввести `manual_tax_penalty` (дефолт 0.5) для ручной уплаты налога.  
- Оставить `-2.0` за переполнение жилья (можно позже параметризовать, но пока не трогаем).

**Файлы:**  
- `include/colony/env.h` – добавить `error_penalty`, `preserve_penalty`, `manual_tax_penalty` (уже добавлены в Task F)  
- `src/env.cpp` – заменить все `-0.1` и `-1.0` (ошибки) на `cfg_.error_penalty`; заменить `-0.5` за консервацию и ручной налог на соответствующие параметры.  
- `rl/config.py`, bindings, cpp_env – синхронизировать.  
- `tests/test_error_penalty.py` – новый тест.

**Шаги:**

1. В `ColonyEnvCpp::step()` найти все места, где начисляется `-0.1` или `-1.0` за неудачу, и заменить на `cfg_.error_penalty`.  
   Примеры:  
   - `build` неудачна: `rew += cfg_.error_penalty;`  
   - `upgrade_land` неудачна: `rew += cfg_.error_penalty;`  
   - `repair` неудачен: `rew += cfg_.error_penalty;`  
   - `sell_surplus` неудачна: `rew += cfg_.error_penalty;`  
   - `take_credit` неудачен: `rew += cfg_.error_penalty;`  
   - и т.д.  
2. Найти места, где начисляется `-0.5` за успешную консервацию/разконсервацию и заменить на `-cfg_.preserve_penalty`.  
3. Найти ручную уплату налога (действие `A_PAY_TAX`) и заменить `-0.5` на `-cfg_.manual_tax_penalty`.  
4. Оставить `-2.0` за переполнение жилья без изменений.  
5. Написать тест, проверяющий, что неудачная постройка даёт `error_penalty`, а успешная консервация – `-preserve_penalty`.  
6. Запустить тесты.

**Критерий готовности:**  
- Все неудачные действия (кроме особых) дают одинаковый штраф `error_penalty`.  
- Консервация/разконсервация и ручной налог наказываются отдельными параметрами.  
- Тесты подтверждают это.

**Коммит:**  
`git commit -m "feat: unified error_penalty, preserve_penalty, manual_tax_penalty"`

---

# Task D: Штраф за стоимость строительства

**Проблема:**  
Награда за строительство не вычитает затраты на ресурсы. Агент может строить дорогие здания, даже если они не окупаются, так как получает бонус только за доход, а не за чистую прибыль.

**Решение:**  
Добавить новый параметр `build_cost_penalty` (дефолт 0.01). При успешном строительстве вычитать `build_cost_penalty * total_cost`, где `total_cost` – денежный эквивалент затраченных ресурсов. Для этого добавить метод `Game::total_build_cost(building_type)`.

**Файлы:**  
- `include/colony/game.h` – объявить `double total_build_cost(int building_type) const;`  
- `src/game.cpp` – реализовать метод.  
- `src/env.cpp` – в блоке успешного строительства добавить вычитание.  
- `include/colony/env.h` – добавить `build_cost_penalty` (уже есть).  
- `tests/test_build_cost.py` – новый тест.

**Шаги:**

1. В `game.h` добавить:
   ```cpp
   double total_build_cost(int building_type) const;
   ```
2. В `game.cpp` реализовать:
   ```cpp
   double Game::total_build_cost(int building_type) const {
       double cost = 0.0;
       for (int r = 0; r < 9; ++r) {
           cost += build_cost_[building_type][r] * SALE_SUNDUK[r];
       }
       return cost;
   }
   ```
3. В `ColonyEnvCpp::step()` в блоке успешного строительства (после `g.build(...)`) добавить:
   ```cpp
   double cost = g.total_build_cost(g.what_build());
   rew -= cfg_.build_cost_penalty * cost;
   ```
4. Написать тест, строящий дорогое здание (например, BigFarm) и проверяющий, что награда уменьшена на `build_cost_penalty * cost`.  
5. Собрать и запустить тесты.

**Критерий готовности:**  
- При строительстве здания с ненулевой стоимостью награда уменьшается пропорционально `build_cost_penalty`.  
- Если `build_cost_penalty = 0`, поведение не меняется.

**Коммит:**  
`git commit -m "feat: subtract build cost penalty from build reward"`

---

# Task E: Milestone-бонусы (долгосрочные цели)

**Проблема:**  
Нет стимула для достижения пороговых значений (количество баз, людей, дней). Агент может действовать краткосрочно, не заботясь о масштабировании.

**Решение:**  
Ввести награды за достижение milestones:
- `+milestone_base_bonus` за каждые 5 баз (5, 10, 15, ...)
- `+milestone_people_bonus` за каждые 50 человек (50, 100, 150, ...)
- `+milestone_day_bonus` за каждые 100 дней (100, 200, 300, ...)
- `+milestone_year_bonus` за первый год (365 дней)

Состояние milestones хранится в `Game` и сбрасывается при `reset()`.

**Файлы:**  
- `include/colony/game.h` – добавить поля и метод `check_milestones()`  
- `src/game.cpp` – реализовать проверку и начисление (возвращать сумму бонусов)  
- `src/env.cpp` – в `step()` после `advance_day()` вызвать `check_milestones()` и добавить возвращённый бонус к `rew`.  
- `include/colony/env.h` – добавить параметры (уже есть).  
- `tests/test_milestones.py` – новый тест.

**Шаги:**

1. В `Game` добавить поля:
   ```cpp
   int last_base_milestone_ = 0;
   int last_people_milestone_ = 0;
   int last_day_milestone_ = 0;
   bool year_bonus_given_ = false;
   ```
2. В `reset()` обнулить их.
3. Добавить метод `double check_milestones(int day, int people, int bases, const RewardConfig& cfg)` который проверяет условия, обновляет счётчики и возвращает сумму начисленных бонусов.
4. В `ColonyEnvCpp::step()` после `advance_day()` вызвать этот метод и добавить возвращённое значение к `rew`.
5. Написать тест, проверяющий, что при достижении 5 баз награда увеличивается на `milestone_base_bonus`, и что бонус начисляется только один раз за уровень.
6. Собрать и запустить тесты.

**Критерий готовности:**  
- При достижении milestone награда увеличивается на соответствующий бонус.  
- Повторные достижения (например, 10 баз) дают бонус снова.  
- При `reset()` счётчики обнуляются.

**Коммит:**  
`git commit -m "feat: milestone bonuses for bases, people, days, year"`

---

# Task G: Пространственные бонусы (опционально)

**Проблема:**  
Агент не учитывает расположение зданий относительно ресурсов. Это не влияет на выживание, но в долгосрочной перспективе оптимальное размещение повышает эффективность.

**Решение:**  
Добавить бонус за строительство здания в радиусе 3 клеток от соответствующего ресурса. Параметр `proximity_bonus` по умолчанию 0 (отключено). Если >0, при постройке здания проверяется наличие ресурса нужного типа в радиусе и добавляется бонус.

**Файлы:**  
- `include/colony/env.h` – добавить `proximity_bonus` (уже есть).  
- `src/env.cpp` – в блоке успешного строительства добавить проверку близости.  
- `tests/test_proximity.py` – новый тест.

**Шаги:**

1. Определить соответствие здания → требуемый ресурс (например, CoalMine → уголь, Farm → вода). Можно захардкодить в небольшой функции в `ColonyEnvCpp`.  
2. При успешном строительстве проверить, есть ли в радиусе 3 клеток клетка с нужным типом ресурса (использовать `g.map().cell(coord).get_land_type()`).  
3. Если есть – добавить `cfg_.proximity_bonus`.  
4. Написать тест: построить CoalMine рядом с углём – получить бонус, построить вдалеке – без бонуса.  
5. Собрать и запустить тесты.

**Критерий готовности:**  
- При `proximity_bonus=0` поведение не меняется.  
- При положительном значении бонус начисляется только при близости к ресурсу.  
- Тесты проходят.

**Коммит:**  
`git commit -m "feat: proximity bonus for building near matching resources (optional)"`

---

## Итоговый порядок выполнения и зависимости

1. **Task F** – меняем дефолты (ни от чего не зависит).  
2. **Task A** – клиппинг (использует новые дефолты, но может выполняться после F).  
3. **Task B** – налоговые бонусы (не зависит от A).  
4. **Task C** – штрафы (использует новые параметры из F, но может выполняться после F).  
5. **Task D** – стоимость строительства (использует `build_cost_penalty` из F).  
6. **Task E** – milestones (использует параметры из F).  
7. **Task G** – пространственные бонусы (использует `proximity_bonus` из F).

Все задачи независимы в том смысле, что изменения в одном файле не конфликтуют с другими (кроме добавления новых полей в `RewardConfig`, которые уже добавлены в Task F). Поэтому их можно выполнять параллельно, но я рекомендую последовательно для упрощения отладки.

Теперь вы можете использовать эти спецификации для выполнения задач через subagent-driven подход или вручную. Если нужны дополнительные пояснения по какой-либо задаче – дайте знать.