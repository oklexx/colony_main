# РЕВЬЮ 02 — GUI (raylib), pybind11-биндинги и консольное меню (src/gui.cpp, bindings.cpp, main.cpp)

**Объём:** пользовательский интерфейс, мост C++↔Python, точка входа консольной игры.
**Игнорируемые папки:** `extracted*`, `resources_out`, `promer`.

> Важно: размерность observation/action между C++ и RL-кодом **согласована** (проверено
> покомпонентно: `obs_size()==203`, `n_actions==45`, порядок менеджеров совпадает с
> `cpp_env.py`). Этот класс багов (частая причина поломки RL) здесь **отсутствует**.

---

## CRITICAL

### [C-C1] Диалоги, открытые мышью, мгновенно закрываются (gui.cpp)

**Места:** `src/gui.cpp:380` (`panel()`) и `src/gui.cpp:1114` (обновление `g_dlg_age`).

```cpp
380: void panel(int x, int y, int w, int h, const char* title) {
381:     DrawRectangle(0, 0, WIN_W, WIN_H, {0, 0, 0, 160});
...
     if (cur_dlg != DLG_NONE && g_dlg_age > 2 && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
         Vector2 mp = GetMousePosition();
         if (mp.x < x || mp.x > x + w || mp.y < y || mp.y > y + h) cur_dlg = DLG_NONE;
     }
```

```cpp
1114: {
1114:    static int prev_dlg = DLG_NONE;
1114:    if (cur_dlg != prev_dlg) { g_dlg_age = 0; prev_dlg = cur_dlg; }
1114:    else if (cur_dlg != DLG_NONE) g_dlg_age++;
       }
```

**Сценарий бага:** пользователь кликает по пункту меню мышью → в секции ввода
`cur_dlg = DLG_MARKET` (напр. `gui.cpp:741` `else if (act == 9) { dlg_mode = 0; cur_dlg = DLG_MARKET; }`).
В том же кадре `Draw` вызывает `panel()`, который видит тот же «нажатый ЛКМ»
(`IsMouseButtonPressed` не «съедается») и, поскольку клик был в топ-панели (вне
прямоугольника диалога), ставит `cur_dlg = DLG_NONE`. Поле `g_dlg_age` к моменту `panel()`
уже `>2` (сохранилось с прошлого открытия), гвард не срабатывает.

**Итог:** все диалоги, открытые мышью (Меню→Новая/Загрузить/Сохранить, Market «B/S»,
Bank «K», NALOG и т.д.) **открываются и тут же закрываются**. Работают только открытые с
клавиатуры (F2/F4/F5/B/S/K) — у них `IsMouseButtonPressed` ложно. GUI фактически неюзабелен
мышью.

**Исправление:** обновлять `g_dlg_age`/`prev_dlg` **до** вызова `panel()` (перенести блок
1114 выше `switch`), либо в `panel()` не проверять закрытие в том же кадре, когда `cur_dlg`
только что сменился (передавать флаг «just opened»).
**Серьёзность: critical.**

---

### [C-C2] Висячий указатель `selected_base` → use-after-free / segfault (gui.cpp)

**Места:** `src/gui.cpp:663` (объявление), `gui.cpp:773` (присваивание), `gui.cpp:1176` (использование).

```cpp
663: static const Base* selected_base = nullptr;
...
773: const Base* b = (cx >= 0 && cy >= 0 && cx < g.map_size() && cy < g.map_size())
774:                ? g.base_in_box(cx, cy) : nullptr;
775: if (b) { selected_base = b; has_sel = false; status = b->data->caption; }
...
1176: if (!show_bd && selected_base) show_bd = selected_base->data;
```

**Проблема:** `g.bases` — `std::vector<Base>`. Любая постройка (`do_build_area` →
`g.build` → `push_back`) может перераспределить вектор, делая `selected_base` висячим.
После этого наведение мыши на пустую клетку (пока `show_bd` не перекрыт ховером) приведёт к
разыменованию освобождённой памяти. То же — после `env.reset(42)` (DLG_NEW, `gui.cpp:521`):
`Game` пересобирает `bases`, а `selected_base` не сбрасывается.

**Воспроизведение:** клик по зданию (select) → затем ПКМ-постройка другого здания (realloc)
→ навести на пустую клетку → краш/UB.
**Исправление:** хранить координаты `(x,y)` вместо сырого указателя и резолвить
`g.base_in_box(x,y)` при отрисовке; либо `std::shared_ptr`/`weak_ptr`; обнулять
`selected_base` в `reset` и после любого `g.build/destroy`.
**Серьёзность: critical.**

---

### [C-C3] Off-by-one в нумерации меню → OOB и недоступный Quit (main.cpp)

**Места:** `src/main.cpp:67, 76, 78, 150, 179`.

```cpp
67:  int mgr_base = 3 + env.n_build();
...
76:  std::cout << "  [" << mgr_base + i + 1 << "]  " << mgr_names[i] << "\n";
...
78:  int quit_key = mgr_base + N_MANAGERS + 1;
...
150: int max_key = 2 + env.n_build() + N_MANAGERS + 1;
...
179: int action = action_map[choice - 1];
```

Пусть `n_build = nb`. Тогда `n_actions = 2+nb+11`, `action_map` имеет индексы
`0..(2+nb+10)` (размер `2+nb+11`). Менеджеры должны печататься под клавишами
`3+nb .. 13+nb`, quit — `14+nb`. Но:
- менеджеры печатаются как `mgr_base+i+1 = 4+nb+i` → `4+nb .. 14+nb` (сдвинуты на +1,
  клавиша `3+nb` пропущена);
- `quit_key = 15+nb`, а `max_key = 14+nb` ⇒ **Quit всегда даёт «Invalid choice»**
  (никогда недостижим).

Последний напечатанный менеджер (клавиша `14+nb`) → `action_map[14+nb-1] = action_map[13+nb]`,
а размер `action_map = 13+nb` ⇒ **чтение за пределами вектора (UB / краш)**. Даже до этого
каждый менеджерский пункт меню триггерит действие на +1 (ветка `mgr_idx = choice - 3 -
n_build` делает правильное смещение, но сам `choice` сдвинут, поэтому нажатая клавиша
запускает *следующий* менеджер).

**Исправление:** печатать менеджеры как `mgr_base + i` (без `+1`), а
`quit_key = mgr_base + N_MANAGERS` (= `max_key`); при необходимости
`max_key = 3 + env.n_build() + N_MANAGERS`.
**Серьёзность: critical.**

---

## MAJOR

### [M-C1] `obs_buffer()` возвращает numpy, разделяющий сырой буфер C++ (bindings.cpp)

**Место:** `src/bindings.cpp:432`
```cpp
432: .def("obs_buffer", [](const ColonyVecEnvCpp& v) {
433:     return py::array_t<float>(
434:         {(int)v.n_envs(), v.obs_size()},
435:         v.obs_buffer());
436: });
```
`py::array_t` оборачивает **тот же** `float*` (`ColonyVecEnvCpp::obs_buffer_.data()`) без
копии и без `keep_alive`. Следующий `step_wait_batch` перезаписывает `obs_buffer_`, и ранее
возвращённый массив «покажет» новые данные (тихая порча observation в RL). Также если объект
`ColonyVecEnvCpp` соберёт GC раньше массива — висячий указатель.

**Исправление:** возвращать копию:
```cpp
py::array_t<float> a({(int)v.n_envs(), v.obs_size()});
std::memcpy(a.mutable_data(), v.obs_buffer(),
            (size_t)v.n_envs() * v.obs_size() * sizeof(float));
return a;
```
(или `py::cast(std::vector<float>(...))`).
**Серьёзность: major** (тихая порча данных в обучении — один из худших видов багов).

---

### [M-G1] Неверный текст сезона в диалоге (gui.cpp)

**Место:** `src/gui.cpp:465`
```cpp
465: const char* txt = dlg_season == 0 ? "Наступило лето" : dlg_season == 1 ? "Наступила осень"
466:                : dlg_season == 2 ? "Наступила зима" : "Наступила весна";
```
`SEASON_SPRING = 0` (constants.h:71), поэтому при наступлении **весны** (dlg_season==0)
пишет «Наступило лето». Сдвиг на единицу для всех сезонов.
**Исправление:** `0→«весна», 1→«лето», 2→«осень», 3→«зима»` (см. также m-E5 в `01_game_logic_cpp.md`).
**Серьёзность: minor** (косметика, но сбивает).

---

### [M-G2] `draw_newgame` игнорирует выбранную карту (gui.cpp)

**Место:** `src/gui.cpp:521`
```cpp
521: if (btn(x + 330, y + 64, 110, 32, "ОК")) { env.reset(42); cur_dlg = DLG_NONE; }
```
В списке три «карты» («Старый город», «Новый город 2», «Сахалин»), но `reset` всегда с
`seed=42`, т.е. генерит один и тот же мир; выбор не влияет. И `prev_season` после reset не
обновляется (остаётся старый) → может спровоцировать лишний диалог сезона на первом кадре.
**Исправление:** передавать выбранный сид/карту и обновлять `prev_season = g.season` после `reset`.
**Серьёзность: minor.**

---

## MINOR

### [m-C1] Утечка текстур/шрифта при выходе (gui.cpp)
`load_assets()` (gui.cpp:188) и `g_mini_tex`/`gFont` нигде не выгружаются перед
`CloseWindow()` (gui.cpp:1192). Одноразовая утечка при выходе (ОС вернёт), но `g_mini_tex`
пересоздаётся при смене `map_size` — в `draw_minimap` (gui.cpp:303) старый корректно
`UnloadTexture`, а вот при выходе — нет. Рекомендуется `UnloadTexture` для всех
`baseTex/earthTex/.../g_mini_tex` и `UnloadFont(gFont)` перед `CloseWindow`.

### [m-C2] Лаг на 1 кадр у интерактивных зон (gui.cpp)
Прямоугольники `g_tool_rect`, `g_time_btn`, `g_date_strip`, `g_mini_rect`, `g_popup_rect`
заполняются во время отрисовки, а обрабатываются в секции ввода **следующего** кадра
(типичный immediate-mode, но в первом кадре они нулевые). Заметно только в первом кадре.

### [m-B1] `RunningMeanStd::mean()/var()` при size==0 (bindings.cpp)
`bindings.cpp:374-378`: если `size()==0`, `r.mean()` может быть `nullptr`, и
`std::vector<double>(nullptr, nullptr+0)` — граничный случай. Безопаснее гвард
`if (r.size()==0) return {};`.

### [m-B2] `game()` возвращает `Game&` по ссылке без `keep_alive` (bindings.cpp)
`bindings.cpp:338`: если Python удержит `Game` после уничтожения `ColonyEnvCpp` — висячая
ссылка. Стоит `py::keep_alive<0,1>`.

### [m-B3] `obs` в `StepOut` и `obs()` как Python-list, а не numpy (bindings.cpp)
`bindings.cpp:72` `d["obs"] = s.obs;` и `bindings.cpp:328` `def("obs", ... return env.obs())`.
`cpp_env.py` делает `np.array(result["obs"])` — работает, но это list-of-float (медленно).
Не баг, но для throughput рекомендуется `py::array_t<float>` с копией.

### [m-B4] GIL
Батч-методы корректно используют `py::call_guard<py::gil_scoped_release>()`
(bindings.cpp:412,415,418), одиночный `step` не держит GIL без нужды. Явных утечек GIL нет.

---

## Позитивные паттерны
- Точное совпадение `obs_size()` (203) с фактическим наполнением `obs()` (нет рассинхрона,
  который обычно ломает RL).
- Корректная диспетчеризация менеджер-действий и согласованный порядок с `cpp_env.py`.
- Границы координат проверяются перед `g.build`/доступом к тайлам
  (`do_build_area` gui.cpp:183, проверки `cx>=0 && cy>=0 && cx<map_size()`).
- GIL релизнут в многопоточном батч-окружении.
- Умная гвардия `g_dlg_age` (хотя из-за порядка вызовов она не работает — см. C-C1).

---

## VERDICT: REQUEST CHANGES
Блокирующие: C-C1 (мгновенное закрытие мыш-opened диалогов), C-C2 (UAF `selected_base`),
C-C3 (off-by-one консольного меню с OOB-чтением и недостижимым Quit), M-C1 (aliasing
`obs_buffer` в bindings). Остальное — minor/косметика.
