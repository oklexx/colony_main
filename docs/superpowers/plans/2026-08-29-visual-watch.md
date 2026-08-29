# Визуальное наблюдение за моделью — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Кнопка «Наблюдать» в Qt UI открывает raylib-окно, в котором обученная модель играет в реальном времени (карта, здания, статистика), с управлением скоростью.

**Architecture:** Путь C — готовый `sakhalin_colony_gui.exe` + Python-драйвер. C++ GUI получает новый режим `--headless-ai`: читает action из `actions.txt`, пишет `state.json`, игнорирует ввод мыши/клавиатуры (кроме ESC), камера авто-центр, подсветка выбранного действия в палитре. Python (`watch_champion.py --visual`) запускает exe как subprocess, каждый шаг: policy → action → `actions.txt`, читает `state.json` для статистики, управляет скоростью (пауза/×1/×2/×5). Qt UI передаёт `--visual`.

**Tech Stack:** C++17 + raylib 6.0 (gui.cpp), Python 3 + PyTorch (watch_champion.py), PySide6 (main_window.py). Файлы IPC: `actions.txt` (Python→C++), `state.json` (C++→Python).

**Spec:** `docs/superpowers/specs/2026-08-29-visual-watch-design.md`

## Global Constraints

- C++: C++17, raylib 6.0, сборка через `build_gui.bat` (не CMake). Исходники: `src/gui.cpp`.
- Константы действий: `A_DAY=0`, `A_WEEK=1`, `A_BUILD0=2` (`include/colony/constants.h:126-128`).
- `ColonyEnvCpp::step(int action)` возвращает `StepOut{obs, rew, terminated, truncated, days, people, money, n_bases, ...}` (`include/colony/env.h:57-67`).
- Python: `CppColonyEnv` (gym.Env), `policy(obs)→(logits, values)`, `argmax` = действие.
- PySide6 (не PyQt). Весь UI-текст на русском.
- Тесты: `python -m pytest tests/ -x -q`. Без новых зависимостей.
- IPC файлы: `actions.txt` (одно целое число = action), `state.json` (JSON-объект).
- Скорость: 0 = пауза, 1/2/5 = действий в секунду.
- Камера: авто-центр карты по умолчанию, панорама средней кнопкой мыши, зум колесом (как сейчас).
- Подсветка: текущее действие модели подсвечивается в палитре зданий (`sel_action`).
- При `terminated`: экран «Игра окончена», автосброс через 5 секунд (новый эпизод).
- Рабочая директория запуска exe = корень проекта (нужно для `assets/`, `configs/`).

---

### Task 1: C++ headless-ai mode in gui.cpp

**Files:**
- Modify: `src/gui.cpp` (добавить ~80 строк)
- No Python tests (C++ compiled separately)

**Interfaces:**
- Consumes: `ColonyEnvCpp::step(int)`, `env.game()`, `env.reset(seed)`, `env.build_data()`, existing `cam`, `sel_action`, `game_over`, `want_close`
- Produces: CLI flags `--headless-ai --actions-file <path> --state-file <path>`; GUI reads action from file, writes state JSON, highlights action in palette, auto-resets on game over

- [ ] **Step 1: Add headless-ai globals and helper functions**

In `src/gui.cpp`, after the existing `static bool want_close = false;` (line 428), add:

```cpp
// ═══ Headless AI mode (policy-driven) ═══
static bool headless_ai = false;
static std::string ai_actions_path;
static std::string ai_state_path;
static int ai_last_action = -1;   // last action written by Python
static bool ai_terminated = false;
static float ai_reset_timer = 0.0f;

static int ai_read_action() {
    // Read action from file. Returns -1 if no new action available.
    std::ifstream f(ai_actions_path);
    if (!f.is_open()) return -1;
    int action = -1;
    f >> action;
    f.close();
    // Clear the file so next read gets the new action
    std::ofstream out(ai_actions_path, std::ios::trunc);
    out.close();
    return action;
}

static void ai_write_state(const Game& g, const std::vector<float>& obs, int action, bool terminated) {
    // Write state JSON to file (includes obs for Python policy)
    std::ofstream f(ai_state_path);
    if (!f.is_open()) return;
    f << "{"
      << "\"day\":" << g.day << ","
      << "\"month\":" << g.month << ","
      << "\"year\":" << g.year << ","
      << "\"people\":" << g.people << ","
      << "\"bases\":" << g.bases.size() << ","
      << "\"money\":" << g.money << ","
      << "\"action\":" << action << ","
      << "\"terminated\":" << (terminated ? "true" : "false") << ","
      << "\"obs\":[";
    for (size_t i = 0; i < obs.size(); i++) {
        f << obs[i];
        if (i + 1 < obs.size()) f << ",";
    }
    f << "]";
    f << "}";
    f.close();
}

static void ai_reset_env(ColonyEnvCpp& env) {
    int64_t new_seed = (int64_t)GetRandomValue(1, 999999999);
    env.reset(new_seed);
    game_over = false;
    ai_terminated = false;
    ai_reset_timer = 0.0f;
    sel_bx = sel_by = -1;
    cur_dlg = DLG_NONE;
    stat_built.clear(); stat_earned = 0; stat_spent = 0;
    stat_started = false; stat_tax_over = false;
    cam.target = {(float)env.game().earth.init_sel_x * TILE,
                  (float)env.game().earth.init_sel_y * TILE};
    cam.zoom = 1.0f;
    // Clear action file so Python knows to send a new one
    std::ofstream out(ai_actions_path, std::ios::trunc);
    out.close();
}
```

- [ ] **Step 2: Parse headless-ai CLI args**

In `main()`, in the CLI arg parsing loop (lines 717-722), add:

```cpp
else if (a == "--headless-ai") headless_ai = true;
else if (a == "--actions-file" && i+1 < argc) ai_actions_path = argv[++i];
else if (a == "--state-file" && i+1 < argc) ai_state_path = argv[++i];
```

- [ ] **Step 3: Add headless-ai input handling in the game loop**

In the game loop, after the `// ─── Input ───` comment (line 760), before the mouse/keyboard handling, add:

```cpp
// ─── Headless AI mode: read action from file, step env ───
if (headless_ai) {
    if (IsKeyPressed(KEY_ESCAPE)) { want_close = true; }

    if (game_over) {
        // Auto-reset after 5 seconds
        ai_reset_timer += GetFrameTime();
        if (ai_reset_timer > 5.0f) {
            ai_reset_env(env);
        }
        // Draw game-over screen (reuse existing code below)
        // ... fall through to draw ...
    } else {
        int action = ai_read_action();
        if (action >= 0) {
            ai_last_action = action;
            sel_action = action;  // highlight in palette
            auto out = env.step(action);
            if (out.terminated) {
                game_over = true;
                ai_terminated = true;
            }
            // Write state including obs for Python policy
            std::vector<float> obs = env.obs();
            ai_write_state(env.game(), obs, action, out.terminated);
        }
    }
}
```

- [ ] **Step 4: Skip mouse/keyboard input in headless mode**

Wrap the mouse/keyboard input block (lines 760-1038) in `if (!headless_ai) { ... }`. The headless-ai block from Step 3 goes before this.

- [ ] **Step 5: Auto-center camera in headless mode**

After the camera setup (lines 731-733), add:

```cpp
if (headless_ai) {
    cam.target = {(float)g.map_size() * TILE / 2.0f, (float)g.map_size() * TILE / 2.0f};
    cam.zoom = 0.8f;
}
```

- [ ] **Step 6: Build and verify**

Run:
```
build_gui.bat
```
Expected: `Build OK: ...sakhalin_colony_gui.exe`

- [ ] **Step 7: Manual test**

Create a test action file and run the GUI:
```
echo 0 > actions.txt
sakhalin_colony_gui.exe --headless-ai --actions-file actions.txt --state-file state.json
```
Expected: Window opens, map is rendered, after a few seconds state.json appears with day/people/bases/money. Press ESC to close.

- [ ] **Step 8: Commit**

```bash
git add src/gui.cpp
git commit -m "feat: headless-ai mode in gui.cpp for policy-driven visual watch"
```

---

### Task 2: Python visual watch mode in watch_champion.py

**Files:**
- Modify: `watch_champion.py` (add ~60 lines)
- Test: `tests/test_watch_champion.py` (add tests)

**Interfaces:**
- Consumes: `policy(obs)→(logits, values)`, `CppColonyEnv`, `Path`, `subprocess`, `json`
- Produces: `--visual` flag; launches `sakhalin_colony_gui.exe --headless-ai ...`; writes `actions.txt` per step; reads `state.json`; speed control (0/1/2/5)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_watch_champion.py`:

```python
def test_visual_arg_accepted():
    """--visual should be accepted by the argument parser."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "watch_champion.py"), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "--visual" in result.stdout


def test_visual_launches_gui_exe(tmp_path):
    """When --visual is set, watch_champion launches the GUI exe."""
    import watch_champion
    importlib.reload(watch_champion)

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running
    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        watch_champion.launch_visual_watch(
            model_dir=tmp_path,
            exe_path="test_gui.exe",
            actions_file=tmp_path / "actions.txt",
            state_file=tmp_path / "state.json",
            seed=42,
            map_size=200,
        )
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert "test_gui.exe" in args
        assert "--headless-ai" in args
        assert "--actions-file" in args
        assert "--state-file" in args


def test_write_action_file(tmp_path):
    """write_action should write the action int to the file."""
    import watch_champion
    importlib.reload(watch_champion)

    action_file = tmp_path / "actions.txt"
    watch_champion.write_action(action_file, 5)

    assert action_file.exists()
    assert int(action_file.read_text().strip()) == 5


def test_read_state_file(tmp_path):
    """read_state should parse the state JSON file including obs."""
    import json
    import watch_champion
    importlib.reload(watch_champion)

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({
        "day": 10, "month": 3, "year": 2026,
        "people": 50, "bases": 5, "money": 1000,
        "action": 2, "terminated": False,
        "obs": [0.1, 0.2, 0.3, 0.4, 0.5],
    }))

    state = watch_champion.read_state(state_file)
    assert state["day"] == 10
    assert state["people"] == 50
    assert state["terminated"] is False
    assert len(state["obs"]) == 5
    assert state["obs"][0] == 0.1


def test_read_state_missing_file(tmp_path):
    """read_state should return None if file doesn't exist."""
    import watch_champion
    importlib.reload(watch_champion)

    state = watch_champion.read_state(tmp_path / "nonexistent.json")
    assert state is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_watch_champion.py::test_visual_arg_accepted tests/test_watch_champion.py::test_visual_launches_gui_exe tests/test_watch_champion.py::test_write_action_file tests/test_watch_champion.py::test_read_state_file tests/test_watch_champion.py::test_read_state_missing_file -v`
Expected: FAIL — `launch_visual_watch`, `write_action`, `read_state` don't exist

- [ ] **Step 3: Add visual watch functions**

In `watch_champion.py`, after the `read_stage_from_meta` function (line 59), add:

```python
def write_action(action_file: Path, action: int) -> None:
    """Write action int to the IPC file."""
    action_file.write_text(str(action), encoding="utf-8")


def read_state(state_file: Path) -> dict | None:
    """Read state JSON from the IPC file. Returns None if not found."""
    import json
    if not state_file.exists():
        return None
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def launch_visual_watch(
    model_dir: Path,
    exe_path: str,
    actions_file: Path,
    state_file: Path,
    seed: int,
    map_size: int,
    curriculum_stage: int | None = None,
) -> "subprocess.Popen":
    """Launch the GUI exe in headless-ai mode."""
    import subprocess
    args = [
        exe_path,
        "--headless-ai",
        "--actions-file", str(actions_file),
        "--state-file", str(state_file),
        "--seed", str(seed),
        "--map-size", str(map_size),
    ]
    if curriculum_stage is not None:
        args.extend(["--stage", str(curriculum_stage)])
    workdir = str(PROJECT_ROOT)
    return subprocess.Popen(args, cwd=workdir)
```

- [ ] **Step 4: Add --visual CLI arg**

In `main()`, add after the `--curriculum-stage` arg:

```python
parser.add_argument("--visual", action="store_true",
                    help="Open visual GUI window (requires sakhalin_colony_gui.exe)")
```

- [ ] **Step 5: Add visual watch loop**

In `main()`, after the existing console watch loop (after line 198), add:

```python
    if args.visual:
        exe_path = str(PROJECT_ROOT / "sakhalin_colony_gui.exe")
        if not Path(exe_path).exists():
            print(f"ERROR: GUI exe not found: {exe_path}")
            print("Build it first: build_gui.bat")
            sys.exit(1)

        import tempfile
        tmp_dir = Path(tempfile.gettempdir()) / "colony_watch"
        tmp_dir.mkdir(exist_ok=True)
        actions_file = tmp_dir / "actions.txt"
        state_file = tmp_dir / "state.json"

        print(f"Launching visual watch: {exe_path}")
        proc = launch_visual_watch(
            model_dir=model_dir,
            exe_path=exe_path,
            actions_file=actions_file,
            state_file=state_file,
            seed=args.seed,
            map_size=args.map_size,
            curriculum_stage=stage,
        )

        # Clean up IPC files
        for f in (actions_file, state_file):
            if f.exists():
                f.unlink()
        actions_file.touch()

        speed = args.speed if args.speed > 0 else 1.0
        print(f"Visual watch running. Speed: {speed} steps/s. Close GUI window to stop.")

        last_state_hash = None
        try:
            while proc.poll() is None:
                state = read_state(state_file)
                if state is None:
                    time.sleep(0.05)
                    continue

                if state.get("terminated"):
                    print(f"Episode terminated at day {state.get('day')}")
                    break

                # Only compute action if state changed (new obs from C++)
                state_hash = hash(tuple(state.get("obs", [])))
                if state_hash == last_state_hash:
                    time.sleep(0.05)
                    continue
                last_state_hash = state_hash

                # Compute action from policy using obs from state
                obs = state.get("obs", [])
                if obs:
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=dev)
                        obs_t = obs_t.reshape(1, -1)
                        logits, _ = policy(obs_t)
                        action = int(logits.argmax(dim=-1).item())
                    write_action(actions_file, action)

                time.sleep(1.0 / speed)

        except KeyboardInterrupt:
            pass
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
            for f in (actions_file, state_file):
                if f.exists():
                    f.unlink()
            print("Visual watch stopped.")
        return
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_watch_champion.py -v`
Expected: ALL PASS

- [ ] **Step 7: Run all tests**

Run: `python -m pytest tests/ -x -q`
Expected: 102 passed (97 existing + 5 new)

- [ ] **Step 8: Commit**

```bash
git add watch_champion.py tests/test_watch_champion.py
git commit -m "feat: visual watch mode in watch_champion.py (launch GUI exe + IPC)"
```

---

### Task 3: Qt UI visual watch toggle

**Files:**
- Modify: `train_ui/main_window.py` (add ~20 lines)

**Interfaces:**
- Consumes: `_watch_selected()`, `_watch_pid`, `_watch_log`, existing watch panel
- Produces: `self.watch_visual_chk` (QCheckBox); `--visual` flag passed to watch_champion.py

- [ ] **Step 1: Add visual checkbox to watch panel**

In `_build_left_panel` (or wherever the watch panel is built), after the existing watch settings, add:

```python
self.watch_visual_chk = QCheckBox("Визуальный режим (GUI окно)")
self.watch_visual_chk.setObjectName("watch_visual_chk")
self.watch_visual_chk.setChecked(False)
self.watch_visual_chk.setToolTip("Открыть raylib-окно с визуализацией игры")
w_layout.addWidget(self.watch_visual_chk)
```

- [ ] **Step 2: Pass --visual flag in _watch_selected**

In `_watch_selected` (line 808), after building the `args` list, add:

```python
if self.watch_visual_chk.isChecked():
    args.append("--visual")
```

- [ ] **Step 3: Verify compilation**

Run: `python -c "import py_compile; py_compile.compile('train_ui/main_window.py', doraise=True); print('OK')"`
Expected: OK

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -x -q`
Expected: 102 passed

- [ ] **Step 5: Commit**

```bash
git add train_ui/main_window.py
git commit -m "feat: visual watch toggle in Qt UI"
```

---

### Task 4: Integration test + final verification

**Files:**
- No new files (manual test)

- [ ] **Step 1: Build GUI exe**

Run: `build_gui.bat`
Expected: Build OK

- [ ] **Step 2: Run visual watch end-to-end**

```
python watch_champion.py --model-dir C:\Users\oklex\colony_runs\models\run_004 --visual --speed 2 --max-steps 50
```
Expected:
- GUI window opens
- Map is rendered
- Model plays (buildings appear, day advances)
- State.json is written
- Close window to stop

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: 102 passed

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: visual watch integration verified"
```
