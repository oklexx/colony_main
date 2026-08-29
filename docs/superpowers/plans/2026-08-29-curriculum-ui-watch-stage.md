# Curriculum UI + Watch Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add curriculum schedule table with profiles to training UI, add stage override with explanations to watch UI, and persist `curriculum_stage_at_best` in model meta.

**Architecture:** All UI changes go into `train_ui/main_window.py` (existing PySide6 pattern). `watch_champion.py` gains `--curriculum-stage` CLI arg. `async_trainer.py` writes the stage into `best_model.meta.json` at save time. No new files.

**Tech Stack:** PySide6, Python 3.14, pytest

**Spec:** User-provided task description in session (curriculum UI + watch stage + meta persistence)

## Global Constraints

- PySide6 (not PyQt) — existing pattern
- All UI text in Russian
- Stage values: 0 = disabled (all buildings), 1-3 = curriculum stages
- `curriculum_schedule` format: `list[tuple[int, int]]` = `[(threshold_steps, stage), ...]`
- Empty list `[]` = curriculum disabled
- Tests must pass: `python -m pytest tests/ -x -q`
- No new dependencies

---

### Task 1: Write `curriculum_stage_at_best` into best_model.meta.json

**Files:**
- Modify: `rl/async_trainer.py:224-235`
- Test: `tests/test_async_trainer.py`

**Interfaces:**
- Consumes: `self._curriculum_stage` (int, already tracked in `AsyncTrainer.__init__` line 66)
- Produces: `best_model.meta.json` gains key `curriculum_stage_at_best: int`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_async_trainer.py`:

```python
def test_curriculum_stage_in_meta(tmp_path):
    """best_model.meta.json must contain curriculum_stage_at_best."""
    import json
    from unittest.mock import MagicMock, patch
    from rl.async_trainer import AsyncTrainer
    from rl.config import Config

    cfg = Config(
        model_dir=str(tmp_path),
        eval_episodes=1,
        eval_freq=100,
        n_steps=128,
        batch_size=64,
        total_timesteps=200,
        curriculum_schedule=[(0, 1), (100, 2)],
    )

    mock_em = MagicMock()
    mock_em.device = torch.device("cpu")
    mock_em.cfg = cfg

    mock_ppo = MagicMock()
    mock_ppo.save = MagicMock()

    trainer = AsyncTrainer(
        cfg=cfg,
        env_manager=mock_em,
    )
    trainer.em = mock_em
    trainer.em.ppo = mock_ppo
    trainer._curriculum_stage = 2
    trainer.best_score = None

    with patch("train_ui.evaluator.run_eval") as mock_run_eval:
        mock_run_eval.return_value = {
            "days": 100.0, "bases": 10.0, "people": 50.0,
            "avg_return": 1000.0,
            "episode_days": [100.0], "episode_bases": [10.0],
            "episode_people": [50.0], "episode_returns": [1000.0],
        }
        trainer._eval(200)

    meta_path = tmp_path / "best_model.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["curriculum_stage_at_best"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_async_trainer.py::test_curriculum_stage_in_meta -v`
Expected: FAIL with `KeyError: 'curriculum_stage_at_best'`

- [ ] **Step 3: Write minimal implementation**

In `rl/async_trainer.py`, in the `_eval` method, add `curriculum_stage_at_best` to the meta dict (around line 224-235):

```python
meta = {
    "best_score": score,
    "best_days": days_agg,
    "best_bases": bases_agg,
    "best_people": people_agg,
    "best_return": return_agg,
    "score_weights": list(getattr(self.cfg, "eval_score_weights", (0.4, 3.0, 0.2, 0.0001))),
    "min_bases": min_bases,
    "min_return": min_return,
    "total_timesteps": total_done,
    "episodes": len(all_days),
    "curriculum_stage_at_best": self._curriculum_stage,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_async_trainer.py::test_curriculum_stage_in_meta -v`
Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/ -x -q`
Expected: 91 passed (90 existing + 1 new)

- [ ] **Step 6: Commit**

```bash
git add rl/async_trainer.py tests/test_async_trainer.py
git commit -m "feat: persist curriculum_stage_at_best in best_model.meta.json"
```

---

### Task 2: Add `--curriculum-stage` to `watch_champion.py`

**Files:**
- Modify: `watch_champion.py`
- Test: `tests/test_watch_champion.py` (new file)

**Interfaces:**
- Consumes: `CppColonyEnv` (already has `curriculum_stage` param in constructor), `colony_cpp.ColonyEnvCpp.set_curriculum_stage(int)`
- Produces: CLI arg `--curriculum-stage` (int, default=None). When set, calls `env.cpp_env.set_curriculum_stage(stage)` after env creation.

- [ ] **Step 1: Write the failing test**

Create `tests/test_watch_champion.py`:

```python
"""Tests for watch_champion.py --curriculum-stage handling."""
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "python"))


def _get_parser():
    """Import watch_champion and return its argparse parser."""
    import watch_champion
    importlib.reload(watch_champion)
    # Re-create the parser by calling main with --help and catching SystemExit
    # Instead, we test the stage application logic directly
    return watch_champion


def test_curriculum_stage_arg_accepted():
    """--curriculum-stage should be accepted by the argument parser."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "watch_champion.py"), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "--curriculum-stage" in result.stdout


def test_curriculum_stage_applied_to_env():
    """When --curriculum-stage is set, env.cpp_env.set_curriculum_stage is called."""
    import watch_champion
    importlib.reload(watch_champion)

    mock_env = MagicMock()
    mock_env.cpp_env = MagicMock()
    mock_env.reset.return_value = (MagicMock(), {})
    mock_env.step.return_value = (MagicMock(), 0.0, True, False, {})
    mock_env._action_names = ["Day"]

    mock_policy = MagicMock()
    mock_policy.return_value = (MagicMock(), MagicMock())

    with patch("watch_champion.CppColonyEnv", return_value=mock_env), \
         patch("watch_champion._load_policy", return_value=mock_policy), \
         patch("torch.no_grad", lambda: _nullcontext()):
        # Simulate: stage=2 should call set_curriculum_stage(2)
        watch_champion.apply_curriculum_stage(mock_env, 2)

    mock_env.cpp_env.set_curriculum_stage.assert_called_once_with(2)


def test_curriculum_stage_none_skips():
    """When stage is None, set_curriculum_stage is NOT called."""
    import watch_champion
    importlib.reload(watch_champion)

    mock_env = MagicMock()
    mock_env.cpp_env = MagicMock()

    watch_champion.apply_curriculum_stage(mock_env, None)

    mock_env.cpp_env.set_curriculum_stage.assert_not_called()


def test_curriculum_stage_zero_disables():
    """When stage is 0, set_curriculum_stage(0) is called (all buildings)."""
    import watch_champion
    importlib.reload(watch_champion)

    mock_env = MagicMock()
    mock_env.cpp_env = MagicMock()

    watch_champion.apply_curriculum_stage(mock_env, 0)

    mock_env.cpp_env.set_curriculum_stage.assert_called_once_with(0)


def test_curriculum_stage_from_meta():
    """When stage is None and meta exists, read stage from meta."""
    import json
    import watch_champion
    importlib.reload(watch_champion)

    mock_env = MagicMock()
    mock_env.cpp_env = MagicMock()

    with patch("builtins.open", create=True) as mock_open:
        mock_file = MagicMock()
        mock_file.__enter__ = lambda s: s
        mock_file.__exit__ = lambda *a: None
        mock_open.return_value = mock_file
        mock_file.read.return_value = json.dumps({"curriculum_stage_at_best": 3})
        stage = watch_champion.read_stage_from_meta(MagicMock())

    assert stage == 3


class _nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_watch_champion.py -v`
Expected: FAIL — `apply_curriculum_stage` and `read_stage_from_meta` don't exist yet; `--curriculum-stage` not in help

- [ ] **Step 3: Write minimal implementation**

In `watch_champion.py`, add two helper functions and the CLI arg:

```python
# After the TeeWriter class, before main():

def apply_curriculum_stage(env, stage):
    """Apply curriculum stage to the env. None = skip, 0 = all buildings, 1-3 = stages."""
    if stage is not None:
        env.cpp_env.set_curriculum_stage(stage)


def read_stage_from_meta(model_dir):
    """Read curriculum_stage_at_best from best_model.meta.json. Returns None if not found."""
    import json
    meta_path = model_dir / "best_model.meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("curriculum_stage_at_best")
    except (json.JSONDecodeError, OSError):
        return None
```

In `main()`, add the CLI arg (after `--log-file`):

```python
parser.add_argument("--curriculum-stage", type=int, default=None,
                    help="Override curriculum stage (0=all buildings, 1-3). "
                         "If not set, reads from best_model.meta.json")
```

After env creation (after `env = CppColonyEnv(...)`), add:

```python
stage = args.curriculum_stage
if stage is None:
    stage = read_stage_from_meta(model_dir)
if stage is not None:
    apply_curriculum_stage(env, stage)
    print(f"Curriculum stage: {stage}")
else:
    print("Curriculum stage: not set (all buildings)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_watch_champion.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/ -x -q`
Expected: 96 passed (91 + 5 new)

- [ ] **Step 6: Commit**

```bash
git add watch_champion.py tests/test_watch_champion.py
git commit -m "feat: watch_champion.py --curriculum-stage with meta fallback"
```

---

### Task 3: Add curriculum table + profiles to training UI

**Files:**
- Modify: `train_ui/main_window.py` (center panel, `_collect_config`, `save_state`, `_restore_state`, `_reset_params`)

**Interfaces:**
- Consumes: existing `_build_center_panel`, `_collect_config`, `save_state`, `_restore_state`
- Produces: `self._get_curriculum_data() -> list[tuple[int,int]]`, `self.curriculum_table`, `self.curriculum_profile_combo`, `self.btn_cur_add/del/clear/save/load`

**Profiles:**
- "Отключён" → `[]`
- "Стандартный" → `[(0,1), (5_000_000,2), (15_000_000,3)]`
- "Быстрый" → `[(0,2), (3_000_000,3)]`
- "Медленный" → `[(0,1), (10_000_000,2), (30_000_000,3)]`

- [ ] **Step 1: Add curriculum UI widgets to `_build_center_panel`**

In `train_ui/main_window.py`, in `_build_center_panel`, after the `checks` layout (line ~306) and before the `actions` layout, insert:

```python
# --- Curriculum ---
curr_box = QGroupBox("Автокурикулум (расписание этапов)")
curr_box.setObjectName("curriculum_box")
curr_layout = QVBoxLayout(curr_box)

self.curriculum_table = QTableWidget(0, 2)
self.curriculum_table.setObjectName("curriculum_table")
self.curriculum_table.setHorizontalHeaderLabels(["Порог шагов", "Этап (1-3)"])
self.curriculum_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
self.curriculum_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
self.curriculum_table.setMinimumHeight(120)
self.curriculum_table.setMaximumHeight(200)
curr_layout.addWidget(self.curriculum_table)

curr_btns = QHBoxLayout()
self.btn_cur_add = QPushButton("Добавить")
self.btn_cur_add.setObjectName("btn_cur_add")
self.btn_cur_add.clicked.connect(self._cur_add_row)
self.btn_cur_del = QPushButton("Удалить")
self.btn_cur_del.setObjectName("btn_cur_del")
self.btn_cur_del.clicked.connect(self._cur_del_row)
self.btn_cur_clear = QPushButton("Очистить")
self.btn_cur_clear.setObjectName("btn_cur_clear")
self.btn_cur_clear.clicked.connect(self._cur_clear)
curr_btns.addWidget(self.btn_cur_add)
curr_btns.addWidget(self.btn_cur_del)
curr_btns.addWidget(self.btn_cur_clear)
curr_btns.addStretch(1)
curr_layout.addLayout(curr_btns)

curr_profile_row = QHBoxLayout()
profile_lbl = QLabel("Профиль:")
profile_lbl.setObjectName("curriculum_profile_label")
self.curriculum_profile_combo = QComboBox()
self.curriculum_profile_combo.setObjectName("curriculum_profile_combo")
self.curriculum_profile_combo.addItems(["Отключён", "Стандартный", "Быстрый", "Медленный"])
self.curriculum_profile_combo.setCurrentIndex(1)  # Стандартный
self.curriculum_profile_combo.currentIndexChanged.connect(self._cur_profile_changed)
curr_profile_row.addWidget(profile_lbl)
curr_profile_row.addWidget(self.curriculum_profile_combo)
curr_profile_row.addStretch(1)
curr_layout.addLayout(curr_profile_row)

curr_save_row = QHBoxLayout()
self.btn_cur_save = QPushButton("Сохранить")
self.btn_cur_save.setObjectName("btn_cur_save")
self.btn_cur_save.setToolTip("Сохранить расписание в JSON-файл")
self.btn_cur_save.clicked.connect(self._cur_save)
self.btn_cur_load = QPushButton("Загрузить")
self.btn_cur_load.setObjectName("btn_cur_load")
self.btn_cur_load.setToolTip("Загрузить расписание из JSON-файла")
self.btn_cur_load.clicked.connect(self._cur_load)
curr_save_row.addWidget(self.btn_cur_save)
curr_save_row.addWidget(self.btn_cur_load)
curr_save_row.addStretch(1)
curr_layout.addLayout(curr_save_row)

layout.addWidget(curr_box)
```

Also add `QComboBox` to the imports at the top of the file (line 14-20):

```python
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)
```

- [ ] **Step 2: Add curriculum helper methods**

After `_build_right_panel` method (or before `_on_param_changed`), add:

```python
# ---------- curriculum ----------

CURRICULUM_PROFILES: Dict[str, List[List[int]]] = {
    "Отключён": [],
    "Стандартный": [[0, 1], [5_000_000, 2], [15_000_000, 3]],
    "Быстрый": [[0, 2], [3_000_000, 3]],
    "Медленный": [[0, 1], [10_000_000, 2], [30_000_000, 3]],
}

def _cur_add_row(self, threshold: int = 0, stage: int = 1):
    row = self.curriculum_table.rowCount()
    self.curriculum_table.insertRow(row)
    self.curriculum_table.setItem(row, 0, QTableWidgetItem(str(threshold)))
    self.curriculum_table.setItem(row, 1, QTableWidgetItem(str(stage)))

def _cur_del_row(self):
    row = self.curriculum_table.currentRow()
    if row >= 0:
        self.curriculum_table.removeRow(row)

def _cur_clear(self):
    self.curriculum_table.setRowCount(0)

def _cur_profile_changed(self, _idx: int):
    name = self.curriculum_profile_combo.currentText()
    profile = self.CURRICULUM_PROFILES.get(name, [])
    self._cur_clear()
    for th, st in profile:
        self._cur_add_row(th, st)

def _cur_save(self):
    path, _ = QFileDialog.getSaveFileName(
        self, "Сохранить расписание", "curriculum.json", "JSON (*.json)"
    )
    if not path:
        return
    data = self._get_curriculum_data()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    self.log("info", f"Расписание сохранено: {path}")

def _cur_load(self):
    path, _ = QFileDialog.getOpenFileName(
        self, "Загрузить расписание", "", "JSON (*.json)"
    )
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            self.log("error", "Файл должен содержать список пар [порог, этап]")
            return
        self._cur_clear()
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                self._cur_add_row(int(item[0]), int(item[1]))
        self.log("info", f"Расписание загружено: {path}")
    except (json.JSONDecodeError, OSError, ValueError) as e:
        self.log("error", f"Не удалось загрузить: {e}")

def _get_curriculum_data(self) -> List[List[int]]:
    data: List[List[int]] = []
    for row in range(self.curriculum_table.rowCount()):
        th_item = self.curriculum_table.item(row, 0)
        st_item = self.curriculum_table.item(row, 1)
        if th_item and st_item:
            try:
                th = int(th_item.text())
                st = int(st_item.text())
                if 1 <= st <= 3:
                    data.append([th, st])
            except ValueError:
                continue
    data.sort(key=lambda x: x[0])
    return data

def _validate_curriculum(self, data: List[List[int]]) -> tuple[bool, str]:
    for i in range(1, len(data)):
        if data[i][0] <= data[i - 1][0]:
            return False, f"Порог {data[i][0]:,} должен быть больше предыдущего {data[i-1][0]:,}"
    return True, "OK"
```

- [ ] **Step 3: Initialize curriculum table with default profile**

In `_build_center_panel`, after adding `curr_box` to layout, call:

```python
self._cur_profile_changed(1)  # Стандартный
```

- [ ] **Step 4: Wire into `_collect_config`**

In `_collect_config` (line ~775), add before `return cfg`:

```python
curriculum = self._get_curriculum_data()
if curriculum:
    ok, msg = self._validate_curriculum(curriculum)
    if not ok:
        QMessageBox.warning(self, "Курикулум", msg)
        curriculum = []
cfg["curriculum_schedule"] = curriculum
```

- [ ] **Step 5: Wire into `save_state` and `_restore_state`**

In `save_state` (line ~992), add before `return state`:

```python
state["curriculum_schedule"] = self._get_curriculum_data()
```

In `_restore_state` (line ~971), add after the existing restore logic:

```python
curriculum = cfg.get("curriculum_schedule")
if curriculum is not None:
    self._cur_clear()
    for item in curriculum:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            self._cur_add_row(int(item[0]), int(item[1]))
else:
    self._cur_profile_changed(1)
```

- [ ] **Step 6: Wire into `_reset_params`**

In `_reset_params` (line ~959), add before the log line:

```python
self.curriculum_profile_combo.blockSignals(True)
self.curriculum_profile_combo.setCurrentIndex(1)  # Стандартный
self.curriculum_profile_combo.blockSignals(False)
self._cur_profile_changed(1)
```

- [ ] **Step 7: Verify compilation**

Run: `python -c "import py_compile; py_compile.compile('train_ui/main_window.py', doraise=True); print('OK')"`
Expected: OK

- [ ] **Step 8: Run all tests**

Run: `python -m pytest tests/ -x -q`
Expected: 96 passed

- [ ] **Step 9: Commit**

```bash
git add train_ui/main_window.py
git commit -m "feat: curriculum schedule table with profiles in training UI"
```

---

### Task 4: Add watch stage override UI with explanations

**Files:**
- Modify: `train_ui/main_window.py` (left panel, `_watch_selected`)

**Interfaces:**
- Consumes: `self._selected_model()`, `watch_champion.py --curriculum-stage`
- Produces: `self.watch_use_model_stage` (QCheckBox), `self.watch_override_check` (QCheckBox), `self.watch_stage_combo` (QComboBox), `self.watch_stage_label` (QLabel)

- [ ] **Step 1: Add watch settings group to left panel**

In `_build_left_panel`, after the `btns` layout (after `layout.addLayout(btns)`) and before `layout.addWidget(self.models_hint)`, insert:

```python
# --- Watch settings ---
watch_settings = QGroupBox("Настройки наблюдения (для диагностики)")
watch_settings.setObjectName("watch_settings")
ws_layout = QVBoxLayout(watch_settings)

self.watch_use_model_stage = QCheckBox("Использовать stage модели (по умолчанию)")
self.watch_use_model_stage.setObjectName("watch_use_model_stage")
self.watch_use_model_stage.setChecked(True)
self.watch_use_model_stage.setToolTip(
    "Рекомендуется для обычного просмотра. "
    "Использует этап, на котором модель была сохранена."
)
ws_layout.addWidget(self.watch_use_model_stage)

label_use = QLabel("→ Используется этап, на котором модель была сохранена. Обычно этого достаточно.")
label_use.setObjectName("watch_use_label")
label_use.setStyleSheet("color: gray; font-size: 9pt; margin-left: 20px;")
ws_layout.addWidget(label_use)

override_row = QHBoxLayout()
self.watch_override_check = QCheckBox("Переопределить stage:")
self.watch_override_check.setObjectName("watch_override_check")
self.watch_override_check.setChecked(False)
self.watch_override_check.setToolTip(
    "Принудительно запустить модель на указанном этапе. "
    "Для диагностики: проверьте поведение модели на разных этапах курикулума."
)
override_row.addWidget(self.watch_override_check)

self.watch_stage_combo = QComboBox()
self.watch_stage_combo.setObjectName("watch_stage_combo")
self.watch_stage_combo.addItems(["Stage 1", "Stage 2", "Stage 3", "Отключить (все здания)"])
self.watch_stage_combo.setEnabled(False)
self.watch_stage_combo.setToolTip(
    "Для диагностики: проверьте, как модель ведёт себя с ограниченным набором зданий "
    "(stage 1) или на полном наборе (stage 3)."
)
override_row.addWidget(self.watch_stage_combo)
ws_layout.addLayout(override_row)

label_override = QLabel(
    "→ Например, если модель обучена на всех зданиях, "
    "посмотрите, как она справляется только с базовыми."
)
label_override.setObjectName("watch_override_label")
label_override.setStyleSheet("color: gray; font-size: 9pt; margin-left: 20px;")
ws_layout.addWidget(label_override)

self.watch_stage_label = QLabel("Stage модели: —")
self.watch_stage_label.setObjectName("watch_stage_label")
ws_layout.addWidget(self.watch_stage_label)

layout.addWidget(watch_settings)

# Connect override check to enable/disable combo
self.watch_override_check.toggled.connect(self._on_watch_override_toggled)
self.watch_use_model_stage.toggled.connect(self._on_watch_use_model_toggled)
```

- [ ] **Step 2: Add watch toggle handlers**

```python
def _on_watch_override_toggled(self, checked: bool):
    self.watch_stage_combo.setEnabled(checked)

def _on_watch_use_model_toggled(self, checked: bool):
    self.watch_override_check.setEnabled(not checked)
    self.watch_stage_combo.setEnabled(not checked and self.watch_override_check.isChecked())
    if checked:
        self.watch_override_check.setChecked(False)
```

- [ ] **Step 3: Update `_watch_selected` to pass stage**

In `_watch_selected`, after building the `args` list, add the stage logic:

```python
# Determine curriculum stage
use_model_stage = self.watch_use_model_stage.isChecked()
override = self.watch_override_check.isChecked()

if override:
    idx = self.watch_stage_combo.currentIndex()
    # 0=Stage1, 1=Stage2, 2=Stage3, 3=All
    stage = 0 if idx == 3 else (idx + 1)
    args.extend(["--curriculum-stage", str(stage)])
elif use_model_stage:
    # Read from meta
    import json as _json
    meta_path = model_dir / "best_model.meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = _json.load(f)
            stage = meta.get("curriculum_stage_at_best")
            if stage is not None:
                args.extend(["--curriculum-stage", str(stage)])
        except (json.JSONDecodeError, OSError):
            pass
# else: no stage → watch_champion reads from meta itself
```

- [ ] **Step 4: Update model stage label on selection**

Add a method and connect it:

```python
def _update_watch_stage_label(self):
    m = self._selected_model()
    if m is None:
        self.watch_stage_label.setText("Stage модели: —")
        return
    import json as _json
    meta_path = m.model_file.parent / "best_model.meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = _json.load(f)
            stage = meta.get("curriculum_stage_at_best", "неизвестно")
            self.watch_stage_label.setText(f"Stage модели: {stage}")
            return
        except (json.JSONDecodeError, OSError):
            pass
    self.watch_stage_label.setText("Stage модели: неизвестно")
```

Connect in `_build_left_panel` after adding `watch_settings`:

```python
self.model_table.cellClicked.connect(self._update_watch_stage_label)
```

Also call it in `_on_model_clicked` (existing method) and `refresh_models`.

- [ ] **Step 5: Verify compilation**

Run: `python -c "import py_compile; py_compile.compile('train_ui/main_window.py', doraise=True); print('OK')"`
Expected: OK

- [ ] **Step 6: Run all tests**

Run: `python -m pytest tests/ -x -q`
Expected: 96 passed

- [ ] **Step 7: Commit**

```bash
git add train_ui/main_window.py
git commit -m "feat: watch stage override UI with diagnostic explanations"
```

---

### Task 5: Final verification + push

**Files:** All modified files

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS (96 tests)

- [ ] **Step 2: Verify watch_champion --help**

Run: `python watch_champion.py --help`
Expected: `--curriculum-stage` visible in help

- [ ] **Step 3: Commit and push**

```bash
git add -A
git commit -m "feat: curriculum UI + watch stage override + meta persistence"
git push origin master
```
