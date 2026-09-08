"""UI 2.0 main window.

Layout principles (vs the old UI):
  * thematic tabs instead of one scrolling wall;
  * every metric has exactly ONE home (no duplicated readouts);
  * 22px controls, 11px font, 4px grids — nothing overlaps;
  * live charts + KPI cards + eval tracking (old UI never showed eval);
  * separate state file → runs in parallel with the old UI.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from train_ui import protocol as P
from train_ui.models import ModelInfo, ModelRegistry
from rl.config import RewardConfig as _RC

from train_ui2 import theme as T
from train_ui2.charts import Bars, Chart
from train_ui2.controls import ParamGroup, StatCard

CONFIG_PATH = Path.home() / "colony_runs" / "sakhalin_colony_ui2" / "config.json"
CONFIG_VERSION = 1

# eval lines look like:
# [Eval @ 1,048,576] days=731.0 people=69.0 bases=5.0 return=-3408.7 score=85.00 ...
_EVAL_RE = re.compile(
    r"\[Eval @ ([\d,]+)\] days=([\d.]+) people=([\d.]+) bases=([\d.]+) "
    r"return=(-?[\d.]+) score=(-?[\d.]+)")
_BEST_RE = re.compile(
    r"\[Best\] Saved best_model\.pt \(score=([\d.]+), days=([\d.]+), bases=([\d.]+)\)")

PARAM_GROUPS = {
    "Среда": ["n_envs", "map_size", "seed", "total_timesteps"],
    "PPO": ["learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef",
            "vf_coef", "max_grad_norm", "target_kl", "n_steps", "batch_size",
            "n_epochs"],
    "Оценка и сохранение": ["eval_freq", "eval_episodes", "eval_min_days",
                            "eval_min_bases", "save_freq",
                            "early_stopping_patience"],
}
REWARD_GROUPS = {
    "Стройка": ["build_bonus", "chain_bonus", "chain_daily", "novelty",
                "diversity_bonus", "proximity_bonus", "build_cost_penalty"],
    "Экономика": ["daily_income", "sale_bonus", "tax_daily_bonus",
                  "manual_tax_penalty", "debt_coeff"],
    "Выживание": ["survival_bonus", "survival_coeff", "game_over_penalty",
                  "death_penalty", "tax_fail_penalty", "base_lost_penalty",
                  "born_bonus", "home_overflow_penalty"],
    "Потребности": ["housing_need_bonus", "food_need_bonus", "water_need_bonus"],
    "Дисциплина": ["error_penalty", "preserve_penalty", "demolish_penalty",
                   "idle_build_penalty", "idle_build_threshold_days"],
    "Milestones и клип": ["milestone_base_bonus", "milestone_people_bonus",
                          "milestone_day_bonus", "milestone_year_bonus",
                          "clip_reward_min", "clip_reward_max"],
}
# boolean ablation switches: off in the v2 profile
REWARD_FLAGS = [
    ("disable_daily_income", "Выкл. daily_income",
     "Не начислять ежедневный доход (абляция)"),
    ("disable_net_worth", "Выкл. net worth бонус",
     "Убрать бонус за чистую стоимость (абляция)"),
    ("disable_provider_bonus", "Выкл. provider бонус",
     "Убрать бонус провайдера потребностей (абляция)"),
]


def _fmt_steps(n: float) -> str:
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1000:.0f}k"
    return str(n)


class MainWindow2(QMainWindow):
    def __init__(self, models_dir: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("Sakhalin Colony — Training UI 2.0")
        self.resize(1180, 760)
        self.registry = ModelRegistry(models_dir)
        self.config: Dict[str, Any] = self._load_state()

        # worker state
        self._train_proc: Optional[subprocess.Popen] = None
        self._msg_file: Optional[str] = None
        self._msg_offset = 0
        self._cmd_file: Optional[str] = None
        self._cfg_tmp: Optional[str] = None
        self._msg_timer = QTimer(self)
        self._msg_timer.timeout.connect(self._poll_worker)
        # watch state
        self._watch_proc: Optional[subprocess.Popen] = None
        self._watch_log: Optional[str] = None
        self._watch_offset = 0
        self._watch_timer = QTimer(self)
        self._watch_timer.timeout.connect(self._poll_watch)
        self._extra_cfg: Dict[str, Any] = {}

        self._build_ui()
        self._restore_state()
        self._refresh_models()

    # ─────────────────────────── UI construction ───────────────────────────

    def _build_ui(self):
        split = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_training(), "Обучение")
        self.tabs.addTab(self._tab_monitor(), "Мониторинг")
        self.tabs.addTab(self._tab_rewards(), "Награды")
        self.tabs.addTab(self._tab_models(), "Модели")
        self.tabs.addTab(self._tab_watch(), "Наблюдение")
        split.addWidget(self.tabs)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(6000)
        self.log_view.setStyleSheet("font-family: Consolas, monospace; font-size:10px;")
        split.addWidget(self.log_view)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setSizes([560, 180])
        self.setCentralWidget(split)
        self.statusBar().showMessage("Готово. Старый UI можно запускать параллельно.")

    # ── Tab: Обучение ──
    def _tab_training(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # run bar
        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(T.label("Имя:", T.DIM))
        self.name_edit = QLineEdit()
        self.name_edit.setFixedWidth(150)
        self.name_edit.setPlaceholderText("run_001")
        bar.addWidget(self.name_edit)
        self.btn_start = T.button("▶ Старт", self._start_training, "primary",
                                  "Запустить обучение с текущими параметрами")
        self.btn_pause = T.button("⏸ Пауза", self._toggle_pause)
        self.btn_pause.setEnabled(False)
        self.btn_stop = T.button("■ Стоп", self._stop_training, "danger")
        self.btn_stop.setEnabled(False)
        self.btn_entropy = T.button("Энтропия ×2", self._boost_entropy,
                                    tooltip="Удвоить ent_coef (борьба со схлопыванием)")
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_pause)
        bar.addWidget(self.btn_stop)
        bar.addWidget(self.btn_entropy)
        bar.addStretch(1)
        self.run_status = T.label("остановлено", T.DIM)
        bar.addWidget(self.run_status)
        root.addLayout(bar)

        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m шагов  (%p%)")
        root.addWidget(self.progress)

        # parameter groups in 3 columns
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        cols = QHBoxLayout()
        cols.setSpacing(8)
        col_w: List[QVBoxLayout] = [QVBoxLayout(), QVBoxLayout(), QVBoxLayout()]
        self.pgroups: Dict[str, ParamGroup] = {}
        for i, (title, keys) in enumerate(PARAM_GROUPS.items()):
            g = ParamGroup(title, keys)
            g.value_changed.connect(self._on_param_changed)
            self.pgroups[title] = g
            col_w[i % 3].addWidget(g)
        # difficulty / obs_mode / stage rows appended into the Среда group
        env_box = self.pgroups["Среда"]
        grp_grid = env_box.layout()  # QVBoxLayout (outer)
        inner = grp_grid.itemAt(0).widget().layout()  # QGridLayout of group
        row = inner.rowCount()
        inner.addWidget(T.field_label("Сложность", "light = ×2 деньги, без главного налога, до 1950"), row, 0)
        self.cmb_difficulty = T.combo(["normal", "light"], "normal")
        inner.addWidget(self.cmb_difficulty, row, 1, Qt.AlignLeft)
        row += 1
        inner.addWidget(T.field_label("Режим obs", "flat=MLP 246, minimap=CNN, hybrid=оба"), row, 0)
        self.cmb_obs_mode = T.combo(["flat", "minimap", "hybrid"], "flat")
        inner.addWidget(self.cmb_obs_mode, row, 1, Qt.AlignLeft)
        row += 1
        inner.addWidget(T.field_label("Миникарта R", "Радиус миникарты (сетка 2R+1)"), row, 0)
        self.spn_mm_radius = QSpinBox(); self.spn_mm_radius.setRange(4, 32)
        self.spn_mm_radius.setValue(14); self.spn_mm_radius.setFixedWidth(92)
        inner.addWidget(self.spn_mm_radius, row, 1, Qt.AlignLeft)
        row += 1
        inner.addWidget(T.field_label("Курикулум", "0 = все здания, 1–5 = ограниченные наборы"), row, 0)
        self.cmb_stage = T.combo([0, 1, 2, 3, 4, 5], 0)
        inner.addWidget(self.cmb_stage, row, 1, Qt.AlignLeft)
        row += 1
        inner.addWidget(T.field_label("Карта", "Каждый старт — новый seed (новая карта)"), row, 0)
        seed_bar = QHBoxLayout()
        seed_bar.setSpacing(4)
        self.btn_seed_dice = T.button("🎲", self._randomize_seed,
                                      tooltip="Случайный seed карты")
        self.chk_rand_seed = T.check("случайный при старте", True,
                                     "Генерировать новый seed при каждом запуске обучения")
        seed_bar.addWidget(self.btn_seed_dice)
        seed_bar.addWidget(self.chk_rand_seed)
        seed_bar.addStretch(1)
        seed_w = QWidget(); seed_w.setLayout(seed_bar)
        inner.addWidget(seed_w, row, 1, Qt.AlignLeft)

        # network + performance group (custom widgets)
        perf = T.group("Сеть и производительность")
        pg = perf.layout()
        pg.addWidget(T.field_label("Слоёв"), 0, 0)
        self.spn_layers = QSpinBox(); self.spn_layers.setRange(1, 4)
        self.spn_layers.setValue(2); self.spn_layers.setFixedWidth(92)
        pg.addWidget(self.spn_layers, 0, 1, Qt.AlignLeft)
        pg.addWidget(T.field_label("Ширина слоя"), 1, 0)
        self.spn_width = QSpinBox(); self.spn_width.setRange(32, 2048)
        self.spn_width.setSingleStep(32); self.spn_width.setValue(256)
        self.spn_width.setFixedWidth(92)
        pg.addWidget(self.spn_width, 1, 1, Qt.AlignLeft)
        self.chk_amp = T.check("AMP (bfloat16)", True,
                               "Смешанная точность в update() на CUDA")
        pg.addWidget(self.chk_amp, 2, 0, 1, 2)
        self.chk_compile = T.check("torch.compile", False,
                                   "Только Linux/CUDA; на Windows автооткат в eager")
        pg.addWidget(self.chk_compile, 3, 0, 1, 2)
        pg.addWidget(T.field_label("Потоки C++", "0 = авто"), 4, 0)
        self.spn_cpp_threads = QSpinBox(); self.spn_cpp_threads.setRange(0, 64)
        self.spn_cpp_threads.setValue(0); self.spn_cpp_threads.setFixedWidth(92)
        pg.addWidget(self.spn_cpp_threads, 4, 1, Qt.AlignLeft)
        col_w[2].addWidget(perf)
        for c in col_w:
            c.addStretch(1)
            w = QWidget(); w.setLayout(c)
            cols.addWidget(w, 1)
        scroll_w = QWidget(); scroll_w.setLayout(cols)
        scroll.setWidget(scroll_w)
        root.addWidget(scroll, 1)
        return page

    # ── Tab: Мониторинг ──
    def _tab_monitor(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        cards = QHBoxLayout()
        cards.setSpacing(6)
        self.card_steps = StatCard("Шаги", "0")
        self.card_fps = StatCard("FPS", "—", T.OK)
        self.card_eps = StatCard("Эпизоды", "0")
        self.card_score = StatCard("Лучший eval score", "—", "#c586c0")
        self.card_days = StatCard("Eval дни (мед.)", "—", T.OK)
        self.card_bases = StatCard("Eval базы (мед.)", "—", T.ACCENT)
        self.card_ent = StatCard("Энтропия", "—", "#9cdcfe")
        self.card_kl = StatCard("KL", "—", T.ERR)
        for c in (self.card_steps, self.card_fps, self.card_eps, self.card_score,
                  self.card_days, self.card_bases, self.card_ent, self.card_kl):
            cards.addWidget(c)
        root.addLayout(cards)

        from PySide6.QtWidgets import QGridLayout
        g = QGridLayout()
        g.setSpacing(6)
        self.chart_fps = Chart("шаги/сек", 600)
        self.chart_fps.add_series("fps")
        self.chart_ret = Chart("return эпизодов (последние 50)", 600, y_zero_line=True)
        for s in ("median", "avg", "min", "max"):
            self.chart_ret.add_series(s)
        self.chart_loss = Chart("loss", 600, y_zero_line=True)
        self.chart_loss.add_series("policy_loss")
        self.chart_loss.add_series("value_loss")
        self.chart_ent = Chart("энтропия / KL", 600)
        self.chart_ent.add_series("entropy")
        self.chart_ent.add_series("kl")
        self.chart_eval = Chart("eval: медиана дней (порог 730)", 200)
        self.chart_eval.add_series("days")
        self.chart_eval.add_series("bases")
        g.addWidget(self.chart_fps, 0, 0)
        g.addWidget(self.chart_ret, 0, 1)
        g.addWidget(self.chart_loss, 1, 0)
        g.addWidget(self.chart_ent, 1, 1)
        g.addWidget(self.chart_eval, 2, 0, 1, 2)
        root.addLayout(g, 1)

        bottom = QHBoxLayout()
        self.bars_actions = Bars()
        bottom.addWidget(self.bars_actions, 3)
        side = QVBoxLayout()
        self.lbl_loop = T.label("циклы: нет", T.DIM)
        self.lbl_curric = T.label("курикулум: —", T.DIM)
        self.lbl_thresh = T.label("пороги eval: —", T.DIM)
        side.addWidget(self.lbl_loop)
        side.addWidget(self.lbl_curric)
        side.addWidget(self.lbl_thresh)
        side.addStretch(1)
        bottom.addLayout(side, 1)
        root.addLayout(bottom)
        return page

    # ── Tab: Награды ──
    def _tab_rewards(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        bar = QHBoxLayout()
        bar.addWidget(T.label(
            "Профиль v2 = дефолты rl/config.py (проверен свипами). "
            "Правки сохраняются между запусками.", T.DIM))
        bar.addStretch(1)
        bar.addWidget(T.button("Сбросить к v2", self._reset_rewards_v2,
                               tooltip="Вернуть всем наградам дефолты из rl/config.py"))
        bar.addWidget(T.button("Загрузить JSON…", self._load_reward_json))
        bar.addWidget(T.button("Сохранить JSON…", self._save_reward_json))
        root.addLayout(bar)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout(); grid.setSpacing(8)
        self.rgroups: Dict[str, ParamGroup] = {}
        for i, (title, keys) in enumerate(REWARD_GROUPS.items()):
            g = ParamGroup(title, keys)
            g.value_changed.connect(self._on_param_changed)
            self.rgroups[title] = g
            grid.addWidget(g, i // 2, i % 2)
        # boolean ablation switches (in v2 profile all are off)
        flags_box = T.group("Отключение подсистем (абляции)")
        self.rflags: Dict[str, Any] = {}
        for i, (key, label, tip) in enumerate(REWARD_FLAGS):
            chk = T.check(label, False, tip)
            chk.stateChanged.connect(self._on_param_changed)
            self.rflags[key] = chk
            flags_box.layout().addWidget(chk, i, 0, 1, 2)
        grid.addWidget(flags_box, (len(REWARD_GROUPS) + 1) // 2, 1)
        grid.setRowStretch((len(REWARD_GROUPS) + 1) // 2 + 1, 1)
        w = QWidget(); w.setLayout(grid)
        scroll.setWidget(w)
        root.addWidget(scroll, 1)
        return page

    # ── Tab: Модели ──
    def _tab_models(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        bar = QHBoxLayout()
        bar.addWidget(T.button("⟳ Обновить", self._refresh_models))
        bar.addWidget(T.button("👁 Наблюдать", self._watch_selected_model, "primary"))
        bar.addWidget(T.button("🎓 Дообучить", self._finetune_selected,
                               tooltip="Старт с весами выбранной модели (--resume-model)"))
        bar.addWidget(T.button("📂 Папка", self._open_selected_folder))
        bar.addWidget(T.button("🗑 Удалить", self._delete_selected, "danger"))
        bar.addStretch(1)
        root.addLayout(bar)
        self.tbl_models = QTableWidget(0, 7)
        self.tbl_models.setHorizontalHeaderLabels(
            ["имя", "шаги", "eval score", "eval дни", "eval базы", "эпизоды", "создана"])
        self.tbl_models.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_models.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_models.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_models.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_models.verticalHeader().setVisible(False)
        self.tbl_models.itemSelectionChanged.connect(self._sync_watch_model)
        root.addWidget(self.tbl_models, 1)
        return page

    # ── Tab: Наблюдение ──
    def _tab_watch(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        bar = QHBoxLayout()
        bar.addWidget(T.label("Модель:", T.DIM))
        self.cmb_watch_model = QComboBox()
        self.cmb_watch_model.setMinimumWidth(160)
        bar.addWidget(self.cmb_watch_model)
        bar.addWidget(T.label("Карта (seed):", T.DIM))
        self.spn_watch_seed = QSpinBox()
        self.spn_watch_seed.setRange(0, 999_999_999)
        self.spn_watch_seed.setValue(0)
        self.spn_watch_seed.setFixedWidth(110)
        self.spn_watch_seed.setToolTip("0 = случайная карта при каждом запуске")
        bar.addWidget(self.spn_watch_seed)
        bar.addWidget(T.button("🎲", self._random_watch_seed, tooltip="Случайный seed"))
        bar.addWidget(T.label("Размер:", T.DIM))
        self.spn_watch_map = QSpinBox(); self.spn_watch_map.setRange(100, 500)
        self.spn_watch_map.setValue(280); self.spn_watch_map.setFixedWidth(70)
        self.spn_watch_map.setToolTip("Должен совпадать с тренировочным map_size!")
        bar.addWidget(self.spn_watch_map)
        bar.addWidget(T.label("Скорость:", T.DIM))
        self.cmb_watch_speed = T.combo(["1", "3", "5", "10", "max"], "5")
        self.cmb_watch_speed.setFixedWidth(60)
        bar.addWidget(self.cmb_watch_speed)
        self.chk_watch_visual = T.check("GUI-окно", True,
                                        "raylib-окно игры (нужен sakhalin_colony_gui.exe)")
        bar.addWidget(self.chk_watch_visual)
        self.btn_watch = T.button("👁 Наблюдать", self._toggle_watch, "primary")
        bar.addWidget(self.btn_watch)
        bar.addStretch(1)
        root.addLayout(bar)

        cards = QHBoxLayout()
        self.w_day = StatCard("День", "—")
        self.w_money = StatCard("Касса", "—", T.OK)
        self.w_people = StatCard("Люди", "—")
        self.w_bases = StatCard("Базы", "—", T.ACCENT)
        self.w_action = StatCard("Действие", "—", "#dcdcaa")
        for c in (self.w_day, self.w_money, self.w_people, self.w_bases, self.w_action):
            cards.addWidget(c)
        root.addLayout(cards)

        self.chart_watch = Chart("награда за шаг", 400, y_zero_line=True)
        self.chart_watch.add_series("reward")
        root.addWidget(self.chart_watch, 1)
        return page

    # ─────────────────────────── state ───────────────────────────

    def _load_state(self) -> Dict[str, Any]:
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self):
        cfg = self._collect_config()
        cfg["config_version"] = CONFIG_VERSION
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def _restore_state(self):
        cfg = self.config
        self.name_edit.setText(str(cfg.get("model_name", "")))
        for g in list(self.pgroups.values()) + list(self.rgroups.values()):
            g.set_values(cfg)
        for key, chk in self.rflags.items():
            chk.setChecked(bool(cfg.get(key, False)))
        idx = self.cmb_difficulty.findText(str(cfg.get("difficulty", "normal")))
        if idx >= 0:
            self.cmb_difficulty.setCurrentIndex(idx)
        idx = self.cmb_obs_mode.findText(str(cfg.get("obs_mode", "flat")))
        if idx >= 0:
            self.cmb_obs_mode.setCurrentIndex(idx)
        self.spn_mm_radius.setValue(int(cfg.get("minimap_radius", 14)))
        self.cmb_stage.setCurrentIndex(int(cfg.get("curriculum_stage", 0)))
        net = cfg.get("net_arch", [256, 256])
        if isinstance(net, list) and net:
            self.spn_layers.setValue(len(net))
            self.spn_width.setValue(int(net[0]))
        self.chk_amp.setChecked(bool(cfg.get("use_amp", True)))
        self.chk_compile.setChecked(bool(cfg.get("torch_compile", False)))
        self.spn_cpp_threads.setValue(int(cfg.get("cpp_threads", 0)))
        self.spn_watch_map.setValue(int(cfg.get("watch_map_size", 280)))
        self.spn_watch_seed.setValue(int(cfg.get("watch_seed", 0)))
        self.chk_watch_visual.setChecked(bool(cfg.get("watch_visual", True)))
        idx = self.cmb_watch_speed.findText(str(cfg.get("watch_speed", "5")))
        if idx >= 0:
            self.cmb_watch_speed.setCurrentIndex(idx)
        self._extra_cfg = {k: v for k, v in cfg.get("extra", {}).items()}

    def _collect_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = dict(self._extra_cfg)
        params: Dict[str, Any] = {}
        for g in self.pgroups.values():
            params.update(g.values())
        for g in self.rgroups.values():
            params.update(g.values())
        cfg.update(params)
        for key, chk in self.rflags.items():
            cfg[key] = chk.isChecked()
        cfg.update({
            "model_name": self.name_edit.text().strip(),
            "difficulty": self.cmb_difficulty.currentText(),
            "obs_mode": self.cmb_obs_mode.currentText(),
            "minimap_radius": self.spn_mm_radius.value(),
            "curriculum_stage": self.cmb_stage.currentIndex(),
            "net_arch": [self.spn_width.value()] * self.spn_layers.value(),
            "use_amp": self.chk_amp.isChecked(),
            "torch_compile": self.chk_compile.isChecked(),
            "cpp_threads": self.spn_cpp_threads.value(),
        })
        # watch prefs live in the same state file but are not sent to the worker
        cfg["watch_map_size"] = self.spn_watch_map.value()
        cfg["watch_seed"] = self.spn_watch_seed.value()
        cfg["watch_visual"] = self.chk_watch_visual.isChecked()
        cfg["watch_speed"] = self.cmb_watch_speed.currentText()
        return cfg

    def _on_param_changed(self, *_a):
        QTimer.singleShot(800, self._save_state)

    def closeEvent(self, ev):
        self._save_state()
        if self._train_proc and self._train_proc.poll() is None:
            ret = QMessageBox.question(
                self, "Обучение идёт",
                "Обучение ещё работает. Остановить и выйти?",
                QMessageBox.Yes | QMessageBox.No)
            if ret != QMessageBox.Yes:
                ev.ignore()
                return
            self._stop_training()
        if self._watch_proc and self._watch_proc.poll() is None:
            self._watch_proc.kill()
        super().closeEvent(ev)

    # ─────────────────────────── log ───────────────────────────

    def log(self, level: str, text: str):
        color = {"info": T.TXT, "warn": T.WARN, "error": T.ERR}.get(level, T.TXT)
        ts = time.strftime("%H:%M:%S")
        self.log_view.appendHtml(
            f'<span style="color:{T.DIM}">{ts}</span> '
            f'<span style="color:{color}">{_esc(text)}</span>')
        self.log_view.moveCursor(QTextCursor.End)

    # ─────────────────────────── training ───────────────────────────

    def _randomize_seed(self):
        seed = random.randint(1, 999_999_999)
        self.pgroups["Среда"].rows["seed"].set_value(seed)
        self.log("info", f"Seed карты: {seed}")

    def _start_training(self, resume_model: Optional[Path] = None):
        if self._train_proc and self._train_proc.poll() is None:
            self.log("warn", "Обучение уже запущено")
            return
        if self.chk_rand_seed.isChecked():
            self._randomize_seed()
        cfg = self._collect_config()
        name = cfg.get("model_name") or self._next_run_name()
        cfg["model_name"] = name
        self.name_edit.setText(name)
        if resume_model:
            name = name + "_ft"
            cfg["model_name"] = name
            self.name_edit.setText(name)

        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                          encoding="utf-8")
        json.dump(cfg, tmp, ensure_ascii=False)
        tmp.close()
        self._cfg_tmp = tmp.name
        msg_file = os.path.join(tempfile.gettempdir(),
                                f"colony_ui2_{int(time.time()*1000)}.jsonl")
        cmd_file = os.path.join(tempfile.gettempdir(),
                                f"colony_cmd2_{int(time.time()*1000)}.jsonl")
        self._msg_file, self._msg_offset = msg_file, 0
        self._cmd_file = cmd_file

        args = [sys.executable, "-u",
                str(_PROJECT / "train_ui" / "worker.py"),
                "--config", tmp.name, "--name", name,
                "--output", msg_file, "--command-file", cmd_file]
        if resume_model:
            args += ["--resume-model", str(resume_model)]
        try:
            self._train_proc = subprocess.Popen(
                args, cwd=str(_PROJECT),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            self.log("error", f"не удалось запустить worker: {e}")
            return
        self._msg_timer.start(250)
        self._set_running(True)
        for ch in (self.chart_fps, self.chart_ret, self.chart_loss,
                   self.chart_ent, self.chart_eval):
            ch.clear()
        self.log("info", f"Обучение: {name} (pid={self._train_proc.pid})")

    def _next_run_name(self) -> str:
        max_num = 0
        for m in self.registry.scan():
            if m.name.startswith("run_"):
                try:
                    max_num = max(max_num, int(m.name.split("_", 1)[1]))
                except (ValueError, IndexError):
                    pass
        return f"run_{max_num + 1:03d}"

    def _set_running(self, on: bool):
        self.btn_start.setEnabled(not on)
        self.btn_pause.setEnabled(on)
        self.btn_stop.setEnabled(on)
        self.run_status.setText("обучение…" if on else "остановлено")
        self.run_status.setStyleSheet(
            f"color:{T.OK if on else T.DIM}; font-weight:bold; background:transparent;")
        for g in list(self.pgroups.values()) + list(self.rgroups.values()):
            g.set_enabled(not on)

    def _stop_training(self):
        if self._cmd_file and os.path.exists(self._cmd_file):
            try:
                with open(self._cmd_file, "a", encoding="utf-8") as f:
                    f.write(P.encode_command("stop_training") + "\n")
            except OSError:
                pass
        if self._train_proc and self._train_proc.poll() is None:
            self._train_proc.terminate()
        self.log("info", "Остановка обучения…")

    def _toggle_pause(self):
        if not (self._train_proc and self._train_proc.poll() is None):
            return
        paused = self.btn_pause.text().startswith("▶")
        cmd = "resume_training" if paused else "pause_training"
        self._send_command(cmd)
        self.btn_pause.setText("⏸ Пауза" if paused else "▶ Продолжить")

    def _boost_entropy(self):
        self._send_command("boost_entropy", {"factor": 2.0})
        self.log("info", "Команда: ent_coef ×2")

    def _send_command(self, cmd: str, payload: dict = None):
        if not self._cmd_file:
            return
        try:
            with open(self._cmd_file, "a", encoding="utf-8") as f:
                f.write(P.encode_command(cmd, payload or {}) + "\n")
        except OSError as e:
            self.log("error", f"команда не отправлена: {e}")

    def _poll_worker(self):
        # read messages
        if self._msg_file:
            try:
                size = os.path.getsize(self._msg_file)
            except OSError:
                size = -1
            if size > self._msg_offset:
                try:
                    with open(self._msg_file, "r", encoding="utf-8") as f:
                        f.seek(self._msg_offset)
                        data = f.read()
                        self._msg_offset = f.tell()
                except (OSError, UnicodeDecodeError):
                    data = ""
                for line in data.splitlines():
                    line = line.strip()
                    if line:
                        self._handle_msg(line)
        # liveness
        if self._train_proc and self._train_proc.poll() is not None:
            self._msg_timer.stop()
            self._set_running(False)
            self.log("info", f"Worker завершён (code={self._train_proc.returncode})")
            self._refresh_models()
            self._save_state()

    def _handle_msg(self, line: str):
        try:
            msg = P.decode(line)
        except Exception:
            self.log("info", line)
            return
        d = msg.to_dict() if hasattr(msg, "to_dict") else {}
        t = d.get("type")
        if t == "log":
            text = d.get("message", "")
            level = d.get("level", "info")
            self.log(level, text)
            m = _EVAL_RE.search(text)
            if m:
                step = int(m.group(1).replace(",", ""))
                days, people, bases = float(m.group(2)), float(m.group(3)), float(m.group(4))
                score = float(m.group(6))
                self.chart_eval.push({"days": days, "bases": bases})
                self.card_days.set_value(f"{days:.0f}")
                self.card_bases.set_value(f"{bases:.0f}")
                self.card_score.set_value(f"{score:.1f}")
                ok = "PASS" in text
                self.lbl_thresh.setText(f"пороги eval: {'PASS ✅' if ok else 'FAIL ❌'}")
                self.lbl_thresh.setStyleSheet(
                    f"color:{T.OK if ok else T.ERR}; background:transparent;")
            mb = _BEST_RE.search(text)
            if mb:
                self.card_score.set_value(f"{float(mb.group(1)):.1f}", "#c586c0")
        elif t == "progress":
            done, total = d.get("done", 0), d.get("total", 1)
            self.progress.setMaximum(max(1, int(total)))
            self.progress.setValue(int(done))
            self.card_steps.set_value(_fmt_steps(done))
            fps = d.get("fps", 0)
            self.card_fps.set_value(f"{fps:,.0f}")
            self.chart_fps.push({"fps": fps})
            self.card_eps.set_value(str(d.get("episodes", 0)))
            self.chart_ret.push({
                "median": d.get("median_return"), "avg": d.get("avg_return"),
                "min": d.get("min_return"), "max": d.get("max_return")})
            self.chart_loss.push({"policy_loss": d.get("policy_loss"),
                                  "value_loss": d.get("value_loss")})
            ent, kl = d.get("entropy", 0), d.get("kl", 0)
            self.card_ent.set_value(f"{ent:.2f}")
            self.card_kl.set_value(f"{kl:.4f}")
            self.chart_ent.push({"entropy": ent, "kl": kl})
            self.bars_actions.set_items(d.get("top_actions", {}))
            if d.get("loop_detected"):
                self.lbl_loop.setText(
                    f"циклы: {d.get('envs_with_loops', 0)} env ({d.get('loop_action_name') or '?'})")
                self.lbl_loop.setStyleSheet(f"color:{T.ERR}; background:transparent;")
            else:
                self.lbl_loop.setText("циклы: нет")
                self.lbl_loop.setStyleSheet(f"color:{T.DIM}; background:transparent;")
            stage = d.get("curriculum_stage", 0)
            prog = d.get("curriculum_progress_percent", 0) or 0
            self.lbl_curric.setText(f"курикулум: этап {stage} ({prog*100:.0f}%)")
        elif t == "error":
            self.log("error", d.get("message", "ошибка worker"))
        elif t == "done":
            self.log("info", f"Готово: {d.get('total',0):,} шагов за {d.get('time_s',0):.0f}с")
        elif t == "saved":
            self.log("info", f"Сохранено: {d.get('path')}")

    # ─────────────────────────── rewards tab ───────────────────────────

    def _reset_rewards_v2(self):
        rc = _RC()
        data = rc.to_dict()
        for g in self.rgroups.values():
            g.set_values(data)
        for key, chk in self.rflags.items():
            chk.setChecked(bool(data.get(key, False)))
        self._save_state()
        self.log("info", "Награды сброшены к профилю v2 (rl/config.py)")

    def _load_reward_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Профиль наград", "",
                                              "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data.get("reward"), dict):
                data = data["reward"]
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.warning(self, "Ошибка", str(e))
            return
        for g in self.rgroups.values():
            g.set_values(data)
        for key, chk in self.rflags.items():
            if key in data:
                chk.setChecked(bool(data[key]))
        self._save_state()
        self.log("info", f"Награды загружены: {path}")

    def _save_reward_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить профиль",
                                              "reward_profile.json", "JSON (*.json)")
        if not path:
            return
        data = {}
        for g in self.rgroups.values():
            data.update(g.values())
        for key, chk in self.rflags.items():
            data[key] = chk.isChecked()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            self.log("info", f"Профиль сохранён: {path}")
        except OSError as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    # ─────────────────────────── models tab ───────────────────────────

    def _refresh_models(self):
        models = self.registry.scan()
        self.tbl_models.setRowCount(0)
        self.cmb_watch_model.clear()
        for m in models:
            best_meta = {}
            bm = m.path / "best_model.meta.json"
            if bm.exists():
                try:
                    best_meta = json.loads(bm.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    best_meta = {}
            row = self.tbl_models.rowCount()
            self.tbl_models.insertRow(row)
            vals = [
                m.name,
                _fmt_steps(m.steps),
                f"{best_meta.get('best_score', 0):.1f}" if best_meta else "—",
                f"{best_meta.get('best_days', 0):.0f}" if best_meta else "—",
                f"{best_meta.get('best_bases', 0):.0f}" if best_meta else "—",
                str(m.episodes),
                m.created.strftime("%d.%m %H:%M") if m.created else "—",
            ]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if c == 0:
                    item.setData(Qt.UserRole, str(m.path))
                self.tbl_models.setItem(row, c, item)
            self.cmb_watch_model.addItem(m.name, str(m.path))

    def _selected_model_path(self) -> Optional[Path]:
        row = self.tbl_models.currentRow()
        if row < 0:
            return None
        item = self.tbl_models.item(row, 0)
        return Path(item.data(Qt.UserRole)) if item else None

    def _sync_watch_model(self):
        p = self._selected_model_path()
        if p:
            idx = self.cmb_watch_model.findData(str(p))
            if idx >= 0:
                self.cmb_watch_model.setCurrentIndex(idx)

    def _watch_selected_model(self):
        row = self.tbl_models.currentRow()
        if row < 0:
            QMessageBox.information(self, "Модели", "Выберите модель в таблице")
            return
        self._sync_watch_model()
        self.tabs.setCurrentIndex(4)
        self._toggle_watch(start=True)

    def _finetune_selected(self):
        p = self._selected_model_path()
        if not p:
            QMessageBox.information(self, "Модели", "Выберите модель")
            return
        model_file = p / "final_model.pt"
        if not model_file.exists():
            ck = sorted(p.glob("checkpoint_*_steps.pt"))
            model_file = ck[-1] if ck else p / "best_model.pt"
        if not model_file.exists():
            QMessageBox.warning(self, "Модели", "Нет final/checkpoint/best .pt")
            return
        self._start_training(resume_model=model_file)

    def _open_selected_folder(self):
        p = self._selected_model_path()
        if p:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _delete_selected(self):
        p = self._selected_model_path()
        if not p:
            return
        ret = QMessageBox.question(self, "Удаление",
                                   f"Удалить модель {p.name} безвозвратно?")
        if ret == QMessageBox.Yes:
            try:
                self.registry.delete(p.name)
                self._refresh_models()
            except (OSError, FileNotFoundError) as e:
                self.log("error", str(e))

    # ─────────────────────────── watch ───────────────────────────

    def _random_watch_seed(self):
        self.spn_watch_seed.setValue(random.randint(1, 999_999_999))

    def _toggle_watch(self, start: bool = False):
        if self._watch_proc and self._watch_proc.poll() is None:
            self._watch_proc.kill()
            self._watch_timer.stop()
            self._watch_proc = None
            self.btn_watch.setText("👁 Наблюдать")
            self.log("info", "Наблюдение остановлено")
            return
        if not start and self.btn_watch.text().startswith("■"):
            return
        model_dir = self.cmb_watch_model.currentData()
        if not model_dir:
            QMessageBox.information(self, "Наблюдение", "Нет моделей — сначала обучите")
            return
        seed = self.spn_watch_seed.value()
        speed_txt = self.cmb_watch_speed.currentText()
        args = [sys.executable, "-u", str(_PROJECT / "watch_champion.py"),
                "--model-dir", str(model_dir),
                "--episodes", "1000000", "--max-steps", "1000000",
                "--device", "cpu",
                "--map-size", str(self.spn_watch_map.value()),
                "--speed", "0" if speed_txt == "max" else speed_txt]
        if seed > 0:
            args += ["--seed", str(seed)]
        log_file = os.path.join(tempfile.gettempdir(),
                                f"colony_watch2_{int(time.time()*1000)}.log")
        args += ["--log-file", log_file]
        if self.chk_watch_visual.isChecked():
            args.append("--visual")
        try:
            fh = open(log_file, "a", encoding="utf-8")
            self._watch_proc = subprocess.Popen(
                args, cwd=str(_PROJECT), stdout=fh, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            self.log("error", f"не удалось запустить наблюдение: {e}")
            return
        self._watch_log, self._watch_offset = log_file, 0
        self.chart_watch.clear()
        self._watch_timer.start(500)
        self.btn_watch.setText("■ Стоп")
        self.log("info", f"Наблюдение: {Path(model_dir).name} "
                         f"(seed={'случайная' if seed == 0 else seed}, "
                         f"карта {self.spn_watch_map.value()})")

    def _poll_watch(self):
        if not self._watch_log:
            return
        try:
            size = os.path.getsize(self._watch_log)
        except OSError:
            return
        if size < self._watch_offset:  # файл переписан с нуля
            self._watch_offset = 0
        if size <= self._watch_offset:
            return
        try:
            with open(self._watch_log, "r", encoding="utf-8") as f:
                f.seek(self._watch_offset)
                data = f.read()
                self._watch_offset = f.tell()
        except (OSError, UnicodeDecodeError):
            return
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    self.log("info", line)
                    continue
                if d.get("type") == "step":
                    self.w_day.set_value(str(d.get("day", "—")))
                    self.w_money.set_value(f"{d.get('money', 0):,}")
                    self.w_people.set_value(str(d.get("people", "—")))
                    self.w_bases.set_value(str(d.get("bases", "—")))
                    self.w_action.set_value(str(d.get("action", "—"))[:14])
                    self.chart_watch.push({"reward": d.get("reward", 0)})
                    continue
                if d.get("type") in ("log", "error", "done"):
                    self.log(d.get("level", "info"), d.get("message", ""))
                    continue
            self.log("info", line)
        if self._watch_proc and self._watch_proc.poll() is not None:
            self._watch_timer.stop()
            self.btn_watch.setText("👁 Наблюдать")


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
