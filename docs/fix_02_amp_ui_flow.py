"""
FIX_02: Передача AMP/compile параметров из UI в worker
======================================================

ПРОБЛЕМЫ:
  1. amp_dtype не передаётся из UI в worker (всегда fallback "bfloat16")
  2. UI default для use_amp = False, Config default = True — рассинхрон
  3. Нет выбора float16 vs bfloat16 в UI

ИСПРАВЛЯЕТ: train_ui/main_window.py, train_ui/worker.py
"""


# ============================================================
# ИЗМЕНЕНИЕ 1: main_window.py — добавить выбор amp_dtype
# ============================================================

# СТРОКА ~649 (БЫЛО):
#   self.chk_amp = QCheckBox("AMP (bfloat16)")
#   self.chk_amp.setChecked(bool(self.config.get("use_amp", False)))

# СТРОКА ~649 (ПОСЛЕ ИСПРАВЛЕНИЯ):
#   Добавить выпадающий список для выбора dtype

# --- Новый код для main_window.py, секция "AMP / compile + Buttons" ---

"""
# В imports добавить:
from PySide6.QtWidgets import QComboBox

# В __init__, секция r3 (строка ~645):

        # ── AMP / compile + Buttons ──
        r3 = QHBoxLayout()
        r3.setSpacing(3)

        # AMP controls
        self.chk_amp = QCheckBox("AMP")
        self.chk_amp.setStyleSheet(STYLE_SMALL)
        self.chk_amp.setChecked(bool(self.config.get("use_amp", True)))  # FIX: default True

        self.cmb_amp_dtype = QComboBox()
        self.cmb_amp_dtype.addItems(["bfloat16", "float16"])
        self.cmb_amp_dtype.setStyleSheet(STYLE_SMALL)
        self.cmb_amp_dtype.setMaximumWidth(70)
        # Restore saved dtype
        saved_dtype = self.config.get("amp_dtype", "bfloat16")
        idx = self.cmb_amp_dtype.findText(saved_dtype)
        if idx >= 0:
            self.cmb_amp_dtype.setCurrentIndex(idx)

        self.chk_compile = QCheckBox("torch.compile")
        self.chk_compile.setStyleSheet(STYLE_SMALL)
        self.chk_compile.setChecked(
            bool(self.config.get("torch_compile", False)))

        # ... кнопки ...

        r3.addWidget(self.chk_amp, 0)
        r3.addWidget(self.cmb_amp_dtype, 0)   # <-- НОВЫЙ ВИДЖЕТ
        r3.addWidget(self.chk_compile, 0)
        # ... остальные кнопки ...
"""


# ============================================================
# ИЗМЕНЕНИЕ 2: _collect_config() — передавать amp_dtype
# ============================================================

# СТРОКА 1126-1151 (БЫЛО):
def _collect_config_BROKEN(self) -> dict:
    name = self.name_edit.text().strip()
    if not name:
        name = self._next_run_name()
    cfg = {
        "name": name,
        "use_amp": self.chk_amp.isChecked(),
        "torch_compile": self.chk_compile.isChecked(),
        "obs_mode": getattr(self, "_obs_mode", "flat"),
        "minimap_radius": int(getattr(self, "_minimap_radius", 14)),
    }
    # ... param_rows, reward_rows, net_arch, curriculum ...
    return cfg


# СТРОКА 1126-1151 (ПОСЛЕ ИСПРАВЛЕНИЯ):
def _collect_config_FIXED(self) -> dict:
    name = self.name_edit.text().strip()
    if not name:
        name = self._next_run_name()
    cfg = {
        "name": name,
        "use_amp": self.chk_amp.isChecked(),
        "amp_dtype": self.cmb_amp_dtype.currentText(),  # FIX: передаём dtype
        "torch_compile": self.chk_compile.isChecked(),
        "obs_mode": getattr(self, "_obs_mode", "flat"),
        "minimap_radius": int(getattr(self, "_minimap_radius", 14)),
    }
    # ... param_rows, reward_rows, net_arch, curriculum ...
    return cfg


# ============================================================
# ИЗМЕНЕНИЕ 3: save_state() — сохранять amp_dtype
# ============================================================

# СТРОКА 1561-1578 (БЫЛО):
def save_state_BROKEN(self) -> dict:
    state = {
        "use_amp": self.chk_amp.isChecked(),
        "torch_compile": self.chk_compile.isChecked(),
        # ... без amp_dtype ...
    }


# СТРОКА 1561-1578 (ПОСЛЕ ИСПРАВЛЕНИЯ):
def save_state_FIXED(self) -> dict:
    state = {
        "use_amp": self.chk_amp.isChecked(),
        "amp_dtype": self.cmb_amp_dtype.currentText(),  # FIX
        "torch_compile": self.chk_compile.isChecked(),
        # ...
    }


# ============================================================
# ИЗМЕНЕНИЕ 4: _restore_state() — восстанавливать amp_dtype
# ============================================================

# СТРОКА 1633-1636 (БЫЛО):
def _restore_state_BROKEN(self):
    self.chk_amp.setChecked(bool(cfg.get("use_amp", False)))  # default False
    self.chk_compile.setChecked(bool(cfg.get("torch_compile", False)))


# СТРОКА 1633-1636 (ПОСЛЕ ИСПРАВЛЕНИЯ):
def _restore_state_FIXED(self):
    self.chk_amp.setChecked(bool(cfg.get("use_amp", True)))  # FIX: default True
    saved_dtype = cfg.get("amp_dtype", "bfloat16")
    idx = self.cmb_amp_dtype.findText(saved_dtype)
    if idx >= 0:
        self.cmb_amp_dtype.setCurrentIndex(idx)
    self.chk_compile.setChecked(bool(cfg.get("torch_compile", False)))


# ============================================================
# ИЗМЕНЕНИЕ 5: _load_config_from_file() — восстанавливать amp_dtype
# ============================================================

# СТРОКА ~1600 (добавить после self.chk_amp.setChecked):
#   if "amp_dtype" in cfg:
#       idx = self.cmb_amp_dtype.findText(cfg["amp_dtype"])
#       if idx >= 0:
#           self.cmb_amp_dtype.setCurrentIndex(idx)


# ============================================================
# ИЗМЕНЕНИЕ 6: DEFAULT_PARAMS — обновить default
# ============================================================

# СТРОКА 43-49 (БЫЛО):
DEFAULT_PARAMS = {
    "name": "run_001",
    "use_amp": False,      # <-- НЕПРАВИЛЬНО
    "torch_compile": False,
    # ...
}

# СТРОКА 43-49 (ПОСЛЕ ИСПРАВЛЕНИЯ):
DEFAULT_PARAMS = {
    "name": "run_001",
    "use_amp": True,       # FIX: согласовано с Config()
    "amp_dtype": "bfloat16",  # FIX: добавлено
    "torch_compile": False,
    # ...
}
