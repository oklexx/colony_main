"""
FIX_05: Выравнивание default значений Config и UI
=================================================

ПРОБЛЕМЫ:
  1. Config.use_amp = True, UI DEFAULT_PARAMS.use_amp = False
  2. Config неполный набор параметров дляfrom_dict() —缺少eval_score_weights тип

ИСПРАВЛЯЕТ: rl/config.py, train_ui/main_window.py
"""


# ============================================================
# ИЗМЕНЕНИЕ 1: rl/config.py — проверить default значения
# ============================================================

# СТРОКА 112-165 — Config dataclass:
# Текущие default'ы:
#   use_amp: bool = True          # OK
#   amp_dtype: str = "bfloat16"   # OK
#   torch_compile: bool = False   # OK
#   ent_coef: float = 0.005       # OK (was 0.01, reduced for loop fix)

# Ничего менять не нужно — default'ы в Config корректны.
# Проблема в том, что UI использует другие default'ы.


# ============================================================
# ИЗМЕНЕНИЕ 2: main_window.py — DEFAULT_PARAMS
# ============================================================

# СТРОКА 43-49 (БЫЛО):
DEFAULT_PARAMS = {
    "name": "run_001",
    "use_amp": False,           # <-- РАСХОДСТВО с Config (True)
    "torch_compile": False,
    **{s.key: s.default for s in PARAM_SPECS},
    **{s.key: s.default for s in REWARD_SPECS},
}

# СТРОКА 43-49 (ПОСЛЕ ИСПРАВЛЕНИЯ):
DEFAULT_PARAMS = {
    "name": "run_001",
    "use_amp": True,            # FIX: согласовано с Config()
    "amp_dtype": "bfloat16",    # FIX: добавлено
    "torch_compile": False,
    **{s.key: s.default for s in PARAM_SPECS},
    **{s.key: s.default for s in REWARD_SPECS},
}


# ============================================================
# ИЗМЕНЕНИЕ 3: Config.from_dict() — robust tuple conversion
# ============================================================

# СТРОКА 230 (БЫЛО):
eval_score_weights=tuple(d.get("eval_score_weights", (0.4, 3.0, 0.2, 0.0001))),

# СТРОКА 230 (ПОСЛЕ ИСПРАВЛЕНИЯ):
# FIX: ensure tuple conversion works for both list and tuple inputs
eval_score_weights=tuple(float(x) for x in d.get("eval_score_weights", (0.4, 3.0, 0.2, 0.0001))),


# ============================================================
# ИЗМЕНЕНИЕ 4: Config.to_dict() — добавить eval_seeds
# ============================================================

# СТРОКА 186-201 — to_dict() keys:
# eval_seeds уже есть в списке keys — OK


# ============================================================
# ИЗМЕНЕНИЕ 5: Config.__post_init__() — проверка CUDA для torch_compile
# ============================================================

# СТРОКА 174-184 (ДОБАВИТЬ):
def __post_init__(self):
    if not self.log_dir:
        home = Path.home()
        self.log_dir = str(home / "colony_runs" / "logs")
    if not self.model_dir:
        home = Path.home()
        self.model_dir = str(home / "colony_runs" / "models")
    if self.amp_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"amp_dtype must be bfloat16 or float16, got {self.amp_dtype}")
    if self.obs_mode not in ("flat", "minimap", "hybrid"):
        raise ValueError(f"obs_mode must be 'flat', 'minimap', or 'hybrid', got {self.obs_mode}")

    # FIX: warn if torch_compile enabled without CUDA
    import torch
    if self.torch_compile and not torch.cuda.is_available():
        import warnings
        warnings.warn(
            "torch_compile=True but CUDA is not available. "
            "torch.compile will be disabled at runtime.",
            UserWarning,
            stacklevel=2,
        )

    # FIX: warn if amp_dtype="bfloat16" but GPU doesn't support it
    if self.use_amp and self.amp_dtype == "bfloat16" and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            import warnings
            warnings.warn(
                "amp_dtype='bfloat16' but GPU does not support BF16. "
                "AMP will silently fall back to float32.",
                UserWarning,
                stacklevel=2,
            )
