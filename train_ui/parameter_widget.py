from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Import canonical defaults from rl.config — single source of truth
_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
from rl.config import Config as _Cfg, RewardConfig as _RC

INT_MIN = 1
FLOAT_MIN = 1e-9
FLOAT_ROUND = 8


def scale_value(value: float, factor: float, is_int: bool = True) -> float:
    """Multiply value by factor, clamped and rounded per integer/float rules.

    Integer params: round to nearest int, minimum INT_MIN (1).
    Float params: round to 8 decimals, minimum FLOAT_MIN (1e-9).
    factor must be non-zero.
    """
    if factor == 0:
        raise ValueError("factor must be non-zero")
    v = value * factor
    if is_int:
        return max(INT_MIN, int(round(v)))
    return max(FLOAT_MIN, round(v, FLOAT_ROUND))


@dataclass
class ParamSpec:
    key: str
    label: str
    is_int: bool
    min: float
    max: float
    default: float
    tooltip: str
    step: float = 1.0
    decimals: int = 0
    max_width: int = 0


def spec_for(key: str) -> ParamSpec:
    for s in PARAM_SPECS:
        if s.key == key:
            return s
    for s in REWARD_SPECS:
        if s.key == key:
            return s
    raise KeyError(f"unknown parameter: {key}")


# Helper: build a ParamSpec with default from rl.config dataclass
_cfg = _Cfg()  # canonical defaults
_rc = _RC()    # canonical reward defaults

PARAM_SPECS: list[ParamSpec] = [
    ParamSpec("total_timesteps", "Шагов обучения", True, 1e4, 1e9, float(_cfg.total_timesteps),
              "Общее количество шагов (timesteps) для обучения. "
              "Диапазон: 10 000 – 1 000 000 000. "
              "Пример: 1 000 000 — быстрая проверка, 10 000 000 — полноценный прогон.",
              max_width=70),
    ParamSpec("n_envs", "Параллельных окружений", True, 1, 2048, float(_cfg.n_envs),
              "Количество сред, работающих параллельно. Больше — быстрее сбор данных. "
              "Диапазон: 1 – 2048.", max_width=60),
    ParamSpec("n_steps", "Шагов на rollout", True, 128, 65536, float(_cfg.n_steps),
              "Длина траектории, собираемой за один rollout на окружение. "
              "Диапазон: 128 – 65 536.", max_width=70),
    ParamSpec("batch_size", "Размер батча", True, 256, 262144, float(_cfg.batch_size),
              "Размер мини-батча для обновления PPO. "
              "Диапазон: 256 – 262 144.", max_width=70),
    ParamSpec("n_epochs", "Эпох на rollout", True, 1, 50, float(_cfg.n_epochs),
              "Сколько раз PPO перебирает собранный батч данных. "
              "Диапазон: 1 – 50.", max_width=50),
    ParamSpec("learning_rate", "Коэф. обучения", False, 1e-6, 1e-1, float(_cfg.learning_rate),
              "Шаг оптимизатора AdamW. Диапазон: 0.000001 – 0.1. "
              "Пример: 0.0003 — стандарт для PPO.", max_width=70),
    ParamSpec("gamma", "Дисконт (gamma)", False, 0.9, 0.9999, float(_cfg.gamma),
              "Вес будущих наград. Диапазон: 0.9 – 0.9999. "
              "Пример: 0.995 — длинный горизонт для стратегической игры.", max_width=60),
    ParamSpec("gae_lambda", "GAE lambda", False, 0.8, 1.0, float(_cfg.gae_lambda),
              "Баланс смещения/дисперсии в оценке преимущества. Диапазон: 0.8 – 1.0.",
              max_width=55),
    ParamSpec("clip_range", "Clip range", False, 0.05, 0.5, float(_cfg.clip_range),
              "Ограничение изменения вероятностей действий за шаг PPO. Диапазон: 0.05 – 0.5.",
              max_width=55),
    ParamSpec("ent_coef", "Entropy coef", False, 0.0, 0.1, float(_cfg.ent_coef),
              "Вес энтропии — стимулирует исследование. Диапазон: 0 – 0.1.", max_width=55),
    ParamSpec("vf_coef", "VF coef", False, 0.1, 2.0, float(_cfg.vf_coef),
              "Вес функции ценности в потерях. Диапазон: 0.1 – 2.0.", max_width=55),
    ParamSpec("max_grad_norm", "Max grad norm", False, 0.1, 10.0, float(_cfg.max_grad_norm),
              "Максимальная норма градиента при обрезке. Диапазон: 0.1 – 10.0.", max_width=60),
    ParamSpec("n_layers", "Кол-во слоёв", True, 1, 4, float(len(_cfg.net_arch)),
              "Количество скрытых слоёв сети. Диапазон: 1 – 4.", max_width=40),
    ParamSpec("seed", "Seed", True, 0, 2 ** 31 - 1, float(_cfg.seed),
              "Случайное зерно для воспроизводимости. Диапазон: 0 – 2147483647.", max_width=70),
    ParamSpec("map_size", "Размер карты", True, 100, 500, float(_cfg.map_size),
              "Размер мира в ячейках. Диапазон: 100 – 500. Больше — сложнее задача.",
              max_width=60),
]


REWARD_SPECS: list[ParamSpec] = [
    ParamSpec("build_bonus", "Бонус за постройку", False, 0.0, 50.0, float(_rc.build_bonus),
              "Множитель годовой стоимости здания при постройке.", step=0.1, max_width=60),
    ParamSpec("chain_bonus", "Бонус за цепочку", False, 0.0, 10.0, float(_rc.chain_bonus),
              "Одноразовый бонус за первую цепочку потребления.", step=0.1, max_width=60),
    ParamSpec("chain_daily", "Цепочка ежедн.", False, 0.0, 10.0, float(_rc.chain_daily),
              "Ежедневный бонус за каждую активную цепочку.", step=0.1, max_width=60),
    ParamSpec("novelty", "Новизна здания", False, 0.0, 500.0, float(_rc.novelty),
              "Бонус за первый запуск нового типа здания.", step=5.0, max_width=60),
    ParamSpec("daily_income", "Ежедн. доход ×", False, 0.0, 1.0, float(_rc.daily_income),
              "Множитель ежедневного дохода. Снижайте чтобы стройка была выгоднее.", step=0.01, max_width=60),
    ParamSpec("sale_bonus", "Бонус продажи ×", False, 0.0, 5.0, float(_rc.sale_bonus),
              "Множитель бонуса за продажу ресурсов.", step=0.01, max_width=60),
    ParamSpec("tax_daily_bonus", "Бонус без налога", False, 0.0, 5.0, float(_rc.tax_daily_bonus),
              "Ежедневный бонус когда налог не причитается. Снижайте чтобы стимулировать активность.", step=0.05, max_width=60),
    ParamSpec("survival_bonus", "Бонус выживания", False, 0.0, 10.0, float(_rc.survival_bonus),
              "Ежедневный бонус за каждый шаг без game over.", step=0.5, max_width=60),
    ParamSpec("game_over_penalty", "Штраф game over", False, 0.0, 500.0, float(_rc.game_over_penalty),
              "Штраф при наступлении конца игры (вычитается из награды).", step=5.0, max_width=60),
    ParamSpec("diversity_bonus", "Бонус разнообразия", False, 0.0, 50.0, float(_rc.diversity_bonus),
              "Бонус за каждый новый тип здания при постройке.", step=0.5, max_width=60),
    ParamSpec("error_penalty", "Штраф ошибки", False, -50.0, 0.0, float(_rc.error_penalty),
              "Штраф за неудачное действие (ремонт, снос, продажа и т.д.).", step=1.0, max_width=60),
    ParamSpec("preserve_penalty", "Плата консервации", False, -50.0, 0.0, float(_rc.preserve_penalty),
              "Плата за консервацию/разконсервацию здания (по умолчанию бесплатно).", step=0.5, max_width=60),
    ParamSpec("demolish_penalty", "Штраф сноса", False, -50.0, 0.0, float(_rc.demolish_penalty),
              "Штраф за успешный снос здания.", step=0.5, max_width=60),
    ParamSpec("manual_tax_penalty", "Плата ручного налога", False, -5.0, 0.0, float(_rc.manual_tax_penalty),
              "Плата за ручную уплату налога (налог и так платится автоматом).", step=0.1, max_width=60),
    ParamSpec("build_cost_penalty", "Штраф стоимости", False, 0.0, 1.0, float(_rc.build_cost_penalty),
              "Доля от стоимости строительства, вычитаемая из награды.", step=0.005, max_width=60),
    ParamSpec("idle_build_penalty", "Штраф простоя", False, -50.0, 0.0, float(_rc.idle_build_penalty),
              "Штраф за длительный период без построек.", step=0.5, max_width=60),
    ParamSpec("idle_build_threshold_days", "Порог простоя (дни)", False, 1.0, 365.0, float(_rc.idle_build_threshold_days),
              "Количество дней без построек до начисления штрафа.", step=1.0, max_width=60),
    ParamSpec("milestone_base_bonus", "Milestone базы", False, 0.0, 500.0, float(_rc.milestone_base_bonus),
              "Бонус за каждые 5 баз.", step=10.0, max_width=60),
    ParamSpec("milestone_people_bonus", "Milestone людей", False, 0.0, 200.0, float(_rc.milestone_people_bonus),
              "Бонус за каждые 50 человек.", step=5.0, max_width=60),
    ParamSpec("milestone_day_bonus", "Milestone дней", False, 0.0, 300.0, float(_rc.milestone_day_bonus),
              "Бонус за каждые 100 дней.", step=5.0, max_width=60),
    ParamSpec("milestone_year_bonus", "Milestone года", False, 0.0, 500.0, float(_rc.milestone_year_bonus),
              "Бонус за первый год (365 дней).", step=10.0, max_width=60),
    ParamSpec("proximity_bonus", "Бонус близости", False, 0.0, 50.0, float(_rc.proximity_bonus),
              "Бонус за строительство здания рядом с нужным ресурсом.", step=1.0, max_width=60),
    ParamSpec("clip_reward_min", "Клип мин", False, -500.0, 0.0, float(_rc.clip_reward_min),
              "Нижняя граница клиппинга сырой награды.", step=10.0, max_width=60),
    ParamSpec("clip_reward_max", "Клип макс", False, 0.0, 500.0, float(_rc.clip_reward_max),
              "Верхняя граница клиппинга сырой награды.", step=10.0, max_width=60),
]


def spec_for(key: str) -> ParamSpec:
    for s in PARAM_SPECS:
        if s.key == key:
            return s
    for s in REWARD_SPECS:
        if s.key == key:
            return s
    raise KeyError(f"unknown parameter: {key}")
