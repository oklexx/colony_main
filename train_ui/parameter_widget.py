from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


PARAM_SPECS: list[ParamSpec] = [
    ParamSpec("total_timesteps", "Шагов обучения", True, 1e4, 1e9, 1_000_000,
              "Общее количество шагов (timesteps) для обучения. "
              "Диапазон: 10 000 – 1 000 000 000. "
              "Пример: 1 000 000 — быстрая проверка, 10 000 000 — полноценный прогон."),
    ParamSpec("n_envs", "Параллельных окружений", True, 1, 256, 8,
              "Количество сред, работающих параллельно. Больше — быстрее сбор данных. "
              "Диапазон: 1 – 256."),
    ParamSpec("n_steps", "Шагов на rollout", True, 128, 65536, 2048,
              "Длина траектории, собираемой за один rollout на окружение. "
              "Диапазон: 128 – 65 536."),
    ParamSpec("batch_size", "Размер батча", True, 256, 262144, 2048,
              "Размер мини-батча для обновления PPO. "
              "Диапазон: 256 – 262 144."),
    ParamSpec("n_epochs", "Эпох на rollout", True, 1, 50, 10,
              "Сколько раз PPO перебирает собранный батч данных. "
              "Диапазон: 1 – 50."),
    ParamSpec("learning_rate", "Коэф. обучения", False, 1e-6, 1e-1, 3e-4,
              "Шаг оптимизатора AdamW. Диапазон: 0.000001 – 0.1. "
              "Пример: 0.0003 — стандарт для PPO."),
    ParamSpec("gamma", "Дисконт (gamma)", False, 0.9, 0.9999, 0.99,
              "Вес будущих наград. Диапазон: 0.9 – 0.9999. "
              "Пример: 0.99 — умеренная дальность горизонта."),
    ParamSpec("gae_lambda", "GAE lambda", False, 0.8, 1.0, 0.95,
              "Баланс смещения/дисперсии в оценке преимущества. Диапазон: 0.8 – 1.0."),
    ParamSpec("clip_range", "Clip range", False, 0.05, 0.5, 0.2,
              "Ограничение изменения вероятностей действий за шаг PPO. Диапазон: 0.05 – 0.5."),
    ParamSpec("ent_coef", "Entropy coef", False, 0.0, 0.1, 0.01,
              "Вес энтропии — стимулирует исследование. Диапазон: 0 – 0.1."),
    ParamSpec("vf_coef", "VF coef", False, 0.1, 2.0, 1.0,
              "Вес функции ценности в потерях. Диапазон: 0.1 – 2.0."),
    ParamSpec("max_grad_norm", "Max grad norm", False, 0.1, 10.0, 1.0,
              "Максимальная норма градиента при обрезке. Диапазон: 0.1 – 10.0."),
    ParamSpec("net_arch", "Размер скрытых слоёв", True, 64, 1024, 256,
              "Размер каждого из двух скрытых слоёв сети. Диапазон: 64 – 1024."),
    ParamSpec("seed", "Seed", True, 0, 2 ** 31 - 1, 42,
              "Случайное зерно для воспроизводимости. Диапазон: 0 – 2147483647."),
    ParamSpec("map_size", "Размер карты", True, 100, 500, 200,
              "Размер мира в ячейках. Диапазон: 100 – 500. Больше — сложнее задача."),
]


def spec_for(key: str) -> ParamSpec:
    for s in PARAM_SPECS:
        if s.key == key:
            return s
    raise KeyError(f"unknown parameter: {key}")
