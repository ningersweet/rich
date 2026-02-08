from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .config import Config
from .model import TrainedModel, predict_proba

Signal = Literal["long", "short", "flat"]


@dataclass
class SignalResult:
    signal: Signal
    prob_long: float
    prob_short: float
    prob_flat: float


def generate_signal(
    cfg: Config,
    trained: TrainedModel,
    features_row: pd.DataFrame,
    prev_signal: Signal | None = None,
    prev_signal_time: Optional[pd.Timestamp] = None,
    current_time: Optional[pd.Timestamp] = None,
) -> SignalResult:
    """对单行特征生成交易信号，包含波动过滤、价格行为确认和冷却时间。"""

    proba = predict_proba(trained, features_row)[0]
    # 假定类别顺序为 [-1, 0, 1] → [short, flat, long]
    prob_short, prob_flat, prob_long = proba.tolist()

    strat_cfg = cfg.strategy
    base_min_prob = float(strat_cfg.get("min_prob", 0.5))
    gap_long = float(strat_cfg.get("prob_gap_long", 0.2))
    gap_short = float(strat_cfg.get("prob_gap_short", 0.2))

    # 根据市场状态和波动率动态调整 min_prob（可选）
    min_prob = base_min_prob
    if bool(strat_cfg.get("use_dynamic_min_prob", False)):
        regime_val = 0.0
        if "market_regime" in features_row.columns:
            regime_val = float(features_row["market_regime"].iloc[0])
        volatility = 0.0
        if "atr_pct" in features_row.columns:
            volatility = float(features_row["atr_pct"].iloc[0])
        min_prob = _adjust_min_prob_by_market_regime(
            base_min_prob=base_min_prob,
            regime=regime_val,
            volatility=volatility,
            strat_cfg=strat_cfg,
        )

    # 概率差定义为信号强度，可用于后续动态仓位管理
    long_strength = prob_long - prob_short
    short_strength = prob_short - prob_long

    # 1. 基于概率的初始信号
    if prob_long > min_prob and (prob_long - prob_short) > gap_long:
        sig: Signal = "long"
    elif prob_short > min_prob and (prob_short - prob_long) > gap_short:
        sig = "short"
    else:
        sig = "flat"

    # 1.1 信号强度过滤（可选）
    use_strength = bool(strat_cfg.get("use_signal_strength", False))
    min_strength = float(strat_cfg.get("min_signal_strength", 0.0))
    if use_strength and sig != "flat":
        strength = long_strength if sig == "long" else short_strength
        if strength < min_strength:
            sig = "flat"

    # 2. 波动率过滤（使用 ATR 百分比）
    atr_pct = None
    if "atr_pct" in features_row.columns:
        atr_pct = float(features_row["atr_pct"].iloc[0])
    min_atr_pct = float(strat_cfg.get("min_atr_pct", 0.0005))
    max_atr_pct = float(strat_cfg.get("max_atr_pct", 0.05))
    if atr_pct is not None and (atr_pct < min_atr_pct or atr_pct > max_atr_pct):
        sig = "flat"

    # 3. 市场状态/趋势过滤：过于震荡（绝对收益太小）不交易
    if "return_1" in features_row.columns:
        r1 = float(features_row["return_1"].iloc[0])
        min_trend = float(strat_cfg.get("min_trend_abs_return", 0.0002))
        if abs(r1) < min_trend:
            sig = "flat"

    # 4. 价格行为确认：用价格在近期区间的位置过滤
    if "price_position" in features_row.columns:
        pos = float(features_row["price_position"].iloc[0])
        min_pos_long = float(strat_cfg.get("min_price_pos_long", 0.5))
        max_pos_short = float(strat_cfg.get("max_price_pos_short", 0.5))
        if sig == "long" and pos < min_pos_long:
            sig = "flat"
        elif sig == "short" and pos > max_pos_short:
            sig = "flat"

    # 5. 信号冷却时间：短时间内避免反向频繁开仓
    cooldown_min = float(strat_cfg.get("signal_cooldown_minutes", 0.0))
    if (
        cooldown_min > 0
        and prev_signal is not None
        and prev_signal_time is not None
        and current_time is not None
    ):
        # 若上一次是非空仓信号，且在冷却期内给出反向信号，则忽略本次信号
        if sig != "flat" and sig != prev_signal:
            delta_min = (current_time - prev_signal_time).total_seconds() / 60.0
            if delta_min < cooldown_min:
                sig = "flat"

    return SignalResult(
        signal=sig,
        prob_long=float(prob_long),
        prob_short=float(prob_short),
        prob_flat=float(prob_flat),
    )


def _adjust_min_prob_by_market_regime(
    base_min_prob: float,
    regime: float,
    volatility: float,
    strat_cfg: dict,
) -> float:
    """根据市场状态和波动率调整最小概率阈值。

    - 趋势市（|regime|=1）适当降低阈值，震荡市提高阈值
    - 高波动时略微降低阈值，低波动时略微提高阈值
    """

    adjusted = base_min_prob

    # 市场状态：牛/熊市倾向于放宽，震荡市收紧
    if abs(regime) >= 1.0:
        adjusted *= 0.8
    else:
        adjusted *= 1.2

    high_vol_th = float(strat_cfg.get("high_vol_threshold", 0.02))
    low_vol_th = float(strat_cfg.get("low_vol_threshold", 0.005))

    if volatility > high_vol_th:
        adjusted *= 0.9
    elif 0.0 < volatility < low_vol_th:
        adjusted *= 1.1

    min_bound = float(strat_cfg.get("min_dynamic_min_prob", 0.25))
    max_bound = float(strat_cfg.get("max_dynamic_min_prob", 0.45))

    return float(max(min(adjusted, max_bound), min_bound))


def generate_signals_for_backtest(
    cfg: Config,
    trained: TrainedModel,
    features: pd.DataFrame,
    times: pd.Series,
) -> pd.Series:
    """按时间顺序批量生成信号，应用冷却时间等状态逻辑。"""

    prev_sig: Signal | None = None
    prev_time: Optional[pd.Timestamp] = None
    all_signals: list[str] = []

    for i in range(len(features)):
        row = features.iloc[[i]]
        current_time = times.iloc[i]
        res = generate_signal(
            cfg,
            trained,
            row,
            prev_signal=prev_sig,
            prev_signal_time=prev_time,
            current_time=current_time,
        )
        all_signals.append(res.signal)
        if res.signal != "flat":
            prev_sig = res.signal
            prev_time = current_time

    return pd.Series(all_signals, index=times)
