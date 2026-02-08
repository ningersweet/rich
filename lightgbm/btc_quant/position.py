from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass
class PositionSizingResult:
    position_usdt: float
    leverage: float


def calculate_position_size(
    cfg: Config,
    account_equity: float,
    entry_price: float,
    atr: float | None = None,
) -> PositionSizingResult:
    """根据账户权益和风险参数计算开仓名义价值（USDT）。

    简化版：
    - 单笔风险 = equity * risk_per_trade
    - 若给定 ATR，则以 ATR 作为止损宽度估算仓位；否则按固定杠杆近似估算。
    """

    risk_cfg = cfg.risk
    risk_per_trade = float(risk_cfg.get("risk_per_trade", 0.01))
    max_leverage = float(risk_cfg.get("max_leverage", 3.0))

    risk_capital = account_equity * risk_per_trade

    if atr is not None and atr > 0:
        stop_pct = atr / entry_price
        if stop_pct <= 0:
            notional = account_equity * max_leverage
        else:
            notional = min(risk_capital / stop_pct, account_equity * max_leverage)
    else:
        notional = account_equity * max_leverage * risk_per_trade

    leverage = min(max_leverage, notional / max(account_equity, 1e-6))

    return PositionSizingResult(position_usdt=float(notional), leverage=float(leverage))


def dynamic_position_sizing(
    cfg: Config,
    account_equity: float,
    signal_strength: float,
    volatility: float,
) -> float:
    """根据信号强度和波动率动态调整名义仓位（USDT）。

    - base_risk: config.risk.risk_per_trade
    - signal_strength: 0~1，越大代表信号越强
    - volatility: 使用 ATR% 或收益率波动率作为输入
    """

    risk_cfg = cfg.risk
    base_risk = float(risk_cfg.get("risk_per_trade", 0.01))
    max_leverage = float(risk_cfg.get("max_leverage", 3.0))

    # 信号强度调整（0~1 -> 0.3~1.0倍）
    strength = max(0.0, min(1.0, signal_strength))
    strength_multiplier = 0.3 + 0.7 * strength

    # 波动率调整：高波动降低仓位
    vol = max(0.0, float(volatility))
    vol_adjustment = 1.0 / (1.0 + vol * 10.0)

    adjusted_risk = base_risk * strength_multiplier * vol_adjustment

    risk_capital = account_equity * adjusted_risk
    position_usdt = risk_capital * max_leverage

    # 不超过账户可承受的最大杠杆
    max_notional = account_equity * max_leverage
    position_usdt = float(max(0.0, min(position_usdt, max_notional)))

    return position_usdt


def adaptive_stop_loss_take_profit(
    entry_price: float,
    atr: float,
    signal_strength: float,
    market_regime: float,
) -> tuple[float, float]:
    """根据信号强度和市场状态调整止损止盈距离。

    返回值为 (stop_loss_distance, take_profit_distance)，均为价格绝对距离。
    """

    base_stop_multiplier = 2.0
    base_take_profit_multiplier = 3.0

    # 信号强度调整（强信号放宽止损/止盈）
    if signal_strength > 0.3:
        stop_multiplier = base_stop_multiplier * 1.2
        tp_multiplier = base_take_profit_multiplier * 1.2
    else:
        stop_multiplier = base_stop_multiplier * 0.8
        tp_multiplier = base_take_profit_multiplier * 0.8

    # 市场状态调整：趋势市场给更大的止盈空间
    if market_regime != 0:
        stop_multiplier *= 1.1
        tp_multiplier *= 1.3

    atr = max(0.0, float(atr))
    stop_loss_dist = atr * stop_multiplier
    take_profit_dist = atr * tp_multiplier

    return stop_loss_dist, take_profit_dist
