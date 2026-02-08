from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import Config
from .signals import SignalResult


@dataclass
class BacktestTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # long / short
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[BacktestTrade]
    stats: dict[str, float]


def _compute_backtest_stats(equity: pd.Series, trades: List[BacktestTrade]) -> dict[str, float]:
    """计算回测统计指标（收益、回撤、胜率等）。"""

    stats: dict[str, float] = {}
    if len(equity) > 1:
        start_val = float(equity.iloc[0])
        end_val = float(equity.iloc[-1])
        stats["total_return_pct"] = (end_val / start_val - 1.0) * 100.0

        equity_array = equity.values.astype("float64")
        cummax = np.maximum.accumulate(equity_array)
        dd = equity_array / cummax - 1.0
        stats["max_drawdown_pct"] = float(dd.min() * 100.0)

        bar_returns = equity.pct_change().dropna()
        if not bar_returns.empty and bar_returns.std() > 0:
            stats["avg_bar_return"] = float(bar_returns.mean())
            stats["bar_return_vol"] = float(bar_returns.std())
        else:
            stats["avg_bar_return"] = 0.0
            stats["bar_return_vol"] = 0.0
    else:
        stats["total_return_pct"] = 0.0
        stats["max_drawdown_pct"] = 0.0
        stats["avg_bar_return"] = 0.0
        stats["bar_return_vol"] = 0.0

    if trades:
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        stats["num_trades"] = float(len(trades))
        stats["win_rate_pct"] = float(len(wins) / len(trades) * 100.0)
        stats["avg_pnl_per_trade"] = float(np.mean(pnls))
    else:
        stats["num_trades"] = 0.0
        stats["win_rate_pct"] = 0.0
        stats["avg_pnl_per_trade"] = 0.0

    return stats


def run_backtest(
    cfg: Config,
    klines: pd.DataFrame,
    signals: pd.Series,
    position_usdt: float = None,
    features: pd.DataFrame = None,
    position_ratios: pd.Series = None,
) -> dict:
    """增强版回测：
    
    - 支持动态仓位管理（通过position_ratios参数）
    - 支持固定仓位（通过position_usdt参数，兼容旧版）
    - 只支持单向持仓：要么多、要么空、要么空仓。
    - 使用收盘价成交，考虑手续费和简单滑点。
    
    Args:
        cfg: 配置对象
        klines: K线数据
        signals: 交易信号Series（1=做多，-1=做空，0=观望）
        position_usdt: 固定仓位大小（USDT）
        features: 特征DataFrame（可选）
        position_ratios: 动态仓位比例Series（0-1，可选）
    
    Returns:
        回测结果字典
    """

    fee_rate = float(cfg.backtest.get("fee_rate", 0.0004))
    slippage = float(cfg.backtest.get("slippage", 0.0005))
    initial_balance = float(cfg.backtest.get("initial_balance", 1000.0))
    
    # 确定仓位计算方式
    if position_usdt is None:
        # 使用动态仓位管理
        risk_per_trade = float(cfg.risk.get("risk_per_trade", 0.05))
        base_position_usdt = initial_balance * risk_per_trade
        max_leverage = float(cfg.risk.get("max_leverage", 10.0))
        use_dynamic_position = True
    else:
        # 使用固定仓位
        base_position_usdt = position_usdt
        use_dynamic_position = False

    closes = klines["close"].values.astype("float64")
    times = klines["close_time"].values

    equity = initial_balance
    equity_curve = []
    trades: List[BacktestTrade] = []

    position_side = "flat"
    entry_price = 0.0
    quantity = 0.0
    entry_time = None

    for i, sig in enumerate(signals):
        price = float(closes[i])
        ts = pd.to_datetime(times[i])
        
        # 计算当前仓位大小
        if use_dynamic_position and position_ratios is not None and i < len(position_ratios):
            position_ratio = float(position_ratios.iloc[i])
            current_position_usdt = base_position_usdt * position_ratio * max_leverage
        else:
            current_position_usdt = base_position_usdt

        # 更新权益曲线（逐bar记录）
        equity_curve.append((ts, equity))
        
        # 将信号转换为字符串格式
        if sig == 1:
            sig_str = "long"
        elif sig == -1:
            sig_str = "short"
        else:
            sig_str = "flat"

        if position_side == "flat":
            if sig_str == "long":
                qty = current_position_usdt / price
                cost = current_position_usdt * fee_rate
                equity -= cost
                position_side = "long"
                entry_price = price * (1 + slippage)
                quantity = qty
                entry_time = ts
            elif sig_str == "short":
                qty = current_position_usdt / price
                cost = current_position_usdt * fee_rate
                equity -= cost
                position_side = "short"
                entry_price = price * (1 - slippage)
                quantity = qty
                entry_time = ts
        else:
            # 已有持仓
            should_close = False
            new_side = position_side
            if sig_str == "flat":
                should_close = True
                new_side = "flat"
            elif sig_str == "long" and position_side == "short":
                should_close = True
                new_side = "long"
            elif sig_str == "short" and position_side == "long":
                should_close = True
                new_side = "short"

            if should_close and position_side != "flat":
                if position_side == "long":
                    exit_price = price * (1 - slippage)
                    gross_pnl = (exit_price - entry_price) * quantity
                else:
                    exit_price = price * (1 + slippage)
                    gross_pnl = (entry_price - exit_price) * quantity

                notional = entry_price * quantity
                cost = notional * fee_rate  # 平仓费
                net_pnl = gross_pnl - cost
                equity += net_pnl

                trades.append(
                    BacktestTrade(
                        entry_time=entry_time,
                        exit_time=ts,
                        side=position_side,
                        entry_price=float(entry_price),
                        exit_price=float(exit_price),
                        quantity=float(quantity),
                        pnl=float(net_pnl),
                    )
                )

                position_side = "flat"
                quantity = 0.0
                entry_price = 0.0
                entry_time = None

                # 若有反向信号，则在同一bar开新仓
                if new_side != "flat":
                    if new_side == "long":
                        qty = current_position_usdt / price
                        cost = current_position_usdt * fee_rate
                        equity -= cost
                        position_side = "long"
                        entry_price = price * (1 + slippage)
                        quantity = qty
                        entry_time = ts
                    elif new_side == "short":
                        qty = current_position_usdt / price
                        cost = current_position_usdt * fee_rate
                        equity -= cost
                        position_side = "short"
                        entry_price = price * (1 - slippage)
                        quantity = qty
                        entry_time = ts

    equity_series = pd.Series(
        [e for _, e in equity_curve], index=[t for t, _ in equity_curve], name="equity"
    )

    stats = _compute_backtest_stats(equity_series, trades)
    
    # 返回字典格式（兼容新版调用）
    return {
        'final_equity': float(equity),
        'total_return': stats['total_return_pct'],
        'max_drawdown': stats['max_drawdown_pct'],
        'win_rate': stats['win_rate_pct'],
        'total_trades': int(stats['num_trades']),
        'avg_profit_per_trade': stats['avg_pnl_per_trade'],
        'equity_curve': equity_series,
        'trades': trades,
        'stats': stats,
    }


def run_backtest_with_triple_exit(
    cfg: Config,
    klines: pd.DataFrame,
    signals: pd.Series,
    features: pd.DataFrame = None,
    position_ratios: pd.Series = None,
) -> dict:
    """增强版回测：支持三重出场机制。
    
    三重出场：
    1. 固定止盈：入场价 ± (ATR × take_profit_multiplier)
    2. 移动止损：初始止损(ATR × stop_loss_multiplier) → 盈亏平衡 → 跟随
    3. 时间止损：持仓超过max_holding_bars未达止盈，则平仓
    
    Args:
        cfg: 配置对象
        klines: K线数据
        signals: 交易信号Series（1=做多，-1=做空，0=观望）
        features: 特征DataFrame（用于获取ATR）
        position_ratios: 动态仓位比例Series（0-1）
    
    Returns:
        回测结果字典
    """
    fee_rate = float(cfg.backtest.get("fee_rate", 0.0004))
    slippage = float(cfg.backtest.get("slippage", 0.0005))
    initial_balance = float(cfg.backtest.get("initial_balance", 1000.0))
    
    # 三重出场参数
    use_take_profit = cfg.backtest.get("use_take_profit", True)
    take_profit_atr_multiplier = float(cfg.backtest.get("take_profit_atr_multiplier", 2.0))
    
    use_stop_loss = cfg.backtest.get("use_stop_loss", True)
    stop_loss_atr_multiplier = float(cfg.backtest.get("stop_loss_atr_multiplier", 1.2))
    
    use_trailing_stop = cfg.backtest.get("use_trailing_stop", True)
    trailing_stop_atr_multiplier = float(cfg.backtest.get("trailing_stop_atr_multiplier", 1.2))
    trailing_activation_atr = float(cfg.backtest.get("trailing_activation_atr", 1.0))  # DeepSeek优化：追踪止损激活阈值
    
    use_time_exit = cfg.backtest.get("use_time_exit", True)
    max_holding_bars = int(cfg.backtest.get("max_holding_bars", 12))  # 3小时（12根K线）
    
    # 仓位管理
    risk_per_trade = float(cfg.risk.get("risk_per_trade", 0.01))
    max_leverage = float(cfg.risk.get("max_leverage", 5.0))
    base_position_usdt = initial_balance * risk_per_trade
    
    # 获取ATR数据（假设 features 中有 atr_14 列）
    if features is not None and 'atr_14' in features.columns:
        atr_values = features['atr_14'].values
    else:
        # 如果没有ATR，使用价格的固定比例
        atr_values = klines['close'].values * 0.01  # 1%作为ATR
    
    closes = klines["close"].values.astype("float64")
    highs = klines["high"].values.astype("float64")
    lows = klines["low"].values.astype("float64")
    times = klines["close_time"].values
    
    equity = initial_balance
    equity_curve = []
    trades: List[BacktestTrade] = []
    
    position_side = "flat"
    entry_price = 0.0
    quantity = 0.0
    entry_time = None
    entry_bar_idx = 0
    
    # 止损止盈价格
    stop_loss_price = 0.0
    take_profit_price = 0.0
    trailing_stop_price = 0.0
    trailing_activated = False  # DeepSeek优化：追踪止损是否已激活
    
    for i, sig in enumerate(signals):
        price = float(closes[i])
        high = float(highs[i])
        low = float(lows[i])
        ts = pd.to_datetime(times[i])
        atr = float(atr_values[i]) if i < len(atr_values) else price * 0.01
        
        # 计算当前仓位大小
        if position_ratios is not None and i < len(position_ratios):
            position_ratio = float(position_ratios.iloc[i])
            current_position_usdt = base_position_usdt * position_ratio * max_leverage
        else:
            current_position_usdt = base_position_usdt * max_leverage
        
        equity_curve.append((ts, equity))
        
        # 将信号转换为字符串格式
        if sig == 1:
            sig_str = "long"
        elif sig == -1:
            sig_str = "short"
        else:
            sig_str = "flat"
        
        # === 检查出场条件 ===
        if position_side != "flat":
            should_exit = False
            exit_reason = ""
            
            # 1. 检查固定止盈
            if use_take_profit and take_profit_price > 0:
                if position_side == "long" and high >= take_profit_price:
                    should_exit = True
                    exit_reason = "take_profit"
                elif position_side == "short" and low <= take_profit_price:
                    should_exit = True
                    exit_reason = "take_profit"
            
            # 2. 检查止损
            if not should_exit and use_stop_loss and stop_loss_price > 0:
                if position_side == "long" and low <= stop_loss_price:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif position_side == "short" and high >= stop_loss_price:
                    should_exit = True
                    exit_reason = "stop_loss"
            
            # 3. 检查移动止损（DeepSeek优化：需要先激活）
            if not should_exit and use_trailing_stop:
                # 计算当前盈亏
                if position_side == "long":
                    current_pnl_atr = (price - entry_price) / atr
                elif position_side == "short":
                    current_pnl_atr = (entry_price - price) / atr
                else:
                    current_pnl_atr = 0
                
                # 检查是否需要激活追踪止损
                if not trailing_activated and current_pnl_atr >= trailing_activation_atr:
                    trailing_activated = True
                    # 激活时设置追踪止损价格
                    if position_side == "long":
                        trailing_stop_price = price - atr * trailing_stop_atr_multiplier
                    elif position_side == "short":
                        trailing_stop_price = price + atr * trailing_stop_atr_multiplier
                
                # 如果已激活，检查是否触发止损
                if trailing_activated and trailing_stop_price > 0:
                    if position_side == "long" and low <= trailing_stop_price:
                        should_exit = True
                        exit_reason = "trailing_stop"
                    elif position_side == "short" and high >= trailing_stop_price:
                        should_exit = True
                        exit_reason = "trailing_stop"
                    
                    # 更新移动止损（向盈利方向移动）
                    if not should_exit:
                        if position_side == "long":
                            new_trailing = high - atr * trailing_stop_atr_multiplier
                            trailing_stop_price = max(trailing_stop_price, new_trailing)
                        elif position_side == "short":
                            new_trailing = low + atr * trailing_stop_atr_multiplier
                            trailing_stop_price = min(trailing_stop_price, new_trailing)
            
            # 4. 检查时间止损
            if not should_exit and use_time_exit:
                holding_bars = i - entry_bar_idx
                if holding_bars >= max_holding_bars:
                    should_exit = True
                    exit_reason = "time_exit"
            
            # 5. 检查信号反向
            if not should_exit:
                if sig_str == "flat":
                    should_exit = True
                    exit_reason = "signal_exit"
                elif (sig_str == "long" and position_side == "short") or (sig_str == "short" and position_side == "long"):
                    should_exit = True
                    exit_reason = "reverse_signal"
            
            # === 执行出场 ===
            if should_exit:
                if position_side == "long":
                    exit_price = price * (1 - slippage)
                    gross_pnl = (exit_price - entry_price) * quantity
                else:
                    exit_price = price * (1 + slippage)
                    gross_pnl = (entry_price - exit_price) * quantity
                
                notional = entry_price * quantity
                cost = notional * fee_rate
                net_pnl = gross_pnl - cost
                equity += net_pnl
                
                trades.append(
                    BacktestTrade(
                        entry_time=entry_time,
                        exit_time=ts,
                        side=position_side,
                        entry_price=float(entry_price),
                        exit_price=float(exit_price),
                        quantity=float(quantity),
                        pnl=float(net_pnl),
                    )
                )
                
                position_side = "flat"
                quantity = 0.0
                entry_price = 0.0
                entry_time = None
                stop_loss_price = 0.0
                take_profit_price = 0.0
                trailing_stop_price = 0.0
                trailing_activated = False  # 重置追踪止损激活状态
                
                # 若有反向信号，则开新仓
                if sig_str in ["long", "short"]:
                    sig_str_new = sig_str
                else:
                    sig_str_new = None
            else:
                sig_str_new = None
        else:
            sig_str_new = sig_str if sig_str in ["long", "short"] else None
        
        # === 开仓 ===
        if position_side == "flat" and sig_str_new in ["long", "short"]:
            qty = current_position_usdt / price
            cost = current_position_usdt * fee_rate
            equity -= cost
            
            if sig_str_new == "long":
                position_side = "long"
                entry_price = price * (1 + slippage)
                quantity = qty
                entry_time = ts
                entry_bar_idx = i
                
                # 设置止损止盈
                if use_stop_loss:
                    stop_loss_price = entry_price - atr * stop_loss_atr_multiplier
                if use_take_profit:
                    take_profit_price = entry_price + atr * take_profit_atr_multiplier
                if use_trailing_stop:
                    trailing_stop_price = 0.0  # DeepSeek优化：初始不设置，等待激活
                    trailing_activated = False
                    
            elif sig_str_new == "short":
                position_side = "short"
                entry_price = price * (1 - slippage)
                quantity = qty
                entry_time = ts
                entry_bar_idx = i
                
                # 设置止损止盈
                if use_stop_loss:
                    stop_loss_price = entry_price + atr * stop_loss_atr_multiplier
                if use_take_profit:
                    take_profit_price = entry_price - atr * take_profit_atr_multiplier
                if use_trailing_stop:
                    trailing_stop_price = 0.0  # DeepSeek优化：初始不设置，等待激活
                    trailing_activated = False
    
    equity_series = pd.Series(
        [e for _, e in equity_curve], index=[t for t, _ in equity_curve], name="equity"
    )
    
    stats = _compute_backtest_stats(equity_series, trades)
    
    return {
        'final_equity': float(equity),
        'total_return': stats['total_return_pct'],
        'max_drawdown': stats['max_drawdown_pct'],
        'win_rate': stats['win_rate_pct'],
        'total_trades': int(stats['num_trades']),
        'avg_profit_per_trade': stats['avg_pnl_per_trade'],
        'equity_curve': equity_series,
        'trades': trades,
        'stats': stats,
    }
