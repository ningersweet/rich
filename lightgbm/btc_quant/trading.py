"""
共享交易逻辑模块 - 被回测和实盘交易共用
统一开仓平仓逻辑、风控检查和盈亏计算
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd


@dataclass
class Position:
    """统一持仓数据结构，用于回测和实盘"""
    side: str  # 'long'/'short'
    entry_price: float
    entry_time: pd.Timestamp
    exposure: float  # 敞口倍数（杠杆×仓位）
    hold_period: int  # 预测持仓周期
    quantity: float = 0.0  # 实盘专用：持仓数量
    peak_pnl_pct: float = 0.0  # 最高盈亏百分比（用于追踪止损）
    peak_price: float = 0.0  # 最高价格（用于追踪止损）


@dataclass
class TradingState:
    """统一交易状态管理"""
    equity: float  # 当前权益
    peak_equity: float  # 峰值权益（用于计算回撤）
    daily_start_equity: float  # 每日起始权益
    consecutive_losses: int = 0  # 连续亏损次数
    daily_loss_paused: bool = False  # 每日亏损暂停
    drawdown_paused: bool = False  # 回撤暂停


def calculate_dynamic_exposure(
    predicted_rr: float,
    direction_prob: float,
    current_drawdown: float = 0.0,
    consecutive_losses: int = 0,
    max_exposure: float = 10.0
) -> float:
    """
    根据信号质量动态计算最优敞口
    
    参数:
        predicted_rr: 预测盈亏比
        direction_prob: 方向置信度
        current_drawdown: 当前回撤百分比（0-1）
        consecutive_losses: 连续亏损次数
        max_exposure: 最大敞口限制
    
    返回:
        exposure: 建议敞口（杠杆×仓位），范围 [1.0, max_exposure]
    """
    # 基础敞口：基于盈亏比和置信度
    rr_factor = min(predicted_rr / 2.5, 2.0)
    prob_factor = max((direction_prob - 0.5) / 0.5, 0)
    base_exposure = 2.0 + rr_factor * 3.0 + prob_factor * 3.0
    
    # 回撤惩罚
    if current_drawdown > 0.02:
        drawdown_penalty = 1.0 - (current_drawdown - 0.02) * 15
        drawdown_penalty = max(0.3, drawdown_penalty)
    else:
        drawdown_penalty = 1.0
    
    # 连续亏损惩罚
    if consecutive_losses >= 2:
        loss_penalty = 1.0 - min(consecutive_losses - 1, 5) * 0.15
        loss_penalty = max(0.2, loss_penalty)
    else:
        loss_penalty = 1.0
    
    # 最终敞口
    final_exposure = base_exposure * drawdown_penalty * loss_penalty
    final_exposure = np.clip(final_exposure, 1.0, max_exposure)
    
    return final_exposure


def should_open_position(
    trading_state: TradingState,
    should_trade: bool,
    current_drawdown: float,
    max_drawdown_pause: float = 0.10
) -> bool:
    """
    统一开仓条件判断
    
    参数:
        trading_state: 交易状态
        should_trade: 模型预测是否应该交易
        current_drawdown: 当前回撤百分比（仅用于动态敞口计算）
        max_drawdown_pause: 最大回撤暂停阈值
    
    返回:
        bool: 是否应该开仓
    """
    return (
        not trading_state.daily_loss_paused and
        not trading_state.drawdown_paused and
        should_trade
        # 注意：current_drawdown检查已移除，因为drawdown_paused已提供保护
        # 且calculate_dynamic_exposure会根据current_drawdown调整敞口
    )


def should_close_position(
    position: Position,
    current_price: float,
    stop_loss_pct: float = -0.03,
    use_trailing_stop: bool = True,
    trailing_stop_trigger: float = 0.01,
    trailing_stop_distance: float = 0.02,
    max_hold_period: Optional[int] = None,
    current_hold_period: int = 0
) -> Tuple[bool, str, bool, bool]:
    """
    统一平仓条件判断
    
    参数:
        position: 当前持仓
        current_price: 当前价格
        stop_loss_pct: 固定止损百分比（负数）
        use_trailing_stop: 是否使用追踪止损
        trailing_stop_trigger: 启动追踪止损的最小盈利百分比
        trailing_stop_distance: 追踪止损回撤距离
        max_hold_period: 最大持仓周期（None表示使用预测持仓周期）
        current_hold_period: 当前已持仓周期数
    
    返回:
        Tuple[是否平仓, 平仓原因, 是否固定止损触发, 是否追踪止损触发]
    """
    # 计算价格变化百分比
    if position.side == 'long':
        price_change_pct = (current_price - position.entry_price) / position.entry_price
    else:
        price_change_pct = (position.entry_price - current_price) / position.entry_price
    
    # 计算当前盈亏百分比（考虑杠杆）
    current_pnl_pct = price_change_pct * position.exposure
    
    # 1. 固定止损检查
    if current_pnl_pct <= stop_loss_pct:
        return True, f"固定止损({current_pnl_pct*100:.2f}% ≤ {stop_loss_pct*100:.1f}%)", True, False
    
    # 2. 追踪止损检查
    if use_trailing_stop and current_pnl_pct > trailing_stop_trigger:
        # 更新最高盈利点
        if current_pnl_pct > position.peak_pnl_pct:
            position.peak_pnl_pct = current_pnl_pct
            if position.side == 'long':
                position.peak_price = current_price
            else:
                position.peak_price = current_price
        
        # 计算从最高点的回撤
        pnl_drop_from_peak = position.peak_pnl_pct - current_pnl_pct
        if pnl_drop_from_peak > trailing_stop_distance:
            return True, f"追踪止损(从{position.peak_pnl_pct*100:.2f}%回落{current_pnl_pct*100:.2f}%)", False, True
    
    # 3. 持仓周期检查
    hold_limit = max_hold_period if max_hold_period is not None else position.hold_period
    if current_hold_period >= hold_limit:
        return True, f"持仓周期({current_hold_period}/{hold_limit})", False, False
    
    return False, "", False, False


def calculate_pnl(
    position: Position,
    current_price: float,
    equity: float
) -> float:
    """
    统一盈亏计算（与回测引擎一致）
    
    参数:
        position: 当前持仓
        current_price: 当前价格
        equity: 当前权益
    
    返回:
        pnl: 盈亏金额
    """
    if position.side == 'long':
        price_change_pct = (current_price - position.entry_price) / position.entry_price
    else:
        price_change_pct = (position.entry_price - current_price) / position.entry_price
    
    current_pnl_pct = price_change_pct * position.exposure
    return equity * current_pnl_pct


def update_trading_state(
    trading_state: TradingState,
    pnl: float,
    current_time: pd.Timestamp,
    max_daily_loss_pct: float = -0.20,
    max_drawdown_pause: float = 0.10
) -> None:
    """
    更新交易状态（权益、回撤、暂停状态）
    
    参数:
        trading_state: 交易状态
        pnl: 本次交易盈亏
        current_time: 当前时间
        max_daily_loss_pct: 最大单日亏损百分比
        max_drawdown_pause: 最大回撤暂停阈值
    """
    # 更新权益
    trading_state.equity += pnl
    
    # 更新峰值权益
    if trading_state.equity > trading_state.peak_equity:
        trading_state.peak_equity = trading_state.equity
    
    # 检查每日亏损暂停
    daily_loss_pct = (trading_state.equity - trading_state.daily_start_equity) / trading_state.daily_start_equity
    if daily_loss_pct < max_daily_loss_pct:
        trading_state.daily_loss_paused = True
    
    # 检查回撤暂停
    current_drawdown = (trading_state.peak_equity - trading_state.equity) / trading_state.peak_equity
    if current_drawdown > max_drawdown_pause:
        trading_state.drawdown_paused = True
    
    # 更新连续亏损计数
    if pnl > 0:
        trading_state.consecutive_losses = 0
    else:
        trading_state.consecutive_losses += 1


def reset_daily_state(
    trading_state: TradingState,
    new_day_start_equity: float
) -> None:
    """
    重置每日状态（新的一天开始时调用）
    
    参数:
        trading_state: 交易状态
        new_day_start_equity: 新的一天的起始权益
    """
    trading_state.daily_start_equity = new_day_start_equity
    trading_state.daily_loss_paused = False
    trading_state.drawdown_paused = False


# 兼容性函数：从字典创建Position对象（用于回测引擎迁移）
def position_from_dict(pos_dict: dict) -> Position:
    """从回测引擎的position字典创建Position对象"""
    return Position(
        side=pos_dict['side'],
        entry_price=pos_dict['entry_price'],
        entry_time=pd.Timestamp(pos_dict['entry_time']),
        exposure=pos_dict['exposure'],
        hold_period=pos_dict.get('hold_period', 50),
        peak_pnl_pct=pos_dict.get('peak_pnl_pct', 0.0),
        peak_price=pos_dict.get('peak_price', 0.0)
    )


# 兼容性函数：将Position对象转换为字典（用于回测引擎迁移）
def position_to_dict(position: Position) -> dict:
    """将Position对象转换为回测引擎兼容的字典"""
    return {
        'side': position.side,
        'entry_price': position.entry_price,
        'entry_time': position.entry_time,
        'exposure': position.exposure,
        'hold_period': position.hold_period,
        'peak_pnl_pct': position.peak_pnl_pct,
        'peak_price': position.peak_price
    }


def calculate_total_pnl_for_positions(
    positions: List[Position],
    current_price: float,
    equity: float
) -> Tuple[float, float, float]:
    """
    计算多仓位的总盈亏百分比、总敞口和加权平均入场价
    
    参数:
        positions: Position对象列表
        current_price: 当前价格
        equity: 当前权益（用于计算盈亏金额）
    
    返回:
        Tuple[总盈亏百分比, 总盈亏金额, 加权平均入场价]
    """
    if not positions:
        return 0.0, 0.0, 0.0
    
    total_pnl_pct = 0.0
    total_exposure = 0.0
    total_qty = 0.0
    weighted_entry_sum = 0.0
    
    for pos in positions:
        # 计算单个仓位价格变化百分比
        if pos.side == 'long':
            price_change_pct = (current_price - pos.entry_price) / pos.entry_price
        else:
            price_change_pct = (pos.entry_price - current_price) / pos.entry_price
        
        # 单个仓位盈亏百分比
        pnl_pct = price_change_pct * pos.exposure
        total_pnl_pct += pnl_pct
        total_exposure += pos.exposure
        total_qty += pos.quantity
        weighted_entry_sum += pos.entry_price * pos.quantity
    
    # 计算盈亏金额
    total_pnl = equity * total_pnl_pct
    
    # 计算加权平均入场价
    avg_entry_price = weighted_entry_sum / total_qty if total_qty > 0 else positions[0].entry_price
    
    return total_pnl_pct, total_pnl, avg_entry_price


def check_pyramid_conditions(
    positions: List[Position],
    current_price: float,
    new_signal_direction: int,  # 1 for long, -1 for short
    new_signal_rr: float,
    new_signal_prob: float,
    last_pyramid_time: Optional[pd.Timestamp],
    current_time: pd.Timestamp,
    pyramid_profit_threshold: float = 0.01,
    pyramid_min_rr: float = 3.0,
    pyramid_min_prob: float = 0.75,
    pyramid_min_bars: int = 5,
    max_total_exposure: float = 15.0,
    new_exposure: float = 0.0,
    kline_interval_minutes: int = 15
) -> bool:
    """
    检查金字塔加仓条件
    
    参数:
        positions: 当前仓位列表
        current_price: 当前价格
        new_signal_direction: 新信号方向（1=做多，-1=做空）
        new_signal_rr: 新信号盈亏比
        new_signal_prob: 新信号置信度
        last_pyramid_time: 上次加仓时间
        current_time: 当前时间
        pyramid_profit_threshold: 盈利阈值（默认1%）
        pyramid_min_rr: 最小盈亏比阈值（默认3.0）
        pyramid_min_prob: 最小置信度阈值（默认0.75）
        pyramid_min_bars: 最小K线间隔（默认5根）
        max_total_exposure: 总敞口上限（默认15倍）
        new_exposure: 新仓位敞口
        kline_interval_minutes: K线间隔分钟数（默认15分钟）
    
    返回:
        bool: 是否满足加仓条件
    """
    if not positions:
        return False
    
    # 检查方向一致性
    first_position_side = positions[0].side
    signal_direction_str = 'long' if new_signal_direction == 1 else 'short'
    if signal_direction_str != first_position_side:
        return False
    
    # 计算当前总盈亏百分比
    total_pnl_pct, _, _ = calculate_total_pnl_for_positions(positions, current_price, 1.0)
    
    # 条件1: 总体盈利超过阈值
    if total_pnl_pct <= pyramid_profit_threshold:
        return False
    
    # 条件2: 信号质量高
    if new_signal_rr < pyramid_min_rr or new_signal_prob < pyramid_min_prob:
        return False
    
    # 条件3: 时间间隔足够
    if last_pyramid_time is not None:
        time_diff = (current_time - last_pyramid_time).total_seconds()
        min_seconds = pyramid_min_bars * kline_interval_minutes * 60
        if time_diff < min_seconds:
            return False
    
    # 条件4: 总敞口不超过上限
    total_exposure = sum(p.exposure for p in positions)
    if total_exposure + new_exposure > max_total_exposure:
        return False
    
    return True


def should_close_multiple_positions(
    positions: List[Position],
    current_price: float,
    stop_loss_pct: float = -0.03,
    use_trailing_stop: bool = True,
    trailing_stop_trigger: float = 0.01,
    trailing_stop_distance: float = 0.02,
    current_hold_period: int = 0,
    kline_interval_minutes: int = 15
) -> Tuple[bool, str, bool, bool]:
    """
    多仓位统一平仓条件判断
    
    参数:
        positions: Position对象列表
        current_price: 当前价格
        stop_loss_pct: 固定止损百分比（负数）
        use_trailing_stop: 是否使用追踪止损
        trailing_stop_trigger: 启动追踪止损的最小盈利百分比
        trailing_stop_distance: 追踪止损回撤距离
        current_hold_period: 当前已持仓周期数（以首仓为准）
        kline_interval_minutes: K线间隔分钟数（默认15分钟）
    
    返回:
        Tuple[是否平仓, 平仓原因, 是否固定止损触发, 是否追踪止损触发]
    """
    if not positions:
        return False, "", False, False
    
    # 计算总盈亏百分比和更新仓位峰值
    total_pnl_pct = 0.0
    total_exposure = 0.0
    peak_pnl_pct = 0.0
    peak_updated = False
    
    for pos in positions:
        # 计算单个仓位价格变化百分比
        if pos.side == 'long':
            price_change_pct = (current_price - pos.entry_price) / pos.entry_price
        else:
            price_change_pct = (pos.entry_price - current_price) / pos.entry_price
        
        # 单个仓位盈亏百分比
        pnl_pct = price_change_pct * pos.exposure
        total_pnl_pct += pnl_pct
        total_exposure += pos.exposure
        
        # 更新单个仓位的峰值（用于追踪止损）
        if pnl_pct > pos.peak_pnl_pct:
            pos.peak_pnl_pct = pnl_pct
            pos.peak_price = current_price
            peak_updated = True
        
        # 更新全局峰值
        if pnl_pct > peak_pnl_pct:
            peak_pnl_pct = pnl_pct
    
    # 1. 固定止损检查（基于总盈亏）
    if total_pnl_pct <= stop_loss_pct:
        return True, f"固定止损({total_pnl_pct*100:.2f}% ≤ {stop_loss_pct*100:.1f}%)", True, False
    
    # 2. 追踪止损检查（任一仓位盈利>阈值后启用）
    if use_trailing_stop and peak_pnl_pct > trailing_stop_trigger:
        # 计算从最高点的回撤
        pnl_drop_from_peak = peak_pnl_pct - total_pnl_pct
        if pnl_drop_from_peak > trailing_stop_distance:
            return True, f"追踪止损(从{peak_pnl_pct*100:.2f}%回落{total_pnl_pct*100:.2f}%)", False, True
    
    # 3. 持仓周期检查（以首仓为准）
    if current_hold_period >= positions[0].hold_period:
        return True, f"持仓周期({current_hold_period}/{positions[0].hold_period})", False, False
    
    return False, "", False, False