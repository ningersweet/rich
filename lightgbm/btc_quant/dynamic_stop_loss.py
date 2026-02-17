"""
动态止损模块
基于ATR和移动止盈的风控系统
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def calculate_atr_stop_loss(
    entry_price: float,
    atr: float,
    direction: int,
    k: float = 2.0,
    min_stop_loss_pct: float = 0.01,
    max_stop_loss_pct: float = 0.05
) -> Tuple[float, float]:
    """
    基于ATR动态计算止损价和止损百分比
    
    参数:
        entry_price: 入场价格
        atr: 当前ATR值（绝对价格单位）
        direction: 交易方向（1=做多，-1=做空）
        k: ATR倍数（默认2.0）
        min_stop_loss_pct: 最小止损百分比（1%，防止止损过近）
        max_stop_loss_pct: 最大止损百分比（5%，防止止损过远）
    
    返回:
        (stop_loss_price, stop_loss_pct)
    
    示例:
        >>> # 做多BTC，入场价100,000，ATR=1,500
        >>> stop_price, stop_pct = calculate_atr_stop_loss(100000, 1500, 1, k=2.0)
        >>> # stop_price = 97,000, stop_pct = -0.03 (-3%)
    """
    # 计算止损价格
    if direction == 1:  # 做多
        stop_loss_price = entry_price - k * atr
    else:  # 做空
        stop_loss_price = entry_price + k * atr
    
    # 计算止损百分比
    stop_loss_pct = (stop_loss_price - entry_price) / entry_price * direction
    
    # 限制止损范围（防止过度激进或保守）
    stop_loss_pct = np.clip(stop_loss_pct, -max_stop_loss_pct, -min_stop_loss_pct)
    
    # 重新计算止损价格（基于限制后的百分比）
    if direction == 1:
        stop_loss_price = entry_price * (1 + stop_loss_pct)
    else:
        stop_loss_price = entry_price * (1 - stop_loss_pct)
    
    return stop_loss_price, stop_loss_pct


def calculate_adaptive_atr_multiplier(
    volatility_percentile: float,
    trend_strength: float,
    base_k: float = 2.0
) -> float:
    """
    自适应ATR倍数：根据市场波动率和趋势强度动态调整
    
    参数:
        volatility_percentile: 当前波动率百分位数（0-1）
        trend_strength: 趋势强度（-1到1，负数=下跌趋势，正数=上涨趋势）
        base_k: 基础ATR倍数（默认2.0）
    
    返回:
        调整后的ATR倍数
    
    逻辑:
        - 高波动期（percentile>0.8）：放宽止损（k+0.5）
        - 低波动期（percentile<0.3）：收紧止损（k-0.3）
        - 强趋势期：略放宽止损（避免被噪音洗出）
    """
    k = base_k
    
    # 波动率调整
    if volatility_percentile > 0.8:
        k += 0.5  # 高波动：放宽止损
    elif volatility_percentile < 0.3:
        k -= 0.3  # 低波动：收紧止损
    
    # 趋势强度调整
    if abs(trend_strength) > 0.7:
        k += 0.2  # 强趋势：略放宽（给趋势更多空间）
    
    # 限制范围
    k = np.clip(k, 1.5, 3.5)
    
    return k


class TrailingStopManager:
    """
    移动止盈管理器
    
    功能:
        - 跟踪最高/最低价
        - 利润回撤触发平仓
        - 保护已有利润
    
    使用示例:
        >>> manager = TrailingStopManager(trailing_pct=0.5, min_profit_pct=0.01)
        >>> # 价格上涨，更新trailing stop
        >>> trailing_price = manager.update(102000, 100000, direction=1)
        >>> # 检查是否触发
        >>> if manager.should_exit(101500, direction=1):
        >>>     print("触发移动止盈，平仓")
    """
    
    def __init__(
        self,
        trailing_pct: float = 0.5,
        min_profit_pct: float = 0.01,
        enable_dynamic_trailing: bool = True
    ):
        """
        参数:
            trailing_pct: 利润回撤比例（0.5表示回撤50%时平仓）
            min_profit_pct: 最小利润要求（1%才启动trailing stop）
            enable_dynamic_trailing: 是否启用动态trailing（根据趋势强度调整）
        """
        self.trailing_pct = trailing_pct
        self.min_profit_pct = min_profit_pct
        self.enable_dynamic_trailing = enable_dynamic_trailing
        
        # 状态变量
        self.highest_price = None
        self.lowest_price = None
        self.trailing_stop_price = None
        self.max_profit_reached = 0.0
        self.is_active = False
    
    def reset(self):
        """重置状态（新交易时调用）"""
        self.highest_price = None
        self.lowest_price = None
        self.trailing_stop_price = None
        self.max_profit_reached = 0.0
        self.is_active = False
    
    def update(
        self,
        current_price: float,
        entry_price: float,
        direction: int,
        trend_strength: Optional[float] = None
    ) -> Optional[float]:
        """
        更新移动止盈价
        
        参数:
            current_price: 当前价格
            entry_price: 入场价格
            direction: 方向（1=做多，-1=做空）
            trend_strength: 趋势强度（可选，用于动态调整trailing_pct）
        
        返回:
            当前trailing_stop价格（如果未激活返回None）
        """
        # 计算当前未实现盈亏
        unrealized_pnl_pct = (current_price - entry_price) / entry_price * direction
        
        # 检查是否达到最小利润要求
        if unrealized_pnl_pct < self.min_profit_pct:
            return None
        
        # 激活trailing stop
        if not self.is_active:
            self.is_active = True
        
        # 更新最高/最低价
        if direction == 1:  # 做多
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
                self.max_profit_reached = unrealized_pnl_pct
                
                # 计算trailing stop价格
                self._update_trailing_stop(entry_price, direction, trend_strength)
        
        else:  # 做空
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price
                self.max_profit_reached = unrealized_pnl_pct
                
                # 计算trailing stop价格
                self._update_trailing_stop(entry_price, direction, trend_strength)
        
        return self.trailing_stop_price
    
    def _update_trailing_stop(
        self,
        entry_price: float,
        direction: int,
        trend_strength: Optional[float]
    ):
        """内部方法：更新trailing stop价格"""
        # 动态调整trailing_pct（根据趋势强度）
        trailing_pct = self.trailing_pct
        if self.enable_dynamic_trailing and trend_strength is not None:
            if abs(trend_strength) > 0.7:
                # 强趋势：放宽回撤容忍度（让利润多跑一会）
                trailing_pct += 0.1
            trailing_pct = np.clip(trailing_pct, 0.3, 0.7)
        
        # 计算允许的利润回撤
        allowed_profit = self.max_profit_reached * (1 - trailing_pct)
        
        # 计算trailing stop价格
        if direction == 1:  # 做多
            self.trailing_stop_price = entry_price * (1 + allowed_profit)
        else:  # 做空
            self.trailing_stop_price = entry_price * (1 - allowed_profit)
    
    def should_exit(self, current_price: float, direction: int) -> bool:
        """
        判断是否触发移动止盈
        
        参数:
            current_price: 当前价格
            direction: 方向（1=做多，-1=做空）
        
        返回:
            True表示应该平仓
        """
        if not self.is_active or self.trailing_stop_price is None:
            return False
        
        if direction == 1:  # 做多：价格跌破trailing stop
            return current_price <= self.trailing_stop_price
        else:  # 做空：价格涨破trailing stop
            return current_price >= self.trailing_stop_price
    
    def get_status(self) -> dict:
        """
        获取当前状态（用于日志和监控）
        
        返回:
            状态字典
        """
        return {
            'is_active': self.is_active,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price,
            'trailing_stop_price': self.trailing_stop_price,
            'max_profit_reached': self.max_profit_reached,
            'trailing_pct': self.trailing_pct
        }


def calculate_dynamic_stop_loss_params(
    klines: pd.DataFrame,
    lookback_window: int = 100
) -> dict:
    """
    基于历史数据计算动态止损参数
    
    参数:
        klines: K线数据（必须包含atr_14列）
        lookback_window: 回溯窗口（用于计算波动率百分位）
    
    返回:
        参数字典，包含：
        - atr: 当前ATR值
        - volatility_percentile: 波动率百分位
        - trend_strength: 趋势强度
        - recommended_k: 推荐的ATR倍数
    """
    if 'atr_14' not in klines.columns:
        raise ValueError("klines必须包含atr_14列，请先运行特征工程")
    
    # 获取最新ATR
    current_atr = klines['atr_14'].iloc[-1]
    
    # 计算波动率百分位
    recent_atr = klines['atr_14'].iloc[-lookback_window:]
    volatility_percentile = (recent_atr < current_atr).mean()
    
    # 计算趋势强度（20周期动量）
    if len(klines) >= 20:
        trend_strength = klines['close'].iloc[-1] / klines['close'].iloc[-20] - 1.0
        trend_strength = np.clip(trend_strength, -1.0, 1.0)
    else:
        trend_strength = 0.0
    
    # 计算推荐的ATR倍数
    recommended_k = calculate_adaptive_atr_multiplier(
        volatility_percentile,
        trend_strength,
        base_k=2.0
    )
    
    return {
        'atr': current_atr,
        'volatility_percentile': volatility_percentile,
        'trend_strength': trend_strength,
        'recommended_k': recommended_k
    }
