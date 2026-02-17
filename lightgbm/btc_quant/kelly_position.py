#!/usr/bin/env python3
"""
凯利公式仓位管理模块
基于预测胜率和盈亏比，科学计算最优敞口
"""
import numpy as np
from typing import Tuple, Optional


def calculate_kelly_fraction(
    win_rate: float,
    profit_loss_ratio: float,
    kelly_criterion: float = 0.5
) -> float:
    """
    使用凯利公式计算最优仓位比例
    
    凯利公式:
        f* = (p × b - q) / b
        
        其中:
            f* = 最优仓位比例
            p = 胜率 (win_rate)
            q = 败率 (1 - win_rate)
            b = 盈亏比 (profit_loss_ratio)
    
    参数:
        win_rate: 胜率 (0-1之间，如0.75表示75%胜率)
        profit_loss_ratio: 盈亏比 (平均盈利/平均亏损)
        kelly_criterion: 凯利系数 (0.25-1.0)
            - 0.25: 保守（1/4凯利）
            - 0.50: 中等（半凯利，推荐）
            - 1.00: 激进（全凯利）
    
    返回:
        最优仓位比例 (0-1之间)
    
    示例:
        >>> # 胜率75%，盈亏比1.6，使用半凯利
        >>> kelly_f = calculate_kelly_fraction(0.75, 1.6, kelly_criterion=0.5)
        >>> kelly_f ≈ 0.31 (即31%仓位)
    """
    if not (0 <= win_rate <= 1):
        raise ValueError(f"胜率必须在0-1之间，当前：{win_rate}")
    
    if profit_loss_ratio <= 0:
        return 0.0  # 盈亏比为负或零时不交易
    
    p = win_rate
    q = 1 - p
    b = profit_loss_ratio
    
    # 凯利公式
    kelly_fraction = (p * b - q) / b
    
    # 只有正期望时才下注
    if kelly_fraction <= 0:
        return 0.0
    
    # 应用凯利系数（降低风险）
    adjusted_kelly = kelly_fraction * kelly_criterion
    
    # 限制最大仓位（避免过度集中）
    adjusted_kelly = min(adjusted_kelly, 1.0)
    
    return adjusted_kelly


def calculate_confidence_adjusted_kelly(
    predicted_rr: float,
    direction_prob: float,
    historical_win_rate: Optional[float] = None,
    historical_pl_ratio: Optional[float] = None,
    kelly_criterion: float = 0.5,
    max_exposure: float = 10.0
) -> Tuple[float, str]:
    """
    基于预测信号和历史表现，计算置信度调整的凯利敞口
    
    核心逻辑:
        1. 使用历史数据计算基础凯利仓位
        2. 根据当前信号质量调整（RR和Prob）
        3. 结合动态敞口上限
    
    参数:
        predicted_rr: 预测盈亏比
        direction_prob: 方向置信度
        historical_win_rate: 历史胜率（可选，用于回测）
        historical_pl_ratio: 历史盈亏比（可选，用于回测）
        kelly_criterion: 凯利系数
        max_exposure: 最大敞口限制
    
    返回:
        (optimal_exposure, reasoning)
            - optimal_exposure: 最优敞口
            - reasoning: 决策说明
    """
    
    # 1. 估算当前交易的预期胜率和盈亏比
    if historical_win_rate is not None and historical_pl_ratio is not None:
        # 使用历史数据
        base_win_rate = historical_win_rate
        base_pl_ratio = historical_pl_ratio
    else:
        # 使用预测值估算
        # RR与胜率的经验关系：高RR通常伴随低胜率
        # 假设基准胜率为65%，RR每+1，胜率±2%
        base_win_rate = 0.65 + (direction_prob - 0.65) * 0.5
        base_win_rate = np.clip(base_win_rate, 0.4, 0.85)
        
        base_pl_ratio = predicted_rr
    
    # 2. 计算基础凯利仓位
    base_kelly = calculate_kelly_fraction(
        win_rate=base_win_rate,
        profit_loss_ratio=base_pl_ratio,
        kelly_criterion=kelly_criterion
    )
    
    # 3. 信号质量调整因子
    # RR贡献：RR越高，增加敞口
    if predicted_rr >= 6.0:
        rr_multiplier = 1.5
    elif predicted_rr >= 4.0:
        rr_multiplier = 1.2 + (predicted_rr - 4.0) * 0.15
    elif predicted_rr >= 2.5:
        rr_multiplier = 1.0 + (predicted_rr - 2.5) * 0.13
    else:
        rr_multiplier = 0.5  # RR不足2.5，大幅降低
    
    # Prob贡献：置信度越高，增加敞口
    if direction_prob >= 0.85:
        prob_multiplier = 1.5
    elif direction_prob >= 0.75:
        prob_multiplier = 1.2 + (direction_prob - 0.75) * 3.0
    elif direction_prob >= 0.65:
        prob_multiplier = 1.0 + (direction_prob - 0.65) * 2.0
    else:
        prob_multiplier = 0.5
    
    # 4. 综合调整
    signal_quality_factor = rr_multiplier * prob_multiplier
    signal_quality_factor = np.clip(signal_quality_factor, 0.5, 2.0)
    
    adjusted_exposure = base_kelly * signal_quality_factor * max_exposure
    
    # 5. 最小敞口保护（至少1倍）
    if adjusted_exposure < 1.0 and base_kelly > 0:
        adjusted_exposure = max(1.0, base_kelly * max_exposure * 0.5)
    
    # 6. 生成决策说明
    if adjusted_exposure >= max_exposure * 0.9:
        reasoning = "极高置信度信号，使用接近上限的敞口"
    elif adjusted_exposure >= max_exposure * 0.7:
        reasoning = "高置信度信号，使用较高敞口"
    elif adjusted_exposure >= max_exposure * 0.5:
        reasoning = "中等置信度信号，使用标准敞口"
    elif adjusted_exposure >= 1.0:
        reasoning = "低置信度信号，使用保守敞口"
    else:
        reasoning = "信号质量不足，建议跳过"
    
    return adjusted_exposure, reasoning


class KellyPositionManager:
    """
    凯利公式仓位管理器
    
    功能:
        - 基于历史表现动态调整凯利参数
        - 跟踪连续亏损并降低敞口
        - 支持多种风险模式（保守/中性/激进）
    
    使用示例:
        >>> manager = KellyPositionManager(kelly_criterion=0.5, max_exposure=10.0)
        >>> exposure, reason = manager.calculate_optimal_exposure(
        ...     predicted_rr=4.5,
        ...     direction_prob=0.78,
        ...     current_drawdown=0.03
        ... )
    """
    
    def __init__(
        self,
        kelly_criterion: float = 0.5,
        max_exposure: float = 10.0,
        risk_mode: str = 'balanced'
    ):
        """
        参数:
            kelly_criterion: 凯利系数 (0.25-1.0)
            max_exposure: 最大敞口限制
            risk_mode: 风险模式
                - 'conservative': 保守模式 (kelly×0.5)
                - 'balanced': 平衡模式 (kelly×1.0)
                - 'aggressive': 激进模式 (kelly×1.5)
        """
        self.kelly_criterion = kelly_criterion
        self.max_exposure = max_exposure
        
        if risk_mode == 'conservative':
            self.risk_multiplier = 0.5
        elif risk_mode == 'balanced':
            self.risk_multiplier = 1.0
        elif risk_mode == 'aggressive':
            self.risk_multiplier = 1.5
        else:
            raise ValueError(f"未知风险模式：{risk_mode}")
        
        # 状态跟踪
        self.consecutive_losses = 0
        self.total_trades = 0
        self.win_count = 0
        self.total_pnl = 0.0
        self.peak_equity = 1000.0
        self.current_equity = 1000.0
    
    def reset(self, initial_equity: float = 1000.0):
        """重置状态（新周期或重启时使用）"""
        self.consecutive_losses = 0
        self.total_trades = 0
        self.win_count = 0
        self.total_pnl = 0.0
        self.peak_equity = initial_equity
        self.current_equity = initial_equity
    
    def update_equity(self, new_equity: float, pnl: float):
        """
        更新权益和盈亏记录
        
        参数:
            new_equity: 最新权益
            pnl: 该笔交易的盈亏（USDT）
        """
        self.current_equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.win_count += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
    
    def get_statistics(self) -> dict:
        """获取当前统计指标"""
        if self.total_trades == 0:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_loss_ratio': 0.0
            }
        
        win_rate = self.win_count / self.total_trades
        
        # 简化估算：假设平均盈利和亏损
        # 实际应用中应该从交易记录中计算
        if self.win_count > 0:
            avg_win_estimate = abs(self.total_pnl) / self.win_count * 0.6
        else:
            avg_win_estimate = 0.0
        
        losing_trades = self.total_trades - self.win_count
        if losing_trades > 0:
            avg_loss_estimate = abs(self.total_pnl) / losing_trades * 0.4
        else:
            avg_loss_estimate = 0.001  # 避免除零
        
        pl_ratio = avg_win_estimate / avg_loss_estimate if avg_loss_estimate > 0 else 0.0
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win_estimate,
            'avg_loss': avg_loss_estimate,
            'profit_loss_ratio': pl_ratio
        }
    
    def calculate_optimal_exposure(
        self,
        predicted_rr: float,
        direction_prob: float,
        current_drawdown: float = 0.0
    ) -> Tuple[float, str]:
        """
        计算最优敞口（综合考虑凯利公式、信号质量和风控）
        
        参数:
            predicted_rr: 预测盈亏比
            direction_prob: 方向置信度
            current_drawdown: 当前回撤百分比
        
        返回:
            (optimal_exposure, reasoning)
        """
        # 1. 获取当前统计
        stats = self.get_statistics()
        
        # 2. 基础凯利计算
        base_exposure, reasoning = calculate_confidence_adjusted_kelly(
            predicted_rr=predicted_rr,
            direction_prob=direction_prob,
            historical_win_rate=stats['win_rate'] if self.total_trades >= 5 else None,
            historical_pl_ratio=stats['profit_loss_ratio'] if self.total_trades >= 5 else None,
            kelly_criterion=self.kelly_criterion,
            max_exposure=self.max_exposure
        )
        
        # 3. 连续亏损惩罚
        if self.consecutive_losses >= 2:
            loss_penalty = 1.0 - min(self.consecutive_losses - 1, 5) * 0.15
            loss_penalty = max(0.3, loss_penalty)
            base_exposure *= loss_penalty
            reasoning += f" (连续亏损{self.consecutive_losses}次，敞口×{loss_penalty:.2f})"
        
        # 4. 回撤惩罚
        if current_drawdown > 0.02:
            drawdown_penalty = 1.0 - (current_drawdown - 0.02) * 15
            drawdown_penalty = max(0.3, drawdown_penalty)
            base_exposure *= drawdown_penalty
            reasoning += f" (回撤{current_drawdown*100:.1f}%，敞口×{drawdown_penalty:.2f})"
        
        # 5. 应用风险模式乘数
        base_exposure *= self.risk_multiplier
        base_exposure = np.clip(base_exposure, 0.0, self.max_exposure)
        
        # 6. 最小敞口保护
        if base_exposure > 0 and base_exposure < 1.0:
            base_exposure = 1.0
            reasoning += " (启用最小敞口保护)"
        
        return base_exposure, reasoning


def backtest_kelly_position_sizing(trades_history, initial_balance=1000.0, 
                                   kelly_criterion=0.5, max_exposure=10.0):
    """
    回测凯利公式仓位管理的效果
    
    参数:
        trades_history: 历史交易列表（每笔交易包含pnl_pct）
        initial_balance: 初始资金
        kelly_criterion: 凯利系数
        max_exposure: 最大敞口
    
    返回:
        回测结果字典
    """
    position_manager = KellyPositionManager(
        kelly_criterion=kelly_criterion,
        max_exposure=max_exposure,
        risk_mode='balanced'
    )
    
    equity = initial_balance
    trade_results = []
    
    for i, trade in enumerate(trades_history):
        # 模拟预测信号（使用历史数据的滞后版本）
        if i < 5:
            # 前5笔交易使用固定敞口
            exposure = 5.0
            reasoning = "初始阶段"
        else:
            # 基于过去5笔交易计算统计
            recent_trades = trades_history[max(0, i-10):i]
            wins = sum(1 for t in recent_trades if t['pnl'] > 0)
            total = len(recent_trades)
            
            if total > 0:
                win_rate = wins / total
                avg_win = np.mean([t['pnl'] for t in recent_trades if t['pnl'] > 0]) if wins > 0 else 0
                avg_loss = abs(np.mean([t['pnl'] for t in recent_trades if t['pnl'] <= 0])) if (total - wins) > 0 else 0.001
                
                # 估算RR
                estimated_rr = avg_win / avg_loss if avg_loss > 0 else 2.5
                
                # 估算置信度（简化：用胜率代替）
                estimated_prob = win_rate
                
                exposure, reasoning = position_manager.calculate_optimal_exposure(
                    predicted_rr=estimated_rr,
                    direction_prob=estimated_prob
                )
        
        # 计算盈亏
        pnl = equity * (trade['pnl_pct'] / 100) * exposure
        equity += pnl
        
        # 更新管理器
        position_manager.update_equity(equity, pnl)
        
        trade_results.append({
            'trade_idx': i,
            'exposure': exposure,
            'pnl': pnl,
            'equity_after': equity,
            'reasoning': reasoning
        })
    
    # 计算汇总指标
    total_return = (equity - initial_balance) / initial_balance * 100
    
    # 计算最大回撤
    equity_curve = [initial_balance]
    for tr in trade_results:
        equity_curve.append(tr['equity_after'])
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 计算夏普比率
    returns = equity_series.pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 96) if returns.std() > 0 else 0
    
    return {
        'initial_balance': initial_balance,
        'final_equity': equity,
        'total_return': total_return,
        'max_drawdown': max_drawdown * 100,
        'sharpe_ratio': sharpe_ratio,
        'trade_results': trade_results
    }


# 需要导入 pandas
import pandas as pd
