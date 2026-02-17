#!/usr/bin/env python3
"""
分批出场策略模块
实现三档位分批止盈，平滑收益曲线并提升盈亏比

三档设计:
- 第一档 (50%仓位): 达到1倍RR时平仓，锁定基础利润
- 第二档 (30%仓位): 达到2倍RR时平仓，扩大战果
- 第三档 (20%仓位): 追踪移动止盈，博取超额收益
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PositionTranche:
    """仓位分档信息"""
    proportion: float  # 该档仓位比例（0-1）
    target_rr_multiple: float  # 目标RR倍数（如1.0、2.0）
    exit_price: Optional[float] = None  # 出场价
    pnl_pct: Optional[float] = None  # 该档盈亏百分比
    pnl: Optional[float] = None  # 该档盈亏金额
    is_closed: bool = False  # 是否已平仓


class TieredExitManager:
    """
    三档位分批出场管理器
    
    使用示例:
        >>> manager = TieredExitManager(
        ...     entry_price=100000,
        ...     predicted_rr=4.5,
        ...     direction=1,
        ...     position_size=1.0,  # 1个BTC
        ...     exposure=5.0
        ... )
        >>> 
        >>> # 价格波动时更新
        >>> actions = manager.update(current_price=104500)
        >>> if actions['should_exit_partial']:
        ...     print(f"部分平仓：{actions['exit_proportion']*100:.0f}%")
    """
    
    def __init__(
        self,
        entry_price: float,
        predicted_rr: float,
        direction: int,
        position_size: float,
        exposure: float,
        tranches_config: Optional[List[Dict]] = None
    ):
        """
        参数:
            entry_price: 入场价格
            predicted_rr: 预测盈亏比
            direction: 方向（1=做多，-1=做空）
            position_size: 仓位大小（BTC数量）
            exposure: 敞口（杠杆倍数）
            tranches_config: 分档配置（可选，默认使用标准三档）
        """
        self.entry_price = entry_price
        self.predicted_rr = predicted_rr
        self.direction = direction
        self.position_size = position_size
        self.exposure = exposure
        
        # 计算止损价和预期止盈价
        self.stop_loss_price = self._calculate_stop_loss()
        self.target_profit_price = self._calculate_target_profit()
        
        # 初始化三档仓位
        if tranches_config is None:
            # 标准三档配置 - 简化版：只用前两档固定止盈，第三档自然到期
            self.tranches = [
                PositionTranche(proportion=0.5, target_rr_multiple=1.0),   # 第一档：50% 仓位，1 倍 RR
                PositionTranche(proportion=0.3, target_rr_multiple=2.0),   # 第二档：30% 仓位，2 倍 RR
                PositionTranche(proportion=0.2, target_rr_multiple=999.0)  # 第三档：20% 仓位，持有周期到期
            ]
        else:
            self.tranches = [PositionTranche(**config) for config in tranches_config]
                
        # 状态跟踪（简化版不需要移动止盈）
        self.remaining_position = position_size
    
    def _calculate_stop_loss(self) -> float:
        """基于预测RR计算止损价"""
        if self.direction == 1:  # 做多
            return self.entry_price * (1 - 1.0 / self.predicted_rr)
        else:  # 做空
            return self.entry_price * (1 + 1.0 / self.predicted_rr)
    
    def _calculate_target_profit(self) -> float:
        """基于预测RR计算目标止盈价"""
        if self.direction == 1:  # 做多
            return self.entry_price * (1 + self.predicted_rr / self.predicted_rr)
        else:  # 做空
            return self.entry_price * (1 - self.predicted_rr / self.predicted_rr)
    
    def update(self, current_price: float) -> Dict:
        """
        更新当前价格并检查是否触发部分平仓
        
        返回:
            {
                'should_exit_partial': bool,
                'exit_proportion': float,
                'exit_reason': str,
                'tranche_idx': int
            }
        """
        result = {
            'should_exit_partial': False,
            'exit_proportion': 0.0,
            'exit_reason': None,
            'tranche_idx': -1
        }
        
        # 检查各档是否触发
        for i, tranche in enumerate(self.tranches):
            if tranche.is_closed:
                continue
            
            # 计算该档的目标价格
            is_trailing_stop = (tranche.target_rr_multiple >= 100)
            
            if not is_trailing_stop:  # 固定目标止盈
                target_price = self._calculate_price_at_rr(tranche.target_rr_multiple)
                
                if self.direction == 1:
                    triggered = current_price >= target_price
                else:
                    triggered = current_price <= target_price
                
                if triggered:
                    tranche.is_closed = True
                    tranche.exit_price = current_price
                    tranche.pnl_pct = (current_price - self.entry_price) / self.entry_price * self.direction * 100
                    
                    result['should_exit_partial'] = True
                    result['exit_proportion'] = tranche.proportion
                    result['exit_reason'] = f'🎯 第{i+1}档止盈 (RR×{tranche.target_rr_multiple})'
                    result['tranche_idx'] = i
                    break
            
            else:  # 第三档：持有周期到期，不主动触发
                # 第三档不参与分批出场，由主逻辑在持仓周期到期时统一处理
                continue
        
        return result
    
    def _calculate_price_at_rr(self, rr_multiple: float) -> float:
        """计算达到RR倍数时的价格"""
        risk_per_unit = abs(self.entry_price - self.stop_loss_price)
        reward_per_unit = risk_per_unit * rr_multiple
        
        if self.direction == 1:
            return self.entry_price + reward_per_unit
        else:
            return self.entry_price - reward_per_unit
    
    def get_status(self) -> Dict:
        """获取当前状态"""
        closed_tranches = sum(1 for t in self.tranches if t.is_closed)
        
        return {
            'remaining_position': self.remaining_position,
            'closed_tranches': closed_tranches,
            'total_tranches': len(self.tranches),
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price,
            'trailing_stop_price': self.trailing_stop_price
        }


def backtest_with_tiered_exit(
    klines, predictions, initial_balance=1000.0, max_exposure=10.0,
    atr_k=3.5, enable_tiered_exit=True
):
    """
    分批出场策略回测
    
    参数:
        klines: K线数据
        predictions: 预测结果
        initial_balance: 初始资金
        max_exposure: 最大敞口
        atr_k: ATR止损倍数
        enable_tiered_exit: 是否启用分批出场
    
    返回:
        回测结果字典
    """
    from btc_quant.dynamic_stop_loss import calculate_atr_stop_loss
    
    equity = initial_balance
    trades = []
    position = None
    tiered_manager = None
    
    for i in range(len(predictions)):
        current_price = klines.iloc[i]['close']
        current_atr = klines.iloc[i]['atr_14']
        
        # 开仓逻辑
        if position is None and predictions.iloc[i]['should_trade']:
            entry_price = current_price
            predicted_rr = predictions.iloc[i]['predicted_rr']
            direction_prob = predictions.iloc[i]['direction_prob']
            direction = predictions.iloc[i]['direction']
            hold_period = int(predictions.iloc[i]['holding_period'])
            
            # 计算动态敞口
            base_exposure = 1.0
            
            if predicted_rr >= 6.0:
                rr_multiplier = 5.0
            elif predicted_rr >= 4.0:
                rr_multiplier = 3.0 + (predicted_rr - 4.0) * 1.0
            elif predicted_rr >= 2.5:
                rr_multiplier = 1.0 + (predicted_rr - 2.5) * 1.33
            else:
                rr_multiplier = 0.0
            
            if direction_prob >= 0.85:
                prob_multiplier = 5.0
            elif direction_prob >= 0.75:
                prob_multiplier = 3.0 + (direction_prob - 0.75) * 20.0
            elif direction_prob >= 0.65:
                prob_multiplier = 1.0 + (direction_prob - 0.65) * 20.0
            else:
                prob_multiplier = 0.0
            
            optimal_exposure = base_exposure + rr_multiplier + prob_multiplier
            optimal_exposure = min(optimal_exposure, max_exposure)
            
            position_value = equity * 1.0
            position_size = position_value / entry_price  # BTC数量
            
            # 计算ATR止损价
            stop_loss_price, stop_loss_pct = calculate_atr_stop_loss(
                entry_price=entry_price,
                atr=current_atr,
                direction=direction,
                k=atr_k,
                min_stop_loss_pct=0.01,
                max_stop_loss_pct=0.05
            )
            
            position = {
                'side': direction,
                'entry_price': entry_price,
                'entry_idx': i,
                'position_size': position_size,
                'exposure': optimal_exposure,
                'stop_loss_price': stop_loss_price,
                'predicted_rr': predicted_rr,
                'hold_period': hold_period
            }
            
            # 初始化分批出场管理器
            if enable_tiered_exit:
                tiered_manager = TieredExitManager(
                    entry_price=entry_price,
                    predicted_rr=predicted_rr,
                    direction=direction,
                    position_size=position_size,
                    exposure=optimal_exposure
                )
        
        # 平仓逻辑
        elif position is not None:
            bars_held = i - position['entry_idx']
            should_exit_all = False
            exit_reason = None
            
            # 1. 检查ATR止损（全仓止损）
            if position['side'] == 1:
                if current_price <= position['stop_loss_price']:
                    should_exit_all = True
                    exit_reason = '🛑 ATR止损'
            else:
                if current_price >= position['stop_loss_price']:
                    should_exit_all = True
                    exit_reason = '🛑 ATR止损'
            
            # 2. 检查分批出场（如果不是一次性平仓）
            partial_exits = []
            if not should_exit_all and enable_tiered_exit and tiered_manager:
                while True:
                    action = tiered_manager.update(current_price)
                    if action['should_exit_partial']:
                        partial_exits.append(action)
                    else:
                        break
            
            # 处理部分平仓
            if partial_exits:
                for action in partial_exits:
                    exit_proportion = action['exit_proportion']
                    exit_size = position_size * exit_proportion
                    
                    # 调试输出
                    if 'pnl_pct' not in action:
                        print(f"WARNING: 缺少 pnl_pct: {action}")
                        continue
                    
                    exit_pnl = initial_balance * (action['pnl_pct'] / 100) * position['exposure'] * exit_proportion
                    equity += exit_pnl
                    
                    trades.append({
                        'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                        'exit_time': klines.iloc[i]['open_time'],
                        'side': 'long' if position['side'] == 1 else 'short',
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'position_size': exit_size,
                        'exposure': position['exposure'],
                        'pnl': exit_pnl,
                        'pnl_pct': action['pnl_pct'],
                        'equity_after': equity,
                        'reason': action['exit_reason'],
                        'is_partial_exit': True
                    })
            
            # 3. 检查持仓周期到期（全仓平仓）
            if not should_exit_all and bars_held >= position['hold_period']:
                should_exit_all = True
                exit_reason = '⏰ 持仓周期'
            
            # 全仓平仓
            if should_exit_all:
                if position['side'] == 1:
                    price_change_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                # 只计算剩余仓位的盈亏
                remaining_ratio = 1.0 - sum(t.proportion for t in tiered_manager.tranches if t.is_closed) if tiered_manager else 1.0
                pnl = initial_balance * price_change_pct * position['exposure'] * remaining_ratio
                equity += pnl
                
                trades.append({
                    'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                    'exit_time': klines.iloc[i]['open_time'],
                    'side': 'long' if position['side'] == 1 else 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'position_size': position_size * remaining_ratio,
                    'exposure': position['exposure'],
                    'pnl': pnl,
                    'pnl_pct': price_change_pct * 100 * position['exposure'],
                    'equity_after': equity,
                    'reason': exit_reason,
                    'is_partial_exit': False
                })
                
                position = None
                tiered_manager = None
    
    # 最后平仓
    if position is not None:
        final_price = klines.iloc[-1]['close']
        if position['side'] == 1:
            price_change_pct = (final_price - position['entry_price']) / position['entry_price']
        else:
            price_change_pct = (position['entry_price'] - final_price) / position['entry_price']
        
        remaining_ratio = 1.0 - sum(t.proportion for t in tiered_manager.tranches if t.is_closed) if tiered_manager else 1.0
        pnl = initial_balance * price_change_pct * position['exposure'] * remaining_ratio
        equity += pnl
        
        trades.append({
            'entry_time': klines.iloc[position['entry_idx']]['open_time'],
            'exit_time': klines.iloc[-1]['open_time'],
            'side': 'long' if position['side'] == 1 else 'short',
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'position_size': position_size * remaining_ratio,
            'exposure': position['exposure'],
            'pnl': pnl,
            'pnl_pct': price_change_pct * 100 * position['exposure'],
            'equity_after': equity,
            'reason': '📊 期末平仓',
            'is_partial_exit': False
        })
    
    # 计算回测指标
    total_return = (equity - initial_balance) / initial_balance * 100
    win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100 if trades else 0
    
    # 计算盈亏比
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    
    avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss = abs(np.mean([t['pnl_pct'] for t in losing_trades])) if losing_trades else 0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    
    return {
        'strategy': '分批出场' if enable_tiered_exit else '一次性平仓',
        'initial_balance': initial_balance,
        'final_equity': equity,
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'trades': trades,
        'profit_loss_ratio': profit_loss_ratio,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }
