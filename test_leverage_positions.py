#!/usr/bin/env python3
"""
测试不同杠杆倍数（仓位比例）的回测效果
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels
from btc_quant.risk_reward_labels import RiskRewardLabelBuilder
from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_backtest_with_position_size(predictions, klines, position_pct):
    """
    使用指定仓位比例执行回测
    
    Args:
        predictions: 预测结果DataFrame
        klines: K线数据
        position_pct: 仓位比例（如0.01表示1%）
    """
    initial_balance = 1000.0
    equity = initial_balance
    trades = []
    position = None
    
    for i, row in predictions.iterrows():
        # 如果有持仓，检查是否应该出场
        if position is not None:
            bars_held = i - position['entry_idx']
            current_price = klines.loc[i, 'close']
            
            # 达到预期持有周期，平仓
            if bars_held >= position['hold_period']:
                if position['side'] == 1:  # 多头
                    pnl = (current_price - position['entry_price']) * position['quantity']
                else:  # 空头
                    pnl = (position['entry_price'] - current_price) * position['quantity']
                
                equity += pnl
                trades.append({
                    'entry_time': klines.loc[position['entry_idx'], 'open_time'],
                    'exit_time': klines.loc[i, 'open_time'],
                    'side': 'long' if position['side'] == 1 else 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'return_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100
                })
                position = None
        
        # 如果无持仓且有交易信号，开仓
        if position is None and row['should_trade']:
            entry_price = klines.loc[i, 'close']
            quantity = (equity * position_pct) / entry_price
            
            position = {
                'side': row['direction'],
                'entry_price': entry_price,
                'entry_idx': i,
                'hold_period': int(row['holding_period']),
                'quantity': quantity
            }
    
    # 最后如果还有持仓，平仓
    if position is not None:
        final_price = klines.iloc[-1]['close']
        if position['side'] == 1:
            pnl = (final_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - final_price) * position['quantity']
        
        equity += pnl
        trades.append({
            'entry_time': klines.loc[position['entry_idx'], 'open_time'],
            'exit_time': klines.iloc[-1]['open_time'],
            'side': 'long' if position['side'] == 1 else 'short',
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'quantity': position['quantity'],
            'pnl': pnl,
            'return_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100
        })
    
    # 计算统计指标
    total_return = (equity / initial_balance - 1) * 100
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] <= 0).sum()
        win_rate = winning_trades / len(trades) * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].mean()) if losing_trades > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 计算最大回撤
        equity_curve = [initial_balance]
        current_eq = initial_balance
        for _, trade in trades_df.iterrows():
            current_eq += trade['pnl']
            equity_curve.append(current_eq)
        
        equity_series = pd.Series(equity_curve)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()
    else:
        win_rate = 0
        winning_trades = 0
        losing_trades = 0
        avg_win = 0
        avg_loss = 0
        win_loss_ratio = 0
        max_drawdown = 0
        trades_df = pd.DataFrame()
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'max_drawdown': max_drawdown,
        'final_equity': equity,
        'trades': trades_df
    }


def main():
    logger.info("=" * 100)
    logger.info("杠杆倍数测试 - 对比不同仓位比例")
    logger.info("=" * 100)
    
    # 读取之前保存的交易记录（1%仓位的基准结果）
    logger.info("\n【步骤1】读取基准交易记录")
    trades_files = list(Path("backtest").glob("backtest_results_rr_strategy_*.csv"))
    
    if not trades_files:
        logger.error("在backtest目录下找不到交易记录文件，请先运行 train_and_backtest_rr_strategy.py")
        return None
    
    # 使用最新的文件
    latest_file = max(trades_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"读取文件: {latest_file}")
    
    trades_df = pd.read_csv(latest_file)
    logger.info(f"总交易数: {len(trades_df)}")
    
    # 基准结果（1%仓位）
    initial_balance = 1000.0
    base_position_pct = 0.01
    
    # 模拟不同仓位比例的结果
    position_sizes = [0.01, 0.02, 0.03, 0.05, 0.10]  # 1%, 2%, 3%, 5%, 10%
    
    logger.info("\n【步骤2】模拟不同杠杆倍数")
    logger.info("=" * 100)
    
    results = []
    for pos_size in position_sizes:
        logger.info(f"\n测试仓位: {pos_size*100:.0f}%")
        
        # 按比例缩放每笔交易的PnL
        scale_factor = pos_size / base_position_pct
        scaled_pnls = trades_df['pnl'] * scale_factor
        
        # 计算统计指标
        equity = initial_balance + scaled_pnls.sum()
        total_return = (equity / initial_balance - 1) * 100
        
        winning_trades = (scaled_pnls > 0).sum()
        losing_trades = (scaled_pnls <= 0).sum()
        win_rate = winning_trades / len(trades_df) * 100
        
        avg_win = scaled_pnls[scaled_pnls > 0].mean() if winning_trades > 0 else 0
        avg_loss = abs(scaled_pnls[scaled_pnls <= 0].mean()) if losing_trades > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 计算最大回撤
        equity_curve = [initial_balance]
        current_eq = initial_balance
        for pnl in scaled_pnls:
            current_eq += pnl
            equity_curve.append(current_eq)
        
        equity_series = pd.Series(equity_curve)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()
        
        result = {
            'position_size': pos_size,
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': len(trades_df),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'max_drawdown': max_drawdown,
            'final_equity': equity
        }
        results.append(result)
        
        logger.info(f"  总收益率: {result['total_return']:.2f}%")
        logger.info(f"  胜率: {result['win_rate']:.2f}%")
        logger.info(f"  最大回撤: {result['max_drawdown']:.2f}%")
        logger.info(f"  盈亏比: {result['win_loss_ratio']:.2f}")
        logger.info(f"  交易次数: {result['total_trades']}")
    
    # 对比总结
    logger.info("\n" + "=" * 100)
    logger.info("【综合对比】")
    logger.info("=" * 100)
    
    print("\n{:<10} {:<15} {:<10} {:<15} {:<10} {:<10}".format(
        "仓位", "总收益率", "胜率", "最大回撤", "盈亏比", "交易次数"
    ))
    print("-" * 80)
    
    for result in results:
        print("{:<10} {:<15.2f}% {:<10.2f}% {:<15.2f}% {:<10.2f} {:<10}".format(
            f"{result['position_size']*100:.0f}%",
            result['total_return'],
            result['win_rate'],
            result['max_drawdown'],
            result['win_loss_ratio'],
            result['total_trades']
        ))
    
    # 找出最优配置
    best_sharpe_idx = 0
    best_sharpe = -999
    for i, result in enumerate(results):
        # 简化夏普比率：收益/回撤
        if result['max_drawdown'] != 0:
            sharpe = -result['total_return'] / result['max_drawdown']
        else:
            sharpe = result['total_return']
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_sharpe_idx = i
    
    best_result = results[best_sharpe_idx]
    print("\n" + "=" * 100)
    print(f"【推荐配置】仓位 {best_result['position_size']*100:.0f}%")
    print(f"  总收益率: {best_result['total_return']:.2f}%")
    print(f"  胜率: {best_result['win_rate']:.2f}%")
    print(f"  最大回撤: {best_result['max_drawdown']:.2f}%")
    print(f"  风险收益比: {-best_result['total_return']/best_result['max_drawdown']:.2f}")
    print("=" * 100)
    
    return results


if __name__ == "__main__":
    results = main()
