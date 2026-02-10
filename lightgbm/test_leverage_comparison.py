#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
杠杆倍数对比测试
测试 6倍/8倍/10倍 杠杆对收益和回撤的影响
实际敞口: 杠杆 × 20%仓位
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_leverage_comparison.log')
    ]
)
logger = logging.getLogger(__name__)


def backtest_fixed_capital(klines, predictions, initial_balance=1000.0, 
                           position_size_pct=0.20, leverage=6,
                           stop_loss_pct=-0.03):
    """
    固定本金策略回测（正确的回撤计算）
    """
    equity = initial_balance
    trades = []
    position = None
    
    fixed_position_value = initial_balance * position_size_pct
    
    for i in range(len(predictions)):
        # 平仓逻辑
        if position is not None:
            bars_held = i - position['entry_idx']
            current_price = klines.iloc[i]['close']
            
            if position['side'] == 1:
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            stop_loss_triggered = (price_change_pct < stop_loss_pct)
            holding_period_reached = (bars_held >= position['hold_period'])
            
            if stop_loss_triggered or holding_period_reached:
                pnl = fixed_position_value * price_change_pct * leverage
                equity += pnl
                
                trades.append({
                    'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                    'exit_time': klines.iloc[i]['open_time'],
                    'side': 'long' if position['side'] == 1 else 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'price_change_pct': price_change_pct * 100,
                    'position_value': fixed_position_value,
                    'pnl': pnl,
                    'equity_after': equity,
                    'bars_held': bars_held,
                    'stop_loss_hit': stop_loss_triggered
                })
                position = None
        
        # 开仓逻辑
        if position is None and predictions.iloc[i]['should_trade']:
            entry_price = klines.iloc[i]['close']
            
            position = {
                'side': predictions.iloc[i]['direction'],
                'entry_price': entry_price,
                'entry_idx': i,
                'hold_period': int(predictions.iloc[i]['holding_period'])
            }
    
    # 最后平仓
    if position is not None:
        final_price = klines.iloc[-1]['close']
        if position['side'] == 1:
            price_change_pct = (final_price - position['entry_price']) / position['entry_price']
        else:
            price_change_pct = (position['entry_price'] - final_price) / position['entry_price']
        
        pnl = fixed_position_value * price_change_pct * leverage
        equity += pnl
        
        trades.append({
            'entry_time': klines.iloc[position['entry_idx']]['open_time'],
            'exit_time': klines.iloc[-1]['open_time'],
            'side': 'long' if position['side'] == 1 else 'short',
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'price_change_pct': price_change_pct * 100,
            'position_value': fixed_position_value,
            'pnl': pnl,
            'equity_after': equity,
            'bars_held': len(predictions) - position['entry_idx'],
            'stop_loss_hit': False
        })
    
    # 计算回测指标（使用正确的回撤计算）
    if trades:
        trades_df = pd.DataFrame(trades)
        total_return = ((equity - initial_balance) / initial_balance) * 100
        
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / len(trades) * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].mean()) if losing_trades > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 正确的回撤计算：基于权益曲线的滚动峰值
        peak = trades_df['equity_after'].expanding().max()
        drawdown_series = (peak - trades_df['equity_after']) / peak * 100
        max_drawdown = drawdown_series.max()
        
        # 止损统计
        stop_loss_count = len(trades_df[trades_df['stop_loss_hit'] == True])
        stop_loss_rate = stop_loss_count / len(trades) * 100 if len(trades) > 0 else 0
        
        # 最大连续亏损
        consecutive_losses = 0
        max_consecutive_losses = 0
        for pnl in trades_df['pnl']:
            if pnl <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
    else:
        win_rate = 0
        winning_trades = 0
        losing_trades = 0
        avg_win = 0
        avg_loss = 0
        profit_loss_ratio = 0
        max_drawdown = 0
        max_consecutive_losses = 0
        stop_loss_count = 0
        stop_loss_rate = 0
        total_return = 0
    
    return {
        'total_return': total_return,
        'final_equity': equity,
        'total_trades': len(trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'max_drawdown': max_drawdown,
        'max_consecutive_losses': max_consecutive_losses,
        'stop_loss_count': stop_loss_count,
        'stop_loss_rate': stop_loss_rate,
        'trades': trades_df if trades else None
    }


def main():
    """主函数：测试不同杠杆倍数"""
    
    logger.info("="*80)
    logger.info("杠杆倍数对比测试（实际敞口 = 杠杆 × 20%）")
    logger.info("="*80)
    
    from btc_quant.config import load_config
    from btc_quant.data import load_klines
    from btc_quant.features import build_features_and_labels
    from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy
    
    cfg = load_config(Path('config.yaml'))
    logger.info("加载数据...")
    klines = load_klines(cfg)
    
    backtest_start = pd.Timestamp(cfg.raw['history_data']['backtest_start'])
    klines_backtest = klines[klines['open_time'] >= backtest_start]
    
    feature_label_data_backtest = build_features_and_labels(cfg, klines_backtest)
    X_backtest_full = feature_label_data_backtest.features.reset_index(drop=True)
    
    min_len_backtest = min(len(X_backtest_full), len(klines_backtest))
    X_backtest_full = X_backtest_full.iloc[:min_len_backtest]
    klines_backtest = klines_backtest.iloc[:min_len_backtest].reset_index(drop=True)
    
    # 加载模型
    model_dir = Path('models/final_6x_fixed_capital')
    strategy = TwoStageRiskRewardStrategy()
    strategy.load(model_dir)
    
    with open(model_dir / 'top30_features.txt', 'r') as f:
        top_30_features = [line.strip() for line in f.readlines()]
    
    X_backtest_top30 = X_backtest_full[top_30_features]
    
    logger.info("生成预测结果（优化阈值: RR>=2.5, prob>=0.75）...")
    predictions_dict = strategy.predict(X_backtest_top30, rr_threshold=2.5, prob_threshold=0.75)
    
    predictions = pd.DataFrame({
        'predicted_rr': predictions_dict['predicted_rr'],
        'direction': predictions_dict['direction'],
        'holding_period': predictions_dict['holding_period'].clip(1, 30),
        'direction_prob': predictions_dict['direction_prob'],
        'should_trade': predictions_dict['should_trade']
    })
    
    # 数据对齐
    klines = klines_backtest
    min_len = min(len(klines), len(predictions))
    klines = klines.iloc[-min_len:].reset_index(drop=True)
    predictions = predictions.iloc[-min_len:].reset_index(drop=True)
    
    logger.info(f"数据对齐完成，样本数: {min_len}")
    logger.info(f"回测时间范围: {klines['open_time'].min()} 到 {klines['open_time'].max()}")
    
    # 测试不同杠杆倍数
    leverage_configs = [
        (4, 80, '4倍杠杆（80%敞口）'),
        (5, 100, '5倍杠杆（100%敞口）'),
        (6, 120, '6倍杠杆（120%敞口）'),
        (7, 140, '7倍杠杆（140%敞口）'),
        (8, 160, '8倍杠杆（160%敞口）'),
        (10, 200, '10倍杠杆（200%敞口）'),
        (12.5, 250, '12.5倍杠杆（250%敞口）- 实际敞口250%'),
        (20, 400, '20倍杠杆（400%敞口）- 实际敞口400%'),
        (50, 1000, '50倍杠杆（1000%敞口）- 实际敞口1000%上限'),
    ]
    
    results = []
    
    for leverage, exposure, desc in leverage_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试配置: {desc}")
        logger.info(f"{'='*60}")
        
        result = backtest_fixed_capital(
            klines=klines,
            predictions=predictions,
            initial_balance=1000.0,
            position_size_pct=0.20,
            leverage=leverage,
            stop_loss_pct=-0.03
        )
        
        logger.info(f"\n配置: {desc}")
        logger.info(f"实际敞口: {exposure}%")
        logger.info(f"总收益率: {result['total_return']:.2f}%")
        logger.info(f"最终权益: {result['final_equity']:.2f} USDT")
        logger.info(f"交易次数: {result['total_trades']}")
        logger.info(f"胜率: {result['win_rate']:.2f}%")
        logger.info(f"盈亏比: {result['profit_loss_ratio']:.2f}")
        logger.info(f"最大回撤: {result['max_drawdown']:.2f}%")
        logger.info(f"最大连续亏损: {result['max_consecutive_losses']} 笔")
        logger.info(f"止损触发次数: {result['stop_loss_count']} ({result['stop_loss_rate']:.2f}%)")
        
        # 保存交易记录
        if result['trades'] is not None:
            output_path = f'backtest/leverage_{int(leverage)}x_trades.csv'
            result['trades'].to_csv(output_path, index=False)
            logger.info(f"交易明细已保存: {output_path}")
        
        results.append({
            'leverage': f'{leverage}x',
            'exposure_pct': exposure,
            'total_return': result['total_return'],
            'win_rate': result['win_rate'],
            'profit_loss_ratio': result['profit_loss_ratio'],
            'max_drawdown': result['max_drawdown'],
            'total_trades': result['total_trades'],
            'return_drawdown_ratio': result['total_return'] / result['max_drawdown'] if result['max_drawdown'] > 0 else 0
        })
    
    # 对比总结
    logger.info(f"\n{'='*80}")
    logger.info("杠杆倍数对比总结")
    logger.info(f"{'='*80}")
    
    results_df = pd.DataFrame(results)
    logger.info(f"\n{results_df.to_string(index=False)}")
    
    comparison_path = 'backtest/leverage_comparison.csv'
    results_df.to_csv(comparison_path, index=False)
    logger.info(f"\n对比结果已保存: {comparison_path}")
    
    # 找出最优配置
    best_return = results_df.loc[results_df['total_return'].idxmax()]
    best_drawdown = results_df.loc[results_df['max_drawdown'].idxmin()]
    best_ratio = results_df.loc[results_df['return_drawdown_ratio'].idxmax()]
    
    logger.info(f"\n{'='*60}")
    logger.info("最优配置分析")
    logger.info(f"{'='*60}")
    
    logger.info(f"\n收益率最高: {best_return['leverage']} ({best_return['exposure_pct']}%敞口)")
    logger.info(f"  - 收益率: {best_return['total_return']:.2f}%")
    logger.info(f"  - 最大回撤: {best_return['max_drawdown']:.2f}%")
    logger.info(f"  - 胜率: {best_return['win_rate']:.2f}%")
    
    logger.info(f"\n最大回撤最小: {best_drawdown['leverage']} ({best_drawdown['exposure_pct']}%敞口)")
    logger.info(f"  - 最大回撤: {best_drawdown['max_drawdown']:.2f}%")
    logger.info(f"  - 收益率: {best_drawdown['total_return']:.2f}%")
    
    logger.info(f"\n风险调整收益最优: {best_ratio['leverage']} ({best_ratio['exposure_pct']}%敞口)")
    logger.info(f"  - 收益率: {best_ratio['total_return']:.2f}%")
    logger.info(f"  - 最大回撤: {best_ratio['max_drawdown']:.2f}%")
    logger.info(f"  - 收益/回撤比: {best_ratio['return_drawdown_ratio']:.2f}")
    
    logger.info(f"\n{'='*80}")
    logger.info("测试完成！")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
