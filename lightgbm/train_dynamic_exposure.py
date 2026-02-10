#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态敞口管理策略
模型自主决定杠杆倍数和仓位大小，总敞口限制在1000%以内
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
        logging.FileHandler('train_dynamic_exposure.log')
    ]
)
logger = logging.getLogger(__name__)


def calculate_dynamic_exposure(predicted_rr, direction_prob, market_volatility=None, 
                               current_drawdown=0, consecutive_wins=0):
    """
    根据信号质量动态计算最优敞口
    
    参数:
        predicted_rr: 预测盈亏比
        direction_prob: 方向置信度
        market_volatility: 市场波动率（可选）
        current_drawdown: 当前回撤百分比
        consecutive_wins: 连续盈利次数
    
    返回:
        exposure: 建议敞口（杠杆×仓位），范围 [1.0, 10.0]
    """
    
    # 基础敞口：基于盈亏比和置信度
    # RR越高，敞口越大
    rr_factor = min(predicted_rr / 2.5, 2.0)  # RR=2.5时为1.0，RR=5.0时为2.0
    
    # prob越高，敞口越大
    prob_factor = (direction_prob - 0.5) / 0.5  # prob=0.5时为0，prob=1.0时为1.0
    prob_factor = max(0, prob_factor)  # 确保非负
    
    # 基础敞口计算
    base_exposure = 2.0 + rr_factor * 3.0 + prob_factor * 3.0
    
    # 调整因子1：回撤控制
    # 当前回撤越大，敞口越小
    if current_drawdown > 0.03:  # 回撤超过3%
        drawdown_penalty = 1.0 - (current_drawdown - 0.03) * 10  # 每增加1%回撤，降低10%敞口
        drawdown_penalty = max(0.5, drawdown_penalty)
    else:
        drawdown_penalty = 1.0
    
    # 调整因子2：连胜加成
    # 连续盈利可以适当增加信心
    if consecutive_wins >= 5:
        win_streak_bonus = 1.0 + min(consecutive_wins - 5, 10) * 0.02  # 每连胜1次加2%，最多20%
    else:
        win_streak_bonus = 1.0
    
    # 最终敞口
    final_exposure = base_exposure * drawdown_penalty * win_streak_bonus
    
    # 限制在合理范围内
    final_exposure = np.clip(final_exposure, 1.0, 10.0)
    
    return final_exposure


def dynamic_exposure_backtest(klines, predictions, initial_balance=1000.0, 
                              max_exposure=10.0, stop_loss_pct=-0.03):
    """
    动态敞口管理回测
    
    核心逻辑：
    1. 根据信号质量计算最优敞口（1-10倍）
    2. 敞口 = 杠杆 × 仓位百分比
    3. 为简化，假设杠杆=敞口，仓位=100%本金
    """
    equity = initial_balance
    trades = []
    position = None
    
    # 风险跟踪
    peak_equity = initial_balance
    consecutive_wins = 0
    
    for i in range(len(predictions)):
        # 平仓逻辑
        if position is not None:
            bars_held = i - position['entry_idx']
            current_price = klines.iloc[i]['close']
            
            if position['side'] == 1:
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            # 止损检查
            stop_loss_triggered = (price_change_pct < stop_loss_pct)
            holding_period_reached = (bars_held >= position['hold_period'])
            
            if stop_loss_triggered or holding_period_reached:
                # 计算盈亏（基于敞口）
                # position_value = 固定使用100%本金
                # leverage = 动态敞口
                pnl = initial_balance * price_change_pct * position['exposure']
                
                equity += pnl
                
                # 更新风控指标
                if pnl > 0:
                    consecutive_wins += 1
                    if equity > peak_equity:
                        peak_equity = equity
                else:
                    consecutive_wins = 0
                
                trades.append({
                    'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                    'exit_time': klines.iloc[i]['open_time'],
                    'side': 'long' if position['side'] == 1 else 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'price_change_pct': price_change_pct * 100,
                    'exposure': position['exposure'],
                    'predicted_rr': position['predicted_rr'],
                    'direction_prob': position['direction_prob'],
                    'pnl': pnl,
                    'equity_after': equity,
                    'bars_held': bars_held,
                    'stop_loss_hit': stop_loss_triggered
                })
                position = None
        
        # 开仓逻辑
        if position is None and predictions.iloc[i]['should_trade']:
            entry_price = klines.iloc[i]['close']
            
            # 计算当前回撤
            current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            
            # 动态计算最优敞口
            optimal_exposure = calculate_dynamic_exposure(
                predicted_rr=predictions.iloc[i]['predicted_rr'],
                direction_prob=predictions.iloc[i]['direction_prob'],
                current_drawdown=current_drawdown,
                consecutive_wins=consecutive_wins
            )
            
            # 限制最大敞口
            optimal_exposure = min(optimal_exposure, max_exposure)
            
            position = {
                'side': predictions.iloc[i]['direction'],
                'entry_price': entry_price,
                'entry_idx': i,
                'hold_period': int(predictions.iloc[i]['holding_period']),
                'exposure': optimal_exposure,
                'predicted_rr': predictions.iloc[i]['predicted_rr'],
                'direction_prob': predictions.iloc[i]['direction_prob']
            }
    
    # 最后平仓
    if position is not None:
        final_price = klines.iloc[-1]['close']
        if position['side'] == 1:
            price_change_pct = (final_price - position['entry_price']) / position['entry_price']
        else:
            price_change_pct = (position['entry_price'] - final_price) / position['entry_price']
        
        pnl = initial_balance * price_change_pct * position['exposure']
        equity += pnl
        
        trades.append({
            'entry_time': klines.iloc[position['entry_idx']]['open_time'],
            'exit_time': klines.iloc[-1]['open_time'],
            'side': 'long' if position['side'] == 1 else 'short',
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'price_change_pct': price_change_pct * 100,
            'exposure': position['exposure'],
            'predicted_rr': position['predicted_rr'],
            'direction_prob': position['direction_prob'],
            'pnl': pnl,
            'equity_after': equity,
            'bars_held': len(predictions) - position['entry_idx'],
            'stop_loss_hit': False
        })
    
    # 计算回测指标
    if trades:
        trades_df = pd.DataFrame(trades)
        total_return = ((equity - initial_balance) / initial_balance) * 100
        
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / len(trades) * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].mean()) if losing_trades > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 正确的回撤计算
        peak = trades_df['equity_after'].expanding().max()
        drawdown_series = (peak - trades_df['equity_after']) / peak * 100
        max_drawdown = drawdown_series.max()
        
        # 敞口统计
        avg_exposure = trades_df['exposure'].mean()
        max_exposure_used = trades_df['exposure'].max()
        min_exposure_used = trades_df['exposure'].min()
        
        # 连续亏损
        consecutive_losses = 0
        max_consecutive_losses = 0
        for pnl in trades_df['pnl']:
            if pnl <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
    else:
        total_return = 0
        win_rate = 0
        winning_trades = 0
        losing_trades = 0
        avg_win = 0
        avg_loss = 0
        profit_loss_ratio = 0
        max_drawdown = 0
        max_consecutive_losses = 0
        avg_exposure = 0
        max_exposure_used = 0
        min_exposure_used = 0
    
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
        'avg_exposure': avg_exposure,
        'max_exposure': max_exposure_used,
        'min_exposure': min_exposure_used,
        'trades': trades_df if trades else None
    }


def main():
    """主函数"""
    
    logger.info("="*80)
    logger.info("动态敞口管理策略测试")
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
    
    logger.info("生成预测结果...")
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
    
    # 测试不同最大敞口限制
    max_exposure_configs = [
        (5, '保守型（最大5倍敞口）'),
        (8, '平衡型（最大8倍敞口）'),
        (10, '激进型（最大10倍敞口）'),
    ]
    
    results = []
    
    for max_exp, desc in max_exposure_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试配置: {desc}")
        logger.info(f"{'='*60}")
        
        result = dynamic_exposure_backtest(
            klines=klines,
            predictions=predictions,
            initial_balance=1000.0,
            max_exposure=max_exp,
            stop_loss_pct=-0.03
        )
        
        logger.info(f"\n配置: {desc}")
        logger.info(f"总收益率: {result['total_return']:.2f}%")
        logger.info(f"最终权益: {result['final_equity']:.2f} USDT")
        logger.info(f"胜率: {result['win_rate']:.2f}%")
        logger.info(f"盈亏比: {result['profit_loss_ratio']:.2f}")
        logger.info(f"最大回撤: {result['max_drawdown']:.2f}%")
        logger.info(f"平均敞口: {result['avg_exposure']:.2f}倍")
        logger.info(f"敞口范围: {result['min_exposure']:.2f} - {result['max_exposure']:.2f}倍")
        
        # 保存交易记录
        if result['trades'] is not None:
            output_path = f'backtest/dynamic_exposure_max{int(max_exp)}x_trades.csv'
            result['trades'].to_csv(output_path, index=False)
            logger.info(f"交易明细已保存: {output_path}")
        
        results.append({
            'config': desc,
            'max_exposure': max_exp,
            'total_return': result['total_return'],
            'win_rate': result['win_rate'],
            'max_drawdown': result['max_drawdown'],
            'avg_exposure': result['avg_exposure'],
            'return_drawdown_ratio': result['total_return'] / result['max_drawdown'] if result['max_drawdown'] > 0 else 0
        })
    
    # 对比总结
    logger.info(f"\n{'='*80}")
    logger.info("动态敞口策略对比")
    logger.info(f"{'='*80}")
    
    results_df = pd.DataFrame(results)
    logger.info(f"\n{results_df.to_string(index=False)}")
    
    comparison_path = 'backtest/dynamic_exposure_comparison.csv'
    results_df.to_csv(comparison_path, index=False)
    logger.info(f"\n对比结果已保存: {comparison_path}")
    
    logger.info(f"\n{'='*80}")
    logger.info("测试完成！")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
