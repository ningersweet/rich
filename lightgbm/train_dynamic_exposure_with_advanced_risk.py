#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态敞口管理策略 + 多层风控系统
1. 固定止损：-3%
2. 追踪止损：盈利后自动保护利润
3. 每日最大亏损：-20%
4. 回撤止损：从峰值回撤>6%暂停交易
5. 动态敞口：根据回撤自动降低敞口
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
        logging.FileHandler('train_dynamic_exposure_advanced_risk.log')
    ]
)
logger = logging.getLogger(__name__)


def calculate_dynamic_exposure(predicted_rr, direction_prob, current_drawdown=0, 
                               consecutive_losses=0, max_exposure=10.0):
    """
    根据信号质量和风险状态动态计算敞口
    
    新增：
    - 连续亏损惩罚
    - 回撤动态降敞口
    """
    
    # 基础敞口计算
    rr_factor = min(predicted_rr / 2.5, 2.0)
    prob_factor = max((direction_prob - 0.5) / 0.5, 0)
    base_exposure = 2.0 + rr_factor * 3.0 + prob_factor * 3.0
    
    # 回撤惩罚（更激进）
    if current_drawdown > 0.02:  # 回撤>2%开始降敞口
        drawdown_penalty = 1.0 - (current_drawdown - 0.02) * 15  # 每增加1%降15%
        drawdown_penalty = max(0.3, drawdown_penalty)  # 最低保留30%
    else:
        drawdown_penalty = 1.0
    
    # 连续亏损惩罚（新增）
    if consecutive_losses >= 2:
        loss_penalty = 1.0 - min(consecutive_losses - 1, 5) * 0.15  # 每连亏1笔降15%
        loss_penalty = max(0.2, loss_penalty)  # 最低保留20%
    else:
        loss_penalty = 1.0
    
    # 最终敞口
    final_exposure = base_exposure * drawdown_penalty * loss_penalty
    final_exposure = np.clip(final_exposure, 1.0, max_exposure)
    
    return final_exposure


def advanced_risk_backtest(klines, predictions, initial_balance=1000.0, 
                           max_exposure=10.0, stop_loss_pct=-0.03,
                           max_daily_loss_pct=-0.20, max_drawdown_pause=0.06,
                           use_trailing_stop=True):
    """
    多层风控回测
    
    新增风控机制：
    1. 追踪止损：盈利>1%后，最多回吐50%
    2. 每日最大亏损：单日亏损>20%停止当日交易
    3. 回撤暂停：从峰值回撤>6%暂停交易
    4. 连续亏损敞口降低
    """
    equity = initial_balance
    trades = []
    position = None
    
    # 风控状态
    peak_equity = initial_balance
    consecutive_losses = 0
    daily_start_equity = initial_balance
    current_date = None
    trading_paused = False
    pause_reason = None
    
    for i in range(len(predictions)):
        # 检查日期变化（重置每日亏损）
        current_time = pd.to_datetime(klines.iloc[i]['open_time'])
        if current_date is None or current_time.date() != current_date:
            current_date = current_time.date()
            daily_start_equity = equity
        
        # 每日亏损检查
        daily_loss_pct = (equity - daily_start_equity) / daily_start_equity
        if daily_loss_pct < max_daily_loss_pct:
            if not trading_paused:
                logger.warning(f"[{current_time}] 触发每日最大亏损限制: {daily_loss_pct*100:.2f}%，暂停交易至明日")
                trading_paused = True
                pause_reason = 'daily_loss'
        
        # 回撤暂停检查
        current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        if current_drawdown > max_drawdown_pause:
            if not trading_paused:
                logger.warning(f"[{current_time}] 触发回撤暂停: {current_drawdown*100:.2f}%，暂停交易")
                trading_paused = True
                pause_reason = 'drawdown_pause'
        
        # 新的一天解除暂停
        if trading_paused and pause_reason == 'daily_loss':
            if current_time.date() != current_date:
                trading_paused = False
                pause_reason = None
                logger.info(f"[{current_time}] 新的一天，恢复交易")
        
        # 平仓逻辑
        if position is not None:
            bars_held = i - position['entry_idx']
            current_price = klines.iloc[i]['close']
            
            if position['side'] == 1:
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            # 1. 固定止损
            stop_loss_triggered = (price_change_pct < stop_loss_pct)
            
            # 2. 追踪止损（盈利后保护）
            trailing_stop_triggered = False
            if use_trailing_stop and price_change_pct > 0.01:  # 盈利>1%
                max_profit_pct = position.get('max_profit_pct', price_change_pct)
                position['max_profit_pct'] = max(max_profit_pct, price_change_pct)
                
                # 最多回吐50%利润
                profit_retracement = (max_profit_pct - price_change_pct) / max_profit_pct
                if profit_retracement > 0.5:
                    trailing_stop_triggered = True
            
            # 3. 持仓周期到期
            holding_period_reached = (bars_held >= position['hold_period'])
            
            if stop_loss_triggered or trailing_stop_triggered or holding_period_reached:
                pnl = initial_balance * price_change_pct * position['exposure']
                equity += pnl
                
                # 更新风控指标
                if pnl > 0:
                    consecutive_losses = 0
                    if equity > peak_equity:
                        peak_equity = equity
                        # 回撤降低，可能解除暂停
                        if trading_paused and pause_reason == 'drawdown_pause':
                            new_drawdown = (peak_equity - equity) / peak_equity
                            if new_drawdown < max_drawdown_pause * 0.8:  # 回撤降至阈值的80%
                                trading_paused = False
                                pause_reason = None
                                logger.info(f"回撤降至{new_drawdown*100:.2f}%，恢复交易")
                else:
                    consecutive_losses += 1
                
                trades.append({
                    'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                    'exit_time': klines.iloc[i]['open_time'],
                    'side': 'long' if position['side'] == 1 else 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'price_change_pct': price_change_pct * 100,
                    'exposure': position['exposure'],
                    'pnl': pnl,
                    'equity_after': equity,
                    'bars_held': bars_held,
                    'stop_loss_hit': stop_loss_triggered,
                    'trailing_stop_hit': trailing_stop_triggered,
                    'consecutive_losses': consecutive_losses
                })
                position = None
        
        # 开仓逻辑（如果未暂停）
        if position is None and predictions.iloc[i]['should_trade'] and not trading_paused:
            entry_price = klines.iloc[i]['close']
            current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            
            # 动态计算敞口（考虑连续亏损）
            optimal_exposure = calculate_dynamic_exposure(
                predicted_rr=predictions.iloc[i]['predicted_rr'],
                direction_prob=predictions.iloc[i]['direction_prob'],
                current_drawdown=current_drawdown,
                consecutive_losses=consecutive_losses,
                max_exposure=max_exposure
            )
            
            position = {
                'side': predictions.iloc[i]['direction'],
                'entry_price': entry_price,
                'entry_idx': i,
                'hold_period': int(predictions.iloc[i]['holding_period']),
                'exposure': optimal_exposure,
                'max_profit_pct': 0  # 追踪止损用
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
            'pnl': pnl,
            'equity_after': equity,
            'bars_held': len(predictions) - position['entry_idx'],
            'stop_loss_hit': False,
            'trailing_stop_hit': False,
            'consecutive_losses': consecutive_losses
        })
    
    # 计算统计
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
        
        # 止损统计
        stop_loss_count = len(trades_df[trades_df['stop_loss_hit'] == True])
        trailing_stop_count = len(trades_df[trades_df['trailing_stop_hit'] == True])
        
        avg_exposure = trades_df['exposure'].mean()
        max_consecutive_losses = trades_df['consecutive_losses'].max()
    else:
        total_return = 0
        win_rate = 0
        profit_loss_ratio = 0
        max_drawdown = 0
        stop_loss_count = 0
        trailing_stop_count = 0
        avg_exposure = 0
        max_consecutive_losses = 0
        trades_df = None
    
    return {
        'total_return': total_return,
        'final_equity': equity,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'max_drawdown': max_drawdown,
        'stop_loss_count': stop_loss_count,
        'trailing_stop_count': trailing_stop_count,
        'avg_exposure': avg_exposure,
        'max_consecutive_losses': max_consecutive_losses,
        'trades': trades_df
    }


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("多层风控系统回测")
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
    
    min_len = min(len(X_backtest_full), len(klines_backtest))
    X_backtest_full = X_backtest_full.iloc[:min_len]
    klines_backtest = klines_backtest.iloc[:min_len].reset_index(drop=True)
    
    model_dir = Path('models/final_6x_fixed_capital')
    strategy = TwoStageRiskRewardStrategy()
    strategy.load(model_dir)
    
    with open(model_dir / 'top30_features.txt', 'r') as f:
        top_30_features = [line.strip() for line in f.readlines()]
    
    X_backtest_top30 = X_backtest_full[top_30_features]
    
    logger.info("生成预测...")
    predictions_dict = strategy.predict(X_backtest_top30, rr_threshold=2.5, prob_threshold=0.75)
    
    predictions = pd.DataFrame({
        'predicted_rr': predictions_dict['predicted_rr'],
        'direction': predictions_dict['direction'],
        'holding_period': predictions_dict['holding_period'].clip(1, 30),
        'direction_prob': predictions_dict['direction_prob'],
        'should_trade': predictions_dict['should_trade']
    })
    
    klines = klines_backtest
    min_len = min(len(klines), len(predictions))
    klines = klines.iloc[-min_len:].reset_index(drop=True)
    predictions = predictions.iloc[-min_len:].reset_index(drop=True)
    
    logger.info(f"数据对齐完成，样本数: {min_len}")
    
    # 测试配置
    configs = [
        (False, "基础版（仅固定止损-3%)"),
        (True, "高级版（固定+追踪+每日+回撤多层风控）"),
    ]
    
    results = []
    
    for use_trailing, desc in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试: {desc}")
        logger.info(f"{'='*60}")
        
        result = advanced_risk_backtest(
            klines=klines,
            predictions=predictions,
            initial_balance=1000.0,
            max_exposure=10.0,
            stop_loss_pct=-0.03,
            max_daily_loss_pct=-0.20,
            max_drawdown_pause=0.06,
            use_trailing_stop=use_trailing
        )
        
        logger.info(f"\n配置: {desc}")
        logger.info(f"总收益率: {result['total_return']:.2f}%")
        logger.info(f"最终权益: {result['final_equity']:.2f} USDT")
        logger.info(f"胜率: {result['win_rate']:.2f}%")
        logger.info(f"盈亏比: {result['profit_loss_ratio']:.2f}")
        logger.info(f"最大回撤: {result['max_drawdown']:.2f}%")
        logger.info(f"平均敞口: {result['avg_exposure']:.2f}倍")
        logger.info(f"固定止损触发: {result['stop_loss_count']} 次")
        logger.info(f"追踪止损触发: {result['trailing_stop_count']} 次")
        logger.info(f"最大连续亏损: {result['max_consecutive_losses']} 笔")
        
        results.append({
            'config': desc,
            'total_return': result['total_return'],
            'max_drawdown': result['max_drawdown'],
            'stop_loss_count': result['stop_loss_count'],
            'trailing_stop_count': result['trailing_stop_count']
        })
    
    logger.info(f"\n{'='*80}")
    logger.info("对比完成")
    logger.info(f"{'='*80}")
    
    results_df = pd.DataFrame(results)
    logger.info(f"\n{results_df.to_string(index=False)}")


if __name__ == '__main__':
    main()
