#!/usr/bin/env python3
"""
ATR动态止损回测脚本
对比固定止损 vs ATR动态止损的效果
"""
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels
from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy
from btc_quant.dynamic_stop_loss import (
    calculate_atr_stop_loss,
    calculate_dynamic_stop_loss_params,
    TrailingStopManager
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def backtest_with_atr_stop(
    klines,
    predictions,
    initial_balance=1000.0,
    max_exposure=10.0,
    enable_trailing_stop=True,
    atr_k=2.0,
    trailing_pct=0.5
):
    """
    使用ATR动态止损和移动止盈的回测
    
    参数:
        klines: K线数据（必须包含atr_14列）
        predictions: 预测结果
        initial_balance: 初始资金
        max_exposure: 最大敞口
        enable_trailing_stop: 是否启用移动止盈
        atr_k: ATR倍数
        trailing_pct: 移动止盈回撤比例
    """
    equity = initial_balance
    trades = []
    position = None
    trailing_manager = TrailingStopManager(
        trailing_pct=trailing_pct,
        min_profit_pct=0.01
    ) if enable_trailing_stop else None
    
    for i in range(len(predictions)):
        current_price = klines.iloc[i]['close']
        current_atr = klines.iloc[i]['atr_14']
        
        # 平仓逻辑
        if position is not None:
            bars_held = i - position['entry_idx']
            
            # 计算当前盈亏
            if position['side'] == 1:
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            should_exit = False
            exit_reason = None
            
            # 1. 检查ATR止损
            if position['side'] == 1:  # 做多
                if current_price <= position['stop_loss_price']:
                    should_exit = True
                    exit_reason = '🛑 ATR止损'
            else:  # 做空
                if current_price >= position['stop_loss_price']:
                    should_exit = True
                    exit_reason = '🛑 ATR止损'
            
            # 2. 检查移动止盈
            if not should_exit and enable_trailing_stop and trailing_manager.is_active:
                if trailing_manager.should_exit(current_price, position['side']):
                    should_exit = True
                    exit_reason = '📈 移动止盈'
            
            # 3. 检查holding_period
            if not should_exit and bars_held >= position['hold_period']:
                should_exit = True
                exit_reason = '⏰ 持仓周期'
            
            # 更新trailing stop
            if enable_trailing_stop:
                trailing_manager.update(
                    current_price,
                    position['entry_price'],
                    position['side']
                )
            
            # 执行平仓
            if should_exit:
                pnl = position['position_value'] * price_change_pct * position['exposure']
                equity += pnl
                
                # 检查爆仓
                if equity <= 0:
                    logger.warning(f"⚠️  爆仓！")
                    trades.append({
                        'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                        'exit_time': klines.iloc[i]['open_time'],
                        'side': 'long' if position['side'] == 1 else 'short',
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'stop_loss_price': position['stop_loss_price'],
                        'exposure': position['exposure'],
                        'bars_held': bars_held,
                        'pnl': pnl,
                        'pnl_pct': price_change_pct * 100,
                        'equity_after': 0,
                        'reason': exit_reason,
                        'liquidated': True
                    })
                    return {
                        'initial_balance': initial_balance,
                        'final_equity': 0,
                        'total_return': -100.0,
                        'liquidated': True,
                        'trades': trades
                    }
                
                trades.append({
                    'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                    'exit_time': klines.iloc[i]['open_time'],
                    'side': 'long' if position['side'] == 1 else 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'stop_loss_price': position['stop_loss_price'],
                    'stop_loss_pct': position['stop_loss_pct'] * 100,
                    'exposure': position['exposure'],
                    'bars_held': bars_held,
                    'pnl': pnl,
                    'pnl_pct': price_change_pct * 100,
                    'equity_after': equity,
                    'reason': exit_reason,
                    'liquidated': False
                })
                
                position = None
                if enable_trailing_stop:
                    trailing_manager.reset()
        
        # 开仓逻辑
        if position is None and predictions.iloc[i]['should_trade']:
            entry_price = klines.iloc[i]['close']
            predicted_rr = predictions.iloc[i]['predicted_rr']
            direction_prob = predictions.iloc[i]['direction_prob']
            direction = int(predictions.iloc[i]['direction'])
            hold_period = int(predictions.iloc[i]['holding_period'])
            
            # 计算动态敞口
            rr_factor = min(predicted_rr / 2.5, 2.0)
            prob_factor = max((direction_prob - 0.5) / 0.5, 0)
            exposure = 2.0 + rr_factor * 3.0 + prob_factor * 3.0
            exposure = np.clip(exposure, 1.0, max_exposure)
            
            # 计算ATR止损
            stop_loss_price, stop_loss_pct = calculate_atr_stop_loss(
                entry_price,
                current_atr,
                direction,
                k=atr_k,
                min_stop_loss_pct=0.01,
                max_stop_loss_pct=0.05
            )
            
            position = {
                'entry_idx': i,
                'entry_price': entry_price,
                'side': direction,
                'hold_period': hold_period,
                'exposure': exposure,
                'position_value': initial_balance,
                'stop_loss_price': stop_loss_price,
                'stop_loss_pct': stop_loss_pct,
                'atr_k': atr_k,
                'current_atr': current_atr
            }
    
    # 计算统计指标
    if not trades:
        return {
            'initial_balance': initial_balance,
            'final_equity': equity,
            'total_return': 0.0,
            'liquidated': False,
            'trades': []
        }
    
    trades_df = pd.DataFrame(trades)
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    
    # 按退出原因分组统计
    exit_reasons = trades_df.groupby('reason').agg({
        'pnl': ['count', 'sum', 'mean']
    }).round(4)
    
    stats = {
        'initial_balance': initial_balance,
        'final_equity': equity,
        'total_return': (equity - initial_balance) / initial_balance * 100,
        'liquidated': False,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
        'avg_win': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
        'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0,
        'exit_reasons': exit_reasons.to_dict() if len(exit_reasons) > 0 else {},
        'trades': trades
    }
    
    # ATR止损触发率
    atr_stop_trades = trades_df[trades_df['reason'] == '🛑 ATR止损']
    stats['atr_stop_trigger_rate'] = len(atr_stop_trades) / len(trades) * 100 if trades else 0
    
    if enable_trailing_stop:
        trailing_trades = trades_df[trades_df['reason'] == '📈 移动止盈']
        stats['trailing_stop_trigger_rate'] = len(trailing_trades) / len(trades) * 100 if trades else 0
    
    return stats


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("ATR动态止损回测")
    logger.info("="*60)
    
    # 1. 加载配置和数据
    cfg = load_config(Path('config.yaml'))
    logger.info("✅ 配置加载完成")
    
    klines = load_klines(cfg)
    logger.info(f"✅ K线加载完成，共 {len(klines)} 根")
    
    # 2. 构建特征
    logger.info("构建特征...")
    feature_label_data = build_features_and_labels(cfg, klines)
    X_full = feature_label_data.features.reset_index(drop=True)
    
    # 对齐K线和特征长度
    min_len = min(len(X_full), len(klines))
    X_full = X_full.iloc[:min_len]
    klines = klines.iloc[:min_len].reset_index(drop=True)
    
    # 重要：将ATR特征添加到klines中（用于动态止损）
    if 'atr_14' in X_full.columns:
        klines['atr_14'] = X_full['atr_14'].values
    else:
        raise ValueError("特征中缺少atr_14列")
    
    logger.info(f"✅ 特征构建完成，样本数: {min_len}")
    
    # 3. 筛选样本外数据（2025-01-01之后）
    sample_out_mask = klines['open_time'] >= pd.Timestamp('2025-01-01', tz='UTC')
    klines_test = klines[sample_out_mask].reset_index(drop=True)
    X_test = X_full[sample_out_mask].reset_index(drop=True)
    logger.info(f"✅ 样本外数据筛选完成: {len(klines_test)}根K线")
    
    # 4. 加载模型
    model_dir = Path('models/final_6x_fixed_capital')
    strategy = TwoStageRiskRewardStrategy()
    strategy.load(model_dir)
    logger.info(f"✅ 模型加载完成: {model_dir}")
    
    # 5. 生成预测
    top30_features_file = model_dir / 'top30_features.txt'
    with open(top30_features_file, 'r') as f:
        top_30_features = [line.strip() for line in f.readlines()]
    
    X_top30 = X_test[top_30_features]
    predictions_dict = strategy.predict(
        X_top30,
        rr_threshold=2.5,
        prob_threshold=0.75
    )
    
    predictions = pd.DataFrame({
        'predicted_rr': predictions_dict['predicted_rr'],
        'direction': predictions_dict['direction'],
        'holding_period': predictions_dict['holding_period'].clip(1, 30),
        'direction_prob': predictions_dict['direction_prob'],
        'should_trade': predictions_dict['should_trade']
    })
    logger.info(f"✅ 预测完成，信号数: {predictions['should_trade'].sum()}")
    
    # 6. 对比回测：固定止损 vs ATR动态止损
    logger.info("\n" + "="*60)
    logger.info("方案A: 固定止损 -3%（基线）")
    logger.info("="*60)
    
    # 方案A：使用固定止损（模拟）- 这里简化为只用holding_period
    from backtest_best_params import backtest_holding_period_only
    baseline_result = backtest_holding_period_only(klines_test, predictions)
    
    logger.info(f"初始资金: {baseline_result['initial_balance']:.2f} USDT")
    logger.info(f"最终权益: {baseline_result['final_equity']:.2f} USDT")
    logger.info(f"总收益率: {baseline_result['total_return']:.2f}%")
    logger.info(f"总交易数: {baseline_result.get('total_trades', len(baseline_result['trades']))}")
    logger.info(f"胜率: {baseline_result.get('win_rate', 0):.2f}%")
    
    # 方案B：ATR动态止损（无移动止盈）
    logger.info("\n" + "="*60)
    logger.info("方案B: ATR动态止损 (k=2.0)")
    logger.info("="*60)
    
    atr_result = backtest_with_atr_stop(
        klines_test,
        predictions,
        initial_balance=1000.0,
        max_exposure=10.0,
        enable_trailing_stop=False,
        atr_k=2.0
    )
    
    logger.info(f"初始资金: {atr_result['initial_balance']:.2f} USDT")
    logger.info(f"最终权益: {atr_result['final_equity']:.2f} USDT")
    logger.info(f"总收益率: {atr_result['total_return']:.2f}%")
    logger.info(f"总交易数: {atr_result['total_trades']}")
    logger.info(f"胜率: {atr_result['win_rate']:.2f}%")
    logger.info(f"ATR止损触发率: {atr_result['atr_stop_trigger_rate']:.2f}%")
    
    # 方案C：ATR动态止损 + 移动止盈
    logger.info("\n" + "="*60)
    logger.info("方案C: ATR动态止损 + 移动止盈 (k=2.0, trailing=50%)")
    logger.info("="*60)
    
    atr_trailing_result = backtest_with_atr_stop(
        klines_test,
        predictions,
        initial_balance=1000.0,
        max_exposure=10.0,
        enable_trailing_stop=True,
        atr_k=2.0,
        trailing_pct=0.5
    )
    
    logger.info(f"初始资金: {atr_trailing_result['initial_balance']:.2f} USDT")
    logger.info(f"最终权益: {atr_trailing_result['final_equity']:.2f} USDT")
    logger.info(f"总收益率: {atr_trailing_result['total_return']:.2f}%")
    logger.info(f"总交易数: {atr_trailing_result['total_trades']}")
    logger.info(f"胜率: {atr_trailing_result['win_rate']:.2f}%")
    logger.info(f"ATR止损触发率: {atr_trailing_result['atr_stop_trigger_rate']:.2f}%")
    logger.info(f"移动止盈触发率: {atr_trailing_result['trailing_stop_trigger_rate']:.2f}%")
    
    # 7. 对比总结
    logger.info("\n" + "="*60)
    logger.info("📊 对比总结")
    logger.info("="*60)
    
    comparison = pd.DataFrame({
        '指标': ['总收益率(%)', '胜率(%)', '总交易数'],
        '固定止损': [
            f"{baseline_result['total_return']:.2f}",
            f"{baseline_result.get('win_rate', 0):.2f}",
            baseline_result.get('total_trades', len(baseline_result['trades']))
        ],
        'ATR止损': [
            f"{atr_result['total_return']:.2f}",
            f"{atr_result['win_rate']:.2f}",
            atr_result['total_trades']
        ],
        'ATR+移动止盈': [
            f"{atr_trailing_result['total_return']:.2f}",
            f"{atr_trailing_result['win_rate']:.2f}",
            atr_trailing_result['total_trades']
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # 8. 保存结果
    trades_df = pd.DataFrame(atr_trailing_result['trades'])
    output_file = Path('backtest/atr_stop_loss_trades.csv')
    trades_df.to_csv(output_file, index=False)
    logger.info(f"\n✅ 交易记录已保存: {output_file}")
    
    logger.info("\n✅ 回测完成！")


if __name__ == '__main__':
    main()
