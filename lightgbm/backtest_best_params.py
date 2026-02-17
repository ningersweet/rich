#!/usr/bin/env python3
"""
使用最佳参数回测：prob=0.75, rr=2.5
纯holding_period策略（无止损止盈）
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
from train_dynamic_exposure_with_advanced_risk import advanced_risk_backtest

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def backtest_holding_period_only(klines, predictions, initial_balance=1000.0, max_exposure=10.0):
    """
    纯holding_period策略回测
    
    平仓规则：
    - 达到预测的holding_period时平仓（无任何止损止盈）
    """
    equity = initial_balance
    trades = []
    position = None
    
    for i in range(len(predictions)):
        current_price = klines.iloc[i]['close']
        
        # 平仓逻辑：只检查holding_period
        if position is not None:
            bars_held = i - position['entry_idx']
            
            if bars_held >= position['hold_period']:
                # 计算盈亏
                if position['side'] == 1:
                    price_change_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
                
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
                        'exposure': position['exposure'],
                        'pnl': pnl,
                        'pnl_pct': price_change_pct * 100,
                        'equity_after': 0,
                        'reason': '⏰ 持仓周期',
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
                    'exposure': position['exposure'],
                    'pnl': pnl,
                    'pnl_pct': price_change_pct * 100,
                    'equity_after': equity,
                    'reason': '⏰ 持仓周期',
                    'liquidated': False
                })
                
                position = None
        
        # 开仓逻辑
        if position is None and predictions.iloc[i]['should_trade']:
            entry_price = klines.iloc[i]['close']
            predicted_rr = predictions.iloc[i]['predicted_rr']
            direction_prob = predictions.iloc[i]['direction_prob']
            
            # 计算动态敞口（与final_10x一致）
            base_exposure = 1.0
            
            # RR贡献
            if predicted_rr >= 6.0:
                rr_multiplier = 5.0
            elif predicted_rr >= 4.0:
                rr_multiplier = 3.0 + (predicted_rr - 4.0) * 1.0
            elif predicted_rr >= 2.5:
                rr_multiplier = 1.0 + (predicted_rr - 2.5) * 1.33
            else:
                rr_multiplier = 0.0
            
            # Prob贡献
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
            
            position = {
                'side': predictions.iloc[i]['direction'],
                'entry_price': entry_price,
                'entry_idx': i,
                'position_value': position_value,
                'exposure': optimal_exposure,
                'predicted_rr': predicted_rr,
                'direction_prob': direction_prob,
                'hold_period': int(predictions.iloc[i]['holding_period'])
            }
    
    # 最后平仓
    if position is not None:
        final_price = klines.iloc[-1]['close']
        if position['side'] == 1:
            price_change_pct = (final_price - position['entry_price']) / position['entry_price']
        else:
            price_change_pct = (position['entry_price'] - final_price) / position['entry_price']
        
        pnl = position['position_value'] * price_change_pct * position['exposure']
        equity += pnl
        
        if equity <= 0:
            equity = 0
        
        trades.append({
            'entry_time': klines.iloc[position['entry_idx']]['open_time'],
            'exit_time': klines.iloc[-1]['open_time'],
            'side': 'long' if position['side'] == 1 else 'short',
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'exposure': position['exposure'],
            'pnl': pnl,
            'pnl_pct': price_change_pct * 100,
            'equity_after': equity,
            'reason': '回测结束',
            'liquidated': False
        })
    
    total_return = ((equity - initial_balance) / initial_balance) * 100
    
    return {
        'initial_balance': initial_balance,
        'final_equity': equity,
        'total_return': total_return,
        'liquidated': False,
        'trades': trades
    }


def main():
    logger.info("=" * 60)
    logger.info("最佳参数回测：prob=0.75, rr=2.5 + holding_period")
    logger.info("=" * 60)
    
    cfg = load_config()
    
    logger.info("加载历史数据...")
    klines = load_klines(cfg)
    
    # 使用样本外数据回测（2025-01-01之后）
    backtest_start = pd.Timestamp('2025-01-01T00:00:00Z')
    klines = klines[klines['open_time'] >= backtest_start].reset_index(drop=True)
    
    logger.info(f"样本外数据加载完成，共 {len(klines)} 根K线")
    logger.info(f"回测时间范围: {klines.iloc[0]['open_time']} 至 {klines.iloc[-1]['open_time']}")
    
    logger.info("构建特征...")
    feature_label_data = build_features_and_labels(cfg, klines)
    X_full = feature_label_data.features.reset_index(drop=True)
    
    # 关键：对齐klines和特征长度（与visualize一致）
    min_len = min(len(X_full), len(klines))
    X_full = X_full.iloc[:min_len]
    klines = klines.iloc[:min_len].reset_index(drop=True)
    
    logger.info(f"数据对齐完成，样本数: {min_len}")
    
    model_path = Path('models/final_6x_fixed_capital')
    logger.info(f"加载模型: {model_path}")
    strategy = TwoStageRiskRewardStrategy()
    strategy.load(model_path)
    
    top30_features_file = model_path / 'top30_features.txt'
    with open(top30_features_file, 'r') as f:
        top_30_features = [line.strip() for line in f.readlines()]
    
    X_top30 = X_full[top_30_features]
    
    logger.info("生成预测信号（prob=0.75, rr=2.5）...")
    predictions_dict = strategy.predict(
        X_top30,
        rr_threshold=2.5,
        prob_threshold=0.75
    )
    
    # 转换为DataFrame（与visualize一致）
    predictions = pd.DataFrame({
        'predicted_rr': predictions_dict['predicted_rr'],
        'direction': predictions_dict['direction'],
        'holding_period': predictions_dict['holding_period'].clip(1, 30),
        'direction_prob': predictions_dict['direction_prob'],
        'should_trade': predictions_dict['should_trade']
    })
    
    # 数据对齐
    min_len = min(len(klines), len(predictions))
    klines_aligned = klines.iloc[-min_len:].reset_index(drop=True)
    predictions_aligned = predictions.iloc[-min_len:].reset_index(drop=True)
    
    logger.info(f"信号统计: should_trade=True 的数量: {predictions_aligned['should_trade'].sum()}")
    
    logger.info("\n" + "=" * 60)
    logger.info("开始回测（多层风控）...")
    logger.info("=" * 60)
    
    result = advanced_risk_backtest(
        klines_aligned,
        predictions_aligned,
        initial_balance=1000.0,
        max_exposure=10.0,
        stop_loss_pct=-0.03,
        max_daily_loss_pct=-0.20,
        max_drawdown_pause=0.06,
        use_trailing_stop=True
    )
    
    trades_df = pd.DataFrame(result['trades'])
    
    logger.info("\n" + "=" * 60)
    logger.info("回测结果")
    logger.info("=" * 60)
    
    logger.info(f"初始资金: 1000.00 USDT")
    logger.info(f"最终权益: {result['final_equity']:.2f} USDT")
    logger.info(f"总收益率: {result['total_return']:.2f}%")
    logger.info(f"是否爆仓: {'是' if result.get('liquidated', False) else '否'}")
    
    if len(trades_df) > 0:
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        logger.info(f"\n总交易数: {len(trades_df)}")
        logger.info(f"盈利交易: {len(winning_trades)}")
        logger.info(f"亏损交易: {len(losing_trades)}")
        logger.info(f"胜率: {len(winning_trades) / len(trades_df) * 100:.2f}%")
        
        if 'price_change_pct' in trades_df.columns:
            logger.info(f"\n平均盈亏: {trades_df['price_change_pct'].mean():.2f}%")
            if len(winning_trades) > 0:
                logger.info(f"平均盈利: {winning_trades['price_change_pct'].mean():.2f}%")
            if len(losing_trades) > 0:
                logger.info(f"平均亏损: {losing_trades['price_change_pct'].mean():.2f}%")
        
        # 保存交易记录
        output_file = 'backtest/best_params_trades.csv'
        trades_df.to_csv(output_file, index=False)
        logger.info(f"\n交易记录已保存: {output_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("回测完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
