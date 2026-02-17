#!/usr/bin/env python3
"""
每分钟预测回测
模拟实盘运行：每分钟都预测，每次都可以下单
使用最佳参数：prob=0.75, rr=2.5
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def backtest_minute_trading(klines, strategy, top_30_features, cfg,
                            initial_balance=1000.0, max_exposure=10.0,
                            prob_threshold=0.75, rr_threshold=2.5):
    """
    每分钟预测回测（模拟实盘）
    
    优化版：只在新K线时预测，但每根K线都可以开仓
    """
    equity = initial_balance
    trades = []
    position = None
    
    logger.info(f"开始回测，初始资金: {initial_balance} USDT")
    logger.info(f"参数：prob_threshold={prob_threshold}, rr_threshold={rr_threshold}")
    logger.info("优化版：提前构建所有特征，只预测一次")
    
    # 提前构建所有特征（优化性能）
    logger.info("构建特征...")
    feature_label_data = build_features_and_labels(cfg, klines)
    X_full = feature_label_data.features.reset_index(drop=True)
    
    # 数据对齐
    min_len = min(len(klines), len(X_full))
    klines = klines.iloc[:min_len].reset_index(drop=True)
    X_full = X_full.iloc[:min_len]
    
    # 提取TOP30特征
    X_top30 = X_full[top_30_features]
    
    # 生成所有预测（一次性）
    logger.info("生成预测...")
    predictions_dict = strategy.predict(
        X_top30,
        rr_threshold=rr_threshold,
        prob_threshold=prob_threshold
    )
    
    predictions = pd.DataFrame({
        'should_trade': predictions_dict['should_trade'],
        'predicted_rr': predictions_dict['predicted_rr'],
        'direction': predictions_dict['direction'],
        'direction_prob': predictions_dict['direction_prob'],
        'holding_period': predictions_dict['holding_period'].clip(1, 30)
    })
    
    logger.info(f"数据对齐完成，开始回测 {len(klines)} 根K线...")
    
    for i in range(len(klines)):
        current_time = klines.iloc[i]['open_time']
        current_price = klines.iloc[i]['close']
        
        # 直接使用预先计算的预测
        should_trade = predictions.iloc[i]['should_trade']
        predicted_rr = predictions.iloc[i]['predicted_rr']
        direction = predictions.iloc[i]['direction']
        direction_prob = predictions.iloc[i]['direction_prob']
        holding_period = predictions.iloc[i]['holding_period']
        
        # 平仓逻辑：检查holding_period
        if position is not None:
            bars_held = i - position['entry_idx']
            
            if bars_held >= position['hold_period']:
                # 计算盈亏
                if position['side'] == 1:
                    price_change_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                pnl = initial_balance * price_change_pct * position['exposure']
                equity += pnl
                
                # 检查爆仓
                if equity <= 0:
                    logger.warning(f"⚠️  爆仓！时间: {current_time}")
                    trades.append({
                        'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                        'exit_time': current_time,
                        'side': 'long' if position['side'] == 1 else 'short',
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'exposure': position['exposure'],
                        'pnl': pnl,
                        'price_change_pct': price_change_pct * 100,
                        'equity_after': 0,
                        'reason': '⏰ 持仓周期',
                        'liquidated': True
                    })
                    return {
                        'final_equity': 0,
                        'total_return': -100.0,
                        'liquidated': True,
                        'trades': trades
                    }
                
                trades.append({
                    'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                    'exit_time': current_time,
                    'side': 'long' if position['side'] == 1 else 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'exposure': position['exposure'],
                    'pnl': pnl,
                    'price_change_pct': price_change_pct * 100,
                    'equity_after': equity,
                    'reason': '⏰ 持仓周期',
                    'liquidated': False
                })
                
                side_name = 'long' if position['side'] == 1 else 'short'
                logger.info(f"[{current_time}] 平仓 {side_name}, 盈亏={price_change_pct*100:.2f}%, 权益={equity:.2f}")
                
                position = None
        
        # 开仓逻辑：每次预测都可以开仓
        if position is None and should_trade:
            entry_price = current_price
            
            # 计算动态敞口
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
            
            position = {
                'side': direction,
                'entry_price': entry_price,
                'entry_idx': i,
                'exposure': optimal_exposure,
                'predicted_rr': predicted_rr,
                'direction_prob': direction_prob,
                'hold_period': int(holding_period)
            }
            
            side_name = 'long' if direction == 1 else 'short'
            
            logger.info(f"[{current_time}] 开仓 {side_name}, 敞口={optimal_exposure:.2f}倍, RR={predicted_rr:.2f}, prob={direction_prob:.3f}, 周期={holding_period}K")
    
    # 最后平仓
    if position is not None:
        final_price = klines.iloc[-1]['close']
        final_time = klines.iloc[-1]['open_time']
        
        if position['side'] == 1:
            price_change_pct = (final_price - position['entry_price']) / position['entry_price']
        else:
            price_change_pct = (position['entry_price'] - final_price) / position['entry_price']
        
        pnl = initial_balance * price_change_pct * position['exposure']
        equity += pnl
        
        if equity <= 0:
            equity = 0
        
        trades.append({
            'entry_time': klines.iloc[position['entry_idx']]['open_time'],
            'exit_time': final_time,
            'side': 'long' if position['side'] == 1 else 'short',
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'exposure': position['exposure'],
            'pnl': pnl,
            'price_change_pct': price_change_pct * 100,
            'equity_after': equity,
            'reason': '回测结束',
            'liquidated': False
        })
    
    total_return = ((equity - initial_balance) / initial_balance) * 100
    
    return {
        'final_equity': equity,
        'total_return': total_return,
        'liquidated': False,
        'trades': trades
    }


def main():
    logger.info("=" * 60)
    logger.info("每分钟预测回测（模拟实盘运行）")
    logger.info("=" * 60)
    
    cfg = load_config()
    
    logger.info("加载历史数据...")
    klines = load_klines(cfg)
    
    # 使用样本外数据（2025-01-01之后）
    backtest_start = pd.Timestamp('2025-01-01T00:00:00Z')
    klines = klines[klines['open_time'] >= backtest_start].reset_index(drop=True)
    
    logger.info(f"样本外数据加载完成，共 {len(klines)} 根K线")
    logger.info(f"回测时间范围: {klines.iloc[0]['open_time']} 至 {klines.iloc[-1]['open_time']}")
    
    # 加载模型
    model_path = Path('models/final_6x_fixed_capital')
    logger.info(f"加载模型: {model_path}")
    strategy = TwoStageRiskRewardStrategy()
    strategy.load(model_path)
    
    top30_features_file = model_path / 'top30_features.txt'
    with open(top30_features_file, 'r') as f:
        top_30_features = [line.strip() for line in f.readlines()]
    
    logger.info("\n" + "=" * 60)
    logger.info("开始回测（每根K线都预测）...")
    logger.info("=" * 60)
    
    result = backtest_minute_trading(
        klines,
        strategy,
        top_30_features,
        cfg,
        initial_balance=1000.0,
        max_exposure=10.0,
        prob_threshold=0.75,
        rr_threshold=2.5
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
        output_file = 'backtest/minute_by_minute_trades.csv'
        trades_df.to_csv(output_file, index=False)
        logger.info(f"\n交易记录已保存: {output_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("回测完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
