#!/usr/bin/env python3
"""
自适应ATR倍数k值回测脚本
根据市场波动率自动调整止损倍数k（1.5-3.5）
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels
from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy
from backtest_atr_stop_loss import (
    backtest_with_atr_stop, 
    logger
)
from btc_quant.dynamic_stop_loss import calculate_adaptive_atr_multiplier, calculate_atr_stop_loss


def backtest_with_adaptive_atr_k(klines, predictions, initial_balance=1000.0, max_exposure=10.0):
    """
    自适应ATR倍数k值回测
    
    核心逻辑:
        - 低波动市场：k=2.0（紧止损，防震荡洗盘）
        - 中等波动：k=2.5（平衡）
        - 高波动市场：k=3.5（宽松止损，追踪趋势）
    
    参数:
        klines: K线数据
        predictions: 预测结果
        initial_balance: 初始资金
        max_exposure: 最大敞口
    
    返回:
        回测结果字典
    """
    equity = initial_balance
    trades = []
    position = None
    
    # 计算历史ATR统计量（用于自适应k值）
    if 'atr_14' not in klines.columns:
        raise ValueError("K线数据中缺少atr_14列")
    
    atr_series = klines['atr_14']
    atr_median = atr_series.median()
    atr_min = atr_series.min()
    atr_max = atr_series.max()
    
    # 滚动窗口计算ATR百分位（使用过去30天的数据）
    rolling_window = 30 * 96  # 30天（96根K线/天）
    
    for i in range(len(predictions)):
        current_price = klines.iloc[i]['close']
        current_atr = klines.iloc[i]['atr_14']
        
        # 计算当前ATR在滚动窗口中的百分位
        start_idx = max(0, i - rolling_window)
        recent_atr = atr_series.iloc[start_idx:i+1]
        atr_percentile = (recent_atr <= current_atr).sum() / len(recent_atr)
        
        # 自适应k值（需要波动率百分位和趋势强度）
        # 简化版：假设趋势强度为0（中性），仅使用波动率调整
        trend_strength = 0.0
        adaptive_k = calculate_adaptive_atr_multiplier(
            atr_percentile,
            trend_strength,
            base_k=2.0
        )
        
        # 开仓逻辑
        if position is None and predictions.iloc[i]['should_trade']:
            entry_price = current_price
            predicted_rr = predictions.iloc[i]['predicted_rr']
            direction_prob = predictions.iloc[i]['direction_prob']
            
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
            
            # 计算自适应止损价
            stop_loss_price, stop_loss_pct = calculate_atr_stop_loss(
                entry_price=entry_price,
                atr=current_atr,
                direction=predictions.iloc[i]['direction'],
                k=adaptive_k,
                min_stop_loss_pct=0.01,
                max_stop_loss_pct=0.05
            )
            
            position = {
                'side': predictions.iloc[i]['direction'],
                'entry_price': entry_price,
                'entry_idx': i,
                'position_value': position_value,
                'exposure': optimal_exposure,
                'stop_loss_price': stop_loss_price,
                'stop_loss_pct': stop_loss_pct,
                'predicted_rr': predicted_rr,
                'direction_prob': direction_prob,
                'hold_period': int(predictions.iloc[i]['holding_period']),
                'adaptive_k': adaptive_k  # 记录使用的k值
            }
        
        # 平仓逻辑
        elif position is not None:
            bars_held = i - position['entry_idx']
            should_exit = False
            exit_reason = None
            
            # 1. 检查ATR止损（使用自适应k值计算的止损价）
            if position['side'] == 1:  # 做多
                if current_price <= position['stop_loss_price']:
                    should_exit = True
                    exit_reason = '🛑 ATR止损'
            else:  # 做空
                if current_price >= position['stop_loss_price']:
                    should_exit = True
                    exit_reason = '🛑 ATR止损'
            
            # 2. 检查持仓周期到期
            if not should_exit and bars_held >= position['hold_period']:
                should_exit = True
                exit_reason = '⏰ 持仓周期'
            
            # 执行平仓
            if should_exit:
                if position['side'] == 1:
                    price_change_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                pnl = initial_balance * price_change_pct * position['exposure']
                equity += pnl
                
                trades.append({
                    'entry_time': klines.iloc[position['entry_idx']]['open_time'],
                    'exit_time': klines.iloc[i]['open_time'],
                    'side': 'long' if position['side'] == 1 else 'short',
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'price_change_pct': price_change_pct * 100,
                    'exposure': position['exposure'],
                    'pnl': pnl,
                    'pnl_pct': price_change_pct * 100 * position['exposure'],
                    'equity_after': equity,
                    'bars_held': bars_held,
                    'reason': exit_reason,
                    'used_k': position['adaptive_k'],
                    'stop_loss_triggered': (exit_reason == '🛑 ATR止损')
                })
                
                position = None
    
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
            'pnl_pct': price_change_pct * 100 * position['exposure'],
            'equity_after': equity,
            'bars_held': len(klines) - position['entry_idx'],
            'reason': '📊 期末平仓',
            'used_k': position['adaptive_k'],
            'stop_loss_triggered': False
        })
    
    # 计算回测指标
    total_return = (equity - initial_balance) / initial_balance * 100
    win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100 if trades else 0
    atr_stop_trigger_rate = sum(1 for t in trades if t.get('stop_loss_triggered', False)) / len(trades) * 100 if trades else 0
    
    return {
        'initial_balance': initial_balance,
        'final_equity': equity,
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'trades': trades,
        'atr_stop_trigger_rate': atr_stop_trigger_rate
    }


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("自适应ATR倍数k值回测")
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
    
    # 对齐长度
    min_len = min(len(X_full), len(klines))
    X_full = X_full.iloc[:min_len]
    klines = klines.iloc[:min_len].reset_index(drop=True)
    
    # 添加ATR到klines
    if 'atr_14' in X_full.columns:
        klines['atr_14'] = X_full['atr_14'].values
    else:
        raise ValueError("特征中缺少atr_14列")
    
    logger.info(f"✅ 特征构建完成，样本数：{min_len}")
    
    # 3. 筛选样本外数据
    sample_out_mask = klines['open_time'] >= pd.Timestamp('2025-01-01', tz='UTC')
    klines_test = klines[sample_out_mask].reset_index(drop=True)
    X_test = X_full[sample_out_mask].reset_index(drop=True)
    logger.info(f"✅ 样本外数据筛选完成：{len(klines_test)}根K线")
    
    # 4. 加载模型
    model_dir = Path('models/final_6x_fixed_capital')
    strategy = TwoStageRiskRewardStrategy()
    strategy.load(model_dir)
    logger.info(f"✅ 模型加载完成：{model_dir}")
    
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
    logger.info(f"✅ 预测完成，信号数：{predictions['should_trade'].sum()}")
    
    # 6. 运行自适应k值回测
    logger.info("\n" + "="*60)
    logger.info("开始自适应ATR倍数k值回测")
    logger.info("="*60)
    
    result = backtest_with_adaptive_atr_k(
        klines_test,
        predictions,
        initial_balance=1000.0,
        max_exposure=10.0
    )
    
    # 7. 输出结果
    logger.info("\n" + "="*60)
    logger.info("📊 自适应k值回测结果")
    logger.info("="*60)
    
    logger.info(f"初始资金：{result['initial_balance']:.2f} USDT")
    logger.info(f"最终权益：{result['final_equity']:.2f} USDT")
    logger.info(f"总收益率：{result['total_return']:.2f}%")
    logger.info(f"胜率：{result['win_rate']:.2f}%")
    logger.info(f"总交易数：{result['total_trades']}")
    logger.info(f"ATR止损触发率：{result['atr_stop_trigger_rate']:.2f}%")
    
    # 分析k值分布
    trades_df = pd.DataFrame(result['trades'])
    k_stats = trades_df['used_k'].describe()
    logger.info("\n" + "="*60)
    logger.info("📈 自适应k值使用统计")
    logger.info("="*60)
    logger.info(f"k值均值：{k_stats['mean']:.2f}")
    logger.info(f"k值标准差：{k_stats['std']:.2f}")
    logger.info(f"k值最小值：{k_stats['min']:.2f}")
    logger.info(f"k值最大值：{k_stats['max']:.2f}")
    
    # k值分档统计
    logger.info("\nk值分档统计:")
    low_k = len(trades_df[trades_df['used_k'] < 2.5]) / len(trades_df) * 100
    mid_k = len(trades_df[(trades_df['used_k'] >= 2.5) & (trades_df['used_k'] < 3.0)]) / len(trades_df) * 100
    high_k = len(trades_df[trades_df['used_k'] >= 3.0]) / len(trades_df) * 100
    logger.info(f"  低波动 (k<2.5): {low_k:.1f}%")
    logger.info(f"  中波动 (2.5≤k<3.0): {mid_k:.1f}%")
    logger.info(f"  高波动 (k≥3.0): {high_k:.1f}%")
    
    # 不同k值区间的表现
    logger.info("\n" + "="*60)
    logger.info("📊 不同k值区间表现对比")
    logger.info("="*60)
    
    for k_range, mask, label in [
        ((1.5, 2.5), trades_df['used_k'] < 2.5, "低波动"),
        ((2.5, 3.0), (trades_df['used_k'] >= 2.5) & (trades_df['used_k'] < 3.0), "中波动"),
        ((3.0, 3.5), trades_df['used_k'] >= 3.0, "高波动")
    ]:
        subset = trades_df[mask]
        if len(subset) > 0:
            avg_return = subset['pnl_pct'].mean()
            win_rate = (subset['pnl'] > 0).sum() / len(subset) * 100
            logger.info(f"{label} (k={k_range[0]:.1f}-{k_range[1]:.1f}):")
            logger.info(f"  交易数：{len(subset)}, 胜率：{win_rate:.2f}%, 平均收益：{avg_return:.2f}%")
    
    # 保存结果
    output_dir = Path('backtest')
    output_dir.mkdir(exist_ok=True)
    trades_df.to_csv(output_dir / 'adaptive_atr_k_trades.csv', index=False)
    logger.info(f"\n✅ 交易记录已保存：{output_dir / 'adaptive_atr_k_trades.csv'}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ 自适应ATR倍数k值回测完成！")
    logger.info("="*60)
    
    # 对比固定k值
    logger.info("\n" + "="*60)
    logger.info("🆚 与固定k值对比")
    logger.info("="*60)
    
    from optimize_atr_k import optimize_atr_k
    k_values = [2.0, 3.5]
    fixed_results = optimize_atr_k(klines_test, predictions, k_values, enable_trailing_stop=False)
    
    logger.info(f"固定k=2.0: 收益率 {fixed_results[fixed_results['k']==2.0]['total_return'].values[0]:.2f}%")
    logger.info(f"固定k=3.5: 收益率 {fixed_results[fixed_results['k']==3.5]['total_return'].values[0]:.2f}%")
    logger.info(f"自适应k:  收益率 {result['total_return']:.2f}%")
    
    adaptive_vs_k2 = result['total_return'] - fixed_results[fixed_results['k']==2.0]['total_return'].values[0]
    adaptive_vs_k35 = result['total_return'] - fixed_results[fixed_results['k']==3.5]['total_return'].values[0]
    
    logger.info(f"\n相比k=2.0提升：{adaptive_vs_k2:+.2f}%")
    logger.info(f"相比k=3.5变化：{adaptive_vs_k35:+.2f}%")


if __name__ == '__main__':
    import pandas as pd
    main()
