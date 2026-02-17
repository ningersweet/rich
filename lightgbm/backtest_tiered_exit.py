#!/usr/bin/env python3
"""
分批出场策略回测对比
对比两种出场方式：
1. 一次性平仓（当前方案）
2. 三档位分批出场（优化方案）
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels
from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy
from backtest_atr_stop_loss import logger
from btc_quant.tiered_exit import backtest_with_tiered_exit


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("分批出场策略回测对比")
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
    
    min_len = min(len(X_full), len(klines))
    X_full = X_full.iloc[:min_len]
    klines = klines.iloc[:min_len].reset_index(drop=True)
    
    if 'atr_14' in X_full.columns:
        klines['atr_14'] = X_full['atr_14'].values
    else:
        raise ValueError("特征中缺少 atr_14 列")
    
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
    
    # 6. 运行两种策略回测
    logger.info("\n" + "="*60)
    logger.info("开始回测对比")
    logger.info("="*60)
    
    results = []
    
    # 方案 A：一次性平仓
    logger.info("\n【方案 A】一次性平仓（基准）")
    result_a = backtest_with_tiered_exit(
        klines_test, predictions,
        initial_balance=1000.0,
        max_exposure=10.0,
        atr_k=3.5,
        enable_tiered_exit=False
    )
    results.append(result_a)
    logger.info(f"总收益率：{result_a['total_return']:.2f}%")
    logger.info(f"胜率：{result_a['win_rate']:.2f}%")
    logger.info(f"交易数：{result_a['total_trades']}")
    logger.info(f"盈亏比：{result_a['profit_loss_ratio']:.2f}")
    logger.info(f"平均盈利：{result_a['avg_win']:.2f}%")
    logger.info(f"平均亏损：{result_a['avg_loss']:.2f}%")
    
    # 方案 B：三档位分批出场
    logger.info("\n【方案 B】三档位分批出场（优化）")
    result_b = backtest_with_tiered_exit(
        klines_test, predictions,
        initial_balance=1000.0,
        max_exposure=10.0,
        atr_k=3.5,
        enable_tiered_exit=True
    )
    results.append(result_b)
    logger.info(f"总收益率：{result_b['total_return']:.2f}%")
    logger.info(f"胜率：{result_b['win_rate']:.2f}%")
    logger.info(f"交易数：{result_b['total_trades']}")
    logger.info(f"盈亏比：{result_b['profit_loss_ratio']:.2f}")
    logger.info(f"平均盈利：{result_b['avg_win']:.2f}%")
    logger.info(f"平均亏损：{result_b['avg_loss']:.2f}%")
    
    # 7. 汇总对比
    logger.info("\n" + "="*60)
    logger.info("📊 策略对比总结")
    logger.info("="*60)
    
    summary_data = []
    for r in results:
        summary_data.append({
            '策略': r['strategy'],
            '总收益 (%)': f"{r['total_return']:.2f}",
            '胜率 (%)': f"{r['win_rate']:.2f}",
            '交易数': r['total_trades'],
            '盈亏比': f"{r['profit_loss_ratio']:.2f}",
            '平均盈利 (%)': f"{r['avg_win']:.2f}",
            '平均亏损 (%)': f"{r['avg_loss']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # 8. 计算提升效果
    baseline_return = result_a['total_return']
    tiered_return = result_b['total_return']
    return_improvement = tiered_return - baseline_return
    
    baseline_pl_ratio = result_a['profit_loss_ratio']
    tiered_pl_ratio = result_b['profit_loss_ratio']
    pl_ratio_improvement = tiered_pl_ratio - baseline_pl_ratio
    
    logger.info(f"\n📈 分批出场 vs 一次性平仓:")
    logger.info(f"   收益变化：{return_improvement:+.2f}%")
    logger.info(f"   盈亏比提升：{pl_ratio_improvement:+.2f} ({pl_ratio_improvement/baseline_pl_ratio*100:+.1f}%)")
    
    # 9. 保存结果
    output_dir = Path('backtest')
    output_dir.mkdir(exist_ok=True)
    
    for r in results:
        trades_df = pd.DataFrame(r['trades'])
        filename = f"tiered_exit_{r['strategy'].replace('(', '').replace(')', '').replace(' ', '_')}.csv"
        trades_df.to_csv(output_dir / filename, index=False)
    
    logger.info(f"\n✅ 回测结果已保存：{output_dir}")
    
    logger.info("\n" + "="*60)
    logger.info("💡 结论与建议")
    logger.info("="*60)
    
    if return_improvement > 0:
        logger.info("✅ 分批出场提升了总收益！")
        logger.info(f"   收益提升：{return_improvement:+.2f}%")
        logger.info(f"   盈亏比改善：{pl_ratio_improvement:+.2f}")
        logger.info("\n   建议：可以切换到分批出场策略")
    else:
        logger.info("⚠️  分批出场降低了总收益")
        logger.info(f"   收益下降：{return_improvement:+.2f}%")
        logger.info("\n   可能原因:")
        logger.info("   1. 趋势行情较强，过早平仓错失利润")
        logger.info("   2. 三档设置过紧（50%@1RR, 30%@2RR）")
        logger.info("   3. BTC市场特性不适合分批出场")
        logger.info("\n   建议:")
        logger.info("   - 调整分档比例（如减少第一档比例）")
        logger.info("   - 提高止盈目标（如 1.5RR、2.5RR）")
        logger.info("   - 或者放弃分批出场，使用一次性平仓")
    
    # 分析分批出场的细节
    partial_exits = [t for t in result_b['trades'] if t.get('is_partial_exit', False)]
    full_exits = [t for t in result_b['trades'] if not t.get('is_partial_exit', False)]
    
    logger.info(f"\n📊 分批出场详情:")
    logger.info(f"   部分平仓次数：{len(partial_exits)}")
    logger.info(f"   全仓平仓次数：{len(full_exits)}")
    
    if partial_exits:
        avg_partial_pnl = np.mean([t['pnl'] for t in partial_exits])
        logger.info(f"   部分平仓平均盈利：{avg_partial_pnl:.2f} USDT")


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    main()
