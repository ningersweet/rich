#!/usr/bin/env python3
"""
ATR倍数k值调优脚本
测试不同k值对回测性能的影响
"""
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels
from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy
from backtest_atr_stop_loss import backtest_with_atr_stop

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def optimize_atr_k(klines, predictions, k_values, enable_trailing_stop=False):
    """
    测试不同ATR倍数k值的回测性能
    
    参数:
        klines: K线数据
        predictions: 预测结果
        k_values: 要测试的k值列表
        enable_trailing_stop: 是否启用移动止盈
    
    返回:
        结果DataFrame
    """
    results = []
    
    for k in k_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试 k={k}")
        logger.info(f"{'='*60}")
        
        result = backtest_with_atr_stop(
            klines,
            predictions,
            initial_balance=1000.0,
            max_exposure=10.0,
            enable_trailing_stop=enable_trailing_stop,
            atr_k=k
        )
        
        # 计算额外指标
        if result['total_trades'] > 0:
            trades_df = pd.DataFrame(result['trades'])
            
            # 计算盈亏比
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl_pct'].mean()) if len(losing_trades) > 0 else 0
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            # 计算最大回撤
            equity_curve = [1000.0]
            for trade in result['trades']:
                equity_curve.append(trade['equity_after'])
            equity_series = pd.Series(equity_curve)
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 计算夏普比率（简化版）
            returns = equity_series.pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 96) if returns.std() > 0 else 0
            
        else:
            avg_win = 0
            avg_loss = 0
            profit_loss_ratio = 0
            max_drawdown = 0
            sharpe_ratio = 0
        
        results.append({
            'k': k,
            'total_return': result['total_return'],
            'final_equity': result['final_equity'],
            'win_rate': result['win_rate'],
            'total_trades': result['total_trades'],
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'atr_stop_trigger_rate': result['atr_stop_trigger_rate']
        })
        
        logger.info(f"总收益率: {result['total_return']:.2f}%")
        logger.info(f"胜率: {result['win_rate']:.2f}%")
        logger.info(f"盈亏比: {profit_loss_ratio:.2f}")
        logger.info(f"最大回撤: {max_drawdown:.2f}%")
        logger.info(f"夏普比率: {sharpe_ratio:.2f}")
        logger.info(f"ATR止损触发率: {result['atr_stop_trigger_rate']:.2f}%")
    
    return pd.DataFrame(results)


def plot_optimization_results(results_df, output_dir='backtest'):
    """
    绘制优化结果图表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('ATR倍数k值优化结果', fontsize=16, fontweight='bold')
    
    # 1. 总收益率
    axes[0, 0].plot(results_df['k'], results_df['total_return'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('ATR倍数 k', fontsize=12)
    axes[0, 0].set_ylabel('总收益率 (%)', fontsize=12)
    axes[0, 0].set_title('总收益率 vs k值', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 标注最优点
    max_return_idx = results_df['total_return'].idxmax()
    max_k = results_df.loc[max_return_idx, 'k']
    max_return = results_df.loc[max_return_idx, 'total_return']
    axes[0, 0].annotate(f'最优: k={max_k}\n{max_return:.2f}%',
                        xy=(max_k, max_return),
                        xytext=(max_k + 0.2, max_return + 200),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, color='red')
    
    # 2. 胜率
    axes[0, 1].plot(results_df['k'], results_df['win_rate'], 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('ATR倍数 k', fontsize=12)
    axes[0, 1].set_ylabel('胜率 (%)', fontsize=12)
    axes[0, 1].set_title('胜率 vs k值', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=80, color='r', linestyle='--', alpha=0.5, label='目标胜率80%')
    axes[0, 1].legend()
    
    # 3. 盈亏比
    axes[0, 2].plot(results_df['k'], results_df['profit_loss_ratio'], 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 2].set_xlabel('ATR倍数 k', fontsize=12)
    axes[0, 2].set_ylabel('盈亏比', fontsize=12)
    axes[0, 2].set_title('盈亏比 vs k值', fontsize=14)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 最大回撤
    axes[1, 0].plot(results_df['k'], results_df['max_drawdown'].abs(), 'o-', linewidth=2, markersize=8, color='red')
    axes[1, 0].set_xlabel('ATR倍数 k', fontsize=12)
    axes[1, 0].set_ylabel('最大回撤 (%)', fontsize=12)
    axes[1, 0].set_title('最大回撤 vs k值', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].invert_yaxis()
    
    # 5. 夏普比率
    axes[1, 1].plot(results_df['k'], results_df['sharpe_ratio'], 'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 1].set_xlabel('ATR倍数 k', fontsize=12)
    axes[1, 1].set_ylabel('夏普比率', fontsize=12)
    axes[1, 1].set_title('夏普比率 vs k值', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=3.0, color='r', linestyle='--', alpha=0.5, label='目标夏普3.0')
    axes[1, 1].legend()
    
    # 6. ATR止损触发率
    axes[1, 2].plot(results_df['k'], results_df['atr_stop_trigger_rate'], 'o-', linewidth=2, markersize=8, color='brown')
    axes[1, 2].set_xlabel('ATR倍数 k', fontsize=12)
    axes[1, 2].set_ylabel('ATR止损触发率 (%)', fontsize=12)
    axes[1, 2].set_title('ATR止损触发率 vs k值', fontsize=14)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / 'atr_k_optimization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"✅ 优化结果图表已保存: {output_file}")
    
    return str(output_file)


def calculate_综合得分(results_df):
    """
    计算综合得分（加权多指标评估）
    
    权重分配:
        - 总收益率: 30%
        - 胜率: 25%
        - 盈亏比: 20%
        - 夏普比率: 15%
        - 最大回撤: 10% (负向，越小越好)
    """
    # 归一化各指标到0-1
    normalized = pd.DataFrame()
    
    # 正向指标（越大越好）
    for col in ['total_return', 'win_rate', 'profit_loss_ratio', 'sharpe_ratio']:
        min_val = results_df[col].min()
        max_val = results_df[col].max()
        if max_val > min_val:
            normalized[col] = (results_df[col] - min_val) / (max_val - min_val)
        else:
            normalized[col] = 1.0
    
    # 负向指标（越小越好，需要反转）
    min_dd = results_df['max_drawdown'].min()
    max_dd = results_df['max_drawdown'].max()
    if max_dd > min_dd:
        normalized['max_drawdown'] = 1.0 - (results_df['max_drawdown'] - min_dd) / (max_dd - min_dd)
    else:
        normalized['max_drawdown'] = 1.0
    
    # 加权计算综合得分
    weights = {
        'total_return': 0.30,
        'win_rate': 0.25,
        'profit_loss_ratio': 0.20,
        'sharpe_ratio': 0.15,
        'max_drawdown': 0.10
    }
    
    composite_score = (
        normalized['total_return'] * weights['total_return'] +
        normalized['win_rate'] * weights['win_rate'] +
        normalized['profit_loss_ratio'] * weights['profit_loss_ratio'] +
        normalized['sharpe_ratio'] * weights['sharpe_ratio'] +
        normalized['max_drawdown'] * weights['max_drawdown']
    ) * 100
    
    return composite_score


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("ATR倍数k值调优")
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
    
    # 添加ATR到klines
    if 'atr_14' in X_full.columns:
        klines['atr_14'] = X_full['atr_14'].values
    else:
        raise ValueError("特征中缺少atr_14列")
    
    logger.info(f"✅ 特征构建完成，样本数: {min_len}")
    
    # 3. 筛选样本外数据
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
    
    # 6. 测试不同k值
    k_values = [1.5, 2.0, 2.5, 3.0, 3.5]
    
    logger.info("\n" + "="*60)
    logger.info("开始ATR倍数k值优化（不含移动止盈）")
    logger.info("="*60)
    
    results_df = optimize_atr_k(klines_test, predictions, k_values, enable_trailing_stop=False)
    
    # 7. 计算综合得分
    results_df['composite_score'] = calculate_综合得分(results_df)
    
    # 8. 输出结果表格
    logger.info("\n" + "="*60)
    logger.info("📊 优化结果汇总")
    logger.info("="*60)
    
    # 格式化输出
    display_df = results_df.copy()
    display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x:.2f}%")
    display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.2f}%")
    display_df['avg_win'] = display_df['avg_win'].apply(lambda x: f"{x:.2f}%")
    display_df['avg_loss'] = display_df['avg_loss'].apply(lambda x: f"{x:.2f}%")
    display_df['profit_loss_ratio'] = display_df['profit_loss_ratio'].apply(lambda x: f"{x:.2f}")
    display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.2f}%")
    display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
    display_df['atr_stop_trigger_rate'] = display_df['atr_stop_trigger_rate'].apply(lambda x: f"{x:.2f}%")
    display_df['composite_score'] = display_df['composite_score'].apply(lambda x: f"{x:.2f}")
    
    print("\n" + display_df.to_string(index=False))
    
    # 9. 找出最优k值
    best_idx = results_df['composite_score'].idxmax()
    best_k = results_df.loc[best_idx, 'k']
    best_score = results_df.loc[best_idx, 'composite_score']
    
    logger.info(f"\n🏆 最优ATR倍数: k = {best_k}")
    logger.info(f"📊 综合得分: {best_score:.2f}")
    logger.info(f"💰 总收益率: {results_df.loc[best_idx, 'total_return']:.2f}%")
    logger.info(f"🎯 胜率: {results_df.loc[best_idx, 'win_rate']:.2f}%")
    logger.info(f"📈 盈亏比: {results_df.loc[best_idx, 'profit_loss_ratio']:.2f}")
    logger.info(f"📉 最大回撤: {results_df.loc[best_idx, 'max_drawdown']:.2f}%")
    logger.info(f"⚡ 夏普比率: {results_df.loc[best_idx, 'sharpe_ratio']:.2f}")
    
    # 10. 保存结果
    output_dir = Path('backtest')
    output_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(output_dir / 'atr_k_optimization_results.csv', index=False)
    logger.info(f"\n✅ 优化结果已保存: {output_dir / 'atr_k_optimization_results.csv'}")
    
    # 11. 绘制图表
    plot_optimization_results(results_df, output_dir)
    
    logger.info("\n✅ ATR倍数k值优化完成！")
    
    # 12. 给出建议
    logger.info("\n" + "="*60)
    logger.info("💡 优化建议")
    logger.info("="*60)
    
    if best_k == 1.5:
        logger.info("⚠️  最优k=1.5偏小，止损较紧，适合低波动市场")
        logger.info("   建议: 考虑启用自适应ATR倍数，根据波动率动态调整")
    elif best_k == 2.0:
        logger.info("✅ 最优k=2.0适中，平衡了止损保护和利润空间")
        logger.info("   建议: 这是经典ATR止损倍数，可以直接应用")
    elif best_k >= 3.0:
        logger.info("⚠️  最优k>=3.0较大，止损较松，可能在趋势市表现更好")
        logger.info("   建议: 警惕单笔大亏损，建议配合移动止盈使用")
    
    # 对比k=2.0的表现
    if best_k != 2.0:
        k2_idx = results_df[results_df['k'] == 2.0].index[0]
        return_diff = results_df.loc[best_idx, 'total_return'] - results_df.loc[k2_idx, 'total_return']
        logger.info(f"\n📊 相比k=2.0（基准）:")
        logger.info(f"   收益提升: {return_diff:+.2f}%")
        logger.info(f"   胜率变化: {results_df.loc[best_idx, 'win_rate'] - results_df.loc[k2_idx, 'win_rate']:+.2f}%")


if __name__ == '__main__':
    main()
