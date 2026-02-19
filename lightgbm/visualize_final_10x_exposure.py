#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【可视化报告】10倍动态敞口 + 多层风控 - 完整图表分析

用途：
    生成回测结果的可视化报告，包含5个核心图表

功能：
    1. 读取回测结果CSV文件
    2. 生成5个分析图表：
       - 资金曲线 + 回撤
       - 敞口分布
       - 交易盈亏分布
       - 月度统计
       - 胜率/盈亏比趋势
    3. 保存为PNG图片

使用方法：
    python visualize_final_10x_exposure.py
    
输入文件：
    backtest/final_6x_fixed_capital_results_*.csv
    
输出文件：
    backtest_visualization_*.png (5张图表)

最后更新：2026-02-20
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 使用matplotlib内置样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_final_backtest():
    """运行最终回测并保存交易数据"""
    from btc_quant.config import load_config
    from btc_quant.data import load_klines
    from btc_quant.features import build_features_and_labels
    from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy
    from train_dynamic_exposure_with_advanced_risk import advanced_risk_backtest
    
    print("="*80)
    print("10倍动态敞口 + 多层风控 - 最终回测")
    print("="*80)
    
    cfg = load_config(Path('config.yaml'))
    print("\n加载数据...")
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
    
    print("生成预测...")
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
    
    print(f"数据对齐完成，样本数: {min_len}\n")
    print("执行回测...")
    
    result = advanced_risk_backtest(
        klines=klines,
        predictions=predictions,
        initial_balance=1000.0,
        max_exposure=10.0,
        stop_loss_pct=-0.03,
        max_daily_loss_pct=-0.20,
        max_drawdown_pause=0.10,  # 回撤暂停10%（已从6%调整）
        use_trailing_stop=True
    )
    
    # 保存交易数据
    trades_df = result['trades']
    output_path = 'backtest/final_10x_exposure_trades.csv'
    trades_df.to_csv(output_path, index=False)
    print(f"\n交易数据已保存: {output_path}")
    
    return trades_df, result


def plot_equity_curve(trades_df, output_path='backtest/final_10x_equity_curve.png'):
    """绘制权益曲线"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # 1. 权益曲线
    ax1 = axes[0]
    ax1.plot(trades_df.index, trades_df['equity_after'], 
             linewidth=2, color='#2E86C1', label='账户权益')
    ax1.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, label='初始本金')
    ax1.fill_between(trades_df.index, 1000, trades_df['equity_after'], 
                      alpha=0.2, color='#2E86C1')
    ax1.set_title('账户权益曲线（10倍动态敞口 + 多层风控）', fontsize=14, fontweight='bold')
    ax1.set_ylabel('权益 (USDT)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 累计收益率
    ax2 = axes[1]
    trades_df['cumulative_return'] = ((trades_df['equity_after'] - 1000) / 1000) * 100
    ax2.plot(trades_df.index, trades_df['cumulative_return'], 
             linewidth=2, color='#27AE60', label='累计收益率')
    ax2.fill_between(trades_df.index, 0, trades_df['cumulative_return'], 
                      alpha=0.2, color='#27AE60')
    ax2.set_title('累计收益率曲线', fontsize=14, fontweight='bold')
    ax2.set_ylabel('收益率 (%)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. 回撤曲线
    ax3 = axes[2]
    peak = trades_df['equity_after'].expanding().max()
    drawdown = (peak - trades_df['equity_after']) / peak * 100
    trades_df['drawdown'] = drawdown
    ax3.fill_between(trades_df.index, 0, -trades_df['drawdown'], 
                      alpha=0.3, color='#E74C3C', label='回撤')
    ax3.plot(trades_df.index, -trades_df['drawdown'], 
             linewidth=2, color='#C0392B')
    ax3.set_title('回撤曲线', fontsize=14, fontweight='bold')
    ax3.set_ylabel('回撤 (%)', fontsize=12)
    ax3.set_xlabel('交易序号', fontsize=12)
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"权益曲线已保存: {output_path}")
    plt.close()


def plot_risk_analysis(trades_df, output_path='backtest/final_10x_risk_analysis.png'):
    """风控分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. 敞口分布
    ax1 = axes[0, 0]
    ax1.hist(trades_df['exposure'], bins=30, color='#3498DB', alpha=0.7, edgecolor='black')
    ax1.axvline(x=trades_df['exposure'].mean(), color='red', linestyle='--', 
                label=f'平均: {trades_df["exposure"].mean():.2f}倍')
    ax1.set_title('敞口倍数分布', fontsize=12, fontweight='bold')
    ax1.set_xlabel('敞口倍数', fontsize=10)
    ax1.set_ylabel('交易次数', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 止损类型统计
    ax2 = axes[0, 1]
    stop_loss_fixed = len(trades_df[trades_df['stop_loss_hit'] == True])
    stop_loss_trailing = len(trades_df[trades_df['trailing_stop_hit'] == True])
    normal_exit = len(trades_df[(trades_df['stop_loss_hit'] == False) & 
                                 (trades_df['trailing_stop_hit'] == False)])
    
    labels = ['固定止损', '追踪止损', '正常退出']
    sizes = [stop_loss_fixed, stop_loss_trailing, normal_exit]
    colors = ['#E74C3C', '#F39C12', '#2ECC71']
    explode = (0.1, 0.1, 0)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax2.set_title('平仓类型分布', fontsize=12, fontweight='bold')
    
    # 3. 单笔盈亏分布
    ax3 = axes[1, 0]
    wins = trades_df[trades_df['pnl'] > 0]['pnl']
    losses = trades_df[trades_df['pnl'] <= 0]['pnl']
    
    ax3.hist([wins, losses], bins=30, label=['盈利', '亏损'], 
             color=['#2ECC71', '#E74C3C'], alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax3.set_title('单笔盈亏分布', fontsize=12, fontweight='bold')
    ax3.set_xlabel('盈亏 (USDT)', fontsize=10)
    ax3.set_ylabel('交易次数', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 连续亏损演变
    ax4 = axes[1, 1]
    ax4.plot(trades_df.index, trades_df['consecutive_losses'], 
             linewidth=2, color='#E74C3C', marker='o', markersize=3)
    ax4.fill_between(trades_df.index, 0, trades_df['consecutive_losses'], 
                      alpha=0.3, color='#E74C3C')
    ax4.axhline(y=2, color='orange', linestyle='--', label='开始降敞口阈值')
    ax4.set_title('连续亏损次数演变', fontsize=12, fontweight='bold')
    ax4.set_xlabel('交易序号', fontsize=10)
    ax4.set_ylabel('连续亏损次数', fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"风控分析图已保存: {output_path}")
    plt.close()


def generate_summary_report(trades_df, result, output_path='backtest/final_10x_summary.txt'):
    """生成文字总结报告"""
    
    initial_balance = 1000.0
    final_equity = trades_df['equity_after'].iloc[-1]
    
    # 时间统计
    start_time = pd.to_datetime(trades_df['entry_time'].iloc[0])
    end_time = pd.to_datetime(trades_df['exit_time'].iloc[-1])
    days = (end_time - start_time).days
    
    # 止损统计
    stop_loss_count = len(trades_df[trades_df['stop_loss_hit'] == True])
    trailing_stop_count = len(trades_df[trades_df['trailing_stop_hit'] == True])
    
    # 月度统计
    trades_df['entry_month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
    monthly_pnl = trades_df.groupby('entry_month')['pnl'].sum()
    monthly_trades = trades_df.groupby('entry_month').size()
    
    report = f"""
{'='*80}
10倍动态敞口 + 多层风控策略 - 完整回测报告
{'='*80}

【配置信息】
  策略类型: 动态敞口管理（基于信号质量自适应）
  最大敞口: 10倍（1000%）
  固定止损: -3%
  追踪止损: 盈利>1%后保护（价格距最高点下降2%）
  每日最大亏损: -20%
  回撤暂停阈值: 10%
  连续亏损惩罚: 开启

【时间统计】
  回测开始: {start_time.strftime('%Y-%m-%d')}
  回测结束: {end_time.strftime('%Y-%m-%d')}
  回测天数: {days} 天 ({days/30:.1f} 个月)

【收益统计】
  初始本金: {initial_balance:,.2f} USDT
  最终权益: {final_equity:,.2f} USDT
  总收益: {final_equity - initial_balance:,.2f} USDT
  总收益率: {result['total_return']:.2f}%
  年化收益: {result['total_return'] / (days/365):.2f}%

【交易统计】
  总交易数: {result['total_trades']:,} 笔
  盈利交易: {int(result['total_trades'] * result['win_rate'] / 100)} 笔
  亏损交易: {int(result['total_trades'] * (1 - result['win_rate'] / 100))} 笔
  胜率: {result['win_rate']:.2f}%
  盈亏比: {result['profit_loss_ratio']:.2f}
  平均持仓: {trades_df['bars_held'].mean():.1f} 根K线 ({trades_df['bars_held'].mean()*15/60:.1f} 小时)

【风险统计】
  最大回撤: {result['max_drawdown']:.2f}%
  收益/回撤比: {result['total_return']/result['max_drawdown']:.2f}
  最大连续亏损: {result['max_consecutive_losses']} 笔
  平均敞口: {result['avg_exposure']:.2f}倍

【止损分析】
  固定止损触发: {stop_loss_count} 次 ({stop_loss_count/result['total_trades']*100:.2f}%)
  追踪止损触发: {trailing_stop_count} 次 ({trailing_stop_count/result['total_trades']*100:.2f}%)
  正常退出: {result['total_trades'] - stop_loss_count - trailing_stop_count} 次 ({(1-stop_loss_count/result['total_trades']-trailing_stop_count/result['total_trades'])*100:.2f}%)

【月度表现】
  最佳月份: {monthly_pnl.idxmax()} (+{monthly_pnl.max():,.2f} USDT, {monthly_trades[monthly_pnl.idxmax()]} 笔)
  最差月份: {monthly_pnl.idxmin()} ({monthly_pnl.min():,.2f} USDT, {monthly_trades[monthly_pnl.idxmin()]} 笔)
  月均收益: {monthly_pnl.mean():,.2f} USDT
  盈利月份: {len(monthly_pnl[monthly_pnl > 0])} / {len(monthly_pnl)}

【最佳交易TOP5】
{trades_df.nlargest(5, 'pnl')[['entry_time', 'side', 'exposure', 'pnl', 'price_change_pct']].to_string()}

【最差交易TOP5】
{trades_df.nsmallest(5, 'pnl')[['entry_time', 'side', 'exposure', 'pnl', 'price_change_pct']].to_string()}

{'='*80}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"文字报告已保存: {output_path}")
    print(report)


def main():
    """主函数"""
    
    # 1. 运行回测
    trades_df, result = run_final_backtest()
    
    # 2. 生成可视化
    print("\n生成可视化图表...")
    plot_equity_curve(trades_df)
    plot_risk_analysis(trades_df)
    
    # 3. 生成文字报告
    print("\n生成文字报告...")
    generate_summary_report(trades_df, result)
    
    print(f"\n{'='*80}")
    print("所有报告已生成完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
