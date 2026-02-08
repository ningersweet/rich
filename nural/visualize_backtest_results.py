#!/usr/bin/env python3
"""
可视化回测结果：K线图、买卖点、收益曲线
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import mplfinance as mpf
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载回测数据和K线数据"""
    # 加载交易记录
    trades_files = list(Path("backtest").glob("backtest_results_rr_strategy_*.csv"))
    if not trades_files:
        print("错误：在backtest目录下找不到交易记录文件")
        return None, None
    
    latest_file = max(trades_files, key=lambda p: p.stat().st_mtime)
    print(f"读取交易记录: {latest_file}")
    trades_df = pd.read_csv(latest_file)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    # 加载K线数据
    from btc_quant.config import load_config
    from btc_quant.data import load_klines
    
    cfg = load_config()
    klines = load_klines(cfg)
    
    # 只保留回测期间的K线
    backtest_start = pd.Timestamp(cfg.raw['history_data']['backtest_start'])
    klines_backtest = klines[klines['open_time'] >= backtest_start].copy()
    
    return trades_df, klines_backtest


def plot_kline_with_trades(trades_df, klines_df, max_candles=500):
    """绘制K线图和买卖点"""
    print("\n生成K线图和买卖点...")
    
    # 只取前max_candles根K线（避免图太密集）
    klines_plot = klines_df.head(max_candles).copy()
    klines_plot.set_index('open_time', inplace=True)
    
    # 筛选该时间段内的交易
    time_range = (klines_plot.index.min(), klines_plot.index.max())
    trades_in_range = trades_df[
        (trades_df['entry_time'] >= time_range[0]) & 
        (trades_df['entry_time'] <= time_range[1])
    ]
    
    print(f"时间范围: {time_range[0]} ~ {time_range[1]}")
    print(f"该时段交易数: {len(trades_in_range)}")
    
    # 准备mplfinance数据格式
    klines_mpf = klines_plot[['open', 'high', 'low', 'close', 'volume']].copy()
    klines_mpf.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # 绘制K线图（不使用addplot）
    fig, axes = mpf.plot(
        klines_mpf,
        type='candle',
        style='charles',
        title=f'BTC/USDT 回测 K线图 (前{max_candles}根)',
        ylabel='价格 (USDT)',
        volume=True,
        figsize=(16, 10),
        returnfig=True
    )
    
    # 手动添加买卖点
    ax = axes[0]
    
    for _, trade in trades_in_range.iterrows():
        # 找到最接近的K线索引
        entry_idx = klines_mpf.index.get_indexer([trade['entry_time']], method='nearest')[0]
        exit_idx = klines_mpf.index.get_indexer([trade['exit_time']], method='nearest')[0]
        
        if entry_idx >= 0 and entry_idx < len(klines_mpf):
            entry_time = klines_mpf.index[entry_idx]
            entry_price = trade['entry_price']
            
            if trade['side'] == 'long':
                # 做多入场：绿色向上箭头
                ax.scatter(entry_time, entry_price * 0.998, marker='^', color='green', 
                          s=150, zorder=5, alpha=0.8, edgecolors='darkgreen', linewidths=1.5)
            else:
                # 做空入场：红色向下箭头
                ax.scatter(entry_time, entry_price * 1.002, marker='v', color='red', 
                          s=150, zorder=5, alpha=0.8, edgecolors='darkred', linewidths=1.5)
        
        if exit_idx >= 0 and exit_idx < len(klines_mpf):
            exit_time = klines_mpf.index[exit_idx]
            exit_price = trade['exit_price']
            
            # 出场点：圆圈
            if trade['side'] == 'long':
                color = 'lime' if trade['pnl'] > 0 else 'orange'
            else:
                color = 'cyan' if trade['pnl'] > 0 else 'yellow'
            
            ax.scatter(exit_time, exit_price, marker='o', color=color, 
                      s=80, zorder=5, alpha=0.7, edgecolors='black', linewidths=1)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', 
               markersize=12, label='做多入场', markeredgecolor='darkgreen', markeredgewidth=1.5),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', 
               markersize=12, label='做空入场', markeredgecolor='darkred', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', 
               markersize=10, label='盈利出场', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
               markersize=10, label='亏损出场', markeredgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('backtest/kline_with_trades.png', dpi=150, bbox_inches='tight')
    print("K线图已保存: backtest/kline_with_trades.png")
    plt.close()


def plot_equity_curve(trades_df, initial_balance=1000.0):
    """绘制收益曲线"""
    print("\n生成收益曲线...")
    
    # 计算权益曲线
    trades_sorted = trades_df.sort_values('exit_time')
    equity_curve = [initial_balance]
    equity_times = [trades_sorted.iloc[0]['entry_time']]
    
    current_equity = initial_balance
    for _, trade in trades_sorted.iterrows():
        current_equity += trade['pnl']
        equity_curve.append(current_equity)
        equity_times.append(trade['exit_time'])
    
    # 计算累计收益率
    returns_pct = [(eq / initial_balance - 1) * 100 for eq in equity_curve]
    
    # 计算最大回撤
    equity_series = pd.Series(equity_curve)
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax * 100
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 子图1：权益曲线
    ax1 = axes[0]
    ax1.plot(equity_times, equity_curve, linewidth=2, color='blue', label='账户权益')
    ax1.axhline(y=initial_balance, color='gray', linestyle='--', alpha=0.5, label='初始资金')
    ax1.set_ylabel('权益 (USDT)', fontsize=12)
    ax1.set_title('盈亏比驱动策略 - 回测收益曲线', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # 子图2：累计收益率
    ax2 = axes[1]
    ax2.fill_between(equity_times, 0, returns_pct, alpha=0.3, color='green')
    ax2.plot(equity_times, returns_pct, linewidth=2, color='green', label='累计收益率')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('收益率 (%)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # 子图3：回撤
    ax3 = axes[2]
    ax3.fill_between(equity_times, 0, drawdown, alpha=0.3, color='red')
    ax3.plot(equity_times, drawdown, linewidth=2, color='red', label='回撤')
    ax3.set_xlabel('日期', fontsize=12)
    ax3.set_ylabel('回撤 (%)', fontsize=12)
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # 旋转日期标签
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('backtest/equity_curve.png', dpi=150, bbox_inches='tight')
    print("收益曲线已保存: backtest/equity_curve.png")
    plt.close()


def plot_trade_statistics(trades_df):
    """绘制交易统计图表"""
    print("\n生成交易统计图表...")
    
    # 计算统计数据
    trades_df['return_pct'] = (trades_df['pnl'] / (trades_df['entry_price'] * trades_df['quantity'])) * 100
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1：盈亏分布
    ax1 = axes[0, 0]
    wins = trades_df[trades_df['pnl'] > 0]['return_pct']
    losses = trades_df[trades_df['pnl'] <= 0]['return_pct']
    ax1.hist(wins, bins=50, alpha=0.6, color='green', label=f'盈利 ({len(wins)}笔)')
    ax1.hist(losses, bins=50, alpha=0.6, color='red', label=f'亏损 ({len(losses)}笔)')
    ax1.set_xlabel('收益率 (%)')
    ax1.set_ylabel('交易次数')
    ax1.set_title('盈亏分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：做多vs做空表现
    ax2 = axes[0, 1]
    long_trades = trades_df[trades_df['side'] == 'long']
    short_trades = trades_df[trades_df['side'] == 'short']
    
    long_win_rate = (long_trades['pnl'] > 0).sum() / len(long_trades) * 100
    short_win_rate = (short_trades['pnl'] > 0).sum() / len(short_trades) * 100
    
    categories = ['做多', '做空']
    win_rates = [long_win_rate, short_win_rate]
    colors = ['green', 'red']
    
    bars = ax2.bar(categories, win_rates, color=colors, alpha=0.6)
    ax2.set_ylabel('胜率 (%)')
    ax2.set_title('做多 vs 做空胜率')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # 子图3：累计PnL走势
    ax3 = axes[1, 0]
    trades_sorted = trades_df.sort_values('exit_time')
    cumulative_pnl = trades_sorted['pnl'].cumsum()
    ax3.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='blue')
    ax3.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, alpha=0.3, color='blue')
    ax3.set_xlabel('交易序号')
    ax3.set_ylabel('累计PnL (USDT)')
    ax3.set_title('累计盈亏走势')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 子图4：每日交易次数
    ax4 = axes[1, 1]
    trades_df['date'] = pd.to_datetime(trades_df['exit_time']).dt.date
    daily_trades = trades_df.groupby('date').size()
    ax4.bar(range(len(daily_trades)), daily_trades.values, color='steelblue', alpha=0.7)
    ax4.set_xlabel('日期')
    ax4.set_ylabel('交易次数')
    ax4.set_title('每日交易频率')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('backtest/trade_statistics.png', dpi=150, bbox_inches='tight')
    print("交易统计图已保存: backtest/trade_statistics.png")
    plt.close()


def main():
    print("=" * 80)
    print("盈亏比驱动策略 - 回测结果可视化")
    print("=" * 80)
    
    # 加载数据
    trades_df, klines_df = load_data()
    if trades_df is None or klines_df is None:
        return
    
    print(f"\n总交易数: {len(trades_df)}")
    print(f"总K线数: {len(klines_df)}")
    
    # 生成图表
    plot_kline_with_trades(trades_df, klines_df, max_candles=500)
    plot_equity_curve(trades_df)
    plot_trade_statistics(trades_df)
    
    print("\n" + "=" * 80)
    print("所有图表生成完成！")
    print("=" * 80)
    print("\n生成的文件:")
    print("  1. kline_with_trades.png - K线图与买卖点")
    print("  2. equity_curve.png - 收益曲线与回撤")
    print("  3. trade_statistics.png - 交易统计分析")
    print()


if __name__ == "__main__":
    main()
