#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('backtest_results/final_2024_dynamic_20260221_102849.csv')
df['exit_time'] = pd.to_datetime(df['exit_time'])
df['month'] = df['exit_time'].dt.to_period('M')
df['year'] = df['exit_time'].dt.year

print('='*120)
print('月度收益统计')
print('='*120)
print(f"{'月份':<10} {'交易数':>6} {'盈利':>4} {'亏损':>4} {'胜率':>7} {'月收益率':>12} {'月初权益':>18} {'月末权益':>20}")
print('-'*120)

for m in sorted(df['month'].unique()):
    month_data = df[df['month']==m]
    start_eq = month_data.iloc[0]['equity_after'] - month_data.iloc[0]['pnl']
    end_eq = month_data.iloc[-1]['equity_after']
    ret = (end_eq/start_eq-1)*100
    wins = (month_data['pnl']>0).sum()
    losses = len(month_data) - wins
    wr = wins/len(month_data)*100
    
    print(f"{str(m):<10} {len(month_data):>6} {wins:>4} {losses:>4} {wr:>6.1f}% {ret:>11.2f}% {start_eq:>18,.0f} {end_eq:>20,.0f}")

print('='*120)
print('\n年度汇总')
print('='*120)

for y in sorted(df['year'].unique()):
    year_data = df[df['year']==y]
    start_eq = year_data.iloc[0]['equity_after'] - year_data.iloc[0]['pnl']
    end_eq = year_data.iloc[-1]['equity_after']
    ret = (end_eq/start_eq-1)*100
    wins = (year_data['pnl']>0).sum()
    
    print(f"\n{y}年:")
    print(f"  交易笔数: {len(year_data)}")
    print(f"  胜率: {wins/len(year_data)*100:.1f}%")
    print(f"  年收益率: {ret:,.2f}%")
    print(f"  年初权益: {start_eq:,.0f} USDT")
    print(f"  年末权益: {end_eq:,.0f} USDT")
    print(f"  年度增长: {(end_eq/start_eq):.1f}倍")
