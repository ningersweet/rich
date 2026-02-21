import pandas as pd

df = pd.read_csv('backtest_results/final_2024_dynamic_20260221_102849.csv')

print('='*100)
print('敞口策略深度分析')
print('='*100)

# 按敞口区间分组
bins = [0, 3, 5, 7, 9, 11]
labels = ['低敞口(1-3倍)', '中敞口(3-5倍)', '高敞口(5-7倍)', '超高(7-9倍)', '满仓(9-10倍)']
df['exposure_range'] = pd.cut(df['exposure'], bins=bins, labels=labels)

print('\n【敞口分级表现】')
print(f"{'敞口级别':<15} {'交易数':>6} {'胜率':>7} {'平均盈亏':>15} {'总盈亏':>18}")
print('-'*100)

for label in labels:
    group = df[df['exposure_range'] == label]
    if len(group) > 0:
        wr = (group['pnl'] > 0).sum() / len(group) * 100
        avg_pnl = group['pnl'].mean()
        total_pnl = group['pnl'].sum()
        print(f"{label:<15} {len(group):>6} {wr:>6.1f}% {avg_pnl:>15,.0f} {total_pnl:>18,.0f}")

print('\n【高敞口问题诊断】')
high_exp = df[df['exposure'] > 8]
print(f'\n高敞口(>8倍)交易统计:')
print(f'  总数: {len(high_exp)}笔 ({len(high_exp)/len(df)*100:.1f}%)')
print(f'  胜率: {(high_exp["pnl"]>0).sum()/len(high_exp)*100:.1f}%')
print(f'  平均盈亏: {high_exp["pnl"].mean():,.0f} USDT')

high_exp_win = high_exp[high_exp['pnl'] > 0]
high_exp_loss = high_exp[high_exp['pnl'] <= 0]
print(f'\n高敞口盈利交易:')
print(f'  数量: {len(high_exp_win)}笔')
print(f'  平均盈利: {high_exp_win["pnl"].mean():,.0f} USDT')
print(f'\n高敞口亏损交易:')
print(f'  数量: {len(high_exp_loss)}笔')
print(f'  平均亏损: {high_exp_loss["pnl"].mean():,.0f} USDT')

print('\n【关键发现】')
print(f'❌ 高敞口胜率{(high_exp["pnl"]>0).sum()/len(high_exp)*100:.1f}%低于整体胜率55.5%')
print(f'⚠️  说明模型在高信心交易上表现不佳')
print(f'💡 优化方向: 降低敞口上限或提高高敞口阈值')
