#!/usr/bin/env python3
"""验证新特征并启动回测"""
import sys
sys.path.insert(0, '.')

from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels

print("="*80)
print("验证新特征是否生效")
print("="*80)

cfg = load_config('config.yaml')
klines = load_klines(cfg)
test_klines = klines.tail(1000)
feature_data = build_features_and_labels(cfg, test_klines)

print(f"\n特征总数: {len(feature_data.features.columns)}")
print(f"样本数: {len(feature_data.features)}")

new_features = [
    'buy_sell_pressure', 'volume_weighted_pressure', 'large_order_ratio',
    'cumulative_pressure_5', 'cumulative_pressure_10', 'volume_imbalance',
    'price_vwap_diff', 'oversold_volume_effect', 'overbought_volume_effect',
    'trend_momentum_interaction', 'breakout_in_volatility', 
    'squeeze_breakout_potential', 'price_volume_divergence_strength'
]

print("\n新增特征检查:")
found = 0
for feat in new_features:
    status = "✅" if feat in feature_data.features.columns else "❌"
    print(f"  {status} {feat}")
    if feat in feature_data.features.columns:
        found += 1

print(f"\n结果: {found}/{len(new_features)} 个新特征已添加")

if found == len(new_features):
    print("\n" + "="*80)
    print("✅ 所有新特征验证通过！")
    print("📊 特征数量变化: 54 → " + str(len(feature_data.features.columns)))
    print("="*80)
    print("\n现在开始完整回测，预计需要 5-10 分钟...")
    print("="*80)
    
    # 启动完整回测
    import subprocess
    result = subprocess.run(
        ["python3", "train_and_backtest_rr_strategy.py"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("错误信息:", result.stderr)
else:
    print("\n❌ 部分新特征缺失，请检查 features.py")
