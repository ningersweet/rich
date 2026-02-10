#!/usr/bin/env python3
"""检查新增特征是否正常工作"""

import sys
sys.path.insert(0, '.')
from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels

# 加载配置和数据
cfg = load_config('config.yaml')
klines = load_klines(cfg)

# 构建特征（只取最近1000条测试）
test_klines = klines.tail(1000)
feature_data = build_features_and_labels(cfg, test_klines)

print('=' * 80)
print('特征工程更新统计')
print('=' * 80)
print(f'特征总数: {len(feature_data.features.columns)}')
print(f'数据样本数: {len(feature_data.features)}')
print()

print('新增特征列表（DeepSeek建议）:')
print('-' * 80)

new_features = [
    # 订单簿微观结构特征
    'buy_sell_pressure', 
    'volume_weighted_pressure', 
    'large_order_ratio',
    'cumulative_pressure_5', 
    'cumulative_pressure_10', 
    'volume_imbalance',
    'price_vwap_diff',
    # 条件交互特征
    'oversold_volume_effect', 
    'overbought_volume_effect',
    'trend_momentum_interaction', 
    'breakout_in_volatility', 
    'squeeze_breakout_potential', 
    'price_volume_divergence_strength'
]

for i, feat in enumerate(new_features, 1):
    if feat in feature_data.features.columns:
        print(f'{i:2d}. ✅ {feat}')
    else:
        print(f'{i:2d}. ❌ {feat} (未找到)')

print()
print('特征值统计（前5个新特征）:')
print('-' * 80)
for feat in new_features[:5]:
    if feat in feature_data.features.columns:
        data = feature_data.features[feat]
        print(f'\n{feat}:')
        print(f'  均值: {data.mean():.6f}')
        print(f'  标准差: {data.std():.6f}')
        print(f'  范围: [{data.min():.6f}, {data.max():.6f}]')
        print(f'  NaN数量: {data.isna().sum()}')

print('\n' + '=' * 80)
print('✅ 特征工程更新完成！')
print(f'📊 从原有 ~50 个特征增加到 {len(feature_data.features.columns)} 个特征')
print('🚀 准备进行回测验证...')
print('=' * 80)
