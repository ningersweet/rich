import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels
from btc_quant.risk_reward_model import TwoStageRiskRewardStrategy
from pathlib import Path

# 加载模型和数据
cfg = load_config()
klines = load_klines(cfg)
klines['close_time'] = pd.to_datetime(klines['close_time'])

start = pd.Timestamp('2025-01-01T00:00:00Z')
end = pd.Timestamp('2026-02-20T23:59:59Z')
klines_bt = klines[(klines['close_time']>=start)&(klines['close_time']<=end)].reset_index(drop=True)

features = build_features_and_labels(cfg, klines_bt).features.reset_index(drop=True)
min_len = min(len(features), len(klines_bt))
features = features.iloc[:min_len]
klines_bt = klines_bt.iloc[:min_len].reset_index(drop=True)

# 加载模型和Top30特征
strategy = TwoStageRiskRewardStrategy()
strategy.load(Path('models/final_2025_dynamic'))

# 读取Top30特征列表
with open('models/final_2025_dynamic/top30_features.txt', 'r') as f:
    top30 = [line.strip() for line in f if line.strip()]

# 预测
X_bt = features[top30]
predictions = strategy.predict(X_bt, rr_threshold=2.5, prob_threshold=0.70)

# 模拟统计加仓机会
position = None
pyramid_opportunities = 0
total_trades = 0
last_pyramid_idx = -999

for i in range(len(predictions)):
    price = klines_bt.iloc[i]['close']
    
    if position is not None:
        if position['side'] == 1:
            pnl_pct = (price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - price) / position['entry_price']
        
        # 加仓机会检测
        if (pnl_pct > 0.01 and 
            predictions.iloc[i]['should_trade'] and
            predictions.iloc[i]['direction'] == position['side'] and
            predictions.iloc[i]['predicted_rr'] > 3.0 and
            predictions.iloc[i]['direction_prob'] > 0.75 and
            i - last_pyramid_idx >= 5):  # 距上次加仓至少5根K线
            pyramid_opportunities += 1
            last_pyramid_idx = i
        
        if i - position['entry_idx'] >= 10:
            position = None
            total_trades += 1
            last_pyramid_idx = -999
    else:
        if predictions.iloc[i]['should_trade']:
            position = {'side': predictions.iloc[i]['direction'], 'entry_price': price, 'entry_idx': i}

print(f"回测期间统计（2025-2026）:")
print(f"总交易数: {total_trades}")
print(f"加仓机会: {pyramid_opportunities}")
if total_trades > 0:
    print(f"平均每笔交易加仓机会: {pyramid_opportunities/total_trades:.2f}次")
    print(f"\n💡 潜在收益提升:")
    print(f"  如果每次加仓用等比例敞口，可能增加 {pyramid_opportunities} 次额外交易")
    print(f"  假设平均敞口5倍，总敞口提升: {pyramid_opportunities * 5} 倍·次")
