# 盈亏比驱动策略实验 ⭐推荐

## 快速开始

```bash
# 1. 训练和回测
cd experiments/risk_reward_strategy
python src/train_and_backtest_rr_strategy.py

# 2. 查看结果
ls results/*.csv
```

## 模型说明

两阶段预测模型，基于盈亏比驱动的量化交易策略。

## 目录结构

```
risk_reward_strategy/
├── src/                           # 源代码
│   └── train_and_backtest_rr_strategy.py
├── models/                       # 训练好的模型
│   ├── risk_reward_model.txt
│   ├── direction_model.txt
│   └── period_model.txt
├── data/                         # 符号链接到项目根目录data/
├── results/                      # 回测结果
└── README.md
```

### 两阶段架构

```
【阶段1：盈亏比预测】
输入: 54维特征
  ↓
[LightGBM回归]
  ↓
预测盈亏比(Risk/Reward Ratio)
  ↓
过滤: 仅保留 RR > 2.0 的样本
  ↓
【阶段2：方向+周期预测】
  ↓
┌────────────────┬────────────────┐
│   方向预测      │   周期预测      │
│ (分类模型)     │  (回归模型)     │
└────────────────┴────────────────┘
  ↓              ↓
做多/观望/做空   持有周期(K线数)
  ↓
最终交易信号
```

### 性能指标 ⭐

- **验证准确率**: 64.3%
- **回测胜率**: 78.02% 🎯
- **盈亏比**: 2.94
- **总收益率**: 1.62% (2个月测试期)
- **交易次数**: 182笔
- **算法**: LightGBM (两阶段)
- **训练时间**: ~5分钟

### 核心优势

- ✅ **高胜率**: 78.02% (远超目标45%)
- ✅ **高盈亏比**: 2.94 (平均盈利是亏损的3倍)
- ✅ **风控先行**: 先预测盈亏比再决策方向
- ✅ **动态持仓**: 根据市场预测持有周期
- ✅ **样本质量**: 仅交易高RR机会

### 策略逻辑

1. **预测盈亏比**: 评估潜在交易的风险收益比
2. **过滤低RR**: 仅保留RR>2.0的高质量机会
3. **预测方向**: 决定做多、做空还是观望
4. **预测周期**: 确定最优持有时间
5. **执行交易**: 按预测方向和周期开仓平仓

### 特征重要性 Top 10

#### 盈亏比模型
1. `ma_99` - 99日均线
2. `bb_upper` - 布林带上轨
3. `ema_25` - 25日指数均线
4. `dist_to_high` - 距离最高点
5. `ma_25` - 25日均线

#### 方向模型
1. `ma_99_slope_5` - 99日均线斜率
2. `dist_to_low` - 距离最低点
3. `dist_to_high` - 距离最高点
4. `close_ma_99_diff` - 收盘价与均线差
5. `atr_pct` - ATR百分比

#### 周期模型
1. `atr_pct` - ATR百分比（波动率）
2. `atr_14` - 14日ATR
3. `close_ma_99_diff` - 价格均线差
4. `volatility_regime` - 波动率状态
5. `ma_99` - 99日均线

### 使用方法

```python
from btc_quant.risk_reward_model import RiskRewardModel

# 1. 加载模型
model = RiskRewardModel.load('models/risk_reward_strategy/')

# 2. 预测交易信号
signals = model.predict(features)

# signals包含:
# - direction: 1(做多), 0(观望), -1(做空)
# - predicted_rr: 预测的盈亏比
# - holding_period: 建议持有周期
# - should_trade: 是否应该交易(RR>2.0)
```

### 回测结果

```
总收益率: 1.62%
胜率: 78.02%
总交易次数: 182
盈利交易: 142
亏损交易: 40
平均盈利: 0.13
平均亏损: 0.04
盈亏比: 2.94
基准收益(买入持有): 31.29%
超额收益: -29.68%
```

### 优化方向

1. ✅ 类别权重已优化 (做空0.6, 做多/观望1.5)
2. ✅ LightGBM正则化已增强
3. 🔄 可尝试更长的历史数据训练
4. 🔄 可调整最小盈亏比阈值(当前2.0)
5. 🔄 可优化特征选择(当前54维)
