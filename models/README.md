# 模型目录说明

本目录包含多个算法模型用于量化交易实验和对比。

## 📁 目录结构

```
models/
├── baseline_lgbm/           # 基线LightGBM模型
│   ├── model_latest.pkl
│   ├── enhanced_ensemble.pkl
│   └── README.md
│
├── neural_hybrid/           # 神经网络混合模型
│   ├── neural_lstm_latest.pt
│   ├── hybrid_model_latest.pkl
│   └── README.md
│
└── risk_reward_strategy/    # ⭐ 盈亏比驱动策略（推荐）
    ├── risk_reward_model.txt
    ├── direction_model.txt
    ├── period_model.txt
    └── README.md
```

## 🎯 模型对比

| 模型 | 准确率 | 胜率 | 训练时间 | 推荐度 | 说明 |
|------|--------|------|----------|--------|------|
| **盈亏比驱动策略** | 64.3% | 78.02% ⭐ | ~5分钟 | ⭐⭐⭐⭐⭐ | 两阶段预测，风控优先 |
| 基线LightGBM | 待测试 | 待测试 | ~5分钟 | ⭐⭐⭐ | 简单快速，适合快速验证 |
| 神经网络混合 | 58.9% | 待测试 | ~30分钟 | ⭐⭐ | 需GPU优化，潜力大 |

## 🏆 当前最优方案

**推荐使用**: `risk_reward_strategy/`

**理由**:
1. ✅ 验证准确率最高 (64.3%)
2. ✅ 回测胜率最高 (78.02%)
3. ✅ 盈亏比优秀 (2.94)
4. ✅ 训练速度快
5. ✅ 策略逻辑清晰可控

## 🔬 实验建议

### 快速验证
```bash
# 使用盈亏比驱动策略
python train_and_backtest_rr_strategy.py
```

### 基线对比
```bash
# 训练基线LightGBM
# TODO: 创建对应训练脚本
```

### 深度学习实验
```bash
# 训练神经网络混合模型（需Docker）
docker compose -f docker-compose.train.yml up --build
```

## 📊 性能基准

### 回测数据
- **时间范围**: 2021-01-01 ~ 2024-12-31 (3年)
- **训练集**: 2021-01-01 ~ 2024-10-31
- **测试集**: 2024-11-01 ~ 2024-12-31 (2个月)
- **K线周期**: 15分钟
- **样本数**: 140,256条

### 评估指标
- **准确率**: 分类预测正确率
- **胜率**: 盈利交易占比
- **盈亏比**: 平均盈利/平均亏损
- **总收益率**: 资金曲线增长率
- **交易次数**: 总交易笔数

## 🚀 下一步优化方向

### 1. 数据层面
- [ ] 使用更长历史数据 (5年+)
- [ ] 添加多时间周期特征 (5分钟, 1小时)
- [ ] 整合市场情绪数据

### 2. 特征层面
- [ ] 深度特征工程
- [ ] 特征选择优化
- [ ] 自动特征生成

### 3. 模型层面
- [ ] Transformer模型实验
- [ ] 强化学习策略
- [ ] 模型集成优化

### 4. 策略层面
- [ ] 动态止损止盈
- [ ] 仓位管理优化
- [ ] 多策略组合

## 📝 训练新模型

### 添加新算法步骤

1. **创建模型目录**
```bash
mkdir -p models/新算法名称/
```

2. **实现训练脚本**
```python
# train_新算法.py
# 训练逻辑...
# 保存到 models/新算法名称/
```

3. **添加README说明**
```bash
# models/新算法名称/README.md
# 包含：算法说明、性能指标、使用方法
```

4. **运行对比实验**
```bash
python train_新算法.py
python compare_models.py  # 对比不同模型
```

## 🔧 模型管理

### 版本控制
- ✅ 所有模型文件已添加到 `.gitignore`
- ✅ 仅提交模型架构代码和配置
- ✅ 模型文件在服务器上训练和部署

### 备份策略
```bash
# 备份当前最优模型
cp -r models/risk_reward_strategy/ models/risk_reward_strategy.backup_$(date +%Y%m%d)
```

### 清理旧模型
```bash
# 清理测试模型
find models/ -name "*test*" -delete
```
