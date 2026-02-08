# 实验目录说明

本目录包含多个算法模型的实验，每个模型都有独立的代码、数据和结果。

## 📁 目录结构

```
experiments/
├── risk_reward_strategy/   # ⭐ 盈亏比驱动策略（推荐）
│   ├── src/                 # 训练和回测脚本
│   ├── models/              # 三阶段模型文件
│   ├── data/                # 符号链接 -> ../../data/
│   ├── results/             # 回测结果 CSV
│   └── README.md
│
├── neural_hybrid/          # 神经网络混合模型
│   ├── src/                 # 训练、测试、验证脚本
│   ├── models/              # LSTM + LightGBM模型
│   ├── data/                # 符号链接 -> ../../data/
│   ├── results/             # 训练日志和结果
│   ├── docker-compose.train.yml
│   └── README.md
│
├── baseline_lgbm/          # 基线LightGBM模型
│   ├── src/                 # 训练脚本（待实现）
│   ├── models/              # 模型文件
│   ├── data/                # 符号链接 -> ../../data/
│   ├── results/
│   └── README.md
│
└── README.md               # 本文件
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

## 🚀 快速开始

### 1、盈亏比驱动策略（推荐）

```bash
cd experiments/risk_reward_strategy
python src/train_and_backtest_rr_strategy.py
```

### 2、神经网络混合模型

```bash
# 本地快速验证
cd experiments/neural_hybrid
python src/quick_validate.py

# Docker训练
cd experiments/neural_hybrid
docker compose -f docker-compose.train.yml up --build
```

### 3、基线LightGBM模型

```bash
cd experiments/baseline_lgbm
# TODO: 待实现训练脚本
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

## 📝 添加新实验

```bash
# 1. 创建实验目录
mkdir -p experiments/新算法名称/{src,models,data,results}

# 2. 创建符号链接
cd experiments/新算法名称
ln -s ../../../data data

# 3. 编写训练脚本
# src/train.py
# src/backtest.py

# 4. 创建 README.md
# 包含：算法说明、性能指标、使用方法
```

## 🔧 模型管理

### 版本控制
- ✅ 所有大型模型文件已添加到 `.gitignore`
- ✅ 仅提交模型架构代码和配置
- ✅ 小型模型(<1MB)可以提交以便快速验证

### 备份策略
```bash
# 备份实验结果
cp -r experiments/模型名 experiments/模型名.backup_$(date +%Y%m%d)

# 备份到服务器
scp -r experiments/模型名 root@server:/path/to/backup/
```

### 清理临时文件
```bash
# 清理测试结果
find experiments/*/results/ -name "*test*" -delete

# 清理Python缓存
find experiments/ -type d -name "__pycache__" -exec rm -rf {} +
```
