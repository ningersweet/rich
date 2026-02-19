# BTC量化交易系统

> **版本**: v3.1  
> **最后更新**: 2026-02-20  
> **核心策略**: 盈亏比驱动的两阶段预测架构 + 动态敞口管理

---

## 🚀 快速开始

### 最佳回测结果（2026-02-20验证）

| 指标 | 数值 |
|------|------|
| **总收益率** | **2,838,309%** (约28384倍) |
| **交易数** | 160笔（0.39笔/天） |
| **胜率** | 78.75% |
| **盈亏比** | 1.45 |
| **最大回撤** | 10.00% |
| **回测周期** | 2025-01-01 至 2026-02-17（13.5个月） |

### 核心配置

```yaml
# 信号过滤
prob_threshold: 0.70     # 方向置信度
rr_threshold: 2.5        # 盈亏比阈值

# 风控参数
stop_loss_pct: -0.03     # 固定止损-3%
max_exposure: 10.0       # 最大敞口10倍
max_drawdown_pause: 0.10 # 回撤暂停10%

# 追踪止损
trailing_stop: 
  enable: true
  min_profit: 0.01       # 盈利>1%启动
  price_drop: 0.02       # 价格距最高点下降2%触发
```

---

## 📚 文档导航

### 核心文档
- **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)** - 完整技术文档（1600+行）
  - 系统概览
  - 最新回测结果
  - 技术架构详解
  - 核心模块说明
  - 风控机制详解
  - 部署指南
  - 优化计划

### 快速链接
- [系统概览](TECHNICAL_DOCUMENTATION.md#1-系统概览)
- [最新回测结果](TECHNICAL_DOCUMENTATION.md#2-最新回测结果)
- [风控机制](TECHNICAL_DOCUMENTATION.md#6-风控机制)
- [部署架构](TECHNICAL_DOCUMENTATION.md#7-部署架构)
- [优化计划](TECHNICAL_DOCUMENTATION.md#10-性能优化计划)

---

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 编程语言 | Python 3.10 |
| 机器学习 | LightGBM GBDT |
| 数据处理 | Pandas, NumPy |
| API交互 | Requests (币安API) |
| 容器化 | Docker, Docker Compose |
| K线周期 | 15分钟 |

---

## 📦 快速部署

### 1. 配置API密钥
编辑 `config.yaml`：
```yaml
api:
  paper:
    key: "YOUR_TESTNET_KEY"
    secret: "YOUR_TESTNET_SECRET"
  mode: "paper"
```

### 2. Docker部署（推荐）
```bash
# 构建镜像
docker-compose build

# 启动模拟盘
docker-compose up -d paper_trading

# 查看日志
docker-compose logs -f paper_trading
```

### 3. 本地运行
```bash
# 安装依赖
pip install -r requirements.txt

# 运行回测
python train_dynamic_exposure_with_advanced_risk.py

# 运行实盘
export MODE=paper
python run_live_dynamic_exposure.py
```

---

## 📊 核心特性

### 两阶段预测架构
1. **阶段1**: 预测盈亏比（筛选高质量机会）
2. **阶段2**: 预测方向和持有周期

### 动态敞口管理
- 根据信号质量动态调整（1-10倍）
- 回撤时自动降低敞口
- 连续亏损敞口惩罚

### 多层风控体系
1. 固定止损（-3%）
2. 追踪止损（盈利>1%启动，价格下降2%触发）
3. 持仓周期到期
4. 每日最大亏损限制（-20%）
5. 回撤暂停（10%）
6. 动态敞口管理

---

## ⚙️ 项目结构

```
lightgbm/
├── btc_quant/                          # 核心量化模块
│   ├── config.py                       # 配置管理
│   ├── data.py                         # 数据下载
│   ├── features.py                     # 特征工程（485维）
│   ├── risk_reward_labels.py           # 盈亏比标签
│   ├── risk_reward_model.py            # 两阶段模型
│   ├── execution.py                    # 交易执行
│   └── monitor.py                      # 日志监控
│
├── models/                             # 训练模型
│   └── final_6x_fixed_capital/         # 生产模型（2019-2024训练）
│       ├── risk_reward_model.txt
│       ├── direction_model.txt
│       └── period_model.txt
│
├── data/                               # 历史K线数据
├── logs/                               # 运行日志
├── backtest/                           # 回测结果
│
├── config.yaml                         # 主配置文件
├── docker-compose.yml                  # Docker编排
├── Dockerfile                          # 镜像构建
│
├── train_dynamic_exposure_with_advanced_risk.py  # 主回测脚本（生产）
├── run_live_dynamic_exposure.py                  # 实盘/模拟盘运行脚本
├── visualize_final_10x_exposure.py               # 回测可视化报告
├── README.md                           # 本文件
└── TECHNICAL_DOCUMENTATION.md          # 完整技术文档
```

---

## 🔥 最新更新（v3.1 - 2026-02-20）

- ✅ 追踪止损优化：价格距最高点下降2%触发
- ✅ 数据泄露验证：训练集/回测集完全独立

### 关键性能（v3.0 优化）

- 总收益率：**2,838,309%**（326% → 28384倍）
- 交易频率：0.39笔/天（1.6笔/月 → 提升7.3倍）
- 胜率：78.75% | 盈亏比：1.45

---

## ⚠️ 风险警示

**请在使用本系统前仔细阅读：**

1. **高风险投资**：加密货币合约交易风险极高，可能导致本金全部损失
2. **杠杆风险**：10倍杠杆意味着价格波动10%即可爆仓
3. **回测不代表未来**：历史数据表现不保证实盘收益
4. **充分测试**：实盘前务必在模拟盘测试至少1-2周

**免责声明**：
- 本系统仅供学习和研究使用
- 使用本系统进行实盘交易的一切后果由用户自行承担

---

## 📞 联系方式

- **项目路径**: `/Users/lemonshwang/project/rich/lightgbm`
- **作者**: lemonshwang
- **最后更新**: 2026-02-20

---

## 📖 详细文档

更多详细信息请查看 **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)**

包含内容：
- 完整的系统架构设计
- 核心模块详细说明
- 风控机制深度解析
- 潜在风险全面分析
- 性能优化详细计划
- 运维指南和故障排查

---

**祝交易顺利！💰**
