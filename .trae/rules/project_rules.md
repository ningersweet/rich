# BTC量化交易系统 - 项目运维规则

## 项目概述

BTC量化交易系统是一个基于机器学习的盈亏比驱动策略系统，针对币安BTC永续合约（BTCUSDT）进行自动化交易。系统采用两阶段预测架构，集成动态敞口管理和多层风控机制。

**核心特性**：
- 两阶段预测：盈亏比筛选 + 方向周期预测
- 动态敞口管理：1-10倍杠杆自适应调整
- 多层风控：固定止损、追踪止损、每日亏损限制、回撤暂停
- 容器化部署：Docker + Docker Compose

## 开发环境

### Conda环境配置
本地开发和测试使用名为`rich`的Conda环境，确保所有依赖版本一致。

**环境激活和依赖安装**：
```bash
# 激活conda环境
conda activate rich

# 安装项目依赖
pip install -r requirements.txt

# 验证环境
python -c "import lightgbm; import pandas; print('环境验证通过')"
```

**环境管理命令**：
```bash
# 查看当前环境
conda info --envs

# 创建新环境（如果需要）
conda create -n rich python=3.10

# 导出环境配置
conda env export > environment.yml

# 从环境配置恢复
conda env create -f environment.yml
```

**开发工作流程**：
1. 激活conda环境：`conda activate rich`
2. 运行训练脚本：`python training_scripts/train_2024_model.py`
3. 运行回测：`python backtest_scripts/backtest_2024_model.py`
4. 运行测试：`python -m pytest tests/`
5. 退出环境：`conda deactivate`

### 调试脚本管理
为保持项目结构整洁，所有临时调试和分析脚本应放置在 `debug_scripts/` 目录中。

**规范**：
1. 临时调试脚本必须放置在 `debug_scripts/` 目录
2. 脚本中需正确配置Python导入路径，确保能访问项目模块
3. 示例路径配置：
   ```python
   import sys
   import os
   script_dir = os.path.dirname(os.path.abspath(__file__))
   project_root = os.path.dirname(script_dir)  # lightgbm目录
   sys.path.insert(0, project_root)
   ```
4. 调试完成后，可选择性删除或归档脚本

## 部署架构

### 运行模式
- **回测模式**：历史数据回测验证
- **模拟盘模式**：币安测试网虚拟资金交易
- **实盘模式**：币安主网真实资金交易

### 容器服务
```yaml
# docker-compose.yml 服务配置
paper_trading:    # 模拟盘交易（动态敞口策略）
live_trading:     # 实盘交易（动态敞口策略）
backtest:         # 回测服务
```

### 目录结构
```
lightgbm/
├── btc_quant/          # 核心量化模块
├── models/             # 训练模型存储
├── data/               # K线历史数据
├── logs/               # 运行日志
├── backtest/           # 回测结果
├── debug_scripts/      # 临时调试脚本
├── config.yaml         # 主配置文件
├── docker-compose.yml  # Docker编排
└── run_live_dynamic_exposure.py  # 实盘主脚本
```

**目录说明**：
- `debug_scripts/`：存放临时调试和分析脚本，避免项目根目录混乱。所有调试脚本应放置在此目录，并确保导入路径正确配置。

## 服务器部署

### 服务器连接
生产服务器地址：`47.236.94.252`

**SSH登录与工作目录**：
```bash
# SSH登录到服务器
ssh root@47.236.94.252

# 登录后默认在/root目录，需要进入项目工作目录
cd /root/workspace/rich

# 项目代码在lightgbm子目录中
cd lightgbm
```

### 项目目录结构
服务器项目路径：`/root/workspace/rich/lightgbm/`
```
/root/workspace/rich/lightgbm/
├── btc_quant/          # 核心量化模块
├── models/             # 训练模型存储（需通过scp更新）
├── data/               # K线历史数据
├── logs/               # 运行日志
├── debug_scripts/      # 临时调试脚本
├── config.yaml         # 主配置文件
├── docker-compose.yml  # Docker编排
└── run_live_dynamic_exposure.py  # 实盘主脚本
```

### 文件更新策略
#### 1. 模型文件更新（scp）
模型文件存储在 `models/` 目录下，需要从本地同步到服务器：

```bash
# 同步单个模型文件
scp /Users/lemonshwang/project/rich/lightgbm/models/final_6x_fixed_capital/your_model.txt root@47.236.94.252:/root/workspace/rich/lightgbm/models/final_6x_fixed_capital/

# 同步整个模型目录
scp -r /Users/lemonshwang/project/rich/lightgbm/models/final_6x_fixed_capital/ root@47.236.94.252:/root/workspace/rich/lightgbm/models/
```

#### 2. 代码文件更新（git）
代码文件通过Git进行版本控制和更新：

**重要规则：服务器上不准修改代码**
- 所有代码修改必须在本地开发环境进行
- 提交到Git仓库后，在服务器上通过`git pull`拉取更新
- 禁止在服务器上直接编辑源代码文件
- 配置模板文件的更新通过Git模板（config.yaml.template）管理，重要配置变更需更新模板文件

```bash
# 登录服务器后，确保在项目目录中（/root/workspace/rich/lightgbm）
cd /root/workspace/rich/lightgbm

# 拉取最新代码
git pull origin main

# 如果存在本地修改，先保存
git stash
git pull origin main
git stash pop

# 重启服务使代码生效
docker-compose down
docker-compose up -d
```

#### 3. 配置文件更新（挂载方式）
**重要：配置文件通过Docker卷挂载方式使用，不能提交到git，也不能打进镜像**

**配置文件挂载特点**：
- `config.yaml`文件通过Docker卷挂载到容器中
- 配置文件不进入Git版本控制，也不打包进Docker镜像
- 本地和服务器分别维护各自的`config.yaml`文件
- 重要配置变更需要同步更新本地和服务器两端的配置文件

**配置文件同步方法**：
1. **本地修改**：在本地开发环境的`config.yaml`中进行配置变更
2. **服务器同步**：使用`scp`命令将配置同步到服务器
3. **服务重启**：重启Docker服务使新配置生效

**配置文件同步示例**：
```bash
# 同步配置文件到服务器
scp /Users/lemonshwang/project/rich/lightgbm/config.yaml root@47.236.94.252:/root/workspace/rich/lightgbm/config.yaml

# 登录服务器重启服务
ssh root@47.236.94.252 'cd /root/workspace/rich/lightgbm && docker-compose restart paper_trading'
```

**配置变更注意事项**：
- 敏感信息（如API密钥、邮箱密码）存储在配置文件中，确保文件权限安全
- 配置变更后务必重启相关服务
- 建议先备份服务器原配置文件：`cp config.yaml config.yaml.backup.$(date +%Y%m%d_%H%M%S)`

### 服务运行
模拟盘和实盘交易需要在服务器上运行：

#### 启动模拟盘（纸交易）
```bash
# 确保在项目目录中（/root/workspace/rich/lightgbm）
cd /root/workspace/rich/lightgbm
docker-compose up -d paper_trading
```

#### 启动实盘交易
```bash
# 确保在项目目录中（/root/workspace/rich/lightgbm）
cd /root/workspace/rich/lightgbm
docker-compose up -d live_trading
```

#### 查看服务状态
```bash
# 查看容器状态
docker ps | grep btc_quant

# 查看实时日志
docker-compose logs -f paper_trading

# 停止所有服务
docker-compose down
```
#### 服务更新流程
1. **更新代码**：`git pull origin main`
2. **更新模型**：使用`scp`同步最新模型文件
3. **重启服务**：`docker-compose down && docker-compose up -d`
4. **验证状态**：检查容器状态和日志