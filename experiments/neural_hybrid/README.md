# 神经网络混合模型实验

## 快速开始

```bash
# 1. 本地快速验证（模拟数据）
cd experiments/neural_hybrid
python src/quick_validate.py

# 2. Docker训练（真实数据）
cd experiments/neural_hybrid
docker compose -f docker-compose.train.yml up --build

# 3. 查看模型
ls models/
```

## 模型说明

结合LSTM神经网络和LightGBM的混合深度学习模型。

## 目录结构

```
neural_hybrid/
├── src/                           # 源代码
│   ├── train_hybrid_model.py      # 训练脚本
│   ├── quick_validate.py          # 快速验证
│   └── test_neural_model.py       # 模型测试
├── models/                       # 训练好的模型
│   ├── neural_lstm_latest.pt      # LSTM特征提取器
│   └── hybrid_model_latest.pkl    # 混合LightGBM
├── data/                         # 符号链接到项目根目录data/
├── results/                      # 训练结果
├── docker-compose.train.yml      # Docker训练环境
└── README.md
```

### 架构

```
输入特征(54维) 
    ↓
[序列化: 15步时序窗口]
    ↓
[LSTM特征提取器]
    ↓
神经网络特征(32维)
    ↓
[特征融合: 54+32=86维]
    ↓
[LightGBM最终分类]
    ↓
预测结果(做多/观望/做空)
```

### 性能指标

- **验证准确率**: 58.86% (LSTM)
- **交叉验证准确率**: 54% (混合模型)
- **特征维度**: 86维 (54原始 + 32神经网络)
- **算法**: LSTM + LightGBM
- **训练时间**: ~30分钟 (CPU)

### 优势

- ✅ 捕捉时序依赖关系
- ✅ 深层特征自动学习
- ✅ 集成学习提升稳定性

### 劣势

- ⚠️ 训练时间较长
- ⚠️ 当前性能低于基线模型
- ⚠️ 需要更多数据和GPU加速

### 使用方法

```python
import torch
import joblib

# 1. 加载神经网络
from btc_quant.neural_model import load_neural_model
neural_model = load_neural_model('models/neural_hybrid/neural_lstm_latest.pt')

# 2. 提取神经网络特征
from btc_quant.neural_model import predict_neural
neural_features = predict_neural(neural_model, features)

# 3. 特征融合
import pandas as pd
combined_features = pd.concat([features, neural_features], axis=1)

# 4. LightGBM预测
hybrid_model = joblib.load('models/neural_hybrid/hybrid_model_latest.pkl')
predictions = hybrid_model.predict(combined_features)
```

### 优化建议

1. **增加训练数据**: 使用5年以上历史数据
2. **GPU训练**: 在服务器上使用GPU加速
3. **增大网络**: 增加hidden_dim和sequence_length
4. **更多轮数**: 训练50-100轮
5. **SMOTE过采样**: 处理样本不平衡
