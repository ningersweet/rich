# 神经网络混合模型

## 模型说明

结合LSTM神经网络和LightGBM的混合深度学习模型。

### 文件列表

- `neural_lstm_latest.pt` - LSTM神经网络特征提取器 (267KB)
- `hybrid_model_latest.pkl` - 混合LightGBM分类器 (2.0MB)

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
