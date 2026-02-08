# 基线LightGBM模型

## 模型说明

使用传统的LightGBM集成学习算法训练的基线模型。

### 文件列表

- `model_latest.pkl` - 最新训练的LightGBM模型
- `enhanced_ensemble.pkl` - 增强的集成学习模型

### 性能指标

- **准确率**: 待测试
- **特征维度**: 54维原始特征
- **算法**: LightGBM (GBDT)
- **训练时间**: ~5分钟

### 优势

- ✅ 训练速度快
- ✅ 特征重要性可解释
- ✅ 无需GPU支持
- ✅ 内存占用小

### 使用方法

```python
import joblib
model = joblib.load('models/baseline_lgbm/model_latest.pkl')
predictions = model.predict(features)
```
