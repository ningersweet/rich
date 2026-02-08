"""快速验证脚本：使用小规模数据测试混合模型训练流程"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from btc_quant.config import load_config
from btc_quant.neural_model import train_neural_model, save_neural_model
from btc_quant.features import FeatureLabelData

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_dummy_data(n_samples=5000, n_features=20):
    """生成模拟数据用于快速测试"""
    logger.info(f"生成模拟数据: {n_samples}样本, {n_features}特征")
    
    # 生成随机特征
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)]
    )
    
    # 生成标签（模拟趋势）
    labels = []
    for i in range(n_samples):
        # 简单规则：前5个特征均值决定标签
        mean_val = features.iloc[i, :5].mean()
        if mean_val > 0.3:
            labels.append(1)  # 做多
        elif mean_val < -0.3:
            labels.append(-1)  # 做空
        else:
            labels.append(0)  # 观望
    
    labels = pd.Series(labels)
    
    logger.info(f"标签分布:\n{labels.value_counts().sort_index()}")
    
    return FeatureLabelData(
        features=features,
        labels=labels,
    )

def quick_validation():
    """快速验证混合模型训练流程"""
    logger.info("="*60)
    logger.info("【快速验证】混合模型训练流程")
    logger.info("="*60)
    
    # 加载配置
    cfg = load_config()
    
    # 生成模拟数据
    logger.info("\n【步骤1】生成模拟数据")
    fl_data = generate_dummy_data(n_samples=5000, n_features=20)
    
    # 训练神经网络（小规模）
    logger.info("\n【步骤2】训练LSTM神经网络")
    neural_model = train_neural_model(
        cfg=cfg,
        features=fl_data.features,
        labels=fl_data.labels,
        model_type="lstm",
        sequence_length=10,  # 减小序列长度
        epochs=10,  # 减少训练轮数
        batch_size=128,  # 增大批次
        learning_rate=0.001,
    )
    
    logger.info(f"✅ 神经网络训练完成！")
    logger.info(f"   设备: {neural_model.device}")
    logger.info(f"   序列长度: {neural_model.sequence_length}")
    
    # 保存模型（测试）
    logger.info("\n【步骤3】保存模型")
    model_path = save_neural_model(cfg, neural_model, name="neural_lstm_quick_test.pt")
    logger.info(f"✅ 模型已保存: {model_path}")
    
    # 测试推理
    logger.info("\n【步骤4】测试推理")
    from btc_quant.neural_model import predict_neural
    
    test_features = fl_data.features.iloc[:100]
    probs = predict_neural(neural_model, test_features)
    
    logger.info(f"预测概率形状: {probs.shape}")
    logger.info(f"预测结果前5行:\n{probs[:5]}")
    
    # 总结
    logger.info("\n" + "="*60)
    logger.info("🎉 快速验证成功！所有流程正常")
    logger.info("="*60)
    logger.info("\n可以使用真实数据进行完整训练:")
    logger.info("  python train_hybrid_model.py")

if __name__ == "__main__":
    quick_validation()
