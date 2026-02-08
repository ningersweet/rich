"""混合模型训练脚本：LightGBM + 神经网络（LSTM/Transformer）

训练流程：
1. 神经网络提取时序深层特征
2. 将神经网络特征与原始特征合并
3. 使用LightGBM进行最终预测（集成学习）
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import joblib

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from btc_quant.config import load_config
from btc_quant.data import load_klines
from btc_quant.features import build_features_and_labels
from btc_quant.neural_model import (
    train_neural_model,
    save_neural_model,
    create_sequences,
    NeuralTrainedModel,
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_neural_features(
    neural_model: NeuralTrainedModel,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """使用训练好的神经网络提取深层特征
    
    Returns:
        (n_samples, neural_feature_dim) DataFrame
    """
    import torch
    
    model = neural_model.model
    model.eval()
    
    # 创建序列
    X_seq, _ = create_sequences(
        features,
        pd.Series([0] * len(features)),  # 占位标签
        neural_model.sequence_length
    )
    
    # 转换为Tensor
    X_t = torch.from_numpy(X_seq).to(neural_model.device)
    
    # 提取特征（不做分类）
    with torch.no_grad():
        neural_feats = model.extract_features(X_t)
    
    # 转换为DataFrame
    neural_feats_np = neural_feats.cpu().numpy()
    feature_names = [f"neural_feat_{i}" for i in range(neural_feats_np.shape[1])]
    
    # 注意：序列化会导致前sequence_length-1个样本无特征
    # 用0填充前面的样本
    pad_rows = neural_model.sequence_length - 1
    padded_feats = np.vstack([
        np.zeros((pad_rows, neural_feats_np.shape[1]), dtype=np.float32),
        neural_feats_np
    ])
    
    return pd.DataFrame(padded_feats, columns=feature_names, index=features.index)


def train_hybrid_model(cfg):
    """训练混合模型（神经网络 + LightGBM）"""
    
    logger.info("\n" + "="*60)
    logger.info("【混合模型训练】神经网络 + LightGBM 集成")
    logger.info("="*60)
    
    # ========== 步骤1：加载数据 ==========
    logger.info("\n【步骤1】加载K线数据")
    try:
        klines = load_klines(cfg)
    except FileNotFoundError:
        logger.error("未找到K线数据文件，请先下载数据")
        logger.info("运行命令: python -c 'from btc_quant.data import download_historical_klines; from btc_quant.config import load_config; download_historical_klines(load_config())'")
        sys.exit(1)
    
    logger.info(f"总K线数: {len(klines):,}")
    logger.info(f"时间范围: {klines.index[0]} ~ {klines.index[-1]}")
    
    # ========== 步骤2：构建特征和标签 ==========
    logger.info("\n【步骤2】构建特征和标签")
    fl_data = build_features_and_labels(cfg, klines)
    logger.info(f"特征数: {fl_data.features.shape[1]}")
    logger.info(f"样本数: {len(fl_data.features)}")
    logger.info(f"标签分布:\n{fl_data.labels.value_counts().sort_index()}")
    
    # ========== 步骤3：训练神经网络（时序特征提取器）==========
    logger.info("\n【步骤3】训练神经网络模型（LSTM）")
    
    neural_cfg = cfg.model.get("neural", {})
    model_type = neural_cfg.get("type", "lstm")  # lstm or transformer
    sequence_length = neural_cfg.get("sequence_length", 20)
    epochs = neural_cfg.get("epochs", 50)
    batch_size = neural_cfg.get("batch_size", 64)
    learning_rate = neural_cfg.get("learning_rate", 0.001)
    
    neural_model = train_neural_model(
        cfg=cfg,
        features=fl_data.features,
        labels=fl_data.labels,
        model_type=model_type,
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    # 保存神经网络模型
    neural_path = save_neural_model(cfg, neural_model, name=f"neural_{model_type}_latest.pt")
    logger.info(f"神经网络模型已保存: {neural_path}")
    
    # ========== 步骤4：提取神经网络深层特征 ==========
    logger.info("\n【步骤4】提取神经网络深层特征")
    neural_features = extract_neural_features(neural_model, fl_data.features)
    logger.info(f"神经网络特征维度: {neural_features.shape[1]}")
    
    # ========== 步骤5：特征融合 ==========
    logger.info("\n【步骤5】特征融合（原始特征 + 神经网络特征）")
    
    # 合并特征
    combined_features = pd.concat([
        fl_data.features,
        neural_features
    ], axis=1)
    
    logger.info(f"融合后特征总数: {combined_features.shape[1]}")
    logger.info(f"  - 原始特征: {fl_data.features.shape[1]}")
    logger.info(f"  - 神经网络特征: {neural_features.shape[1]}")
    
    # ========== 步骤6：训练LightGBM（最终分类器）==========
    logger.info("\n【步骤6】训练LightGBM分类器")
    
    # LightGBM参数
    lgbm_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    
    # 类别权重（缓解样本不平衡）
    class_weight_cfg = cfg.model.get("train", {}).get("class_weight", {})
    if class_weight_cfg:
        class_weight = {int(k): float(v) for k, v in class_weight_cfg.items()}
        lgbm_params["class_weight"] = class_weight
        logger.info(f"类别权重: {class_weight}")
    
    # 时间序列交叉验证
    cv_folds = 3
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # 标签映射：-1 -> 0, 0 -> 1, 1 -> 2
    y = fl_data.labels + 1
    
    clf = lgb.LGBMClassifier(**lgbm_params)
    
    logger.info(f"开始{cv_folds}折交叉验证...")
    all_reports = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(combined_features), start=1):
        X_train, X_val = combined_features.iloc[train_idx], combined_features.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        # 映射回原始标签
        y_val_orig = y_val - 1
        y_pred_orig = y_pred - 1
        
        report = classification_report(
            y_val_orig,
            y_pred_orig,
            target_names=["空(-1)", "观望(0)", "多(1)"],
            digits=3
        )
        all_reports.append(f"\nFold {fold} 分类报告:\n{report}")
    
    logger.info("\n".join(all_reports))
    
    # ========== 步骤7：全量数据训练最终模型 ==========
    logger.info("\n【步骤7】使用全量数据训练最终LightGBM模型")
    
    clf_final = lgb.LGBMClassifier(**lgbm_params)
    clf_final.fit(combined_features, y)
    
    # 特征重要性分析
    feature_importance = pd.DataFrame({
        'feature': combined_features.columns,
        'importance': clf_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n前20个重要特征:")
    logger.info(feature_importance.head(20).to_string(index=False))
    
    # ========== 步骤8：保存模型 ==========
    logger.info("\n【步骤8】保存混合模型")
    
    model_dir = Path(cfg.paths["model_dir"]).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    
    hybrid_model_path = model_dir / "hybrid_model_latest.pkl"
    joblib.dump({
        "lgbm_model": clf_final,
        "feature_names": list(combined_features.columns),
        "neural_model_name": f"neural_{model_type}_latest.pt",
        "original_feature_names": list(fl_data.features.columns),
    }, hybrid_model_path)
    
    logger.info(f"混合模型已保存: {hybrid_model_path}")
    
    # ========== 完成 ==========
    logger.info("\n" + "="*60)
    logger.info("✅ 混合模型训练完成！")
    logger.info("="*60)
    logger.info(f"\n模型文件:")
    logger.info(f"  - 神经网络: {neural_path}")
    logger.info(f"  - LightGBM: {hybrid_model_path}")
    logger.info(f"\n使用说明:")
    logger.info(f"  1. 神经网络提取时序深层特征")
    logger.info(f"  2. LightGBM基于融合特征进行最终分类")
    logger.info(f"  3. 集成学习提升预测准确性")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练混合模型（神经网络 + LightGBM）")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 训练混合模型
    train_hybrid_model(cfg)
