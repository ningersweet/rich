"""神经网络模型：LSTM和Transformer用于时序特征提取"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from .config import Config


class LSTMFeatureExtractor(nn.Module):
    """LSTM时序特征提取器"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 32,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, features)
        返回: (batch, output_dim)
        """
        # LSTM输出: output, (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后时刻的隐状态
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        
        # 全连接层
        out = self.fc(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        
        return out


class TransformerFeatureExtractor(nn.Module):
    """Transformer时序特征提取器（用于多头注意力机制）"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        output_dim: int = 32,
    ):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # 输出层
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, features)
        返回: (batch, output_dim)
        """
        # 输入投影
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # 全局平均池化
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        
        # 输出层
        out = self.fc(pooled)
        out = self.relu(out)
        out = self.dropout(out)
        
        return out


class PositionalEncoding(nn.Module):
    """位置编码（Transformer必须）"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class HybridNeuralModel(nn.Module):
    """混合神经网络模型：结合LSTM和Transformer"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        model_type: str = "lstm",  # "lstm" or "transformer"
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.model_type = model_type
        
        if model_type == "lstm":
            self.feature_extractor = LSTMFeatureExtractor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
            )
        elif model_type == "transformer":
            self.feature_extractor = TransformerFeatureExtractor(
                input_dim=input_dim,
                d_model=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, features)
        返回: (batch, num_classes)
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """仅提取特征（不做分类）"""
        return self.feature_extractor(x)


@dataclass
class NeuralTrainedModel:
    """训练好的神经网络模型"""
    model: HybridNeuralModel
    feature_names: list[str]
    sequence_length: int
    device: str


def create_sequences(
    features: pd.DataFrame,
    labels: pd.Series,
    sequence_length: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """将特征转换为时序序列格式
    
    Args:
        features: (n_samples, n_features) DataFrame
        labels: (n_samples,) Series
        sequence_length: 序列长度（回看窗口）
    
    Returns:
        X_seq: (n_samples - sequence_length + 1, sequence_length, n_features)
        y_seq: (n_samples - sequence_length + 1,)
    """
    feature_array = features.values
    label_array = labels.values
    
    n_samples = len(feature_array)
    n_features = feature_array.shape[1]
    
    # 创建序列
    X_seq = []
    y_seq = []
    
    for i in range(sequence_length - 1, n_samples):
        # 取过去sequence_length个时间步的特征
        seq = feature_array[i - sequence_length + 1:i + 1]
        X_seq.append(seq)
        # 标签取当前时刻
        y_seq.append(label_array[i])
    
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int64)
    
    return X_seq, y_seq


def train_neural_model(
    cfg: Config,
    features: pd.DataFrame,
    labels: pd.Series,
    model_type: str = "lstm",
    sequence_length: int = 20,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: Optional[str] = None,
) -> NeuralTrainedModel:
    """训练神经网络模型
    
    Args:
        cfg: 配置对象
        features: 特征DataFrame
        labels: 标签Series（-1/0/1）
        model_type: "lstm" 或 "transformer"
        sequence_length: 时序序列长度
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        device: 设备（"cpu" 或 "cuda"）
    
    Returns:
        训练好的模型
    """
    # 设备设置
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"训练设备: {device}")
    print(f"模型类型: {model_type}")
    print(f"序列长度: {sequence_length}")
    
    # 标签映射：-1 -> 0, 0 -> 1, 1 -> 2
    labels_mapped = labels + 1
    num_classes = 3
    
    # 创建序列
    X_seq, y_seq = create_sequences(features, labels_mapped, sequence_length)
    print(f"序列数据形状: X={X_seq.shape}, y={y_seq.shape}")
    
    # 清理数据：替换inf和nan
    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 标准化输入数据（防止数值过大导致nan）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Reshape为(n_samples * seq_len, n_features)进行标准化
    original_shape = X_seq.shape
    X_seq_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
    X_seq_scaled = scaler.fit_transform(X_seq_reshaped)
    X_seq = X_seq_scaled.reshape(original_shape).astype(np.float32)
    
    print(f"标准化后数据范围: [{X_seq.min():.2f}, {X_seq.max():.2f}]")
    
    # 划分训练集和验证集（80/20）
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
    
    # 转换为Tensor
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    
    # 创建模型
    input_dim = X_seq.shape[2]
    model = HybridNeuralModel(
        input_dim=input_dim,
        num_classes=num_classes,
        model_type=model_type,
        hidden_dim=64,
        output_dim=32,
        dropout=0.2,
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        
        # 随机打乱训练数据
        perm = torch.randperm(len(X_train_t))
        
        epoch_loss = 0.0
        for i in range(0, len(X_train_t), batch_size):
            batch_idx = perm[i:i + batch_size]
            batch_X = X_train_t[batch_idx]
            batch_y = y_train_t[batch_idx]
            
            # 前向传播
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸和nan）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t)
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == y_val_t).float().mean().item()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {epoch_loss/len(X_train_t):.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")
    
    return NeuralTrainedModel(
        model=model,
        feature_names=list(features.columns),
        sequence_length=sequence_length,
        device=device,
    )


def save_neural_model(cfg: Config, trained: NeuralTrainedModel, name: str = "neural_model_latest.pt"):
    """保存神经网络模型"""
    model_dir = Path(cfg.paths["model_dir"]).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / name
    
    torch.save({
        "model_state_dict": trained.model.state_dict(),
        "feature_names": trained.feature_names,
        "sequence_length": trained.sequence_length,
        "model_type": trained.model.model_type,
    }, out_path)
    
    print(f"神经网络模型已保存: {out_path}")
    return out_path


def load_neural_model(cfg: Config, name: str = "neural_model_latest.pt", device: Optional[str] = None) -> NeuralTrainedModel:
    """加载神经网络模型"""
    model_dir = Path(cfg.paths["model_dir"]).expanduser().resolve()
    path = model_dir / name
    
    if not path.exists():
        raise FileNotFoundError(f"神经网络模型文件不存在: {path}")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(path, map_location=device)
    
    # 重建模型
    input_dim = len(checkpoint["feature_names"])
    model = HybridNeuralModel(
        input_dim=input_dim,
        num_classes=3,
        model_type=checkpoint["model_type"],
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return NeuralTrainedModel(
        model=model,
        feature_names=checkpoint["feature_names"],
        sequence_length=checkpoint["sequence_length"],
        device=device,
    )


def predict_neural(trained: NeuralTrainedModel, features: pd.DataFrame) -> np.ndarray:
    """使用神经网络模型预测
    
    Returns:
        概率分布 (n_samples, 3)，对应 [-1, 0, 1]
    """
    model = trained.model
    model.eval()
    
    # 创建序列
    X_seq, _ = create_sequences(
        features,
        pd.Series([0] * len(features)),  # 占位标签
        trained.sequence_length
    )
    
    # 转换为Tensor
    X_t = torch.from_numpy(X_seq).to(trained.device)
    
    # 预测
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1)
    
    # 转换回numpy（映射回-1, 0, 1）
    return probs.cpu().numpy()
