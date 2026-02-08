"""测试神经网络模型基本功能"""
import torch
import numpy as np
import pandas as pd
from btc_quant.neural_model import (
    LSTMFeatureExtractor,
    TransformerFeatureExtractor,
    HybridNeuralModel,
    create_sequences,
)

print("="*60)
print("神经网络模型功能测试")
print("="*60)

# 测试参数
batch_size = 4
seq_len = 20
input_dim = 10
hidden_dim = 32
output_dim = 16

print(f"\n测试配置:")
print(f"  Batch Size: {batch_size}")
print(f"  Sequence Length: {seq_len}")
print(f"  Input Dim: {input_dim}")
print(f"  Hidden Dim: {hidden_dim}")
print(f"  Output Dim: {output_dim}")

# ========== 测试1: LSTM特征提取器 ==========
print("\n" + "-"*60)
print("【测试1】LSTM特征提取器")
print("-"*60)

lstm_model = LSTMFeatureExtractor(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
)

# 创建随机输入
x_lstm = torch.randn(batch_size, seq_len, input_dim)
print(f"输入形状: {x_lstm.shape}")

# 前向传播
lstm_output = lstm_model(x_lstm)
print(f"输出形状: {lstm_output.shape}")
print(f"✅ LSTM模型测试通过！")

# ========== 测试2: Transformer特征提取器 ==========
print("\n" + "-"*60)
print("【测试2】Transformer特征提取器")
print("-"*60)

transformer_model = TransformerFeatureExtractor(
    input_dim=input_dim,
    d_model=hidden_dim,
    nhead=4,
    output_dim=output_dim,
)

# 创建随机输入
x_transformer = torch.randn(batch_size, seq_len, input_dim)
print(f"输入形状: {x_transformer.shape}")

# 前向传播
transformer_output = transformer_model(x_transformer)
print(f"输出形状: {transformer_output.shape}")
print(f"✅ Transformer模型测试通过！")

# ========== 测试3: 混合模型 ==========
print("\n" + "-"*60)
print("【测试3】混合神经网络模型")
print("-"*60)

# 测试LSTM版本
hybrid_lstm = HybridNeuralModel(
    input_dim=input_dim,
    num_classes=3,
    model_type="lstm",
    hidden_dim=hidden_dim,
    output_dim=output_dim,
)

logits = hybrid_lstm(x_lstm)
print(f"LSTM混合模型输出: {logits.shape}")
print(f"✅ LSTM混合模型测试通过！")

# 测试Transformer版本
hybrid_transformer = HybridNeuralModel(
    input_dim=input_dim,
    num_classes=3,
    model_type="transformer",
    hidden_dim=hidden_dim,
    output_dim=output_dim,
)

logits = hybrid_transformer(x_transformer)
print(f"Transformer混合模型输出: {logits.shape}")
print(f"✅ Transformer混合模型测试通过！")

# ========== 测试4: 序列创建 ==========
print("\n" + "-"*60)
print("【测试4】时序序列创建")
print("-"*60)

# 创建模拟数据
n_samples = 100
n_features = 5
sequence_length = 10

features = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f"feat_{i}" for i in range(n_features)]
)
labels = pd.Series(np.random.choice([-1, 0, 1], size=n_samples))

print(f"原始特征: {features.shape}")
print(f"原始标签: {labels.shape}")

# 创建序列
X_seq, y_seq = create_sequences(features, labels, sequence_length)
print(f"序列化特征: {X_seq.shape}")
print(f"序列化标签: {y_seq.shape}")
print(f"✅ 序列创建测试通过！")

# ========== 测试5: 端到端训练测试 ==========
print("\n" + "-"*60)
print("【测试5】端到端训练（Mini Test）")
print("-"*60)

# 创建小规模数据
train_size = 200
test_size = 50

train_features = pd.DataFrame(
    np.random.randn(train_size, n_features),
    columns=[f"feat_{i}" for i in range(n_features)]
)
train_labels = pd.Series(np.random.choice([-1, 0, 1], size=train_size))

# 标签映射
train_labels_mapped = train_labels + 1

# 创建序列
X_train, y_train = create_sequences(train_features, train_labels_mapped, sequence_length)

# 转换为Tensor
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)

# 创建模型
model = HybridNeuralModel(
    input_dim=n_features,
    num_classes=3,
    model_type="lstm",
)

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练几轮
epochs = 5
batch_size = 32

print(f"训练数据: {X_train.shape}")
print(f"开始训练 {epochs} 轮...")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    
    for i in range(0, len(X_train_t), batch_size):
        batch_X = X_train_t[i:i+batch_size]
        batch_y = y_train_t[i:i+batch_size]
        
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / (len(X_train_t) // batch_size + 1)
    print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

print(f"✅ 端到端训练测试通过！")

# ========== 总结 ==========
print("\n" + "="*60)
print("🎉 所有测试通过！神经网络模型功能正常")
print("="*60)
print("\n可以开始训练完整的混合模型:")
print("  python train_hybrid_model.py")
