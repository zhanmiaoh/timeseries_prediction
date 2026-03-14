import torch
import torch.nn as nn
import math

class TransformerForecast(nn.Module):
    """
    Given the past seq_length values of all features, Predict future horizon=1 (default) value(s) of the target feature
    Using self-attention to capture temporal dependencies in the input sequence.
    input_proj -> positional_encoding -> multi-head encoder -> linear

    Input with shape (batch_size, seq_length, num_features)
    Output with shape (batch_size, horizon)
    """
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dropout=0.2, horizon=1):
        super().__init__()
        
        self.input_proj = nn.Linear(num_features, d_model)  # [B, T, 7] -> [B, T, 64]，每个 token/时间步都在统一的隐藏空间里表示
        self.positional_encoding = PositionalEncoding(d_model) # 补上时序位置信息

        encoder_layer = nn.TransformerEncoderLayer(  #先定义每一层的结构（每一层包含nhead）
            d_model=d_model,
            nhead=nhead,  # 每个头的维度是d_model/nhead=16
            dim_feedforward=d_model*4,  # FFN内部先升维到256，再降回d_model=64
            dropout=dropout,
            batch_first=True
        )  # default: activation='relu', dim_feedforward=2048, norm_first=False
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # stack of N encoder layers
        self.linear = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.input_proj(x)  # [B, T, num_features] -> [B, T, d_model]
        x = self.positional_encoding(x)  # [B, T, d_model]
        x = self.encoder(x) 
        x = x[:, -1, :]  # [B, d_model]，让最后一天这个 token 看完前面所有天，再用它的表示来预测未来。
                         #（另外也可以做 mean pooling、attention pooling）
        x = self.linear(x)  # [B, d_model] -> [B, horizon]

        return x
    
# 这段代码没有传 src_mask，也没有传 is_causal=True, generate_square_subsequent_mask
# PyTorch 文档里，TransformerEncoderLayer 的 forward 支持 is_causal，默认是 False。
# 所以这份实现不是 GPT 那种严格因果掩码自注意力，而是“在给定历史窗口内做全窗口 self-attention”。
# 对你这个任务来说，它通常仍然是合理的，因为输入窗口本身已经全是过去数据，没有把标签未来值喂进去

# 现在的 forward 只返回预测值。
# 所以如果你要做“可解释性”，不能直接从这个类里拿每层注意力权重，你得改实现。

class PositionalEncoding(nn.Module):
    """
    d_model: each timestep is a d_model-dim vector of hidden layer
    max_len (default 500): the max seqence length that this module can handle (longer than your seq_length=60)
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # row: time position, col: dim in hidden space
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape: (max_len, 1)
        # sinusoidal positional encoding (不同维度会对应不同“波长”的正弦/余弦曲线)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # shape: (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * frequency)

        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model), 把 pe 注册成 buffer，不是可训练参数

    def forward(self, x): # x: (B, T, d_model), x.size(1) 是 T=seq_len
        return self.pe[:, :x.size(1), :] + x  # (1, T, d_model) + (B, T, d_model) broadcast-> (B, seq_len, d_model)

    