"""
简化版 Mamba2 模型实现
不依赖 Triton 库，适用于 CPU 和 GPU 环境
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


class RMSNorm(nn.Module):
    """均方根归一化层"""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x / math.sqrt(x.shape[-1])
        x_normed = x / (rms_x + self.eps)
        return self.weight * x_normed


class ResidualBlock(nn.Module):
    """残差连接块"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, expand_factor=2, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expand_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expand_factor, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class SimplifiedSSM(nn.Module):
    """简化版状态空间模型"""
    def __init__(self, d_model, d_state, dropout=0.0):
        super().__init__()
        # 参数矩阵
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        # 修改 B 矩阵的维度，从 (d_state, d_model) 改为 (d_model, d_state)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        d_state = self.A.shape[0]
        
        # 初始化状态
        h = torch.zeros(batch_size, d_state, device=x.device)
        outputs = []
        
        # 序列扫描
        for t in range(seq_len):
            # 修改矩阵乘法，确保维度匹配
            # x[:, t] 形状为 [batch_size, d_model]
            # self.B 形状为 [d_model, d_state]
            # 乘积形状为 [batch_size, d_state]
            h = torch.tanh(h @ self.A + x[:, t] @ self.B)
            # 计算输出
            y = h @ self.C.t() + self.D
            outputs.append(y)
        
        # 堆叠输出
        output = torch.stack(outputs, dim=1)
        return self.dropout(output)

class Mamba2Config:
    """Mamba2模型配置"""
    def __init__(
        self,
        d_model=128,
        n_layers=4,
        d_state=16,
        d_head=32,
        expand_factor=2,
        d_conv=4,
        n_groups=1,
        dropout=0.1,
        device=None
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_head = d_head
        self.expand_factor = expand_factor
        self.d_conv = d_conv
        self.n_groups = n_groups
        self.dropout = dropout
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Mamba2Block(nn.Module):
    """Mamba2块"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 归一化层
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        
        # 状态空间模型
        self.ssm = SimplifiedSSM(config.d_model, config.d_state, dropout=config.dropout)
        
        # 前馈网络
        self.ff = FeedForward(
            config.d_model,
            expand_factor=config.expand_factor,
            dropout=config.dropout
        )
    
    def forward(self, x):
        # 第一个子层: SSM
        x = x + self.ssm(self.norm1(x))
        
        # 第二个子层: 前馈网络
        x = x + self.ff(self.norm2(x))
        
        return x


class Mamba2(nn.Module):
    """简化版Mamba2模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 层堆叠
        self.layers = nn.ModuleList([
            Mamba2Block(config) for _ in range(config.n_layers)
        ])
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        for layer in self.layers:
            x = layer(x)
        return x