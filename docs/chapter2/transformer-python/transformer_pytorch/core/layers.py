"""
基础神经网络层模块

实现 Transformer 模型的基础组件层：
- 层归一化 (LayerNorm)
- 前馈神经网络 (FeedForward)
- Dropout 层
- 残差连接 (ResidualConnection)

所有层都基于 PyTorch nn.Module 实现，支持 GPU 加速和自动微分。

作者: Transformer-PyTorch Team
版本: 1.0.0
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .math_utils import get_activation_function, init_weights


class LayerNorm(nn.Module):
    """
    层归一化 (Layer Normalization)
    
    对每个样本的特征维度进行归一化，实现公式：
    LayerNorm(x) = γ * (x - μ) / σ + β
    
    其中 μ 和 σ 是在最后一个维度上计算的均值和标准差。
    
    Args:
        normalized_shape (int): 归一化的维度大小
        eps (float): 防止除零的小值，默认为 1e-6
        elementwise_affine (bool): 是否使用可学习的仿射变换参数
        
    Examples:
        >>> layer_norm = LayerNorm(512)
        >>> x = torch.randn(2, 10, 512)
        >>> output = layer_norm(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            # 可学习的缩放和偏移参数
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量 [..., normalized_shape]
            
        Returns:
            Tensor: 归一化后的张量，形状与输入相同
        """
        # 计算最后一个维度的均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 归一化
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 应用可学习的仿射变换
        if self.elementwise_affine:
            normalized = normalized * self.weight + self.bias
        
        return normalized
    
    def extra_repr(self) -> str:
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class FeedForward(nn.Module):
    """
    前馈神经网络 (Feed Forward Network)
    
    实现两层全连接网络，中间使用激活函数：
    FFN(x) = activation(xW₁ + b₁)W₂ + b₂
    
    Args:
        d_model (int): 输入和输出维度
        d_ff (int): 隐藏层维度
        activation (str): 激活函数名称，默认为 'relu'
        dropout (float): Dropout 概率，默认为 0.1
        bias (bool): 是否使用偏置，默认为 True
        
    Examples:
        >>> ffn = FeedForward(512, 2048, activation='gelu')
        >>> x = torch.randn(2, 10, 512)
        >>> output = ffn(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = 'relu',
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation_name = activation
        
        # 第一层：d_model -> d_ff
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        
        # 激活函数
        self.activation = get_activation_function(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 第二层：d_ff -> d_model
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        
        # 初始化权重
        self.apply(lambda module: init_weights(module, init_std=0.02))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            Tensor: 输出张量 [batch_size, seq_len, d_model]
        """
        # 第一层线性变换 + 激活函数
        hidden = self.activation(self.linear1(x))
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # 第二层线性变换
        output = self.linear2(hidden)
        
        return output
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, d_ff={self.d_ff}, activation={self.activation_name}'


class ResidualConnection(nn.Module):
    """
    残差连接 (Residual Connection)
    
    实现残差连接和层归一化的组合：
    output = x + dropout(sublayer(layer_norm(x)))
    
    这里使用 Pre-LN 结构，即先进行层归一化，再应用子层。
    
    Args:
        d_model (int): 模型维度
        dropout (float): Dropout 概率，默认为 0.1
        
    Examples:
        >>> residual = ResidualConnection(512, dropout=0.1)
        >>> sublayer = nn.Linear(512, 512)
        >>> x = torch.randn(2, 10, 512)
        >>> output = residual(x, sublayer)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量 [batch_size, seq_len, d_model]
            sublayer (nn.Module): 子层模块
            
        Returns:
            Tensor: 输出张量 [batch_size, seq_len, d_model]
        """
        # Pre-LN: 先归一化，再应用子层，最后残差连接
        return x + self.dropout(sublayer(self.layer_norm(x)))


class PositionwiseFeedForward(nn.Module):
    """
    位置相关的前馈网络
    
    这是 FeedForward 的别名，为了与论文术语保持一致。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ffn = FeedForward(*args, **kwargs)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class GLU(nn.Module):
    """
    门控线性单元 (Gated Linear Unit)
    
    实现公式: GLU(x) = (xW + b) ⊙ σ(xV + c)
    其中 ⊙ 表示逐元素乘法，σ 是 sigmoid 函数。
    
    Args:
        d_model (int): 输入维度
        d_ff (int): 隐藏层维度
        bias (bool): 是否使用偏置
    """
    
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(d_model, d_ff, bias=bias)
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 输出张量
        """
        return self.linear(x) * torch.sigmoid(self.gate(x))


class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数
    
    结合 Swish 激活函数和门控机制：
    SwiGLU(x) = Swish(xW) ⊙ (xV)
    
    Args:
        d_model (int): 输入维度
        d_ff (int): 隐藏层维度
        bias (bool): 是否使用偏置
    """
    
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(d_model, d_ff, bias=bias)
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.swish = nn.SiLU()  # SiLU 就是 Swish
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 输出张量
        """
        return self.swish(self.gate(x)) * self.linear(x)


class RMSNorm(nn.Module):
    """
    RMS 归一化 (Root Mean Square Normalization)
    
    一种简化的层归一化变体，只使用 RMS 进行归一化：
    RMSNorm(x) = x / RMS(x) * γ
    
    其中 RMS(x) = √(mean(x²) + ε)
    
    Args:
        d_model (int): 模型维度
        eps (float): 防止除零的小值
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 归一化后的张量
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# 导出的类列表
__all__ = [
    'LayerNorm',
    'FeedForward',
    'ResidualConnection',
    'PositionwiseFeedForward',
    'GLU',
    'SwiGLU',
    'RMSNorm',
]
