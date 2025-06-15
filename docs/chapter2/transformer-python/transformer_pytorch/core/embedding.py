"""
嵌入层模块

实现 Transformer 模型的嵌入层组件：
- 词嵌入 (Token Embedding)
- 位置编码 (Positional Encoding)
- 正弦余弦位置编码 (Sinusoidal Positional Encoding)
- 可学习位置编码 (Learned Positional Encoding)
- 完整的 Transformer 嵌入层

作者: Transformer-PyTorch Team
版本: 1.0.0
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .math_utils import init_weights


class TokenEmbedding(nn.Module):
    """
    词嵌入层 (Token Embedding)
    
    将词汇索引转换为稠密向量表示。
    
    Args:
        vocab_size (int): 词汇表大小
        d_model (int): 嵌入维度
        padding_idx (Optional[int]): 填充词元的索引，默认为 None
        
    Examples:
        >>> embedding = TokenEmbedding(vocab_size=10000, d_model=512)
        >>> token_ids = torch.randint(0, 10000, (2, 10))
        >>> embeddings = embedding(token_ids)
        >>> print(embeddings.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )
        
        # 初始化权重
        self.apply(lambda module: init_weights(module, init_std=0.02))
    
    def forward(self, token_ids: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            token_ids (Tensor): 词元ID张量 [batch_size, seq_len]
            
        Returns:
            Tensor: 嵌入张量 [batch_size, seq_len, d_model]
        """
        return self.embedding(token_ids)
    
    def extra_repr(self) -> str:
        return f'vocab_size={self.vocab_size}, d_model={self.d_model}'


class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦余弦位置编码 (Sinusoidal Positional Encoding)
    
    使用正弦和余弦函数生成位置编码，实现公式：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model (int): 模型维度
        max_len (int): 最大序列长度，默认为 5000
        
    Examples:
        >>> pos_encoding = SinusoidalPositionalEncoding(d_model=512, max_len=1000)
        >>> x = torch.randn(2, 10, 512)
        >>> output = pos_encoding(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        
        # 添加批次维度并注册为缓冲区（不参与训练）
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            Tensor: 添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # 添加位置编码
        x = x + self.pe[:, :seq_len, :]
        return x
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, max_len={self.max_len}'


class LearnedPositionalEncoding(nn.Module):
    """
    可学习位置编码 (Learned Positional Encoding)
    
    使用可训练的参数矩阵作为位置编码。
    
    Args:
        d_model (int): 模型维度
        max_len (int): 最大序列长度
        
    Examples:
        >>> pos_encoding = LearnedPositionalEncoding(d_model=512, max_len=1000)
        >>> x = torch.randn(2, 10, 512)
        >>> output = pos_encoding(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 可学习的位置嵌入
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # 初始化权重
        self.apply(lambda module: init_weights(module, init_std=0.02))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            Tensor: 添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        
        # 生成位置索引
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # 获取位置嵌入
        position_embeddings = self.position_embeddings(positions)
        
        # 添加位置编码
        x = x + position_embeddings
        return x
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, max_len={self.max_len}'


class PositionalEncoding(nn.Module):
    """
    位置编码基类
    
    根据指定的类型创建相应的位置编码。
    
    Args:
        d_model (int): 模型维度
        max_len (int): 最大序列长度
        encoding_type (str): 位置编码类型，'sinusoidal' 或 'learned'
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int,
        encoding_type: str = 'sinusoidal'
    ):
        super().__init__()
        self.encoding_type = encoding_type
        
        if encoding_type == 'sinusoidal':
            self.encoding = SinusoidalPositionalEncoding(d_model, max_len)
        elif encoding_type == 'learned':
            self.encoding = LearnedPositionalEncoding(d_model, max_len)
        else:
            raise ValueError(f"不支持的位置编码类型: {encoding_type}")
    
    def forward(self, x: Tensor) -> Tensor:
        return self.encoding(x)


class TransformerEmbedding(nn.Module):
    """
    完整的 Transformer 嵌入层
    
    组合词嵌入、位置编码和 dropout。
    
    Args:
        vocab_size (int): 词汇表大小
        d_model (int): 模型维度
        max_len (int): 最大序列长度
        padding_idx (Optional[int]): 填充词元的索引
        position_encoding_type (str): 位置编码类型
        dropout (float): Dropout 概率
        scale_embedding (bool): 是否缩放嵌入
        
    Examples:
        >>> embedding = TransformerEmbedding(
        ...     vocab_size=10000, d_model=512, max_len=1000
        ... )
        >>> token_ids = torch.randint(0, 10000, (2, 10))
        >>> output = embedding(token_ids)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        padding_idx: Optional[int] = None,
        position_encoding_type: str = 'sinusoidal',
        dropout: float = 0.1,
        scale_embedding: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.scale_embedding = scale_embedding
        
        # 词嵌入层
        self.token_embedding = TokenEmbedding(
            vocab_size, d_model, padding_idx
        )
        
        # 位置编码
        self.position_encoding = PositionalEncoding(
            d_model, max_len, position_encoding_type
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 嵌入缩放因子
        self.embedding_scale = math.sqrt(d_model) if scale_embedding else 1.0
    
    def forward(self, token_ids: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            token_ids (Tensor): 词元ID张量 [batch_size, seq_len]
            
        Returns:
            Tensor: 最终嵌入张量 [batch_size, seq_len, d_model]
        """
        # 获取词嵌入
        embeddings = self.token_embedding(token_ids)
        
        # 可选的嵌入缩放
        if self.scale_embedding:
            embeddings = embeddings * self.embedding_scale
        
        # 添加位置编码
        embeddings = self.position_encoding(embeddings)
        
        # 应用 dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, scale_embedding={self.scale_embedding}'


class RotaryPositionalEncoding(nn.Module):
    """
    旋转位置编码 (Rotary Positional Encoding, RoPE)
    
    一种相对位置编码方法，通过旋转变换来编码位置信息。
    
    Args:
        d_model (int): 模型维度
        max_len (int): 最大序列长度
        base (float): 基数，默认为 10000
    """
    
    def __init__(self, d_model: int, max_len: int = 2048, base: float = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: Tensor, seq_len: Optional[int] = None) -> Tensor:
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量 [batch_size, seq_len, d_model]
            seq_len (Optional[int]): 序列长度
            
        Returns:
            Tensor: 应用旋转位置编码后的张量
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        # 生成位置索引
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # 计算频率
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 应用旋转变换（简化实现）
        cos_emb = emb.cos()[None, :, :]
        sin_emb = emb.sin()[None, :, :]
        
        # 这里简化了 RoPE 的实现，实际应用中需要更复杂的旋转操作
        return x * cos_emb + self._rotate_half(x) * sin_emb
    
    def _rotate_half(self, x: Tensor) -> Tensor:
        """旋转张量的一半维度"""
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
        return torch.cat((-x2, x1), dim=-1)


# 导出的类列表
__all__ = [
    'TokenEmbedding',
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEncoding',
    'PositionalEncoding',
    'TransformerEmbedding',
    'RotaryPositionalEncoding',
]
