"""
注意力机制模块

实现各种注意力机制，包括：
- 多头注意力 (Multi-Head Attention)
- 自注意力 (Self-Attention)
- 交叉注意力 (Cross-Attention)
- 注意力掩码工具

所有注意力机制都基于缩放点积注意力实现。

作者: Transformer-PyTorch Team
版本: 1.0.0
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .math_utils import scaled_dot_product_attention, init_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    
    实现论文中的多头注意力：
    MultiHead(Q,K,V) = Concat(head₁, ..., headₕ)W^O
    其中 headᵢ = Attention(QW^Q_i, KW^K_i, VW^V_i)
    
    Args:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        dropout (float): Dropout 概率，默认为 0.1
        bias (bool): 是否在线性层中使用偏置，默认为 True
        
    Examples:
        >>> attention = MultiHeadAttention(d_model=512, num_heads=8)
        >>> x = torch.randn(2, 10, 512)
        >>> output, weights = attention(x, x, x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
        >>> print(weights.shape)  # torch.Size([2, 8, 10, 10])
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q, K, V 的线性变换层
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        
        # 输出投影层
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self.apply(lambda module: init_weights(module, init_std=0.02))
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        前向传播
        
        Args:
            query (Tensor): 查询张量 [batch_size, seq_len, d_model]
            key (Tensor): 键张量 [batch_size, seq_len, d_model]
            value (Tensor): 值张量 [batch_size, seq_len, d_model]
            mask (Optional[Tensor]): 注意力掩码 [batch_size, seq_len, seq_len]
            return_attention (bool): 是否返回注意力权重
            
        Returns:
            Tuple[Tensor, Optional[Tensor]]: (输出张量, 注意力权重)
        """
        batch_size, seq_len, d_model = query.size()
        
        # 1. 线性变换得到 Q, K, V
        Q = self.w_q(query)  # [batch_size, seq_len, d_model]
        K = self.w_k(key)    # [batch_size, seq_len, d_model]
        V = self.w_v(value)  # [batch_size, seq_len, d_model]
        
        # 2. 重塑为多头形状
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        
        # 3. 调整掩码形状以匹配多头
        if mask is not None:
            # 扩展掩码以匹配多头维度
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [batch_size, num_heads, seq_len, seq_len]
        
        # 4. 计算缩放点积注意力
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout if self.training else None
        )
        
        # 5. 拼接多头输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )  # [batch_size, seq_len, d_model]
        
        # 6. 最终的线性投影
        output = self.w_o(attention_output)
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, num_heads={self.num_heads}, d_k={self.d_k}'


class SelfAttention(MultiHeadAttention):
    """
    自注意力机制 (Self-Attention)
    
    多头注意力的特殊情况，其中 query、key 和 value 都来自同一个输入。
    """
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量 [batch_size, seq_len, d_model]
            mask (Optional[Tensor]): 注意力掩码
            return_attention (bool): 是否返回注意力权重
            
        Returns:
            Tuple[Tensor, Optional[Tensor]]: (输出张量, 注意力权重)
        """
        return super().forward(x, x, x, mask, return_attention)


class CrossAttention(MultiHeadAttention):
    """
    交叉注意力机制 (Cross-Attention)
    
    用于编码器-解码器注意力，其中 query 来自解码器，key 和 value 来自编码器。
    """
    
    def forward(
        self,
        query: Tensor,
        encoder_output: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        前向传播
        
        Args:
            query (Tensor): 查询张量（来自解码器）[batch_size, tgt_len, d_model]
            encoder_output (Tensor): 编码器输出 [batch_size, src_len, d_model]
            mask (Optional[Tensor]): 注意力掩码 [batch_size, tgt_len, src_len]
            return_attention (bool): 是否返回注意力权重
            
        Returns:
            Tuple[Tensor, Optional[Tensor]]: (输出张量, 注意力权重)
        """
        return super().forward(query, encoder_output, encoder_output, mask, return_attention)


class AttentionMask:
    """
    注意力掩码工具类
    
    提供各种类型的注意力掩码生成方法。
    """
    
    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device = None) -> Tensor:
        """
        创建因果掩码（下三角掩码）
        
        Args:
            seq_len (int): 序列长度
            device (torch.device): 设备
            
        Returns:
            Tensor: 因果掩码 [seq_len, seq_len]
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    @staticmethod
    def create_padding_mask(
        token_ids: Tensor,
        pad_token_id: int = 0
    ) -> Tensor:
        """
        创建填充掩码
        
        Args:
            token_ids (Tensor): 词元ID张量 [batch_size, seq_len]
            pad_token_id (int): 填充词元的ID
            
        Returns:
            Tensor: 填充掩码 [batch_size, seq_len]
        """
        return (token_ids != pad_token_id).long()
    
    @staticmethod
    def create_look_ahead_mask(seq_len: int, device: torch.device = None) -> Tensor:
        """
        创建前瞻掩码（与因果掩码相同）
        
        Args:
            seq_len (int): 序列长度
            device (torch.device): 设备
            
        Returns:
            Tensor: 前瞻掩码 [seq_len, seq_len]
        """
        return AttentionMask.create_causal_mask(seq_len, device)
    
    @staticmethod
    def combine_masks(*masks: Tensor) -> Tensor:
        """
        组合多个掩码
        
        Args:
            *masks: 要组合的掩码张量
            
        Returns:
            Tensor: 组合后的掩码
        """
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = combined_mask * mask
        return combined_mask


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 (Grouped Query Attention, GQA)

    LLaMA2 中使用的注意力机制，是多头注意力和多查询注意力的折中方案。
    将键值头分组，每组共享相同的键值，但有独立的查询头。

    Args:
        d_model (int): 模型维度
        num_heads (int): 查询头数
        num_kv_heads (int): 键值头数（必须能被 num_heads 整除）
        dropout (float): Dropout 概率，默认为 0.1
        bias (bool): 是否在线性层中使用偏置，默认为 False

    Examples:
        >>> # 标准多头注意力：num_heads = num_kv_heads = 8
        >>> # 多查询注意力：num_heads = 8, num_kv_heads = 1
        >>> # 分组查询注意力：num_heads = 8, num_kv_heads = 2
        >>> gqa = GroupedQueryAttention(d_model=512, num_heads=8, num_kv_heads=2)
        >>> x = torch.randn(2, 10, 512)
        >>> output, weights = gqa(x, x, x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
        bias: bool = False
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) 必须能被 num_kv_heads ({num_kv_heads}) 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads  # 每个查询头的维度
        self.d_kv = d_model // num_kv_heads  # 每个键值头的维度
        self.num_queries_per_kv = num_heads // num_kv_heads  # 每个键值头对应的查询头数

        # Q, K, V 的线性变换层
        self.w_q = nn.Linear(d_model, num_heads * self.d_k, bias=bias)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)

        # 输出投影层
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self.apply(lambda module: init_weights(module, init_std=0.02))

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        前向传播

        Args:
            query (Tensor): 查询张量 [batch_size, seq_len, d_model]
            key (Tensor): 键张量 [batch_size, seq_len, d_model]
            value (Tensor): 值张量 [batch_size, seq_len, d_model]
            mask (Optional[Tensor]): 注意力掩码 [batch_size, seq_len, seq_len]
            return_attention (bool): 是否返回注意力权重

        Returns:
            Tuple[Tensor, Optional[Tensor]]: (输出张量, 注意力权重)
        """
        batch_size, seq_len, d_model = query.size()

        # 1. 线性变换得到 Q, K, V
        Q = self.w_q(query)  # [batch_size, seq_len, num_heads * d_k]
        K = self.w_k(key)    # [batch_size, seq_len, num_kv_heads * d_k]
        V = self.w_v(value)  # [batch_size, seq_len, num_kv_heads * d_k]

        # 2. 重塑为多头形状
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)  # [batch_size, num_kv_heads, seq_len, d_k]
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)  # [batch_size, num_kv_heads, seq_len, d_k]

        # 3. 扩展 K 和 V 以匹配查询头数
        # 每个键值头需要复制 num_queries_per_kv 次
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)  # [batch_size, num_heads, seq_len, d_k]
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)  # [batch_size, num_heads, seq_len, d_k]

        # 4. 调整掩码形状以匹配多头
        if mask is not None:
            # 扩展掩码以匹配多头维度
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [batch_size, num_heads, seq_len, seq_len]

        # 5. 计算缩放点积注意力
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout if self.training else None
        )

        # 6. 拼接多头输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )  # [batch_size, seq_len, d_model]

        # 7. 最终的线性投影
        output = self.w_o(attention_output)

        if return_attention:
            return output, attention_weights
        else:
            return output, None

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, d_k={self.d_k}'


class RelativePositionAttention(nn.Module):
    """
    相对位置注意力机制

    在注意力计算中加入相对位置信息。

    Args:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        max_relative_position (int): 最大相对位置距离
        dropout (float): Dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_relative_position: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_relative_position = max_relative_position

        # 相对位置嵌入
        self.relative_position_embeddings = nn.Embedding(
            2 * max_relative_position + 1, self.d_k
        )

        # 标准的多头注意力组件
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        前向传播

        Args:
            query (Tensor): 查询张量
            key (Tensor): 键张量
            value (Tensor): 值张量
            mask (Optional[Tensor]): 注意力掩码

        Returns:
            Tuple[Tensor, Tensor]: (输出张量, 注意力权重)
        """
        # 这里简化实现，实际的相对位置注意力需要更复杂的计算
        # 详细实现可以参考 T5、DeBERTa 等模型
        return self.attention(query, key, value, mask)


# 导出的类列表
__all__ = [
    'MultiHeadAttention',
    'SelfAttention',
    'CrossAttention',
    'GroupedQueryAttention',
    'AttentionMask',
    'RelativePositionAttention',
]
