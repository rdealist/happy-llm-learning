"""
解码器模块

实现 Transformer 解码器层和解码器块：
- 解码器层 (DecoderLayer)
- Transformer 解码器 (TransformerDecoder)

解码器层包含掩码自注意力、交叉注意力和前馈网络，以及残差连接和层归一化。

作者: Transformer-PyTorch Team
版本: 1.0.0
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .attention import CrossAttention, SelfAttention, AttentionMask
from .layers import FeedForward, LayerNorm
from .math_utils import init_weights


class DecoderLayer(nn.Module):
    """
    Transformer 解码器层
    
    实现单个解码器层，包含：
    1. 掩码多头自注意力机制
    2. 残差连接和层归一化
    3. 多头交叉注意力机制（编码器-解码器注意力）
    4. 残差连接和层归一化
    5. 位置相关的前馈网络
    6. 残差连接和层归一化
    
    Args:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        d_ff (int): 前馈网络隐藏层维度
        dropout (float): Dropout 概率，默认为 0.1
        activation (str): 激活函数类型，默认为 'relu'
        norm_first (bool): 是否使用 Pre-LN 结构，默认为 True
        
    Examples:
        >>> layer = DecoderLayer(d_model=512, num_heads=8, d_ff=2048)
        >>> tgt = torch.randn(2, 10, 512)
        >>> memory = torch.randn(2, 15, 512)
        >>> output = layer(tgt, memory)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        norm_first: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.norm_first = norm_first
        
        # 掩码自注意力
        self.self_attention = SelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 交叉注意力（编码器-解码器注意力）
        self.cross_attention = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 前馈网络
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout
        )
        
        # 层归一化
        self.norm1 = LayerNorm(d_model)  # 自注意力后
        self.norm2 = LayerNorm(d_model)  # 交叉注意力后
        self.norm3 = LayerNorm(d_model)  # 前馈网络后
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # 初始化权重
        self.apply(lambda module: init_weights(module, init_std=0.02))
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        前向传播
        
        Args:
            tgt (Tensor): 目标序列张量 [batch_size, tgt_len, d_model]
            memory (Tensor): 编码器输出（记忆）[batch_size, src_len, d_model]
            tgt_mask (Optional[Tensor]): 目标序列掩码（因果掩码）
            memory_mask (Optional[Tensor]): 记忆掩码
            return_attention (bool): 是否返回注意力权重
            
        Returns:
            Tuple[Tensor, Optional[Dict[str, Tensor]]]: (输出张量, 注意力权重字典)
        """
        attention_weights = {} if return_attention else None
        
        if self.norm_first:
            # Pre-LN 结构
            
            # 第一个子层：掩码自注意力
            norm_tgt = self.norm1(tgt)
            self_attn_output, self_attn_weights = self.self_attention(
                norm_tgt, mask=tgt_mask, return_attention=True
            )
            tgt = tgt + self.dropout1(self_attn_output)
            
            # 第二个子层：交叉注意力
            norm_tgt = self.norm2(tgt)
            cross_attn_output, cross_attn_weights = self.cross_attention(
                norm_tgt, memory, mask=memory_mask, return_attention=True
            )
            tgt = tgt + self.dropout2(cross_attn_output)
            
            # 第三个子层：前馈网络
            norm_tgt = self.norm3(tgt)
            ff_output = self.feed_forward(norm_tgt)
            tgt = tgt + self.dropout3(ff_output)
            
        else:
            # Post-LN 结构
            
            # 第一个子层：掩码自注意力
            self_attn_output, self_attn_weights = self.self_attention(
                tgt, mask=tgt_mask, return_attention=True
            )
            tgt = self.norm1(tgt + self.dropout1(self_attn_output))
            
            # 第二个子层：交叉注意力
            cross_attn_output, cross_attn_weights = self.cross_attention(
                tgt, memory, mask=memory_mask, return_attention=True
            )
            tgt = self.norm2(tgt + self.dropout2(cross_attn_output))
            
            # 第三个子层：前馈网络
            ff_output = self.feed_forward(tgt)
            tgt = self.norm3(tgt + self.dropout3(ff_output))
        
        # 收集注意力权重
        if return_attention:
            attention_weights['self_attention'] = self_attn_weights
            attention_weights['cross_attention'] = cross_attn_weights
        
        return tgt, attention_weights
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, num_heads={self.num_heads}, d_ff={self.d_ff}, norm_first={self.norm_first}'


class TransformerDecoder(nn.Module):
    """
    Transformer 解码器
    
    由多个解码器层堆叠而成，可选择性地在最后添加层归一化。
    
    Args:
        decoder_layer (DecoderLayer): 解码器层实例
        num_layers (int): 解码器层数
        norm (Optional[nn.Module]): 最终的归一化层，默认为 None
        
    Examples:
        >>> decoder_layer = DecoderLayer(d_model=512, num_heads=8, d_ff=2048)
        >>> decoder = TransformerDecoder(decoder_layer, num_layers=6)
        >>> tgt = torch.randn(2, 10, 512)
        >>> memory = torch.randn(2, 15, 512)
        >>> output = decoder(tgt, memory)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        decoder_layer: DecoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # 复制解码器层
        self.layers = nn.ModuleList([
            self._get_cloned_layer(decoder_layer) for _ in range(num_layers)
        ])
        
        # 最终的层归一化
        self.norm = norm
    
    def _get_cloned_layer(self, layer: nn.Module) -> nn.Module:
        """克隆解码器层"""
        import copy
        return copy.deepcopy(layer)
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        return_all_layers: bool = False,
        return_attention: bool = False
    ) -> Tensor:
        """
        前向传播
        
        Args:
            tgt (Tensor): 目标序列张量 [batch_size, tgt_len, d_model]
            memory (Tensor): 编码器输出 [batch_size, src_len, d_model]
            tgt_mask (Optional[Tensor]): 目标序列掩码
            memory_mask (Optional[Tensor]): 记忆掩码
            return_all_layers (bool): 是否返回所有层的输出
            return_attention (bool): 是否返回注意力权重
            
        Returns:
            Tensor: 解码器输出 [batch_size, tgt_len, d_model]
            或者包含所有层输出和注意力权重的字典
        """
        output = tgt
        all_layer_outputs = []
        all_self_attention_weights = []
        all_cross_attention_weights = []
        
        # 如果没有提供目标掩码，自动生成因果掩码
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = AttentionMask.create_causal_mask(tgt_len, device=tgt.device)
            tgt_mask = tgt_mask.unsqueeze(0).expand(tgt.size(0), -1, -1)
        
        # 逐层前向传播
        for layer in self.layers:
            output, attention_weights = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                return_attention=return_attention
            )
            
            if return_all_layers:
                all_layer_outputs.append(output)
            
            if return_attention and attention_weights is not None:
                all_self_attention_weights.append(attention_weights['self_attention'])
                all_cross_attention_weights.append(attention_weights['cross_attention'])
        
        # 最终的层归一化
        if self.norm is not None:
            output = self.norm(output)
        
        # 根据返回选项组织输出
        if return_all_layers or return_attention:
            result = {'last_hidden_state': output}
            
            if return_all_layers:
                result['all_hidden_states'] = all_layer_outputs
            
            if return_attention:
                result['all_self_attention_weights'] = all_self_attention_weights
                result['all_cross_attention_weights'] = all_cross_attention_weights
            
            return result
        else:
            return output
    
    def incremental_forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        cache: Optional[Dict] = None,
        memory_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict]:
        """
        增量前向传播（用于自回归生成）
        
        Args:
            tgt (Tensor): 当前步的目标序列
            memory (Tensor): 编码器输出
            cache (Optional[Dict]): 缓存的键值对
            memory_mask (Optional[Tensor]): 记忆掩码
            
        Returns:
            Tuple[Tensor, Dict]: (输出张量, 更新后的缓存)
        """
        if cache is None:
            cache = {}
        
        output = tgt
        new_cache = {}
        
        for i, layer in enumerate(self.layers):
            layer_cache = cache.get(f'layer_{i}', {})
            
            # 简化的增量计算（实际实现需要更复杂的键值缓存）
            output, _ = layer(
                output, memory,
                tgt_mask=None,  # 增量计算时不需要掩码
                memory_mask=memory_mask,
                return_attention=False
            )
            
            new_cache[f'layer_{i}'] = layer_cache
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output, new_cache


def create_decoder(
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_layers: int,
    dropout: float = 0.1,
    activation: str = 'relu',
    norm_first: bool = True,
    final_norm: bool = True
) -> TransformerDecoder:
    """
    创建 Transformer 解码器的便捷函数
    
    Args:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        d_ff (int): 前馈网络隐藏层维度
        num_layers (int): 解码器层数
        dropout (float): Dropout 概率
        activation (str): 激活函数类型
        norm_first (bool): 是否使用 Pre-LN 结构
        final_norm (bool): 是否在最后添加层归一化
        
    Returns:
        TransformerDecoder: 解码器实例
    """
    # 创建解码器层
    decoder_layer = DecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        norm_first=norm_first
    )
    
    # 创建最终的层归一化（如果需要）
    norm = LayerNorm(d_model) if final_norm else None
    
    # 创建解码器
    decoder = TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=num_layers,
        norm=norm
    )
    
    return decoder


# 导出的类和函数列表
__all__ = [
    'DecoderLayer',
    'TransformerDecoder',
    'create_decoder',
]
