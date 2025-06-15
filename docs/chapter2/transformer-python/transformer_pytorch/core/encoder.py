"""
编码器模块

实现 Transformer 编码器层和编码器块：
- 编码器层 (EncoderLayer)
- Transformer 编码器 (TransformerEncoder)

编码器层包含多头自注意力和前馈网络，以及残差连接和层归一化。

作者: Transformer-PyTorch Team
版本: 1.0.0
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention, SelfAttention
from .layers import FeedForward, LayerNorm, ResidualConnection
from .math_utils import init_weights


class EncoderLayer(nn.Module):
    """
    Transformer 编码器层
    
    实现单个编码器层，包含：
    1. 多头自注意力机制
    2. 残差连接和层归一化
    3. 位置相关的前馈网络
    4. 残差连接和层归一化
    
    Args:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        d_ff (int): 前馈网络隐藏层维度
        dropout (float): Dropout 概率，默认为 0.1
        activation (str): 激活函数类型，默认为 'relu'
        norm_first (bool): 是否使用 Pre-LN 结构，默认为 True
        
    Examples:
        >>> layer = EncoderLayer(d_model=512, num_heads=8, d_ff=2048)
        >>> x = torch.randn(2, 10, 512)
        >>> output, attention = layer(x)
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
        
        # 多头自注意力
        self.self_attention = SelfAttention(
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
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 初始化权重
        self.apply(lambda module: init_weights(module, init_std=0.02))
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量 [batch_size, seq_len, d_model]
            mask (Optional[Tensor]): 注意力掩码 [batch_size, seq_len, seq_len]
            return_attention (bool): 是否返回注意力权重
            
        Returns:
            Tuple[Tensor, Optional[Tensor]]: (输出张量, 注意力权重)
        """
        if self.norm_first:
            # Pre-LN 结构: LayerNorm -> SubLayer -> Dropout -> Residual
            
            # 第一个子层：自注意力
            norm_x = self.norm1(x)
            attn_output, attention_weights = self.self_attention(
                norm_x, mask=mask, return_attention=True
            )
            x = x + self.dropout1(attn_output)
            
            # 第二个子层：前馈网络
            norm_x = self.norm2(x)
            ff_output = self.feed_forward(norm_x)
            x = x + self.dropout2(ff_output)
            
        else:
            # Post-LN 结构: SubLayer -> Dropout -> Residual -> LayerNorm
            
            # 第一个子层：自注意力
            attn_output, attention_weights = self.self_attention(
                x, mask=mask, return_attention=True
            )
            x = self.norm1(x + self.dropout1(attn_output))
            
            # 第二个子层：前馈网络
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_output))
        
        if return_attention:
            return x, attention_weights
        else:
            return x, None
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, num_heads={self.num_heads}, d_ff={self.d_ff}, norm_first={self.norm_first}'


class TransformerEncoder(nn.Module):
    """
    Transformer 编码器
    
    由多个编码器层堆叠而成，可选择性地在最后添加层归一化。
    
    Args:
        encoder_layer (EncoderLayer): 编码器层实例
        num_layers (int): 编码器层数
        norm (Optional[nn.Module]): 最终的归一化层，默认为 None
        
    Examples:
        >>> encoder_layer = EncoderLayer(d_model=512, num_heads=8, d_ff=2048)
        >>> encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> x = torch.randn(2, 10, 512)
        >>> output = encoder(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        encoder_layer: EncoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # 复制编码器层
        self.layers = nn.ModuleList([
            self._get_cloned_layer(encoder_layer) for _ in range(num_layers)
        ])
        
        # 最终的层归一化
        self.norm = norm
    
    def _get_cloned_layer(self, layer: nn.Module) -> nn.Module:
        """克隆编码器层"""
        import copy
        return copy.deepcopy(layer)
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_all_layers: bool = False,
        return_attention: bool = False
    ) -> Tensor:
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量 [batch_size, seq_len, d_model]
            mask (Optional[Tensor]): 注意力掩码
            return_all_layers (bool): 是否返回所有层的输出
            return_attention (bool): 是否返回注意力权重
            
        Returns:
            Tensor: 编码器输出 [batch_size, seq_len, d_model]
            或者包含所有层输出和注意力权重的字典
        """
        output = x
        all_layer_outputs = []
        all_attention_weights = []
        
        # 逐层前向传播
        for layer in self.layers:
            output, attention_weights = layer(
                output, mask=mask, return_attention=return_attention
            )
            
            if return_all_layers:
                all_layer_outputs.append(output)
            
            if return_attention and attention_weights is not None:
                all_attention_weights.append(attention_weights)
        
        # 最终的层归一化
        if self.norm is not None:
            output = self.norm(output)
        
        # 根据返回选项组织输出
        if return_all_layers or return_attention:
            result = {'last_hidden_state': output}
            
            if return_all_layers:
                result['all_hidden_states'] = all_layer_outputs
            
            if return_attention:
                result['all_attention_weights'] = all_attention_weights
            
            return result
        else:
            return output
    
    def get_layer_output(
        self,
        x: Tensor,
        layer_idx: int,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        获取指定层的输出
        
        Args:
            x (Tensor): 输入张量
            layer_idx (int): 层索引（0-based）
            mask (Optional[Tensor]): 注意力掩码
            
        Returns:
            Tensor: 指定层的输出
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"层索引 {layer_idx} 超出范围 [0, {self.num_layers - 1}]")
        
        output = x
        for i, layer in enumerate(self.layers):
            output, _ = layer(output, mask=mask, return_attention=False)
            if i == layer_idx:
                return output
        
        return output


def create_encoder(
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_layers: int,
    dropout: float = 0.1,
    activation: str = 'relu',
    norm_first: bool = True,
    final_norm: bool = True
) -> TransformerEncoder:
    """
    创建 Transformer 编码器的便捷函数
    
    Args:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        d_ff (int): 前馈网络隐藏层维度
        num_layers (int): 编码器层数
        dropout (float): Dropout 概率
        activation (str): 激活函数类型
        norm_first (bool): 是否使用 Pre-LN 结构
        final_norm (bool): 是否在最后添加层归一化
        
    Returns:
        TransformerEncoder: 编码器实例
        
    Examples:
        >>> encoder = create_encoder(
        ...     d_model=512, num_heads=8, d_ff=2048, num_layers=6
        ... )
        >>> x = torch.randn(2, 10, 512)
        >>> output = encoder(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    # 创建编码器层
    encoder_layer = EncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        norm_first=norm_first
    )
    
    # 创建最终的层归一化（如果需要）
    norm = LayerNorm(d_model) if final_norm else None
    
    # 创建编码器
    encoder = TransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=num_layers,
        norm=norm
    )
    
    return encoder


# 导出的类和函数列表
__all__ = [
    'EncoderLayer',
    'TransformerEncoder',
    'create_encoder',
]
