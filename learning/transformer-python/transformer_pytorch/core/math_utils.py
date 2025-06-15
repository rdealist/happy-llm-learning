"""
数学工具函数模块

提供 Transformer 模型所需的基础数学运算函数，包括：
- 激活函数（GELU、ReLU、Swish 等）
- 注意力计算函数
- 初始化函数
- 数值稳定性工具

所有函数都基于 PyTorch 实现，支持 GPU 加速和自动微分。

作者: Transformer-PyTorch Team
版本: 1.0.0
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def gelu_activation(x: Tensor) -> Tensor:
    """
    GELU (Gaussian Error Linear Unit) 激活函数
    
    实现公式: GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    Args:
        x (Tensor): 输入张量
        
    Returns:
        Tensor: 经过 GELU 激活的张量
        
    Examples:
        >>> x = torch.randn(2, 3)
        >>> output = gelu_activation(x)
        >>> print(output.shape)  # torch.Size([2, 3])
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def swish_activation(x: Tensor) -> Tensor:
    """
    Swish 激活函数
    
    实现公式: Swish(x) = x * sigmoid(x)
    
    Args:
        x (Tensor): 输入张量
        
    Returns:
        Tensor: 经过 Swish 激活的张量
    """
    return x * torch.sigmoid(x)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
    temperature: float = 1.0
) -> Tuple[Tensor, Tensor]:
    """
    缩放点积注意力机制
    
    实现公式: Attention(Q,K,V) = softmax(QK^T / √d_k) V
    
    Args:
        query (Tensor): 查询张量 [batch_size, seq_len, d_model] 或 [batch_size, num_heads, seq_len, d_k]
        key (Tensor): 键张量，形状与 query 相同
        value (Tensor): 值张量，形状与 query 相同
        mask (Optional[Tensor]): 注意力掩码 [batch_size, seq_len, seq_len]
        dropout (Optional[nn.Dropout]): Dropout 层
        temperature (float): 温度参数，用于控制注意力分布的尖锐程度
        
    Returns:
        Tuple[Tensor, Tensor]: (注意力输出, 注意力权重)
        
    Examples:
        >>> batch_size, seq_len, d_model = 2, 10, 64
        >>> q = torch.randn(batch_size, seq_len, d_model)
        >>> k = torch.randn(batch_size, seq_len, d_model)
        >>> v = torch.randn(batch_size, seq_len, d_model)
        >>> output, attention_weights = scaled_dot_product_attention(q, k, v)
        >>> print(output.shape)  # torch.Size([2, 10, 64])
        >>> print(attention_weights.shape)  # torch.Size([2, 10, 10])
    """
    d_k = query.size(-1)
    
    # 计算注意力分数: Q @ K^T / √d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / (math.sqrt(d_k) * temperature)
    
    # 应用掩码（如果提供）
    if mask is not None:
        # 将掩码为 0 的位置设置为负无穷，这样 softmax 后会变成 0
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 计算注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    
    # 应用 dropout（如果提供）
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # 计算注意力输出
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


def create_causal_mask(seq_len: int, device: torch.device = None) -> Tensor:
    """
    创建因果掩码（下三角掩码）
    
    用于解码器的自注意力，确保每个位置只能看到之前的位置。
    
    Args:
        seq_len (int): 序列长度
        device (torch.device, optional): 设备
        
    Returns:
        Tensor: 因果掩码 [seq_len, seq_len]，下三角为 1，上三角为 0
        
    Examples:
        >>> mask = create_causal_mask(4)
        >>> print(mask)
        tensor([[1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1]])
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def create_padding_mask(
    token_ids: Tensor,
    pad_token_id: int = 0
) -> Tensor:
    """
    创建填充掩码
    
    用于忽略填充位置的注意力计算。
    
    Args:
        token_ids (Tensor): 词元ID张量 [batch_size, seq_len]
        pad_token_id (int): 填充词元的ID
        
    Returns:
        Tensor: 填充掩码 [batch_size, seq_len]，非填充位置为 1，填充位置为 0
        
    Examples:
        >>> token_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        >>> mask = create_padding_mask(token_ids, pad_token_id=0)
        >>> print(mask)
        tensor([[1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0]])
    """
    return (token_ids != pad_token_id).long()


def create_attention_mask(
    src_mask: Optional[Tensor] = None,
    tgt_mask: Optional[Tensor] = None,
    src_len: Optional[int] = None,
    tgt_len: Optional[int] = None,
    causal: bool = False,
    device: torch.device = None
) -> Tensor:
    """
    创建注意力掩码
    
    Args:
        src_mask (Optional[Tensor]): 源序列掩码 [batch_size, src_len]
        tgt_mask (Optional[Tensor]): 目标序列掩码 [batch_size, tgt_len]
        src_len (Optional[int]): 源序列长度
        tgt_len (Optional[int]): 目标序列长度
        causal (bool): 是否使用因果掩码
        device (torch.device): 设备
        
    Returns:
        Tensor: 注意力掩码
    """
    if src_mask is not None and tgt_mask is not None:
        # 交叉注意力掩码 [batch_size, tgt_len, src_len]
        batch_size = src_mask.size(0)
        mask = tgt_mask.unsqueeze(2) * src_mask.unsqueeze(1)
        return mask
    
    elif tgt_mask is not None:
        # 自注意力掩码
        batch_size, seq_len = tgt_mask.shape
        mask = tgt_mask.unsqueeze(1) * tgt_mask.unsqueeze(2)
        
        if causal:
            # 应用因果掩码
            causal_mask = create_causal_mask(seq_len, device)
            mask = mask * causal_mask.unsqueeze(0)
        
        return mask
    
    elif causal and tgt_len is not None:
        # 仅因果掩码
        return create_causal_mask(tgt_len, device).unsqueeze(0)
    
    else:
        return None


def init_weights(module: nn.Module, init_std: float = 0.02) -> None:
    """
    初始化模型权重
    
    Args:
        module (nn.Module): 要初始化的模块
        init_std (float): 初始化标准差
    """
    if isinstance(module, nn.Linear):
        # 线性层权重使用正态分布初始化
        torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # 嵌入层权重使用正态分布初始化
        torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
    elif isinstance(module, nn.LayerNorm):
        # 层归一化参数初始化
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)


def get_activation_function(activation: str) -> nn.Module:
    """
    根据名称获取激活函数
    
    Args:
        activation (str): 激活函数名称
        
    Returns:
        nn.Module: 激活函数模块
        
    Raises:
        ValueError: 不支持的激活函数
    """
    activation = activation.lower()
    
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'swish' or activation == 'silu':
        return nn.SiLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"不支持的激活函数: {activation}")


def compute_model_size(model: nn.Module) -> dict:
    """
    计算模型大小和参数统计
    
    Args:
        model (nn.Module): PyTorch 模型
        
    Returns:
        dict: 包含模型大小信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算模型大小（假设每个参数 4 字节）
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'total_parameters_M': f"{total_params / 1e6:.2f}M",
        'trainable_parameters_M': f"{trainable_params / 1e6:.2f}M",
    }


def label_smoothing_loss(
    logits: Tensor,
    targets: Tensor,
    smoothing: float = 0.1,
    ignore_index: int = -100
) -> Tensor:
    """
    标签平滑损失函数
    
    Args:
        logits (Tensor): 模型输出 [batch_size, seq_len, vocab_size]
        targets (Tensor): 目标标签 [batch_size, seq_len]
        smoothing (float): 平滑参数
        ignore_index (int): 忽略的索引
        
    Returns:
        Tensor: 损失值
    """
    vocab_size = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 创建平滑标签
    smooth_targets = torch.zeros_like(log_probs)
    smooth_targets.fill_(smoothing / (vocab_size - 1))
    smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - smoothing)
    
    # 计算损失
    loss = -smooth_targets * log_probs
    
    # 处理忽略索引
    if ignore_index >= 0:
        mask = (targets != ignore_index).float()
        loss = loss.sum(dim=-1) * mask
        return loss.sum() / mask.sum()
    else:
        return loss.sum(dim=-1).mean()


# 导出的函数列表
__all__ = [
    'gelu_activation',
    'swish_activation',
    'scaled_dot_product_attention',
    'create_causal_mask',
    'create_padding_mask',
    'create_attention_mask',
    'init_weights',
    'get_activation_function',
    'compute_model_size',
    'label_smoothing_loss',
]
