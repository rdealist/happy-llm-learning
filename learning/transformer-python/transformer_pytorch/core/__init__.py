"""
Transformer-PyTorch 核心模块

包含 Transformer 架构的所有核心组件实现：
- 数学工具函数
- 基础神经网络层
- 注意力机制
- 嵌入层
- 编码器和解码器
- 完整的 Transformer 模型

所有组件都基于 PyTorch 实现，支持 GPU 加速和自动微分。
"""

from .math_utils import *
from .layers import *
from .attention import *
from .embedding import *
from .encoder import *
from .decoder import *
from .transformer import *

__all__ = [
    # 数学工具
    'gelu_activation',
    'scaled_dot_product_attention',
    
    # 基础层
    'LayerNorm',
    'FeedForward',
    'Dropout',
    
    # 注意力机制
    'MultiHeadAttention',
    'SelfAttention',
    'CrossAttention',
    'AttentionMask',
    
    # 嵌入层
    'TokenEmbedding',
    'PositionalEncoding',
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEncoding',
    'TransformerEmbedding',
    
    # 编码器
    'EncoderLayer',
    'TransformerEncoder',
    
    # 解码器
    'DecoderLayer',
    'TransformerDecoder',
    
    # 完整模型
    'Transformer',
    'TransformerForSequenceToSequence',
    'TransformerForLanguageModeling',
]
