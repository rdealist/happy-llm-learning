"""
Transformer-PyTorch 配置模块

提供模型配置管理功能：
- 预设配置（small, default, large 等）
- 自定义配置创建和验证
- 常量定义
- 配置工具函数
"""

from .config import *
from .constants import *

__all__ = [
    # 配置类和函数
    'TransformerConfig',
    'get_config',
    'create_config',
    'validate_config',
    'print_config',
    'estimate_parameters',
    
    # 预设配置
    'SMALL_CONFIG',
    'DEFAULT_CONFIG', 
    'LARGE_CONFIG',
    'MINIPROGRAM_CONFIG',
    
    # 常量
    'SPECIAL_TOKENS',
    'SPECIAL_TOKEN_IDS',
    'ACTIVATION_FUNCTIONS',
    'ATTENTION_TYPES',
    'POSITION_ENCODING_TYPES',
    'MODEL_ARCHITECTURES',
]
