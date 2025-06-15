"""
Transformer-PyTorch: 基于 PyTorch 的 Transformer 架构实现

这是一个完整的 Transformer 实现，基于《第二章 Transformer架构》文档，
提供了模块化、易于理解和使用的 Transformer 组件。

主要特性:
- 🧩 模块化设计 - 每个组件独立实现，支持按需使用
- 🚀 PyTorch 优化 - 充分利用 PyTorch 的自动微分和 GPU 加速
- 📚 详细注释 - 所有代码都有中文注释和详细文档
- 🔧 灵活配置 - 支持多种预设配置和自定义参数
- 🎯 教育友好 - 代码结构清晰，便于学习和理解
- ⚡ 高性能 - 支持 GPU 加速和批处理计算

作者: shihom_wu
版本: 1.0.0
许可: MIT License
"""

__version__ = "1.0.0"
__author__ = "shihom_wu"
__email__ = "transformer-pytorch@example.com"
__license__ = "MIT"

# 导入核心组件
from .core.math_utils import *
from .core.layers import *
from .core.attention import *
from .core.embedding import *
from .core.encoder import *
from .core.decoder import *
from .core.transformer import *

# 导入配置
from .config.config import *
from .config.constants import *

# 版本信息
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'version': __version__,
    'pytorch_version': None,  # 将在运行时检测
    'cuda_available': None,   # 将在运行时检测
}

def get_version_info():
    """
    获取版本信息和环境信息
    
    Returns:
        dict: 包含版本和环境信息的字典
    """
    import torch
    
    VERSION_INFO['pytorch_version'] = torch.__version__
    VERSION_INFO['cuda_available'] = torch.cuda.is_available()
    
    if torch.cuda.is_available():
        VERSION_INFO['cuda_version'] = torch.version.cuda
        VERSION_INFO['gpu_count'] = torch.cuda.device_count()
        VERSION_INFO['current_device'] = torch.cuda.current_device()
    
    return VERSION_INFO

def print_version_info():
    """打印版本信息"""
    info = get_version_info()
    print(f"Transformer-PyTorch v{info['version']}")
    print(f"PyTorch v{info['pytorch_version']}")
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA Version: {info.get('cuda_version', 'Unknown')}")
        print(f"GPU Count: {info.get('gpu_count', 0)}")

# 设置默认的随机种子以确保可重现性
def set_seed(seed: int = 42):
    """
    设置随机种子以确保结果可重现
    
    Args:
        seed (int): 随机种子值
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保 CUDA 操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 自动检测和设置设备
def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    自动检测并返回最佳可用设备
    
    Args:
        prefer_gpu (bool): 是否优先使用 GPU
        
    Returns:
        torch.device: 设备对象
    """
    import torch
    
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用 GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    
    return device

# 内存使用监控
def get_memory_usage():
    """
    获取当前内存使用情况
    
    Returns:
        dict: 内存使用信息
    """
    import torch
    import psutil
    
    memory_info = {
        'cpu_memory_percent': psutil.virtual_memory().percent,
        'cpu_memory_available_gb': psutil.virtual_memory().available / (1024**3),
    }
    
    if torch.cuda.is_available():
        memory_info.update({
            'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
        })
    
    return memory_info

def print_memory_usage():
    """打印内存使用情况"""
    memory = get_memory_usage()
    print(f"CPU 内存使用: {memory['cpu_memory_percent']:.1f}%")
    print(f"CPU 可用内存: {memory['cpu_memory_available_gb']:.2f} GB")
    
    if 'gpu_memory_allocated_gb' in memory:
        print(f"GPU 已分配内存: {memory['gpu_memory_allocated_gb']:.2f} GB")
        print(f"GPU 保留内存: {memory['gpu_memory_reserved_gb']:.2f} GB")
        print(f"GPU 总内存: {memory['gpu_memory_total_gb']:.2f} GB")

# 模型大小计算
def count_parameters(model) -> dict:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        dict: 参数统计信息
    """
    import torch.nn as nn
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'total_parameters_M': f"{total_params / 1e6:.2f}M",
        'trainable_parameters_M': f"{trainable_params / 1e6:.2f}M",
    }

def print_model_info(model):
    """打印模型信息"""
    params = count_parameters(model)
    print(f"模型参数统计:")
    print(f"  总参数: {params['total_parameters_M']}")
    print(f"  可训练参数: {params['trainable_parameters_M']}")
    print(f"  不可训练参数: {params['non_trainable_parameters']:,}")

# 导出的公共接口
__all__ = [
    # 版本和环境
    '__version__',
    'get_version_info',
    'print_version_info',
    'set_seed',
    'get_device',
    
    # 内存和模型工具
    'get_memory_usage',
    'print_memory_usage',
    'count_parameters',
    'print_model_info',
    
    # 核心组件 (将在各模块中定义)
    'TransformerConfig',
    'Transformer',
    'TransformerEncoder',
    'TransformerDecoder',
    'MultiHeadAttention',
    'TransformerEmbedding',
    'PositionalEncoding',
]

# 初始化时的欢迎信息
def _welcome_message():
    """显示欢迎信息"""
    print("🚀 欢迎使用 Transformer-PyTorch!")
    print("📚 基于《第二章 Transformer架构》的完整实现")
    print("💡 使用 help(transformer_pytorch) 查看更多信息")

# 可选的欢迎信息（仅在交互式环境中显示）
import sys
if hasattr(sys, 'ps1'):  # 检查是否在交互式环境中
    _welcome_message()
