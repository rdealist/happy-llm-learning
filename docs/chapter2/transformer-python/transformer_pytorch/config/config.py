"""
Transformer 模型配置模块

提供模型配置管理功能：
- 配置类定义
- 预设配置
- 配置验证和工具函数

作者: Transformer-PyTorch Team
版本: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json


@dataclass
class TransformerConfig:
    """
    Transformer 模型配置类
    
    包含模型的所有超参数和配置选项。
    
    Attributes:
        vocab_size (int): 词汇表大小
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        num_encoder_layers (int): 编码器层数
        num_decoder_layers (int): 解码器层数
        d_ff (int): 前馈网络隐藏层维度
        max_seq_len (int): 最大序列长度
        dropout (float): Dropout 概率
        attention_dropout (float): 注意力 Dropout 概率
        activation (str): 激活函数类型
        norm_first (bool): 是否使用 Pre-LN 结构
        tie_word_embeddings (bool): 是否共享输入输出嵌入权重
        position_encoding_type (str): 位置编码类型
        scale_embedding (bool): 是否缩放嵌入
        init_std (float): 权重初始化标准差
        layer_norm_eps (float): 层归一化的 epsilon
        pad_token_id (int): 填充词元ID
        bos_token_id (int): 序列开始词元ID
        eos_token_id (int): 序列结束词元ID
        use_cache (bool): 是否使用缓存（用于生成）
    """
    
    # 模型基础参数
    vocab_size: int = 30000
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 512
    
    # 正则化参数
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # 激活函数和归一化
    activation: str = 'relu'
    norm_first: bool = True
    layer_norm_eps: float = 1e-6
    
    # 嵌入相关
    tie_word_embeddings: bool = False
    position_encoding_type: str = 'sinusoidal'  # 'sinusoidal' or 'learned'
    scale_embedding: bool = True
    
    # 初始化参数
    init_std: float = 0.02
    
    # 特殊词元
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # 其他配置
    use_cache: bool = True
    
    def __post_init__(self):
        """初始化后的验证"""
        self.validate()
    
    def validate(self) -> None:
        """验证配置参数的有效性"""
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) 必须能被 num_heads ({self.num_heads}) 整除"
            )
        
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size 必须大于 0，当前值: {self.vocab_size}")
        
        if self.num_encoder_layers < 0 or self.num_decoder_layers < 0:
            raise ValueError("编码器和解码器层数必须非负")
        
        if not (0 <= self.dropout <= 1):
            raise ValueError(f"dropout 必须在 [0, 1] 范围内，当前值: {self.dropout}")
        
        if not (0 <= self.attention_dropout <= 1):
            raise ValueError(f"attention_dropout 必须在 [0, 1] 范围内，当前值: {self.attention_dropout}")
        
        if self.activation not in ['relu', 'gelu', 'swish', 'silu']:
            raise ValueError(f"不支持的激活函数: {self.activation}")
        
        if self.position_encoding_type not in ['sinusoidal', 'learned']:
            raise ValueError(f"不支持的位置编码类型: {self.position_encoding_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransformerConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def save_to_file(self, file_path: str) -> None:
        """保存配置到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'TransformerConfig':
        """从文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def estimate_parameters(self) -> Dict[str, Any]:
        """估算模型参数量"""
        # 嵌入层参数
        token_embedding_params = self.vocab_size * self.d_model
        
        if self.position_encoding_type == 'learned':
            position_embedding_params = self.max_seq_len * self.d_model
        else:
            position_embedding_params = 0
        
        # 编码器参数
        encoder_layer_params = (
            4 * self.d_model * self.d_model +  # Q, K, V, O 矩阵
            2 * self.d_model * self.d_ff +     # FFN 两层
            4 * self.d_model                   # 层归一化参数
        )
        encoder_params = self.num_encoder_layers * encoder_layer_params
        
        # 解码器参数
        decoder_layer_params = (
            6 * self.d_model * self.d_model +  # 自注意力 + 交叉注意力
            2 * self.d_model * self.d_ff +     # FFN 两层
            6 * self.d_model                   # 层归一化参数
        )
        decoder_params = self.num_decoder_layers * decoder_layer_params
        
        # 输出层参数
        if self.tie_word_embeddings:
            output_params = 0
        else:
            output_params = self.d_model * self.vocab_size
        
        # 总参数量
        total_params = (
            token_embedding_params +
            position_embedding_params +
            encoder_params +
            decoder_params +
            output_params
        )
        
        return {
            'token_embedding': token_embedding_params,
            'position_embedding': position_embedding_params,
            'encoder': encoder_params,
            'decoder': decoder_params,
            'output_layer': output_params,
            'total': total_params,
            'total_M': f"{total_params / 1e6:.2f}M"
        }


# 预设配置
SMALL_CONFIG = TransformerConfig(
    vocab_size=10000,
    d_model=256,
    num_heads=4,
    num_encoder_layers=4,
    num_decoder_layers=4,
    d_ff=1024,
    max_seq_len=128,
    dropout=0.1,
)

DEFAULT_CONFIG = TransformerConfig(
    vocab_size=30000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_seq_len=512,
    dropout=0.1,
)

LARGE_CONFIG = TransformerConfig(
    vocab_size=50000,
    d_model=1024,
    num_heads=16,
    num_encoder_layers=12,
    num_decoder_layers=12,
    d_ff=4096,
    max_seq_len=1024,
    dropout=0.1,
)

MINIPROGRAM_CONFIG = TransformerConfig(
    vocab_size=8000,
    d_model=128,
    num_heads=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    d_ff=512,
    max_seq_len=64,
    dropout=0.05,
)

# 配置字典
PRESET_CONFIGS = {
    'small': SMALL_CONFIG,
    'default': DEFAULT_CONFIG,
    'large': LARGE_CONFIG,
    'miniprogram': MINIPROGRAM_CONFIG,
}


def get_config(config_name: str = 'default') -> TransformerConfig:
    """
    获取预定义配置
    
    Args:
        config_name (str): 配置名称
        
    Returns:
        TransformerConfig: 配置对象
        
    Raises:
        ValueError: 未知的配置名称
    """
    if config_name not in PRESET_CONFIGS:
        available_configs = list(PRESET_CONFIGS.keys())
        raise ValueError(
            f"未知的配置名称: {config_name}. "
            f"可用配置: {available_configs}"
        )
    
    return PRESET_CONFIGS[config_name]


def create_config(
    base_config: Optional[TransformerConfig] = None,
    **overrides
) -> TransformerConfig:
    """
    创建自定义配置
    
    Args:
        base_config (Optional[TransformerConfig]): 基础配置
        **overrides: 覆盖参数
        
    Returns:
        TransformerConfig: 新的配置对象
    """
    if base_config is None:
        base_config = DEFAULT_CONFIG
    
    # 获取基础配置的字典
    config_dict = base_config.to_dict()
    
    # 应用覆盖参数
    config_dict.update(overrides)
    
    # 创建新配置
    return TransformerConfig.from_dict(config_dict)


def print_config(config: TransformerConfig) -> None:
    """
    打印配置信息
    
    Args:
        config (TransformerConfig): 配置对象
    """
    print("=== Transformer 模型配置 ===")
    print(f"词汇表大小: {config.vocab_size:,}")
    print(f"模型维度: {config.d_model}")
    print(f"注意力头数: {config.num_heads}")
    print(f"编码器层数: {config.num_encoder_layers}")
    print(f"解码器层数: {config.num_decoder_layers}")
    print(f"前馈网络维度: {config.d_ff}")
    print(f"最大序列长度: {config.max_seq_len}")
    print(f"Dropout: {config.dropout}")
    print(f"激活函数: {config.activation}")
    print(f"位置编码: {config.position_encoding_type}")
    
    # 参数估算
    params = config.estimate_parameters()
    print(f"估算参数量: {params['total_M']}")
    print("=" * 30)


def validate_config(config: TransformerConfig) -> bool:
    """
    验证配置
    
    Args:
        config (TransformerConfig): 配置对象
        
    Returns:
        bool: 配置是否有效
    """
    try:
        config.validate()
        return True
    except ValueError as e:
        print(f"配置验证失败: {e}")
        return False


# 导出的类和函数
__all__ = [
    'TransformerConfig',
    'SMALL_CONFIG',
    'DEFAULT_CONFIG',
    'LARGE_CONFIG',
    'MINIPROGRAM_CONFIG',
    'PRESET_CONFIGS',
    'get_config',
    'create_config',
    'print_config',
    'validate_config',
]
