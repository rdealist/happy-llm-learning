"""
常量定义模块

定义 Transformer 模型中使用的各种常量和枚举值。

作者: Transformer-PyTorch Team
版本: 1.0.0
"""

from enum import Enum
from typing import Dict, Any


# 特殊词元常量
class SpecialTokens:
    """特殊词元定义"""
    PAD = '<PAD>'           # 填充标记
    UNK = '<UNK>'           # 未知词标记
    BOS = '<BOS>'           # 序列开始标记
    EOS = '<EOS>'           # 序列结束标记
    MASK = '<MASK>'         # 掩码标记
    SEP = '<SEP>'           # 分隔符标记
    CLS = '<CLS>'           # 分类标记


class SpecialTokenIds:
    """特殊词元ID定义"""
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    MASK_ID = 4
    SEP_ID = 5
    CLS_ID = 6


# 激活函数类型
class ActivationFunction(Enum):
    """激活函数枚举"""
    RELU = 'relu'
    GELU = 'gelu'
    SWISH = 'swish'
    SILU = 'silu'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'


# 注意力机制类型
class AttentionType(Enum):
    """注意力机制类型枚举"""
    SELF_ATTENTION = 'self_attention'
    CROSS_ATTENTION = 'cross_attention'
    MASKED_ATTENTION = 'masked_attention'
    MULTI_HEAD_ATTENTION = 'multi_head_attention'
    RELATIVE_ATTENTION = 'relative_attention'


# 位置编码类型
class PositionEncodingType(Enum):
    """位置编码类型枚举"""
    SINUSOIDAL = 'sinusoidal'      # 正弦余弦位置编码
    LEARNED = 'learned'             # 可学习位置编码
    RELATIVE = 'relative'           # 相对位置编码
    ROTARY = 'rotary'              # 旋转位置编码 (RoPE)


# 归一化类型
class NormalizationType(Enum):
    """归一化类型枚举"""
    LAYER_NORM = 'layer_norm'
    BATCH_NORM = 'batch_norm'
    RMS_NORM = 'rms_norm'
    GROUP_NORM = 'group_norm'


# 模型架构类型
class ModelArchitecture(Enum):
    """模型架构类型枚举"""
    TRANSFORMER = 'transformer'
    ENCODER_ONLY = 'encoder_only'          # 如 BERT
    DECODER_ONLY = 'decoder_only'          # 如 GPT
    ENCODER_DECODER = 'encoder_decoder'    # 如原始 Transformer


# 训练模式
class TrainingMode(Enum):
    """训练模式枚举"""
    TRAIN = 'train'
    EVAL = 'eval'
    INFERENCE = 'inference'


# 损失函数类型
class LossFunction(Enum):
    """损失函数类型枚举"""
    CROSS_ENTROPY = 'cross_entropy'
    LABEL_SMOOTHING = 'label_smoothing'
    FOCAL_LOSS = 'focal_loss'
    MSE = 'mse'
    MAE = 'mae'


# 优化器类型
class OptimizerType(Enum):
    """优化器类型枚举"""
    SGD = 'sgd'
    ADAM = 'adam'
    ADAMW = 'adamw'
    ADAGRAD = 'adagrad'
    RMSPROP = 'rmsprop'


# 学习率调度器类型
class LRSchedulerType(Enum):
    """学习率调度器类型枚举"""
    CONSTANT = 'constant'
    LINEAR = 'linear'
    COSINE = 'cosine'
    EXPONENTIAL = 'exponential'
    STEP = 'step'
    WARMUP = 'warmup'
    COSINE_WITH_RESTARTS = 'cosine_with_restarts'


# 数学常量
class MathConstants:
    """数学常量定义"""
    PI = 3.141592653589793
    E = 2.718281828459045
    SQRT_2 = 1.4142135623730951
    SQRT_PI = 1.7724538509055159
    LOG_2 = 0.6931471805599453
    LOG_10 = 2.302585092994046
    EPSILON = 1e-8              # 防止除零的小值
    LAYER_NORM_EPS = 1e-6       # 层归一化 epsilon
    ATTENTION_SCALE_EPS = 1e-4  # 注意力缩放 epsilon


# 默认超参数
class DefaultHyperParams:
    """默认超参数定义"""
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    EPOCHS = 100
    WARMUP_STEPS = 4000
    WEIGHT_DECAY = 0.01
    GRADIENT_CLIP = 1.0
    DROPOUT = 0.1
    ATTENTION_DROPOUT = 0.1
    LABEL_SMOOTHING = 0.1


# 性能配置
class PerformanceConfig:
    """性能配置常量"""
    MAX_BATCH_SIZE = 64
    MAX_SEQUENCE_LENGTH = 1024
    MEMORY_LIMIT_GB = 8
    COMPUTATION_TIMEOUT_SECONDS = 300
    CACHE_SIZE_LIMIT = 1000


# 设备类型
class DeviceType(Enum):
    """设备类型枚举"""
    CPU = 'cpu'
    CUDA = 'cuda'
    MPS = 'mps'  # Apple Silicon GPU


# 数据类型
class DataType(Enum):
    """数据类型枚举"""
    FLOAT16 = 'float16'
    FLOAT32 = 'float32'
    FLOAT64 = 'float64'
    BFLOAT16 = 'bfloat16'


# 模型状态
class ModelState(Enum):
    """模型状态枚举"""
    INITIALIZED = 'initialized'
    TRAINING = 'training'
    TRAINED = 'trained'
    LOADING = 'loading'
    LOADED = 'loaded'
    SAVING = 'saving'
    SAVED = 'saved'
    ERROR = 'error'


# 文件格式
class FileFormat(Enum):
    """文件格式枚举"""
    JSON = 'json'
    YAML = 'yaml'
    PICKLE = 'pickle'
    TORCH = 'torch'
    SAFETENSORS = 'safetensors'
    ONNX = 'onnx'


# 任务类型
class TaskType(Enum):
    """任务类型枚举"""
    TRANSLATION = 'translation'
    SUMMARIZATION = 'summarization'
    QUESTION_ANSWERING = 'question_answering'
    TEXT_GENERATION = 'text_generation'
    CLASSIFICATION = 'classification'
    LANGUAGE_MODELING = 'language_modeling'


# 生成策略
class GenerationStrategy(Enum):
    """生成策略枚举"""
    GREEDY = 'greedy'
    BEAM_SEARCH = 'beam_search'
    TOP_K = 'top_k'
    TOP_P = 'top_p'
    TEMPERATURE = 'temperature'


# 版本信息
class VersionInfo:
    """版本信息常量"""
    MAJOR = 1
    MINOR = 0
    PATCH = 0
    BUILD = '20240101'
    VERSION_STRING = f"{MAJOR}.{MINOR}.{PATCH}"


# 配置验证规则
VALIDATION_RULES = {
    'vocab_size': {'min': 1, 'max': 1000000},
    'd_model': {'min': 1, 'max': 8192, 'divisible_by': 'num_heads'},
    'num_heads': {'min': 1, 'max': 64},
    'num_layers': {'min': 0, 'max': 100},
    'dropout': {'min': 0.0, 'max': 1.0},
    'learning_rate': {'min': 1e-8, 'max': 1.0},
    'max_seq_len': {'min': 1, 'max': 32768},
}

# 支持的激活函数映射
ACTIVATION_FUNCTIONS = {
    ActivationFunction.RELU.value: 'ReLU',
    ActivationFunction.GELU.value: 'GELU',
    ActivationFunction.SWISH.value: 'SiLU',
    ActivationFunction.SILU.value: 'SiLU',
    ActivationFunction.TANH.value: 'Tanh',
    ActivationFunction.SIGMOID.value: 'Sigmoid',
}

# 位置编码类型映射
POSITION_ENCODING_TYPES = {
    PositionEncodingType.SINUSOIDAL.value: 'SinusoidalPositionalEncoding',
    PositionEncodingType.LEARNED.value: 'LearnedPositionalEncoding',
    PositionEncodingType.RELATIVE.value: 'RelativePositionalEncoding',
    PositionEncodingType.ROTARY.value: 'RotaryPositionalEncoding',
}

# 注意力机制类型映射
ATTENTION_TYPES = {
    AttentionType.SELF_ATTENTION.value: 'SelfAttention',
    AttentionType.CROSS_ATTENTION.value: 'CrossAttention',
    AttentionType.MASKED_ATTENTION.value: 'MaskedAttention',
    AttentionType.MULTI_HEAD_ATTENTION.value: 'MultiHeadAttention',
    AttentionType.RELATIVE_ATTENTION.value: 'RelativePositionAttention',
}

# 模型架构映射
MODEL_ARCHITECTURES = {
    ModelArchitecture.TRANSFORMER.value: 'Transformer',
    ModelArchitecture.ENCODER_ONLY.value: 'TransformerEncoder',
    ModelArchitecture.DECODER_ONLY.value: 'TransformerDecoder',
    ModelArchitecture.ENCODER_DECODER.value: 'Transformer',
}


# 导出的常量和类
__all__ = [
    # 特殊词元
    'SpecialTokens',
    'SpecialTokenIds',
    
    # 枚举类
    'ActivationFunction',
    'AttentionType',
    'PositionEncodingType',
    'NormalizationType',
    'ModelArchitecture',
    'TrainingMode',
    'LossFunction',
    'OptimizerType',
    'LRSchedulerType',
    'DeviceType',
    'DataType',
    'ModelState',
    'FileFormat',
    'TaskType',
    'GenerationStrategy',
    
    # 常量类
    'MathConstants',
    'DefaultHyperParams',
    'PerformanceConfig',
    'VersionInfo',
    
    # 映射字典
    'VALIDATION_RULES',
    'ACTIVATION_FUNCTIONS',
    'POSITION_ENCODING_TYPES',
    'ATTENTION_TYPES',
    'MODEL_ARCHITECTURES',
]
