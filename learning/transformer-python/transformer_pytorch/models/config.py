"""
预训练语言模型配置
定义各种模型的默认配置和工厂函数

作者: shihom_wu
版本: 1.0.0
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BERTConfig:
    """BERT模型配置"""
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    use_cache: bool = True
    classifier_dropout: Optional[float] = None
    tie_word_embeddings: bool = False
    num_labels: int = 2


@dataclass
class GPTConfig:
    """GPT模型配置"""
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: Optional[int] = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    summary_type: str = "cls_index"
    summary_use_proj: bool = True
    summary_activation: Optional[str] = None
    summary_proj_to_labels: bool = True
    summary_first_dropout: float = 0.1
    use_cache: bool = True
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    pad_token_id: int = 50256
    tie_word_embeddings: bool = True


@dataclass
class T5Config:
    """T5模型配置"""
    vocab_size: int = 32128
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_decoder_layers: Optional[int] = None
    num_heads: int = 8
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"
    is_encoder_decoder: bool = True
    use_cache: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0
    tie_word_embeddings: bool = True


class ModelFactory:
    """
    模型工厂类
    提供创建各种预训练模型的统一接口
    """
    
    # 预定义配置
    BERT_CONFIGS = {
        'bert-tiny': BERTConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512
        ),
        'bert-mini': BERTConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024
        ),
        'bert-small': BERTConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=2048
        ),
        'bert-base': BERTConfig(),
        'bert-large': BERTConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096
        )
    }
    
    GPT_CONFIGS = {
        'gpt2-tiny': GPTConfig(
            n_embd=128,
            n_layer=2,
            n_head=2,
            n_inner=512
        ),
        'gpt2-mini': GPTConfig(
            n_embd=256,
            n_layer=4,
            n_head=4,
            n_inner=1024
        ),
        'gpt2-small': GPTConfig(
            n_embd=512,
            n_layer=6,
            n_head=8,
            n_inner=2048
        ),
        'gpt2': GPTConfig(),
        'gpt2-medium': GPTConfig(
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_inner=4096
        ),
        'gpt2-large': GPTConfig(
            n_embd=1280,
            n_layer=36,
            n_head=20,
            n_inner=5120
        ),
        'gpt2-xl': GPTConfig(
            n_embd=1600,
            n_layer=48,
            n_head=25,
            n_inner=6400
        )
    }
    
    T5_CONFIGS = {
        't5-small': T5Config(),
        't5-base': T5Config(
            d_model=768,
            d_ff=3072,
            num_layers=12,
            num_heads=12
        ),
        't5-large': T5Config(
            d_model=1024,
            d_ff=4096,
            num_layers=24,
            num_heads=16
        ),
        't5-3b': T5Config(
            d_model=1024,
            d_ff=16384,
            num_layers=24,
            num_heads=32
        ),
        't5-11b': T5Config(
            d_model=1024,
            d_ff=65536,
            num_layers=24,
            num_heads=128
        )
    }
    
    @classmethod
    def create_bert(
        cls,
        model_name: str = 'bert-base',
        task_type: str = 'base',
        **kwargs
    ):
        """
        创建BERT模型
        
        Args:
            model_name: 模型名称
            task_type: 任务类型 ('base', 'classification', 'token_classification', 'masked_lm')
            **kwargs: 额外配置参数
            
        Returns:
            BERT模型实例
        """
        from .bert import (
            BERTModel, 
            BERTForSequenceClassification,
            BERTForTokenClassification,
            BERTForMaskedLM
        )
        
        config = cls.BERT_CONFIGS[model_name]
        
        # 更新配置
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        if task_type == 'base':
            return BERTModel(config)
        elif task_type == 'classification':
            return BERTForSequenceClassification(config)
        elif task_type == 'token_classification':
            return BERTForTokenClassification(config)
        elif task_type == 'masked_lm':
            return BERTForMaskedLM(config)
        else:
            raise ValueError(f"不支持的BERT任务类型: {task_type}")
    
    @classmethod
    def create_gpt(
        cls,
        model_name: str = 'gpt2',
        task_type: str = 'causal_lm',
        **kwargs
    ):
        """
        创建GPT模型
        
        Args:
            model_name: 模型名称
            task_type: 任务类型 ('base', 'causal_lm')
            **kwargs: 额外配置参数
            
        Returns:
            GPT模型实例
        """
        from .gpt import GPTModel, GPTForCausalLM
        
        config = cls.GPT_CONFIGS[model_name]
        
        # 设置n_inner默认值
        if config.n_inner is None:
            config.n_inner = 4 * config.n_embd
        
        # 更新配置
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        if task_type == 'base':
            return GPTModel(config)
        elif task_type == 'causal_lm':
            return GPTForCausalLM(config)
        else:
            raise ValueError(f"不支持的GPT任务类型: {task_type}")
    
    @classmethod
    def create_t5(
        cls,
        model_name: str = 't5-base',
        task_type: str = 'conditional_generation',
        **kwargs
    ):
        """
        创建T5模型
        
        Args:
            model_name: 模型名称
            task_type: 任务类型 ('base', 'conditional_generation')
            **kwargs: 额外配置参数
            
        Returns:
            T5模型实例
        """
        from .t5 import T5Model, T5ForConditionalGeneration
        
        config = cls.T5_CONFIGS[model_name]
        
        # 设置num_decoder_layers默认值
        if config.num_decoder_layers is None:
            config.num_decoder_layers = config.num_layers
        
        # 更新配置
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        if task_type == 'base':
            return T5Model(config)
        elif task_type == 'conditional_generation':
            return T5ForConditionalGeneration(config)
        else:
            raise ValueError(f"不支持的T5任务类型: {task_type}")
    
    @classmethod
    def get_config(cls, model_type: str, model_name: str) -> Any:
        """
        获取模型配置
        
        Args:
            model_type: 模型类型 ('bert', 'gpt', 't5')
            model_name: 模型名称
            
        Returns:
            模型配置
        """
        if model_type.lower() == 'bert':
            return cls.BERT_CONFIGS[model_name]
        elif model_type.lower() == 'gpt':
            return cls.GPT_CONFIGS[model_name]
        elif model_type.lower() == 't5':
            return cls.T5_CONFIGS[model_name]
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @classmethod
    def list_available_models(cls) -> Dict[str, list]:
        """
        列出所有可用的模型
        
        Returns:
            可用模型列表
        """
        return {
            'bert': list(cls.BERT_CONFIGS.keys()),
            'gpt': list(cls.GPT_CONFIGS.keys()),
            't5': list(cls.T5_CONFIGS.keys())
        }
