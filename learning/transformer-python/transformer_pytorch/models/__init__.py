"""
预训练语言模型模块
包含BERT、GPT、T5等主流预训练模型的实现

作者: shihom_wu
版本: 1.0.0
"""

from .bert import (
    BERTModel,
    BERTForSequenceClassification,
    BERTForTokenClassification,
    BERTForMaskedLM,
    BERTForMultipleChoice,
    BERTForQuestionAnswering
)

from .gpt import (
    GPTModel,
    GPTForCausalLM
)

from .t5 import (
    T5Model,
    T5ForConditionalGeneration
)

from .config import (
    BERTConfig,
    GPTConfig,
    T5Config,
    ModelFactory
)

__all__ = [
    # BERT models
    'BERTModel',
    'BERTForSequenceClassification',
    'BERTForTokenClassification', 
    'BERTForMaskedLM',
    'BERTForMultipleChoice',
    'BERTForQuestionAnswering',
    
    # GPT models
    'GPTModel',
    'GPTForCausalLM',
    
    # T5 models
    'T5Model',
    'T5ForConditionalGeneration',
    
    # Configs
    'BERTConfig',
    'GPTConfig',
    'T5Config',
    'ModelFactory'
]
