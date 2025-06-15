"""
BERT模型实现
基于Transformer Encoder的双向语言模型

作者: shihom_wu
版本: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
from torch import Tensor

from ..core.encoder import TransformerEncoder
from ..core.layers import LayerNorm
from ..core.heads import (
    SequenceClassificationHead,
    TokenClassificationHead,
    LanguageModelingHead,
    MultipleChoiceHead,
    QuestionAnsweringHead
)


class BERTEmbeddings(nn.Module):
    """
    BERT嵌入层
    包含词嵌入、位置嵌入和token类型嵌入
    """
    
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.type_vocab_size = config.type_vocab_size
        
        # 词嵌入
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=config.pad_token_id
        )
        
        # 位置嵌入
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, 
            config.hidden_size
        )
        
        # token类型嵌入
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, 
            config.hidden_size
        )
        
        # 层归一化和dropout
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 注册位置ID缓冲区
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
    
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None
    ) -> Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            token_type_ids: token类型ID [batch_size, seq_len]
            position_ids: 位置ID [batch_size, seq_len]
            
        Returns:
            嵌入输出 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # 默认位置ID
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]
        
        # 默认token类型ID
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # 获取各种嵌入
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # 嵌入相加
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        
        # 层归一化和dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BERTPooler(nn.Module):
    """
    BERT池化层
    用于获取句子级别的表示
    """
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
            
        Returns:
            池化输出 [batch_size, hidden_size]
        """
        # 取第一个token ([CLS]) 的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        
        return pooled_output


class BERTModel(nn.Module):
    """
    BERT模型
    实现Encoder-only的双向语言模型
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embeddings = BERTEmbeddings(config)
        
        # 编码器
        self.encoder = TransformerEncoder(
            n_layers=config.num_hidden_layers,
            n_embd=config.hidden_size,
            n_heads=config.num_attention_heads,
            ffn_hidden_dim=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            attention_dropout=config.attention_probs_dropout_prob,
            activation=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps
        )
        
        # 池化层
        self.pooler = BERTPooler(config)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: token类型ID [batch_size, seq_len]
            position_ids: 位置ID [batch_size, seq_len]
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出所有隐藏状态
            
        Returns:
            模型输出字典
        """
        batch_size, seq_len = input_ids.shape
        
        # 默认注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 扩展注意力掩码
        extended_attention_mask = self._get_extended_attention_mask(attention_mask)
        
        # 嵌入层
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        # 编码器
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        sequence_output = encoder_outputs['last_hidden_state']
        
        # 池化层
        pooled_output = self.pooler(sequence_output)
        
        return {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output,
            'hidden_states': encoder_outputs.get('hidden_states'),
            'attentions': encoder_outputs.get('attentions')
        }
    
    def _get_extended_attention_mask(self, attention_mask: Tensor) -> Tensor:
        """
        创建扩展的注意力掩码
        
        Args:
            attention_mask: 原始注意力掩码 [batch_size, seq_len]
            
        Returns:
            扩展的注意力掩码 [batch_size, 1, 1, seq_len]
        """
        # 扩展维度: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        extended_attention_mask = attention_mask[:, None, None, :]
        
        # 转换为浮点数并应用掩码值
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    @property
    def dtype(self):
        """获取模型的数据类型"""
        return next(self.parameters()).dtype


class BERTForSequenceClassification(nn.Module):
    """
    BERT序列分类模型
    用于文本分类、情感分析等任务
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BERTModel(config)
        self.classifier = SequenceClassificationHead(
            hidden_size=config.hidden_size,
            num_labels=config.num_labels,
            dropout=config.classifier_dropout
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            token_type_ids: token类型ID
            labels: 标签 [batch_size]

        Returns:
            模型输出字典
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 使用pooler输出进行分类
        logits = self.classifier(outputs['pooler_output'].unsqueeze(1))

        result = {'logits': logits}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            result['loss'] = loss_fct(logits, labels)

        return result


class BERTForTokenClassification(nn.Module):
    """
    BERT Token分类模型
    用于命名实体识别、词性标注等任务
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BERTModel(config)
        self.classifier = TokenClassificationHead(
            hidden_size=config.hidden_size,
            num_labels=config.num_labels,
            dropout=config.classifier_dropout
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            token_type_ids: token类型ID
            labels: 标签 [batch_size, seq_len]

        Returns:
            模型输出字典
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        logits = self.classifier(outputs['last_hidden_state'])

        result = {'logits': logits}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            result['loss'] = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return result


class BERTForMaskedLM(nn.Module):
    """
    BERT掩码语言模型
    用于MLM预训练任务
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert = BERTModel(config)
        self.lm_head = LanguageModelingHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            tie_word_embeddings=config.tie_word_embeddings
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            token_type_ids: token类型ID
            labels: MLM标签 [batch_size, seq_len]

        Returns:
            模型输出字典
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 获取词嵌入权重（如果共享权重）
        embedding_weights = None
        if self.lm_head.tie_word_embeddings:
            embedding_weights = self.bert.embeddings.word_embeddings.weight

        logits = self.lm_head(outputs['last_hidden_state'], embedding_weights)

        result = {'logits': logits}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            result['loss'] = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return result
