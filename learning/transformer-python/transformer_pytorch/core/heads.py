"""
分类头模块
实现各种下游任务的分类器和预测头

作者: shihom_wu
版本: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from torch import Tensor


class SequenceClassificationHead(nn.Module):
    """
    序列分类头
    用于文本分类、情感分析等任务
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        pooling_strategy: str = 'cls'
    ):
        """
        初始化序列分类头
        
        Args:
            hidden_size: 隐藏层大小
            num_labels: 标签数量
            dropout: Dropout概率
            pooling_strategy: 池化策略 ('cls', 'mean', 'max')
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.pooling_strategy = pooling_strategy
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 初始化权重
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self, 
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            分类logits [batch_size, num_labels]
        """
        # 根据池化策略获取序列表示
        if self.pooling_strategy == 'cls':
            # 使用第一个token ([CLS])
            pooled_output = hidden_states[:, 0]
        elif self.pooling_strategy == 'mean':
            # 平均池化
            pooled_output = self._mean_pooling(hidden_states, attention_mask)
        elif self.pooling_strategy == 'max':
            # 最大池化
            pooled_output = self._max_pooling(hidden_states, attention_mask)
        else:
            raise ValueError(f"不支持的池化策略: {self.pooling_strategy}")
        
        # 应用dropout和分类器
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def _mean_pooling(
        self, 
        hidden_states: Tensor, 
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """平均池化"""
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        
        # 考虑注意力掩码的平均池化
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def _max_pooling(
        self, 
        hidden_states: Tensor, 
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """最大池化"""
        if attention_mask is None:
            return hidden_states.max(dim=1)[0]
        
        # 考虑注意力掩码的最大池化
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        hidden_states = hidden_states.masked_fill(~mask_expanded.bool(), -1e9)
        
        return hidden_states.max(dim=1)[0]


class TokenClassificationHead(nn.Module):
    """
    Token分类头
    用于命名实体识别、词性标注等token级别任务
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1
    ):
        """
        初始化Token分类头
        
        Args:
            hidden_size: 隐藏层大小
            num_labels: 标签数量
            dropout: Dropout概率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 初始化权重
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
            
        Returns:
            每个token的分类logits [batch_size, seq_len, num_labels]
        """
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        
        return logits


class LanguageModelingHead(nn.Module):
    """
    语言模型头
    用于MLM、CLM等语言建模任务
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        tie_word_embeddings: bool = False,
        bias: bool = True
    ):
        """
        初始化语言模型头
        
        Args:
            hidden_size: 隐藏层大小
            vocab_size: 词汇表大小
            tie_word_embeddings: 是否共享词嵌入权重
            bias: 是否使用偏置
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        
        if not tie_word_embeddings:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=bias)
            # 初始化权重
            nn.init.normal_(self.lm_head.weight, std=0.02)
            if bias:
                nn.init.zeros_(self.lm_head.bias)
    
    def forward(
        self, 
        hidden_states: Tensor,
        embedding_weights: Optional[Tensor] = None
    ) -> Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
            embedding_weights: 词嵌入权重 [vocab_size, hidden_size]（如果共享权重）
            
        Returns:
            词汇表上的logits [batch_size, seq_len, vocab_size]
        """
        if self.tie_word_embeddings and embedding_weights is not None:
            # 使用共享的嵌入权重: hidden @ embedding_weights.T
            logits = torch.matmul(hidden_states, embedding_weights.T)
        else:
            # 使用独立的语言模型头
            logits = self.lm_head(hidden_states)
        
        return logits


class MultipleChoiceHead(nn.Module):
    """
    多选题分类头
    用于多选题回答等任务
    """
    
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1
    ):
        """
        初始化多选题分类头
        
        Args:
            hidden_size: 隐藏层大小
            dropout: Dropout概率
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        
        # 初始化权重
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态 [batch_size, num_choices, hidden_size]
            
        Returns:
            每个选择的分数 [batch_size, num_choices]
        """
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states).squeeze(-1)
        
        return logits


class QuestionAnsweringHead(nn.Module):
    """
    问答任务头
    用于抽取式问答任务，预测答案的开始和结束位置
    """
    
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1
    ):
        """
        初始化问答任务头
        
        Args:
            hidden_size: 隐藏层大小
            dropout: Dropout概率
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.qa_outputs = nn.Linear(hidden_size, 2)  # 开始和结束位置
        
        # 初始化权重
        nn.init.normal_(self.qa_outputs.weight, std=0.02)
        nn.init.zeros_(self.qa_outputs.bias)
    
    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
            
        Returns:
            开始位置logits和结束位置logits
        """
        hidden_states = self.dropout(hidden_states)
        logits = self.qa_outputs(hidden_states)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits


# 导出所有分类头
__all__ = [
    'SequenceClassificationHead',
    'TokenClassificationHead',
    'LanguageModelingHead',
    'MultipleChoiceHead',
    'QuestionAnsweringHead'
]
