"""
预训练任务实现
包含MLM、NSP、SOP、CLM等预训练任务的数据处理和损失计算

作者: shihom_wu
版本: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import random
import numpy as np


class MLMTask:
    """
    掩码语言模型任务 (MLM)
    用于BERT等双向语言模型的预训练
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        mask_ratio: float = 0.15,
        mask_token_id: int = 103,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0
    ):
        """
        初始化MLM任务
        
        Args:
            vocab_size: 词汇表大小
            mask_ratio: 掩码比例，默认0.15
            mask_token_id: [MASK] token的ID
            cls_token_id: [CLS] token的ID
            sep_token_id: [SEP] token的ID
            pad_token_id: [PAD] token的ID
        """
        self.vocab_size = vocab_size
        self.mask_ratio = mask_ratio
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        
        # 特殊token集合，不对其进行掩码
        self.special_tokens = {cls_token_id, sep_token_id, pad_token_id}
    
    def prepare_data(
        self, 
        token_ids: Union[List[int], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        准备MLM训练数据
        
        Args:
            token_ids: 输入token序列
            
        Returns:
            处理后的训练数据字典
        """
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        seq_len = len(token_ids)
        
        # 找到可以掩码的位置（排除特殊token）
        candidate_positions = []
        for i, token_id in enumerate(token_ids):
            if token_id.item() not in self.special_tokens:
                candidate_positions.append(i)
        
        # 计算要掩码的位置数量
        num_mask = max(1, int(len(candidate_positions) * self.mask_ratio))
        
        # 随机选择要掩码的位置
        mask_positions = random.sample(candidate_positions, min(num_mask, len(candidate_positions)))
        
        # 创建掩码后的输入和标签
        masked_tokens = token_ids.clone()
        labels = torch.full_like(token_ids, -100)  # -100表示不计算损失
        
        for pos in mask_positions:
            labels[pos] = token_ids[pos]  # 保存原始token用于计算损失
            
            rand = random.random()
            if rand < 0.8:
                # 80%的概率替换为[MASK]
                masked_tokens[pos] = self.mask_token_id
            elif rand < 0.9:
                # 10%的概率替换为随机token
                masked_tokens[pos] = random.randint(0, self.vocab_size - 1)
            # 10%的概率保持不变
        
        # 创建注意力掩码
        attention_mask = (token_ids != self.pad_token_id).long()
        
        return {
            'input_ids': masked_tokens,
            'labels': labels,
            'attention_mask': attention_mask,
            'mask_positions': torch.tensor(mask_positions, dtype=torch.long)
        }
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算MLM损失
        
        Args:
            predictions: 模型预测结果 [batch_size, seq_len, vocab_size]
            labels: 真实标签 [batch_size, seq_len]
            
        Returns:
            损失值
        """
        # 只计算被掩码位置的损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 重塑张量以适应损失函数
        predictions = predictions.view(-1, predictions.size(-1))
        labels = labels.view(-1)
        
        return loss_fct(predictions, labels)


class NSPTask:
    """
    下一句预测任务 (NSP)
    用于BERT的句子级别理解预训练
    """
    
    def __init__(
        self,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0
    ):
        """
        初始化NSP任务
        
        Args:
            cls_token_id: [CLS] token的ID
            sep_token_id: [SEP] token的ID
            pad_token_id: [PAD] token的ID
        """
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
    
    def prepare_data(
        self,
        sentence_a: Union[List[int], torch.Tensor],
        sentence_b: Union[List[int], torch.Tensor],
        is_next: bool
    ) -> Dict[str, torch.Tensor]:
        """
        准备NSP训练数据
        
        Args:
            sentence_a: 第一个句子的token序列
            sentence_b: 第二个句子的token序列
            is_next: 是否为连续句子
            
        Returns:
            处理后的训练数据字典
        """
        if isinstance(sentence_a, list):
            sentence_a = torch.tensor(sentence_a, dtype=torch.long)
        if isinstance(sentence_b, list):
            sentence_b = torch.tensor(sentence_b, dtype=torch.long)
        
        # 构建输入序列: [CLS] sentence_a [SEP] sentence_b [SEP]
        input_ids = torch.cat([
            torch.tensor([self.cls_token_id]),
            sentence_a,
            torch.tensor([self.sep_token_id]),
            sentence_b,
            torch.tensor([self.sep_token_id])
        ])
        
        # 创建token类型ID (0表示第一个句子，1表示第二个句子)
        token_type_ids = torch.cat([
            torch.zeros(1 + len(sentence_a) + 1, dtype=torch.long),  # [CLS] + sentence_a + [SEP]
            torch.ones(len(sentence_b) + 1, dtype=torch.long)        # sentence_b + [SEP]
        ])
        
        # 创建注意力掩码
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'next_sentence_label': torch.tensor(1 if is_next else 0, dtype=torch.long)
        }
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算NSP损失
        
        Args:
            predictions: 模型预测结果 [batch_size, 2]
            labels: 真实标签 [batch_size]
            
        Returns:
            损失值
        """
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(predictions, labels)


class SOPTask:
    """
    句子顺序预测任务 (SOP)
    用于ALBERT等模型的改进预训练任务
    """
    
    def __init__(
        self,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0
    ):
        """
        初始化SOP任务
        
        Args:
            cls_token_id: [CLS] token的ID
            sep_token_id: [SEP] token的ID
            pad_token_id: [PAD] token的ID
        """
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
    
    def prepare_data(
        self,
        sentence_a: Union[List[int], torch.Tensor],
        sentence_b: Union[List[int], torch.Tensor],
        is_correct_order: bool
    ) -> Dict[str, torch.Tensor]:
        """
        准备SOP训练数据
        
        Args:
            sentence_a: 第一个句子的token序列
            sentence_b: 第二个句子的token序列
            is_correct_order: 是否为正确顺序
            
        Returns:
            处理后的训练数据字典
        """
        if isinstance(sentence_a, list):
            sentence_a = torch.tensor(sentence_a, dtype=torch.long)
        if isinstance(sentence_b, list):
            sentence_b = torch.tensor(sentence_b, dtype=torch.long)
        
        # 如果不是正确顺序，交换两个句子
        if not is_correct_order:
            sentence_a, sentence_b = sentence_b, sentence_a
        
        # 构建输入序列: [CLS] sentence_a [SEP] sentence_b [SEP]
        input_ids = torch.cat([
            torch.tensor([self.cls_token_id]),
            sentence_a,
            torch.tensor([self.sep_token_id]),
            sentence_b,
            torch.tensor([self.sep_token_id])
        ])
        
        # 创建token类型ID
        token_type_ids = torch.cat([
            torch.zeros(1 + len(sentence_a) + 1, dtype=torch.long),
            torch.ones(len(sentence_b) + 1, dtype=torch.long)
        ])
        
        # 创建注意力掩码
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'sentence_order_label': torch.tensor(1 if is_correct_order else 0, dtype=torch.long)
        }
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算SOP损失
        
        Args:
            predictions: 模型预测结果 [batch_size, 2]
            labels: 真实标签 [batch_size]
            
        Returns:
            损失值
        """
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(predictions, labels)


class CLMTask:
    """
    因果语言模型任务 (CLM)
    用于GPT等自回归语言模型的预训练
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        eos_token_id: int = 2,
        pad_token_id: int = 0
    ):
        """
        初始化CLM任务

        Args:
            vocab_size: 词汇表大小
            eos_token_id: 结束token的ID
            pad_token_id: 填充token的ID
        """
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def prepare_data(
        self,
        token_ids: Union[List[int], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        准备CLM训练数据

        Args:
            token_ids: 输入token序列

        Returns:
            处理后的训练数据字典
        """
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, dtype=torch.long)

        # CLM任务中，输入是前n-1个token，标签是后n-1个token
        input_ids = token_ids[:-1]
        labels = token_ids[1:]

        # 创建因果注意力掩码
        seq_len = len(input_ids)
        attention_mask = torch.tril(torch.ones(seq_len, seq_len))

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算CLM损失

        Args:
            predictions: 模型预测结果 [batch_size, seq_len, vocab_size]
            labels: 真实标签 [batch_size, seq_len]

        Returns:
            损失值
        """
        # 忽略填充位置的损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        # 重塑张量以适应损失函数
        predictions = predictions.view(-1, predictions.size(-1))
        labels = labels.view(-1)

        return loss_fct(predictions, labels)


class PretrainingDataProcessor:
    """
    预训练数据处理器
    统一处理各种预训练任务的数据
    """

    def __init__(self, config: Dict):
        """
        初始化数据处理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.mlm_task = MLMTask(**config.get('mlm', {}))
        self.nsp_task = NSPTask(**config.get('nsp', {}))
        self.sop_task = SOPTask(**config.get('sop', {}))
        self.clm_task = CLMTask(**config.get('clm', {}))

    def process_bert_data(
        self,
        sentence_a: Union[List[int], torch.Tensor],
        sentence_b: Union[List[int], torch.Tensor],
        is_next: bool
    ) -> Dict[str, torch.Tensor]:
        """
        处理BERT风格的预训练数据 (MLM + NSP)

        Args:
            sentence_a: 第一个句子
            sentence_b: 第二个句子
            is_next: 是否为连续句子

        Returns:
            处理后的数据字典
        """
        # 准备NSP数据
        nsp_data = self.nsp_task.prepare_data(sentence_a, sentence_b, is_next)

        # 对组合后的序列应用MLM
        mlm_data = self.mlm_task.prepare_data(nsp_data['input_ids'])

        return {
            'input_ids': mlm_data['input_ids'],
            'token_type_ids': nsp_data['token_type_ids'],
            'attention_mask': mlm_data['attention_mask'],
            'mlm_labels': mlm_data['labels'],
            'nsp_labels': nsp_data['next_sentence_label'],
            'mask_positions': mlm_data['mask_positions']
        }

    def process_albert_data(
        self,
        sentence_a: Union[List[int], torch.Tensor],
        sentence_b: Union[List[int], torch.Tensor],
        is_correct_order: bool
    ) -> Dict[str, torch.Tensor]:
        """
        处理ALBERT风格的预训练数据 (MLM + SOP)

        Args:
            sentence_a: 第一个句子
            sentence_b: 第二个句子
            is_correct_order: 是否为正确顺序

        Returns:
            处理后的数据字典
        """
        # 准备SOP数据
        sop_data = self.sop_task.prepare_data(sentence_a, sentence_b, is_correct_order)

        # 对组合后的序列应用MLM
        mlm_data = self.mlm_task.prepare_data(sop_data['input_ids'])

        return {
            'input_ids': mlm_data['input_ids'],
            'token_type_ids': sop_data['token_type_ids'],
            'attention_mask': mlm_data['attention_mask'],
            'mlm_labels': mlm_data['labels'],
            'sop_labels': sop_data['sentence_order_label'],
            'mask_positions': mlm_data['mask_positions']
        }

    def process_gpt_data(
        self,
        token_ids: Union[List[int], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        处理GPT风格的预训练数据 (CLM)

        Args:
            token_ids: 输入token序列

        Returns:
            处理后的数据字典
        """
        return self.clm_task.prepare_data(token_ids)
