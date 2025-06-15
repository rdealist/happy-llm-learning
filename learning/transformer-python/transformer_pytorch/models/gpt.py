"""
GPT模型实现
基于Transformer Decoder的自回归语言模型

作者: shihom_wu
版本: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
from torch import Tensor
import math

from ..core.decoder import TransformerDecoder
from ..core.layers import LayerNorm
from ..core.heads import LanguageModelingHead
from ..core.math_utils import create_causal_mask


class GPTEmbeddings(nn.Module):
    """
    GPT嵌入层
    包含词嵌入和位置嵌入
    """
    
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        
        # 词嵌入
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # 位置嵌入
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Dropout
        self.dropout = nn.Dropout(config.embd_pdrop)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None
    ) -> Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            position_ids: 位置ID [batch_size, seq_len]
            
        Returns:
            嵌入输出 [batch_size, seq_len, n_embd]
        """
        batch_size, seq_len = input_ids.shape
        
        # 默认位置ID
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 获取嵌入
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        
        # 嵌入相加
        embeddings = token_embeddings + position_embeddings
        
        # Dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings


class GPTModel(nn.Module):
    """
    GPT模型
    实现Decoder-only的自回归语言模型
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embeddings = GPTEmbeddings(config)
        
        # 解码器
        self.decoder = TransformerDecoder(
            n_layers=config.n_layer,
            n_embd=config.n_embd,
            n_heads=config.n_head,
            ffn_hidden_dim=config.n_inner,
            dropout=config.resid_pdrop,
            attention_dropout=config.attn_pdrop,
            activation=config.activation_function,
            layer_norm_eps=config.layer_norm_epsilon
        )
        
        # 最终层归一化
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            position_ids: 位置ID [batch_size, seq_len]
            use_cache: 是否使用缓存（用于生成）
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出所有隐藏状态
            
        Returns:
            模型输出字典
        """
        batch_size, seq_len = input_ids.shape
        
        # 嵌入层
        hidden_states = self.embeddings(input_ids, position_ids)
        
        # 创建因果注意力掩码
        causal_mask = create_causal_mask(seq_len, device=input_ids.device)
        
        # 如果提供了attention_mask，与因果掩码结合
        if attention_mask is not None:
            # 扩展attention_mask维度
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
            
            # 与因果掩码结合
            causal_mask = causal_mask + attention_mask
        
        # 解码器
        decoder_outputs = self.decoder(
            hidden_states,
            attention_mask=causal_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        hidden_states = decoder_outputs['last_hidden_state']
        
        # 最终层归一化
        hidden_states = self.ln_f(hidden_states)
        
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': decoder_outputs.get('hidden_states'),
            'attentions': decoder_outputs.get('attentions')
        }
    
    @property
    def dtype(self):
        """获取模型的数据类型"""
        return next(self.parameters()).dtype


class GPTForCausalLM(nn.Module):
    """
    GPT因果语言建模模型
    用于文本生成任务
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.gpt = GPTModel(config)
        self.lm_head = LanguageModelingHead(
            hidden_size=config.n_embd,
            vocab_size=config.vocab_size,
            tie_word_embeddings=config.tie_word_embeddings
        )
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 [batch_size, seq_len]
            
        Returns:
            模型输出字典
        """
        outputs = self.gpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        hidden_states = outputs['last_hidden_state']
        
        # 获取词嵌入权重（如果共享权重）
        embedding_weights = None
        if self.lm_head.tie_word_embeddings:
            embedding_weights = self.gpt.embeddings.wte.weight
        
        logits = self.lm_head(hidden_states, embedding_weights)
        
        result = {
            'logits': logits,
            'hidden_states': outputs.get('hidden_states'),
            'attentions': outputs.get('attentions')
        }
        
        if labels is not None:
            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            result['loss'] = loss
        
        return result
    
    def generate(
        self,
        input_ids: Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Tensor:
        """
        生成文本
        
        Args:
            input_ids: 输入token序列 [batch_size, seq_len]
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: top-k采样
            top_p: top-p采样
            do_sample: 是否采样
            pad_token_id: 填充token ID
            eos_token_id: 结束token ID
            
        Returns:
            生成的token序列 [batch_size, generated_len]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 设置默认值
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        # 生成循环
        generated = input_ids.clone()
        
        for _ in range(max_length):
            # 前向传播
            outputs = self.forward(generated)
            logits = outputs['logits']
            
            # 获取最后一个位置的logits
            next_token_logits = logits[:, -1, :]
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 采样或贪婪选择
            if do_sample:
                # 应用top-k和top-p过滤
                next_token_logits = self._top_k_top_p_filtering(
                    next_token_logits, top_k=top_k, top_p=top_p
                )
                
                # 多项式采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪选择
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 添加到生成序列
            generated = torch.cat([generated, next_tokens], dim=-1)
            
            # 检查是否生成了结束符
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
        
        return generated
    
    def _top_k_top_p_filtering(
        self,
        logits: Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float('Inf')
    ) -> Tensor:
        """
        应用top-k和top-p过滤
        
        Args:
            logits: 输入logits [batch_size, vocab_size]
            top_k: top-k参数
            top_p: top-p参数
            filter_value: 过滤值
            
        Returns:
            过滤后的logits
        """
        if top_k > 0:
            # Top-k过滤
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        if top_p < 1.0:
            # Top-p过滤
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过阈值的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        
        return logits
