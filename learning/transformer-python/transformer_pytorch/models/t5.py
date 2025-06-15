"""
T5模型实现
基于Transformer Encoder-Decoder的文本到文本转换模型

作者: shihom_wu
版本: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
from torch import Tensor

from ..core.encoder import TransformerEncoder
from ..core.decoder import TransformerDecoder
from ..core.layers import RMSNorm
from ..core.heads import LanguageModelingHead
from ..core.math_utils import create_causal_mask


class T5Embeddings(nn.Module):
    """
    T5嵌入层
    只包含词嵌入，不使用位置嵌入（T5使用相对位置编码）
    """
    
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        
        # 共享的词嵌入
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Embedding):
            # T5使用标准正态分布初始化
            factor = 1.0
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            
        Returns:
            嵌入输出 [batch_size, seq_len, d_model]
        """
        return self.shared(input_ids)


class T5Model(nn.Module):
    """
    T5模型
    实现完整的Encoder-Decoder架构，支持文本到文本的转换
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 共享嵌入
        self.shared = T5Embeddings(config)
        
        # 编码器
        self.encoder = TransformerEncoder(
            n_layers=config.num_layers,
            n_embd=config.d_model,
            n_heads=config.num_heads,
            ffn_hidden_dim=config.d_ff,
            dropout=config.dropout_rate,
            activation='relu',
            layer_norm_eps=1e-6,
            use_rms_norm=True
        )
        
        # 解码器
        self.decoder = TransformerDecoder(
            n_layers=config.num_layers,
            n_embd=config.d_model,
            n_heads=config.num_heads,
            ffn_hidden_dim=config.d_ff,
            dropout=config.dropout_rate,
            activation='relu',
            layer_norm_eps=1e-6,
            use_rms_norm=True
        )
        
        # 最终RMS归一化
        self.final_layer_norm = RMSNorm(config.d_model)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # T5使用标准正态分布初始化
            factor = 1.0
            module.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 编码器输入token ID [batch_size, src_len]
            decoder_input_ids: 解码器输入token ID [batch_size, tgt_len]
            attention_mask: 编码器注意力掩码 [batch_size, src_len]
            decoder_attention_mask: 解码器注意力掩码 [batch_size, tgt_len]
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出所有隐藏状态
            
        Returns:
            模型输出字典
        """
        # 编码器嵌入
        encoder_embeddings = self.shared(input_ids)
        
        # 解码器嵌入
        decoder_embeddings = self.shared(decoder_input_ids)
        
        # 创建注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(decoder_input_ids)
        
        # 扩展编码器注意力掩码
        encoder_extended_attention_mask = self._get_extended_attention_mask(
            attention_mask, input_ids.shape
        )
        
        # 创建解码器因果掩码
        decoder_seq_len = decoder_input_ids.shape[-1]
        causal_mask = create_causal_mask(decoder_seq_len, device=decoder_input_ids.device)
        
        # 扩展解码器注意力掩码
        decoder_extended_attention_mask = self._get_extended_attention_mask(
            decoder_attention_mask, decoder_input_ids.shape
        )
        
        # 组合解码器掩码
        decoder_extended_attention_mask = decoder_extended_attention_mask + causal_mask
        
        # 编码器前向传播
        encoder_outputs = self.encoder(
            encoder_embeddings,
            attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        encoder_hidden_states = encoder_outputs['last_hidden_state']
        
        # 解码器前向传播
        decoder_outputs = self.decoder(
            decoder_embeddings,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_extended_attention_mask,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        decoder_hidden_states = decoder_outputs['last_hidden_state']
        
        # 最终层归一化
        decoder_hidden_states = self.final_layer_norm(decoder_hidden_states)
        
        return {
            'last_hidden_state': decoder_hidden_states,
            'encoder_last_hidden_state': encoder_hidden_states,
            'encoder_hidden_states': encoder_outputs.get('hidden_states'),
            'decoder_hidden_states': decoder_outputs.get('hidden_states'),
            'encoder_attentions': encoder_outputs.get('attentions'),
            'decoder_attentions': decoder_outputs.get('attentions'),
            'cross_attentions': decoder_outputs.get('cross_attentions')
        }
    
    def _get_extended_attention_mask(
        self, 
        attention_mask: Tensor, 
        input_shape: Tuple[int, ...]
    ) -> Tensor:
        """
        创建扩展的注意力掩码
        
        Args:
            attention_mask: 原始注意力掩码 [batch_size, seq_len]
            input_shape: 输入形状
            
        Returns:
            扩展的注意力掩码 [batch_size, 1, 1, seq_len]
        """
        # 扩展维度
        extended_attention_mask = attention_mask[:, None, None, :]
        
        # 转换为浮点数并应用掩码值
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        
        return extended_attention_mask
    
    @property
    def dtype(self):
        """获取模型的数据类型"""
        return next(self.parameters()).dtype


class T5ForConditionalGeneration(nn.Module):
    """
    T5条件生成模型
    用于文本到文本的转换任务
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.t5 = T5Model(config)
        
        # 语言模型头（共享嵌入权重）
        self.lm_head = LanguageModelingHead(
            hidden_size=config.d_model,
            vocab_size=config.vocab_size,
            tie_word_embeddings=True,
            bias=False
        )
    
    def forward(
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 编码器输入token ID
            decoder_input_ids: 解码器输入token ID
            attention_mask: 编码器注意力掩码
            decoder_attention_mask: 解码器注意力掩码
            labels: 标签 [batch_size, tgt_len]
            
        Returns:
            模型输出字典
        """
        outputs = self.t5(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs
        )
        
        hidden_states = outputs['last_hidden_state']
        
        # 使用共享的嵌入权重
        embedding_weights = self.t5.shared.shared.weight
        logits = self.lm_head(hidden_states, embedding_weights)
        
        result = {
            'logits': logits,
            'encoder_last_hidden_state': outputs['encoder_last_hidden_state'],
            'encoder_hidden_states': outputs.get('encoder_hidden_states'),
            'decoder_hidden_states': outputs.get('decoder_hidden_states'),
            'encoder_attentions': outputs.get('encoder_attentions'),
            'decoder_attentions': outputs.get('decoder_attentions'),
            'cross_attentions': outputs.get('cross_attentions')
        }
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            result['loss'] = loss
        
        return result
    
    def generate(
        self,
        input_ids: Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
        decoder_start_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Tensor:
        """
        生成文本
        
        Args:
            input_ids: 输入token序列 [batch_size, src_len]
            max_length: 最大生成长度
            temperature: 温度参数
            do_sample: 是否采样
            decoder_start_token_id: 解码器开始token ID
            eos_token_id: 结束token ID
            
        Returns:
            生成的token序列 [batch_size, generated_len]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 设置默认值
        if decoder_start_token_id is None:
            decoder_start_token_id = self.config.decoder_start_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        # 初始化解码器输入
        decoder_input_ids = torch.full(
            (batch_size, 1), 
            decoder_start_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # 生成循环
        for _ in range(max_length):
            # 前向传播
            outputs = self.forward(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids
            )
            
            logits = outputs['logits']
            
            # 获取最后一个位置的logits
            next_token_logits = logits[:, -1, :]
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 采样或贪婪选择
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 添加到解码器输入
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
            
            # 检查是否生成了结束符
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
        
        # 移除开始token
        return decoder_input_ids[:, 1:]
