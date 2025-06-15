"""
完整的 Transformer 模型

实现端到端的 Transformer 架构，包括：
- 完整的编码器-解码器 Transformer
- 仅编码器模型（如 BERT）
- 仅解码器模型（如 GPT）
- 序列到序列任务支持
- 语言建模任务支持

作者: Transformer-PyTorch Team
版本: 1.0.0
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import AttentionMask
from .decoder import TransformerDecoder, create_decoder
from .embedding import TransformerEmbedding
from .encoder import TransformerEncoder, create_encoder
from .layers import LayerNorm
from .math_utils import init_weights
from ..config.config import TransformerConfig


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    
    实现编码器-解码器架构的 Transformer，支持序列到序列任务。
    
    Args:
        config (TransformerConfig): 模型配置
        
    Examples:
        >>> config = TransformerConfig(vocab_size=10000, d_model=512)
        >>> model = Transformer(config)
        >>> src = torch.randint(0, 10000, (2, 10))
        >>> tgt = torch.randint(0, 10000, (2, 8))
        >>> output = model(src, tgt)
        >>> print(output.logits.shape)  # torch.Size([2, 8, 10000])
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # 源序列嵌入层
        self.src_embedding = TransformerEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_len=config.max_seq_len,
            padding_idx=config.pad_token_id,
            position_encoding_type=config.position_encoding_type,
            dropout=config.dropout,
            scale_embedding=config.scale_embedding
        )
        
        # 目标序列嵌入层
        if config.tie_word_embeddings:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = TransformerEmbedding(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                max_len=config.max_seq_len,
                padding_idx=config.pad_token_id,
                position_encoding_type=config.position_encoding_type,
                dropout=config.dropout,
                scale_embedding=config.scale_embedding
            )
        
        # 编码器
        if config.num_encoder_layers > 0:
            self.encoder = create_encoder(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                num_layers=config.num_encoder_layers,
                dropout=config.dropout,
                activation=config.activation,
                norm_first=config.norm_first
            )
        else:
            self.encoder = None
        
        # 解码器
        if config.num_decoder_layers > 0:
            self.decoder = create_decoder(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                num_layers=config.num_decoder_layers,
                dropout=config.dropout,
                activation=config.activation,
                norm_first=config.norm_first
            )
        else:
            self.decoder = None
        
        # 输出投影层
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重共享
        if config.tie_word_embeddings:
            self.output_projection.weight = self.src_embedding.token_embedding.embedding.weight
        
        # 初始化权重
        self.apply(lambda module: init_weights(module, config.init_std))
    
    def forward(
        self,
        src: Optional[Tensor] = None,
        tgt: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        前向传播
        
        Args:
            src (Optional[Tensor]): 源序列 [batch_size, src_len]
            tgt (Optional[Tensor]): 目标序列 [batch_size, tgt_len]
            src_mask (Optional[Tensor]): 源序列掩码
            tgt_mask (Optional[Tensor]): 目标序列掩码
            memory_mask (Optional[Tensor]): 记忆掩码
            return_dict (bool): 是否返回字典格式的输出
            
        Returns:
            Union[Tensor, Dict[str, Tensor]]: 模型输出
        """
        # 编码阶段
        if src is not None and self.encoder is not None:
            # 源序列嵌入
            src_embedded = self.src_embedding(src)
            
            # 创建源序列填充掩码
            if src_mask is None:
                src_mask = self._create_padding_mask(src, self.config.pad_token_id)
            
            # 编码器前向传播
            encoder_output = self.encoder(src_embedded, mask=src_mask)
            
            if isinstance(encoder_output, dict):
                memory = encoder_output['last_hidden_state']
                encoder_attentions = encoder_output.get('all_attention_weights', None)
            else:
                memory = encoder_output
                encoder_attentions = None
        else:
            memory = None
            encoder_attentions = None
        
        # 解码阶段
        if tgt is not None and self.decoder is not None:
            # 目标序列嵌入
            tgt_embedded = self.tgt_embedding(tgt)
            
            # 创建目标序列因果掩码
            if tgt_mask is None:
                tgt_len = tgt.size(1)
                tgt_mask = AttentionMask.create_causal_mask(tgt_len, device=tgt.device)
                tgt_mask = tgt_mask.unsqueeze(0).expand(tgt.size(0), -1, -1)
            
            # 创建记忆掩码（如果有编码器输出）
            if memory is not None and memory_mask is None:
                memory_mask = src_mask
            
            # 解码器前向传播
            decoder_output = self.decoder(
                tgt_embedded, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask
            )
            
            if isinstance(decoder_output, dict):
                hidden_states = decoder_output['last_hidden_state']
                decoder_self_attentions = decoder_output.get('all_self_attention_weights', None)
                decoder_cross_attentions = decoder_output.get('all_cross_attention_weights', None)
            else:
                hidden_states = decoder_output
                decoder_self_attentions = None
                decoder_cross_attentions = None
        elif memory is not None:
            # 仅编码器模式
            hidden_states = memory
            decoder_self_attentions = None
            decoder_cross_attentions = None
        else:
            raise ValueError("必须提供 src 或 tgt 中的至少一个")
        
        # 输出投影
        logits = self.output_projection(hidden_states)
        
        if return_dict:
            return {
                'logits': logits,
                'last_hidden_state': hidden_states,
                'encoder_last_hidden_state': memory,
                'encoder_attentions': encoder_attentions,
                'decoder_self_attentions': decoder_self_attentions,
                'decoder_cross_attentions': decoder_cross_attentions,
            }
        else:
            return logits
    
    def encode(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        仅编码
        
        Args:
            src (Tensor): 源序列
            src_mask (Optional[Tensor]): 源序列掩码
            
        Returns:
            Dict[str, Tensor]: 编码器输出
        """
        if self.encoder is None:
            raise ValueError("模型没有编码器")
        
        src_embedded = self.src_embedding(src)
        
        if src_mask is None:
            src_mask = self._create_padding_mask(src, self.config.pad_token_id)
        
        encoder_output = self.encoder(src_embedded, mask=src_mask)
        
        if isinstance(encoder_output, dict):
            return encoder_output
        else:
            return {'last_hidden_state': encoder_output}
    
    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        仅解码
        
        Args:
            tgt (Tensor): 目标序列
            memory (Tensor): 编码器输出
            tgt_mask (Optional[Tensor]): 目标序列掩码
            memory_mask (Optional[Tensor]): 记忆掩码
            
        Returns:
            Dict[str, Tensor]: 解码器输出
        """
        if self.decoder is None:
            raise ValueError("模型没有解码器")
        
        tgt_embedded = self.tgt_embedding(tgt)
        
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = AttentionMask.create_causal_mask(tgt_len, device=tgt.device)
            tgt_mask = tgt_mask.unsqueeze(0).expand(tgt.size(0), -1, -1)
        
        decoder_output = self.decoder(
            tgt_embedded, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        
        if isinstance(decoder_output, dict):
            hidden_states = decoder_output['last_hidden_state']
        else:
            hidden_states = decoder_output
        
        logits = self.output_projection(hidden_states)
        
        return {
            'logits': logits,
            'last_hidden_state': hidden_states,
        }
    
    def generate(
        self,
        src: Optional[Tensor] = None,
        tgt: Optional[Tensor] = None,
        max_length: int = 50,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = False,
        **kwargs
    ) -> Tensor:
        """
        生成序列
        
        Args:
            src (Optional[Tensor]): 源序列（用于序列到序列任务）
            tgt (Optional[Tensor]): 目标序列前缀
            max_length (int): 最大生成长度
            num_beams (int): 束搜索大小
            temperature (float): 温度参数
            top_k (int): Top-K 采样
            top_p (float): Top-P 采样
            do_sample (bool): 是否使用采样
            
        Returns:
            Tensor: 生成的序列
        """
        self.eval()
        
        with torch.no_grad():
            # 编码阶段（如果有源序列）
            if src is not None:
                encoder_output = self.encode(src)
                memory = encoder_output['last_hidden_state']
            else:
                memory = None
            
            # 初始化目标序列
            if tgt is None:
                batch_size = src.size(0) if src is not None else 1
                tgt = torch.full(
                    (batch_size, 1),
                    self.config.bos_token_id,
                    dtype=torch.long,
                    device=next(self.parameters()).device
                )
            
            # 简单的贪心解码
            for _ in range(max_length):
                # 解码当前序列
                if memory is not None:
                    decoder_output = self.decode(tgt, memory)
                else:
                    decoder_output = self.forward(tgt=tgt, return_dict=True)
                
                # 获取下一个词元的 logits
                next_token_logits = decoder_output['logits'][:, -1, :]
                
                # 应用温度
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # 选择下一个词元
                if do_sample:
                    # 采样
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # 贪心选择
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 添加到序列
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # 检查是否生成了结束标记
                if (next_token == self.config.eos_token_id).all():
                    break
            
            return tgt
    
    def _create_padding_mask(self, token_ids: Tensor, pad_token_id: int) -> Tensor:
        """创建填充掩码"""
        return AttentionMask.create_padding_mask(token_ids, pad_token_id)
    
    def get_input_embeddings(self) -> nn.Module:
        """获取输入嵌入层"""
        return self.src_embedding.token_embedding.embedding
    
    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        """设置输入嵌入层"""
        self.src_embedding.token_embedding.embedding = embeddings
        if not self.config.tie_word_embeddings:
            self.tgt_embedding.token_embedding.embedding = embeddings
    
    def get_output_embeddings(self) -> nn.Module:
        """获取输出嵌入层"""
        return self.output_projection
    
    def set_output_embeddings(self, embeddings: nn.Module) -> None:
        """设置输出嵌入层"""
        self.output_projection = embeddings


class TransformerForSequenceToSequence(Transformer):
    """
    用于序列到序列任务的 Transformer 模型
    """
    
    def __init__(self, config: TransformerConfig):
        if config.num_encoder_layers == 0 or config.num_decoder_layers == 0:
            raise ValueError("序列到序列模型需要编码器和解码器")
        super().__init__(config)


class TransformerForLanguageModeling(nn.Module):
    """
    用于语言建模任务的 Transformer 模型（仅解码器）
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # 修改配置为仅解码器
        config.num_encoder_layers = 0
        
        # 嵌入层
        self.embedding = TransformerEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_len=config.max_seq_len,
            padding_idx=config.pad_token_id,
            position_encoding_type=config.position_encoding_type,
            dropout=config.dropout,
            scale_embedding=config.scale_embedding
        )
        
        # 解码器（作为主干网络）
        self.transformer = create_decoder(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_layers=config.num_decoder_layers,
            dropout=config.dropout,
            activation=config.activation,
            norm_first=config.norm_first
        )
        
        # 语言建模头
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重共享
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.token_embedding.embedding.weight
        
        # 初始化权重
        self.apply(lambda module: init_weights(module, config.init_std))
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_dict: bool = True
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        前向传播
        
        Args:
            input_ids (Tensor): 输入词元ID [batch_size, seq_len]
            attention_mask (Optional[Tensor]): 注意力掩码
            labels (Optional[Tensor]): 标签（用于计算损失）
            return_dict (bool): 是否返回字典格式
            
        Returns:
            Union[Tensor, Dict[str, Tensor]]: 模型输出
        """
        # 嵌入
        hidden_states = self.embedding(input_ids)
        
        # 创建因果掩码
        seq_len = input_ids.size(1)
        causal_mask = AttentionMask.create_causal_mask(seq_len, device=input_ids.device)
        causal_mask = causal_mask.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        
        # 组合掩码
        if attention_mask is not None:
            # 将填充掩码与因果掩码结合
            attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            causal_mask = causal_mask * attention_mask
        
        # Transformer 前向传播（使用解码器作为主干）
        # 注意：这里我们传递 None 作为 memory，因为这是仅解码器模型
        transformer_output = self.transformer(
            hidden_states, memory=None,
            tgt_mask=causal_mask, memory_mask=None
        )
        
        if isinstance(transformer_output, dict):
            hidden_states = transformer_output['last_hidden_state']
        else:
            hidden_states = transformer_output
        
        # 语言建模头
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # 计算交叉熵损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
        
        if return_dict:
            return {
                'logits': logits,
                'loss': loss,
                'last_hidden_state': hidden_states,
            }
        else:
            return logits


# 导出的类
__all__ = [
    'Transformer',
    'TransformerForSequenceToSequence',
    'TransformerForLanguageModeling',
]
