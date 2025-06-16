"""
LLaMA2 模型实现

基于第五章《动手搭建大模型》的理论，实现完整的 LLaMA2 架构。
包含以下关键组件：
- RMSNorm 归一化
- 旋转位置编码 (RoPE)
- 分组查询注意力 (GQA)
- SwiGLU 激活函数
- 完整的 LLaMA2 Transformer 块

作者: shihom_wu
基于: Happy-LLM 项目第五章理论
"""

import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.layers import RMSNorm, SwiGLU
from ..core.attention import GroupedQueryAttention
from ..core.embedding import RotaryPositionalEncoding, TokenEmbedding


@dataclass
class LLaMA2Config:
    """
    LLaMA2 模型配置
    
    Args:
        vocab_size (int): 词汇表大小
        d_model (int): 模型维度
        num_layers (int): Transformer 层数
        num_heads (int): 注意力头数
        num_kv_heads (int): 键值头数（用于 GQA）
        d_ff (int): 前馈网络隐藏层维度
        max_seq_len (int): 最大序列长度
        dropout (float): Dropout 概率
        rope_base (float): RoPE 基数
        norm_eps (float): 归一化的 epsilon
        pad_token_id (int): 填充词元 ID
        bos_token_id (int): 开始词元 ID
        eos_token_id (int): 结束词元 ID
    """
    vocab_size: int = 32000
    d_model: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32  # 标准多头注意力，设为与 num_heads 相同
    d_ff: int = 11008
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    @classmethod
    def llama2_7b(cls) -> 'LLaMA2Config':
        """LLaMA2 7B 模型配置"""
        return cls(
            vocab_size=32000,
            d_model=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=32,
            d_ff=11008,
            max_seq_len=2048
        )
    
    @classmethod
    def llama2_13b(cls) -> 'LLaMA2Config':
        """LLaMA2 13B 模型配置"""
        return cls(
            vocab_size=32000,
            d_model=5120,
            num_layers=40,
            num_heads=40,
            num_kv_heads=40,
            d_ff=13824,
            max_seq_len=2048
        )
    
    @classmethod
    def llama2_70b(cls) -> 'LLaMA2Config':
        """LLaMA2 70B 模型配置（使用 GQA）"""
        return cls(
            vocab_size=32000,
            d_model=8192,
            num_layers=80,
            num_heads=64,
            num_kv_heads=8,  # 使用分组查询注意力
            d_ff=28672,
            max_seq_len=2048
        )


class LLaMA2MLP(nn.Module):
    """
    LLaMA2 前馈网络
    
    使用 SwiGLU 激活函数的三层线性网络：
    MLP(x) = SwiGLU(xW_gate, xW_up)W_down
    
    Args:
        config (LLaMA2Config): 模型配置
    """
    
    def __init__(self, config: LLaMA2Config):
        super().__init__()
        self.config = config
        
        # 三个线性层
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        
        # SwiGLU 激活函数
        self.activation = nn.SiLU()  # SiLU 就是 Swish
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            Tensor: 输出张量 [batch_size, seq_len, d_model]
        """
        # SwiGLU: SiLU(xW_gate) ⊙ (xW_up)
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        
        # 下投影
        output = self.down_proj(hidden)
        return output


class LLaMA2DecoderLayer(nn.Module):
    """
    LLaMA2 解码器层
    
    包含：
    1. 自注意力机制（使用 GQA）
    2. 前馈网络（使用 SwiGLU）
    3. RMSNorm 归一化
    4. 残差连接
    
    Args:
        config (LLaMA2Config): 模型配置
    """
    
    def __init__(self, config: LLaMA2Config):
        super().__init__()
        self.config = config
        
        # 自注意力机制
        self.self_attn = GroupedQueryAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.dropout,
            bias=False
        )
        
        # 前馈网络
        self.mlp = LLaMA2MLP(config)
        
        # RMSNorm 归一化层
        self.input_layernorm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.norm_eps)
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        前向传播
        
        Args:
            hidden_states (Tensor): 输入隐藏状态 [batch_size, seq_len, d_model]
            attention_mask (Optional[Tensor]): 注意力掩码
            position_ids (Optional[Tensor]): 位置 ID
            return_attention (bool): 是否返回注意力权重
            
        Returns:
            Tuple[Tensor, Optional[Tensor]]: (输出隐藏状态, 注意力权重)
        """
        residual = hidden_states
        
        # 1. 自注意力 + 残差连接
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attention_weights = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            mask=attention_mask,
            return_attention=return_attention
        )
        hidden_states = residual + hidden_states
        
        # 2. 前馈网络 + 残差连接
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        if return_attention:
            return hidden_states, attention_weights
        else:
            return hidden_states, None


class LLaMA2Model(nn.Module):
    """
    LLaMA2 基础模型
    
    实现完整的 LLaMA2 Transformer 架构，包括：
    - 词嵌入层
    - 多层 LLaMA2 解码器
    - 最终的 RMSNorm 层
    
    Args:
        config (LLaMA2Config): 模型配置
    """
    
    def __init__(self, config: LLaMA2Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        # 词嵌入层
        self.embed_tokens = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            padding_idx=config.pad_token_id
        )
        
        # 旋转位置编码
        self.rotary_emb = RotaryPositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_len,
            base=config.rope_base
        )
        
        # Transformer 解码器层
        self.layers = nn.ModuleList([
            LLaMA2DecoderLayer(config) for _ in range(config.num_layers)
        ])
        
        # 最终归一化层
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        
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
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, Tensor]:
        """
        前向传播
        
        Args:
            input_ids (Tensor): 输入词元 ID [batch_size, seq_len]
            attention_mask (Optional[Tensor]): 注意力掩码
            position_ids (Optional[Tensor]): 位置 ID
            return_attention (bool): 是否返回注意力权重
            
        Returns:
            Dict[str, Tensor]: 包含 'last_hidden_state' 和可选的 'attentions'
        """
        batch_size, seq_len = input_ids.size()
        
        # 1. 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 2. 应用旋转位置编码（简化实现）
        # 注意：实际的 RoPE 应该在注意力计算中应用
        # hidden_states = self.rotary_emb(hidden_states, seq_len)
        
        # 3. 通过所有 Transformer 层
        all_attentions = [] if return_attention else None
        
        for layer in self.layers:
            hidden_states, attention_weights = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_attention=return_attention
            )
            
            if return_attention and attention_weights is not None:
                all_attentions.append(attention_weights)
        
        # 4. 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 构建输出字典
        outputs = {
            'last_hidden_state': hidden_states
        }
        
        if return_attention and all_attentions:
            outputs['attentions'] = torch.stack(all_attentions, dim=1)
        
        return outputs


class LLaMA2ForCausalLM(nn.Module):
    """
    LLaMA2 因果语言模型

    在 LLaMA2 基础模型上添加语言模型头，用于文本生成任务。

    Args:
        config (LLaMA2Config): 模型配置
    """

    def __init__(self, config: LLaMA2Config):
        super().__init__()
        self.config = config

        # 基础模型
        self.model = LLaMA2Model(config)

        # 语言模型头（输出层）
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重绑定：嵌入层和输出层共享权重
        self.lm_head.weight = self.model.embed_tokens.embedding.weight

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, Tensor]:
        """
        前向传播

        Args:
            input_ids (Tensor): 输入词元 ID [batch_size, seq_len]
            attention_mask (Optional[Tensor]): 注意力掩码
            labels (Optional[Tensor]): 标签，用于计算损失
            return_attention (bool): 是否返回注意力权重

        Returns:
            Dict[str, Tensor]: 包含 'logits'、可选的 'loss' 和 'attentions'
        """
        # 获取基础模型输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_attention=return_attention
        )

        hidden_states = outputs['last_hidden_state']

        # 计算 logits
        logits = self.lm_head(hidden_states)

        # 构建输出字典
        result = {
            'logits': logits
        }

        # 计算损失（如果提供了标签）
        if labels is not None:
            # 移位标签：预测下一个词元
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            result['loss'] = loss

        # 添加注意力权重
        if return_attention and 'attentions' in outputs:
            result['attentions'] = outputs['attentions']

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> Tensor:
        """
        文本生成

        Args:
            input_ids (Tensor): 输入词元 ID [batch_size, seq_len]
            max_new_tokens (int): 最大生成词元数
            temperature (float): 温度参数，控制随机性
            top_k (Optional[int]): Top-K 采样
            top_p (Optional[float]): Top-P (nucleus) 采样
            do_sample (bool): 是否使用采样，否则使用贪心解码
            pad_token_id (Optional[int]): 填充词元 ID
            eos_token_id (Optional[int]): 结束词元 ID

        Returns:
            Tensor: 生成的词元 ID [batch_size, seq_len + max_new_tokens]
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        batch_size = input_ids.size(0)
        device = input_ids.device

        # 生成序列
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # 前向传播
            outputs = self.forward(generated)
            logits = outputs['logits']

            # 获取最后一个位置的 logits
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # 采样或贪心解码
            if do_sample:
                # Top-K 采样
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                # Top-P 采样
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # 移除累积概率超过 top_p 的词元
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # 多项式采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # 贪心解码
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 添加到生成序列
            generated = torch.cat([generated, next_tokens], dim=1)

            # 检查是否生成了结束词元
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

        return generated


# 导出的类列表
__all__ = [
    'LLaMA2Config',
    'LLaMA2MLP',
    'LLaMA2DecoderLayer',
    'LLaMA2Model',
    'LLaMA2ForCausalLM',
]
