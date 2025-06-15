"""
基础功能单元测试

测试 Transformer-PyTorch 的核心组件和基础功能。

作者: Transformer-PyTorch Team
版本: 1.0.0
"""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# 导入要测试的模块
from transformer_pytorch.core.math_utils import (
    gelu_activation,
    scaled_dot_product_attention,
    create_causal_mask,
    create_padding_mask,
)
from transformer_pytorch.core.layers import LayerNorm, FeedForward
from transformer_pytorch.core.attention import MultiHeadAttention, SelfAttention
from transformer_pytorch.core.embedding import (
    TokenEmbedding,
    SinusoidalPositionalEncoding,
    TransformerEmbedding,
)
from transformer_pytorch.core.encoder import EncoderLayer, create_encoder
from transformer_pytorch.core.decoder import DecoderLayer, create_decoder
from transformer_pytorch.core.transformer import Transformer
from transformer_pytorch.config.config import TransformerConfig, get_config


class TestMathUtils:
    """测试数学工具函数"""
    
    def test_gelu_activation(self):
        """测试 GELU 激活函数"""
        x = torch.randn(2, 3, 4)
        output = gelu_activation(x)
        
        assert output.shape == x.shape
        assert output.dtype == x.dtype
        
        # 测试 GELU(0) ≈ 0
        zero_input = torch.zeros(1)
        zero_output = gelu_activation(zero_input)
        assert torch.abs(zero_output).item() < 1e-6
    
    def test_scaled_dot_product_attention(self):
        """测试缩放点积注意力"""
        batch_size, seq_len, d_model = 2, 5, 8
        
        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = scaled_dot_product_attention(q, k, v)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)
        
        # 检查注意力权重是否归一化
        weight_sums = attention_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)
    
    def test_create_causal_mask(self):
        """测试因果掩码生成"""
        seq_len = 4
        mask = create_causal_mask(seq_len)
        
        assert mask.shape == (seq_len, seq_len)
        
        # 检查下三角结构
        expected = torch.tril(torch.ones(seq_len, seq_len))
        assert torch.equal(mask, expected)
    
    def test_create_padding_mask(self):
        """测试填充掩码生成"""
        token_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        mask = create_padding_mask(token_ids, pad_token_id=0)
        
        expected = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        assert torch.equal(mask, expected)


class TestLayers:
    """测试基础层"""
    
    def test_layer_norm(self):
        """测试层归一化"""
        d_model = 8
        layer_norm = LayerNorm(d_model)
        
        x = torch.randn(2, 5, d_model)
        output = layer_norm(x)
        
        assert output.shape == x.shape
        
        # 检查归一化效果（均值接近0，方差接近1）
        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)
    
    def test_feed_forward(self):
        """测试前馈网络"""
        d_model, d_ff = 8, 16
        ffn = FeedForward(d_model, d_ff)
        
        x = torch.randn(2, 5, d_model)
        output = ffn(x)
        
        assert output.shape == x.shape
        
        # 测试参数数量
        expected_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
        actual_params = sum(p.numel() for p in ffn.parameters())
        assert actual_params == expected_params


class TestAttention:
    """测试注意力机制"""
    
    def test_multi_head_attention(self):
        """测试多头注意力"""
        d_model, num_heads = 8, 2
        attention = MultiHeadAttention(d_model, num_heads)
        
        x = torch.randn(2, 5, d_model)
        output, weights = attention(x, x, x)
        
        assert output.shape == x.shape
        assert weights.shape == (2, num_heads, 5, 5)
        
        # 检查注意力权重归一化
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)
    
    def test_self_attention(self):
        """测试自注意力"""
        d_model, num_heads = 8, 2
        self_attn = SelfAttention(d_model, num_heads)
        
        x = torch.randn(2, 5, d_model)
        output, weights = self_attn(x)
        
        assert output.shape == x.shape
        assert weights.shape == (2, num_heads, 5, 5)


class TestEmbedding:
    """测试嵌入层"""
    
    def test_token_embedding(self):
        """测试词嵌入"""
        vocab_size, d_model = 100, 8
        embedding = TokenEmbedding(vocab_size, d_model)
        
        token_ids = torch.randint(0, vocab_size, (2, 5))
        output = embedding(token_ids)
        
        assert output.shape == (2, 5, d_model)
    
    def test_sinusoidal_positional_encoding(self):
        """测试正弦位置编码"""
        d_model, max_len = 8, 10
        pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        x = torch.randn(2, 5, d_model)
        output = pos_encoding(x)
        
        assert output.shape == x.shape
        
        # 检查位置编码是否被正确添加
        assert not torch.equal(output, x)
    
    def test_transformer_embedding(self):
        """测试完整嵌入层"""
        vocab_size, d_model, max_len = 100, 8, 10
        embedding = TransformerEmbedding(vocab_size, d_model, max_len)
        
        token_ids = torch.randint(0, vocab_size, (2, 5))
        output = embedding(token_ids)
        
        assert output.shape == (2, 5, d_model)


class TestEncoder:
    """测试编码器"""
    
    def test_encoder_layer(self):
        """测试编码器层"""
        d_model, num_heads, d_ff = 8, 2, 16
        layer = EncoderLayer(d_model, num_heads, d_ff)
        
        x = torch.randn(2, 5, d_model)
        output, attention = layer(x)
        
        assert output.shape == x.shape
        if attention is not None:
            assert attention.shape == (2, num_heads, 5, 5)
    
    def test_encoder(self):
        """测试编码器"""
        d_model, num_heads, d_ff, num_layers = 8, 2, 16, 3
        encoder = create_encoder(d_model, num_heads, d_ff, num_layers)
        
        x = torch.randn(2, 5, d_model)
        output = encoder(x)
        
        assert output.shape == x.shape


class TestDecoder:
    """测试解码器"""
    
    def test_decoder_layer(self):
        """测试解码器层"""
        d_model, num_heads, d_ff = 8, 2, 16
        layer = DecoderLayer(d_model, num_heads, d_ff)
        
        tgt = torch.randn(2, 4, d_model)
        memory = torch.randn(2, 5, d_model)
        
        output, attention = layer(tgt, memory)
        
        assert output.shape == tgt.shape
    
    def test_decoder(self):
        """测试解码器"""
        d_model, num_heads, d_ff, num_layers = 8, 2, 16, 3
        decoder = create_decoder(d_model, num_heads, d_ff, num_layers)
        
        tgt = torch.randn(2, 4, d_model)
        memory = torch.randn(2, 5, d_model)
        
        output = decoder(tgt, memory)
        
        assert output.shape == tgt.shape


class TestConfig:
    """测试配置系统"""
    
    def test_transformer_config(self):
        """测试配置类"""
        config = TransformerConfig(
            vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        assert config.vocab_size == 1000
        assert config.d_model == 128
        assert config.num_heads == 4
        
        # 测试验证
        config.validate()
    
    def test_get_config(self):
        """测试预设配置"""
        config = get_config('small')
        
        assert isinstance(config, TransformerConfig)
        assert config.vocab_size > 0
        assert config.d_model > 0
        assert config.num_heads > 0
    
    def test_invalid_config(self):
        """测试无效配置"""
        with pytest.raises(ValueError):
            # d_model 不能被 num_heads 整除
            TransformerConfig(d_model=10, num_heads=3)


class TestTransformer:
    """测试完整模型"""
    
    def test_transformer_creation(self):
        """测试模型创建"""
        config = get_config('small')
        model = Transformer(config)
        
        assert isinstance(model, nn.Module)
        
        # 检查模型组件
        assert hasattr(model, 'src_embedding')
        assert hasattr(model, 'tgt_embedding')
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'output_projection')
    
    def test_transformer_forward(self):
        """测试模型前向传播"""
        config = TransformerConfig(
            vocab_size=100,
            d_model=32,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=64,
            max_seq_len=20
        )
        model = Transformer(config)
        
        src = torch.randint(0, 100, (2, 8))
        tgt = torch.randint(0, 100, (2, 6))
        
        output = model(src, tgt)
        
        assert 'logits' in output
        assert output['logits'].shape == (2, 6, 100)
    
    def test_transformer_encode_only(self):
        """测试仅编码"""
        config = get_config('small')
        model = Transformer(config)
        
        src = torch.randint(0, config.vocab_size, (2, 8))
        
        encoder_output = model.encode(src)
        
        assert 'last_hidden_state' in encoder_output
        assert encoder_output['last_hidden_state'].shape == (2, 8, config.d_model)
    
    def test_transformer_generate(self):
        """测试文本生成"""
        config = TransformerConfig(
            vocab_size=100,
            d_model=32,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=64,
            max_seq_len=20
        )
        model = Transformer(config)
        model.eval()
        
        src = torch.randint(0, 100, (1, 5))
        
        with torch.no_grad():
            generated = model.generate(src, max_length=10)
        
        assert generated.shape[0] == 1
        assert generated.shape[1] <= 10 + 1  # +1 for BOS token


class TestDeviceCompatibility:
    """测试设备兼容性"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """测试 CUDA 兼容性"""
        config = get_config('small')
        model = Transformer(config).cuda()
        
        src = torch.randint(0, config.vocab_size, (2, 8)).cuda()
        tgt = torch.randint(0, config.vocab_size, (2, 6)).cuda()
        
        output = model(src, tgt)
        
        assert output['logits'].is_cuda
        assert output['logits'].shape == (2, 6, config.vocab_size)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
