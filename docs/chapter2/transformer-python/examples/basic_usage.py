"""
Transformer-PyTorch 基础使用示例

演示如何使用各个组件和完整模型。

作者: Transformer-PyTorch Team
版本: 1.0.0
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 导入 Transformer-PyTorch 组件
from transformer_pytorch import (
    TransformerConfig,
    Transformer,
    get_config,
    create_config,
    print_config,
    get_device,
    set_seed,
    print_model_info,
)
from transformer_pytorch.core import (
    MultiHeadAttention,
    SelfAttention,
    TransformerEmbedding,
    create_encoder,
    create_decoder,
)


def example1_basic_components():
    """示例1：基础组件使用"""
    print("=== 示例1：基础组件使用 ===")
    
    # 设置随机种子
    set_seed(42)
    
    # 测试多头注意力
    print("测试多头注意力机制...")
    d_model, num_heads = 64, 4
    attention = MultiHeadAttention(d_model, num_heads, dropout=0.0)
    
    # 创建输入数据
    batch_size, seq_len = 2, 8
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output, attention_weights = attention(x, x, x)
    
    print(f"✅ 输入形状: {x.shape}")
    print(f"✅ 输出形状: {output.shape}")
    print(f"✅ 注意力权重形状: {attention_weights.shape}")
    print(f"✅ 参数数量: {sum(p.numel() for p in attention.parameters()):,}")
    
    # 测试嵌入层
    print("\n测试嵌入层...")
    vocab_size, max_len = 1000, 32
    embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout=0.0)
    
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    embedded = embedding(token_ids)
    
    print(f"✅ 词元ID形状: {token_ids.shape}")
    print(f"✅ 嵌入输出形状: {embedded.shape}")
    print(f"✅ 嵌入层参数数量: {sum(p.numel() for p in embedding.parameters()):,}")
    
    print()


def example2_model_configurations():
    """示例2：模型配置"""
    print("=== 示例2：模型配置 ===")
    
    # 预设配置
    configs = ['small', 'default', 'large']
    
    for config_name in configs:
        print(f"\n--- {config_name.upper()} 配置 ---")
        config = get_config(config_name)
        
        print(f"词汇表大小: {config.vocab_size:,}")
        print(f"模型维度: {config.d_model}")
        print(f"编码器层数: {config.num_encoder_layers}")
        print(f"解码器层数: {config.num_decoder_layers}")
        print(f"注意力头数: {config.num_heads}")
        print(f"最大序列长度: {config.max_seq_len}")
        
        # 参数估算
        params = config.estimate_parameters()
        print(f"估算参数量: {params['total_M']}")
    
    # 自定义配置
    print("\n--- 自定义配置 ---")
    custom_config = create_config(
        base_config=get_config('default'),
        vocab_size=8000,
        d_model=256,
        num_encoder_layers=4,
        num_decoder_layers=4,
        activation='gelu',
        dropout=0.05
    )
    
    print_config(custom_config)
    print()


def example3_complete_model():
    """示例3：完整模型使用"""
    print("=== 示例3：完整模型使用 ===")
    
    # 创建小型模型用于演示
    config = TransformerConfig(
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=256,
        max_seq_len=32,
        dropout=0.1
    )
    
    print("创建 Transformer 模型...")
    model = Transformer(config)
    
    # 打印模型信息
    print_model_info(model)
    
    # 准备示例数据
    batch_size = 2
    src_len, tgt_len = 10, 8
    
    src = torch.randint(1, config.vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, config.vocab_size, (batch_size, tgt_len))
    
    print(f"\n输入数据:")
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    
    # 前向传播
    print("\n执行前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)
    
    print(f"✅ 输出 logits 形状: {output['logits'].shape}")
    print(f"✅ 最后隐藏状态形状: {output['last_hidden_state'].shape}")
    
    # 计算困惑度
    logits = output['logits']
    # 移位标签用于语言建模损失
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = tgt[..., 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=config.pad_token_id
    )
    perplexity = torch.exp(loss)
    
    print(f"✅ 损失: {loss.item():.4f}")
    print(f"✅ 困惑度: {perplexity.item():.4f}")
    
    print()


def example4_encoder_decoder_separate():
    """示例4：编码器和解码器分别使用"""
    print("=== 示例4：编码器和解码器分别使用 ===")
    
    config = get_config('small')
    model = Transformer(config)
    model.eval()
    
    # 准备数据
    src = torch.randint(1, config.vocab_size, (1, 12))
    tgt = torch.randint(1, config.vocab_size, (1, 8))
    
    print("分步执行编码-解码...")
    
    with torch.no_grad():
        # 1. 仅编码
        print("1. 编码源序列...")
        encoder_output = model.encode(src)
        memory = encoder_output['last_hidden_state']
        print(f"   编码器输出形状: {memory.shape}")
        
        # 2. 仅解码
        print("2. 解码目标序列...")
        decoder_output = model.decode(tgt, memory)
        logits = decoder_output['logits']
        print(f"   解码器输出形状: {logits.shape}")
        
        # 3. 预测下一个词元
        print("3. 预测下一个词元...")
        last_logits = logits[0, -1, :]  # 最后一个位置的 logits
        probs = F.softmax(last_logits, dim=-1)
        
        # Top-5 预测
        top_probs, top_indices = torch.topk(probs, 5)
        print("   Top-5 预测:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            print(f"     {i+1}. 词元 {idx.item()}: {prob.item():.4f}")
    
    print()


def example5_text_generation():
    """示例5：文本生成"""
    print("=== 示例5：文本生成 ===")
    
    # 使用小型配置
    config = TransformerConfig(
        vocab_size=500,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        max_seq_len=20
    )
    
    model = Transformer(config)
    model.eval()
    
    # 源序列（用于条件生成）
    src = torch.tensor([[config.bos_token_id, 10, 20, 30, config.eos_token_id]])
    
    print(f"源序列: {src.tolist()[0]}")
    
    # 生成文本
    print("\n开始生成...")
    with torch.no_grad():
        generated = model.generate(
            src=src,
            max_length=15,
            temperature=0.8,
            do_sample=True
        )
    
    print(f"生成序列: {generated.tolist()[0]}")
    print(f"生成长度: {generated.size(1)}")
    
    print()


def example6_attention_visualization():
    """示例6：注意力可视化"""
    print("=== 示例6：注意力可视化 ===")
    
    # 创建简单模型
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=64,
        max_seq_len=10
    )
    
    model = Transformer(config)
    model.eval()
    
    # 准备数据
    src = torch.tensor([[1, 5, 10, 15, 2]])  # 包含 BOS 和 EOS
    tgt = torch.tensor([[1, 8, 12]])         # 目标序列
    
    print(f"源序列: {src.tolist()[0]}")
    print(f"目标序列: {tgt.tolist()[0]}")
    
    # 获取注意力权重
    with torch.no_grad():
        output = model(src, tgt, return_dict=True)
    
    # 可视化编码器注意力
    if output['encoder_attentions'] is not None:
        encoder_attn = output['encoder_attentions'][0]  # 第一层
        head_0_attn = encoder_attn[0, 0].numpy()  # 第一个头
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            head_0_attn,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            xticklabels=[f'Pos{i}' for i in range(head_0_attn.shape[1])],
            yticklabels=[f'Pos{i}' for i in range(head_0_attn.shape[0])]
        )
        plt.title('编码器自注意力权重 (第1层, 第1头)')
        plt.xlabel('键位置')
        plt.ylabel('查询位置')
        plt.tight_layout()
        plt.savefig('encoder_attention.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ 编码器注意力可视化已保存为 'encoder_attention.png'")
    
    # 可视化解码器自注意力
    if output['decoder_self_attentions'] is not None:
        decoder_self_attn = output['decoder_self_attentions'][0]  # 第一层
        head_0_attn = decoder_self_attn[0, 0].numpy()  # 第一个头
        
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            head_0_attn,
            annot=True,
            fmt='.3f',
            cmap='Reds',
            xticklabels=[f'Pos{i}' for i in range(head_0_attn.shape[1])],
            yticklabels=[f'Pos{i}' for i in range(head_0_attn.shape[0])]
        )
        plt.title('解码器自注意力权重 (第1层, 第1头)')
        plt.xlabel('键位置')
        plt.ylabel('查询位置')
        plt.tight_layout()
        plt.savefig('decoder_self_attention.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ 解码器自注意力可视化已保存为 'decoder_self_attention.png'")
    
    print()


def example7_device_usage():
    """示例7：设备使用（CPU/GPU）"""
    print("=== 示例7：设备使用 ===")
    
    # 自动检测设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建模型并移动到设备
    config = get_config('small')
    model = Transformer(config).to(device)
    
    # 创建数据并移动到设备
    src = torch.randint(1, config.vocab_size, (2, 8)).to(device)
    tgt = torch.randint(1, config.vocab_size, (2, 6)).to(device)
    
    print(f"模型设备: {next(model.parameters()).device}")
    print(f"数据设备: {src.device}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        if device.type == 'cuda':
            # 使用混合精度
            with torch.cuda.amp.autocast():
                output = model(src, tgt)
        else:
            output = model(src, tgt)
    
    print(f"✅ 输出设备: {output['logits'].device}")
    print(f"✅ 输出形状: {output['logits'].shape}")
    
    # 显示内存使用情况
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
        print(f"GPU 内存使用: {memory_allocated:.1f} MB (已分配)")
        print(f"GPU 内存保留: {memory_reserved:.1f} MB (已保留)")
    
    print()


def run_all_examples():
    """运行所有示例"""
    print("🚀 Transformer-PyTorch 基础使用示例\n")
    
    try:
        example1_basic_components()
        example2_model_configurations()
        example3_complete_model()
        example4_encoder_decoder_separate()
        example5_text_generation()
        example6_attention_visualization()
        example7_device_usage()
        
        print("✅ 所有示例运行完成！")
        print("\n💡 提示:")
        print("- 这些示例展示了 Transformer-PyTorch 的主要功能")
        print("- 可以根据需要修改配置和参数")
        print("- 查看生成的注意力可视化图片")
        print("- 在 GPU 上运行可以获得更好的性能")
        
    except Exception as e:
        print(f"❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_examples()
