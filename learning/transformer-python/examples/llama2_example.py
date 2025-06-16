"""
LLaMA2 模型使用示例

展示如何使用基于第四章和第五章理论实现的 LLaMA2 模型进行：
1. 模型创建和配置
2. 前向传播
3. 文本生成
4. 模型训练（简化版）

作者: shihom_wu
基于: Happy-LLM 项目第四章和第五章理论
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

# 导入我们的 LLaMA2 实现
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_pytorch.models.llama2 import (
    LLaMA2Config, 
    LLaMA2Model, 
    LLaMA2ForCausalLM
)


def create_sample_data(vocab_size: int = 1000, seq_len: int = 64, num_samples: int = 100):
    """
    创建示例训练数据
    
    Args:
        vocab_size: 词汇表大小
        seq_len: 序列长度
        num_samples: 样本数量
        
    Returns:
        DataLoader: 训练数据加载器
    """
    # 生成随机词元序列
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # 标签是输入序列向右移动一位
    labels = torch.cat([input_ids[:, 1:], torch.zeros(num_samples, 1, dtype=torch.long)], dim=1)
    
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    return dataloader


def example_model_creation():
    """
    示例1: 模型创建和配置
    """
    print("=" * 60)
    print("示例1: LLaMA2 模型创建和配置")
    print("=" * 60)
    
    # 创建不同规模的模型配置
    configs = {
        "小型模型": LLaMA2Config(
            vocab_size=1000,
            d_model=256,
            num_layers=6,
            num_heads=8,
            num_kv_heads=8,
            d_ff=1024,
            max_seq_len=128
        ),
        "7B模型配置": LLaMA2Config.llama2_7b(),
        "13B模型配置": LLaMA2Config.llama2_13b(),
        "70B模型配置": LLaMA2Config.llama2_70b()
    }
    
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  词汇表大小: {config.vocab_size}")
        print(f"  模型维度: {config.d_model}")
        print(f"  层数: {config.num_layers}")
        print(f"  注意力头数: {config.num_heads}")
        print(f"  键值头数: {config.num_kv_heads}")
        print(f"  前馈维度: {config.d_ff}")
        print(f"  最大序列长度: {config.max_seq_len}")
        
        if name == "小型模型":
            # 只为小型模型创建实际的模型实例
            model = LLaMA2ForCausalLM(config)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  总参数量: {total_params:,}")


def example_forward_pass():
    """
    示例2: 前向传播
    """
    print("\n" + "=" * 60)
    print("示例2: LLaMA2 前向传播")
    print("=" * 60)
    
    # 创建小型模型
    config = LLaMA2Config(
        vocab_size=1000,
        d_model=256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,  # 使用分组查询注意力
        d_ff=1024,
        max_seq_len=64
    )
    
    model = LLaMA2ForCausalLM(config)
    model.eval()
    
    # 创建示例输入
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"输入形状: {input_ids.shape}")
    print(f"输入内容: {input_ids[0][:10].tolist()}...")
    
    # 前向传播
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_ids)
        end_time = time.time()
    
    logits = outputs['logits']
    print(f"输出logits形状: {logits.shape}")
    print(f"前向传播耗时: {(end_time - start_time) * 1000:.2f} ms")
    
    # 计算下一个词元的概率
    next_token_logits = logits[0, -1, :]  # 第一个样本的最后一个位置
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(next_token_probs, k=5)
    
    print(f"下一个词元的Top-5预测:")
    for i, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices)):
        print(f"  {i+1}. 词元ID {idx.item()}: 概率 {prob.item():.4f}")


def example_text_generation():
    """
    示例3: 文本生成
    """
    print("\n" + "=" * 60)
    print("示例3: LLaMA2 文本生成")
    print("=" * 60)
    
    # 创建小型模型
    config = LLaMA2Config(
        vocab_size=1000,
        d_model=256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        d_ff=1024,
        max_seq_len=64
    )
    
    model = LLaMA2ForCausalLM(config)
    model.eval()
    
    # 输入提示
    prompt = torch.tensor([[1, 123, 456, 789]])  # BOS + 一些词元
    print(f"输入提示: {prompt[0].tolist()}")
    
    # 不同的生成策略
    generation_configs = [
        {"max_new_tokens": 10, "temperature": 1.0, "do_sample": False},  # 贪心
        {"max_new_tokens": 10, "temperature": 0.8, "do_sample": True},   # 采样
        {"max_new_tokens": 10, "temperature": 1.2, "do_sample": True},   # 高温度采样
    ]
    
    for i, gen_config in enumerate(generation_configs):
        print(f"\n生成策略 {i+1}: {gen_config}")
        
        with torch.no_grad():
            start_time = time.time()
            generated = model.generate(prompt, **gen_config)
            end_time = time.time()
        
        print(f"生成结果: {generated[0].tolist()}")
        print(f"生成耗时: {(end_time - start_time) * 1000:.2f} ms")


def example_training():
    """
    示例4: 模型训练（简化版）
    """
    print("\n" + "=" * 60)
    print("示例4: LLaMA2 模型训练")
    print("=" * 60)
    
    # 创建小型模型
    config = LLaMA2Config(
        vocab_size=1000,
        d_model=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=512,
        max_seq_len=32
    )
    
    model = LLaMA2ForCausalLM(config)
    
    # 创建训练数据
    dataloader = create_sample_data(
        vocab_size=config.vocab_size,
        seq_len=config.max_seq_len,
        num_samples=50
    )
    
    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 训练循环
    model.train()
    total_loss = 0
    num_batches = 0
    
    print("开始训练...")
    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        if batch_idx >= 5:  # 只训练5个批次作为示例
            break
            
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        print(f"批次 {batch_idx + 1}, 损失: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"平均训练损失: {avg_loss:.4f}")


def main():
    """
    主函数：运行所有示例
    """
    print("LLaMA2 模型示例")
    print("基于 Happy-LLM 项目第四章和第五章理论实现")
    print("作者: shihom_wu")
    
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        example_model_creation()
        example_forward_pass()
        example_text_generation()
        example_training()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
