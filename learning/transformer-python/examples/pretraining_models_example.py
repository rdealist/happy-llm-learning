"""
预训练语言模型使用示例
演示BERT、GPT、T5等模型的基本使用方法

作者: shihom_wu
版本: 1.0.0
"""

import torch
import torch.nn.functional as F
from transformer_pytorch.models import ModelFactory
from transformer_pytorch.pretraining import PretrainingDataProcessor


def bert_example():
    """BERT模型使用示例"""
    print("=" * 50)
    print("BERT模型示例")
    print("=" * 50)
    
    # 创建BERT分类模型
    model = ModelFactory.create_bert(
        model_name='bert-small',
        task_type='classification',
        num_labels=2
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 准备输入数据
    batch_size = 2
    seq_len = 16
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    token_type_ids = torch.zeros(batch_size, seq_len)
    labels = torch.randint(0, 2, (batch_size,))
    
    print(f"输入形状: {input_ids.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
    
    print(f"输出logits形状: {outputs['logits'].shape}")
    print(f"损失: {outputs['loss'].item():.4f}")
    
    # 预测概率
    probs = F.softmax(outputs['logits'], dim=-1)
    predictions = torch.argmax(probs, dim=-1)
    print(f"预测结果: {predictions.tolist()}")
    print(f"预测概率: {probs.tolist()}")


def gpt_example():
    """GPT模型使用示例"""
    print("\n" + "=" * 50)
    print("GPT模型示例")
    print("=" * 50)
    
    # 创建GPT语言模型
    model = ModelFactory.create_gpt(
        model_name='gpt2-small',
        task_type='causal_lm'
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 准备输入数据
    batch_size = 1
    seq_len = 10
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"输入token IDs: {input_ids[0].tolist()}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    print(f"输出logits形状: {outputs['logits'].shape}")
    
    # 文本生成
    print("\n文本生成示例:")
    prompt = torch.randint(0, 1000, (1, 5))  # 5个token的提示
    print(f"提示token IDs: {prompt[0].tolist()}")
    
    generated = model.generate(
        input_ids=prompt,
        max_length=15,
        temperature=0.8,
        do_sample=True
    )
    
    print(f"生成的token IDs: {generated[0].tolist()}")
    print(f"生成长度: {generated.shape[1]}")


def t5_example():
    """T5模型使用示例"""
    print("\n" + "=" * 50)
    print("T5模型示例")
    print("=" * 50)
    
    # 创建T5条件生成模型
    model = ModelFactory.create_t5(
        model_name='t5-small',
        task_type='conditional_generation'
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 准备输入数据
    batch_size = 1
    src_len = 12
    tgt_len = 8
    
    input_ids = torch.randint(0, 1000, (batch_size, src_len))
    decoder_input_ids = torch.randint(0, 1000, (batch_size, tgt_len))
    attention_mask = torch.ones(batch_size, src_len)
    decoder_attention_mask = torch.ones(batch_size, tgt_len)
    
    print(f"编码器输入形状: {input_ids.shape}")
    print(f"解码器输入形状: {decoder_input_ids.shape}")
    print(f"编码器输入token IDs: {input_ids[0].tolist()}")
    print(f"解码器输入token IDs: {decoder_input_ids[0].tolist()}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask
        )
    
    print(f"输出logits形状: {outputs['logits'].shape}")
    
    # 文本生成
    print("\n条件生成示例:")
    source = torch.randint(0, 1000, (1, 8))  # 源序列
    print(f"源序列token IDs: {source[0].tolist()}")
    
    generated = model.generate(
        input_ids=source,
        max_length=12,
        temperature=0.8,
        do_sample=True
    )
    
    print(f"生成的token IDs: {generated[0].tolist()}")
    print(f"生成长度: {generated.shape[1]}")


def pretraining_tasks_example():
    """预训练任务使用示例"""
    print("\n" + "=" * 50)
    print("预训练任务示例")
    print("=" * 50)
    
    # 创建数据处理器
    config = {
        'mlm': {
            'vocab_size': 1000,
            'mask_ratio': 0.15,
            'mask_token_id': 103,
            'cls_token_id': 101,
            'sep_token_id': 102,
            'pad_token_id': 0
        },
        'nsp': {
            'cls_token_id': 101,
            'sep_token_id': 102,
            'pad_token_id': 0
        }
    }
    
    processor = PretrainingDataProcessor(config)
    
    # BERT风格数据处理示例
    print("BERT风格数据处理:")
    sentence_a = [10, 25, 67, 89, 45]
    sentence_b = [12, 34, 78, 90]
    is_next = True
    
    bert_data = processor.process_bert_data(sentence_a, sentence_b, is_next)
    
    print(f"原始句子A: {sentence_a}")
    print(f"原始句子B: {sentence_b}")
    print(f"是否连续: {is_next}")
    print(f"处理后输入: {bert_data['input_ids'].tolist()}")
    print(f"Token类型: {bert_data['token_type_ids'].tolist()}")
    print(f"MLM标签: {bert_data['mlm_labels'].tolist()}")
    print(f"NSP标签: {bert_data['nsp_labels'].item()}")
    print(f"掩码位置: {bert_data['mask_positions'].tolist()}")
    
    # GPT风格数据处理示例
    print("\nGPT风格数据处理:")
    token_ids = [10, 25, 67, 89, 45, 12, 34, 78]
    
    gpt_data = processor.process_gpt_data(token_ids)
    
    print(f"原始序列: {token_ids}")
    print(f"输入序列: {gpt_data['input_ids'].tolist()}")
    print(f"标签序列: {gpt_data['labels'].tolist()}")
    print(f"注意力掩码形状: {gpt_data['attention_mask'].shape}")


def model_comparison():
    """模型对比示例"""
    print("\n" + "=" * 50)
    print("模型对比")
    print("=" * 50)
    
    models = {
        'BERT-small': ModelFactory.create_bert('bert-small', 'base'),
        'GPT2-small': ModelFactory.create_gpt('gpt2-small', 'base'),
        'T5-small': ModelFactory.create_t5('t5-small', 'base')
    }
    
    print(f"{'模型':<15} {'参数数量':<15} {'类型':<20}")
    print("-" * 50)
    
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        model_type = "Encoder-only" if "BERT" in name else \
                    "Decoder-only" if "GPT" in name else \
                    "Encoder-Decoder"
        
        print(f"{name:<15} {param_count:>10,}     {model_type:<20}")


def main():
    """主函数"""
    print("预训练语言模型使用示例")
    print("作者: shihom_wu")
    print("版本: 1.0.0")
    
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    
    try:
        # 运行各种示例
        bert_example()
        gpt_example()
        t5_example()
        pretraining_tasks_example()
        model_comparison()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
