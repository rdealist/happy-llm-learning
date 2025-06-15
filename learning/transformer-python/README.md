# Transformer-PyTorch

> 基于 PyTorch 的 Transformer 架构实现，专为学习和研究设计
>
> **🆕 第三章扩展：预训练语言模型支持**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.9%2B-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📖 项目简介

Transformer-PyTorch 是一个完整的 Transformer 架构实现，基于《第二章 Transformer架构》和《第三章 预训练语言模型》文档中的理论知识，使用 PyTorch 框架构建。本项目提供了模块化、易于理解和使用的 Transformer 组件，现已扩展支持 BERT、GPT、T5 等主流预训练语言模型，支持 GPU 加速和现代深度学习最佳实践。

### ✨ 主要特性

- 🧩 **模块化设计** - 每个组件独立实现，支持按需使用和自定义
- 🚀 **PyTorch 优化** - 充分利用 PyTorch 的自动微分和 GPU 加速
- 📚 **详细注释** - 所有代码都有中文注释和完整的 docstring
- 🔧 **灵活配置** - 支持多种预设配置和自定义参数
- 🎯 **教育友好** - 代码结构清晰，便于学习和理解 Transformer 原理
- ⚡ **高性能** - 支持 GPU 加速、混合精度训练和批处理计算
- 🔬 **研究导向** - 易于扩展和修改，适合研究和实验
- 🤖 **预训练模型** - 支持 BERT、GPT、T5 等主流预训练语言模型
- 📝 **预训练任务** - 实现 MLM、NSP、SOP、CLM 等预训练任务

### 🏗️ 架构组件

#### 核心组件
- **数学工具** - 激活函数、注意力计算、初始化函数
- **基础层** - 线性层、层归一化、RMSNorm、前馈网络、Dropout
- **注意力机制** - 多头注意力、自注意力、交叉注意力、掩码工具
- **嵌入层** - 词嵌入、正弦位置编码、可学习位置编码
- **编码器** - Transformer 编码器层和编码器块
- **解码器** - Transformer 解码器层和解码器块
- **完整模型** - 端到端的 Transformer 模型，支持多种任务

#### 预训练语言模型
- **BERT 系列** - BERT、RoBERTa、ALBERT 等双向语言模型
- **GPT 系列** - GPT、GPT-2 等自回归语言模型
- **T5 模型** - 文本到文本转换的 Encoder-Decoder 模型
- **分类头** - 序列分类、Token分类、语言建模、问答等任务头
- **预训练任务** - MLM、NSP、SOP、CLM 等预训练任务实现

## 🚀 快速开始

### 安装

```bash
# 从源码安装
git clone https://github.com/transformer-pytorch/transformer-pytorch.git
cd transformer-pytorch
pip install -e .

# 或者直接安装依赖
pip install -r requirements.txt
```

### 基础使用

```python
import torch
from transformer_pytorch import TransformerConfig, Transformer

# 创建配置
config = TransformerConfig(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_seq_len=512
)

# 创建模型
model = Transformer(config)

# 准备数据
src = torch.randint(0, 10000, (2, 10))  # [batch_size, src_len]
tgt = torch.randint(0, 10000, (2, 8))   # [batch_size, tgt_len]

# 前向传播
output = model(src, tgt)
print(f"输出形状: {output['logits'].shape}")  # [2, 8, 10000]
```

### 使用预设配置

```python
from transformer_pytorch.config import get_config

# 使用预设配置
config = get_config('small')  # 'small', 'default', 'large'
model = Transformer(config)

# 查看模型信息
from transformer_pytorch import print_model_info
print_model_info(model)
```

### 预训练语言模型使用

```python
from transformer_pytorch.models import ModelFactory

# 创建BERT分类模型
bert_model = ModelFactory.create_bert(
    model_name='bert-base',
    task_type='classification',
    num_labels=2
)

# 创建GPT语言模型
gpt_model = ModelFactory.create_gpt(
    model_name='gpt2',
    task_type='causal_lm'
)

# 创建T5条件生成模型
t5_model = ModelFactory.create_t5(
    model_name='t5-base',
    task_type='conditional_generation'
)

# BERT文本分类
input_ids = torch.randint(0, 1000, (2, 16))
outputs = bert_model(input_ids=input_ids)
predictions = torch.argmax(outputs['logits'], dim=-1)

# GPT文本生成
prompt = torch.randint(0, 1000, (1, 5))
generated = gpt_model.generate(
    input_ids=prompt,
    max_length=20,
    temperature=0.8,
    do_sample=True
)

# T5文本转换
src_ids = torch.randint(0, 1000, (1, 10))
generated = t5_model.generate(
    input_ids=src_ids,
    max_length=15
)
```

### GPU 加速

```python
from transformer_pytorch import get_device

# 自动检测设备
device = get_device()
model = model.to(device)
src = src.to(device)
tgt = tgt.to(device)

# 前向传播
with torch.cuda.amp.autocast():  # 混合精度
    output = model(src, tgt)
```

## 📁 项目结构

```
transformer-pytorch/
├── transformer_pytorch/           # 主包
│   ├── core/                     # 核心组件
│   │   ├── math_utils.py         # 数学工具函数
│   │   ├── layers.py             # 基础神经网络层
│   │   ├── attention.py          # 注意力机制
│   │   ├── embedding.py          # 嵌入层
│   │   ├── encoder.py            # 编码器
│   │   ├── decoder.py            # 解码器
│   │   └── transformer.py        # 完整模型
│   ├── config/                   # 配置管理
│   │   ├── config.py             # 模型配置
│   │   └── constants.py          # 常量定义
│   └── __init__.py               # 包初始化
├── examples/                     # 使用示例
├── tests/                        # 单元测试
├── notebooks/                    # Jupyter 教程
├── docs/                         # 详细文档
├── setup.py                      # 安装脚本
├── requirements.txt              # 依赖列表
└── README.md                     # 项目说明
```

## 🔧 配置选项

### 预设配置

| 配置名称 | 词汇表 | 模型维度 | 层数 | 注意力头数 | 最大序列长度 | 参数量 |
|----------|--------|----------|------|------------|--------------|--------|
| `small` | 10,000 | 256 | 4 | 4 | 128 | ~11M |
| `default` | 30,000 | 512 | 6 | 8 | 512 | ~65M |
| `large` | 50,000 | 1024 | 12 | 16 | 1024 | ~355M |

### 自定义配置

```python
from transformer_pytorch.config import create_config, get_config

# 基于默认配置创建自定义配置
custom_config = create_config(
    base_config=get_config('default'),
    vocab_size=8000,
    d_model=256,
    num_encoder_layers=4,
    num_decoder_layers=4,
    activation='gelu',
    dropout=0.05
)
```

## 📚 详细教程

### Jupyter Notebooks

- [01_基础概念.ipynb](notebooks/01_basic_concepts.ipynb) - Transformer 基础概念
- [02_注意力机制.ipynb](notebooks/02_attention_mechanism.ipynb) - 注意力机制详解
- [03_完整模型.ipynb](notebooks/03_complete_model.ipynb) - 完整模型使用
- [04_训练示例.ipynb](notebooks/04_training_example.ipynb) - 训练示例
- [05_可视化分析.ipynb](notebooks/05_visualization.ipynb) - 注意力可视化

### 代码示例

```python
# 单独使用注意力机制
from transformer_pytorch.core import MultiHeadAttention

attention = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)
output, weights = attention(x, x, x)

# 单独使用编码器
from transformer_pytorch.core import create_encoder

encoder = create_encoder(
    d_model=512, num_heads=8, d_ff=2048, num_layers=6
)
encoded = encoder(x)

# 文本生成
model.eval()
generated = model.generate(
    src=src,
    max_length=50,
    temperature=0.8,
    do_sample=True
)
```

## 🧪 测试和验证

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_attention.py -v

# 生成覆盖率报告
pytest tests/ --cov=transformer_pytorch --cov-report=html
```

### 性能基准

```bash
# 运行性能测试
python examples/benchmark.py

# 内存使用分析
python examples/memory_analysis.py
```

## 🎯 使用场景

### 教育学习
- 理解 Transformer 架构原理
- 学习注意力机制实现
- 深度学习概念实践

### 研究开发
- 快速原型验证
- 算法改进和实验
- 新架构探索

### 实际应用
- 机器翻译
- 文本摘要
- 问答系统
- 语言建模

## ⚡ 性能优化

### GPU 加速
```python
# 使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 混合精度训练
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(src, tgt)
```

### 内存优化
```python
# 梯度检查点
from torch.utils.checkpoint import checkpoint

# 在模型中使用
output = checkpoint(layer, x)
```

## 🤝 贡献指南

我们欢迎各种形式的贡献！

1. **报告问题** - 在 Issues 中报告 bug 或提出功能请求
2. **代码贡献** - Fork 项目，创建分支，提交 Pull Request
3. **文档改进** - 改进文档、添加示例、修正错误
4. **测试用例** - 添加测试用例，提高代码覆盖率

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/transformer-pytorch/transformer-pytorch.git
cd transformer-pytorch

# 安装开发依赖
pip install -e ".[dev]"

# 安装 pre-commit 钩子
pre-commit install

# 运行代码格式化
black transformer_pytorch/
isort transformer_pytorch/

# 运行类型检查
mypy transformer_pytorch/
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢 Vaswani 等人的开创性论文 "Attention Is All You Need"
- 感谢《第二章 Transformer架构》文档提供的理论基础
- 感谢 PyTorch 团队提供的优秀深度学习框架
- 感谢所有为开源社区做出贡献的开发者

## 📞 联系方式

- **GitHub Issues**: [提交问题](https://github.com/transformer-pytorch/transformer-pytorch/issues)
- **邮件**: transformer-pytorch@example.com
- **文档**: [在线文档](https://transformer-pytorch.readthedocs.io/)

---

**Transformer-PyTorch** - 让 Transformer 架构的学习和研究变得简单高效！
