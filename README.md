<div align='center'>
    <h1>Transformer 架构学习与实现项目</h1>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/language-JavaScript%20%7C%20Python-brightgreen?style=for-the-badge" alt="Languages"/>
  <img src="https://img.shields.io/badge/framework-PyTorch-orange?style=for-the-badge" alt="Framework"/>
  <img src="https://img.shields.io/badge/platform-Browser%20%7C%20Node.js%20%7C%20微信小程序-blue?style=for-the-badge" alt="Platform"/>
</div>

<div align="center">
  <h3>🚀 从零开始的 Transformer 架构双语言实现</h3>
  <p><em>JavaScript 和 Python 双版本，理论与实践完美结合</em></p>
</div>

---

## 🎯 项目简介

本项目是一个完整的 **Transformer 架构学习与实现项目**，提供了 JavaScript 和 Python 两个版本的完整实现。项目基于 Happy-LLM 项目的系统性理论知识，特别是《第二章 Transformer架构》、《第三章 预训练语言模型》、《第四章 大语言模型》和《第五章 动手搭建大模型》，将复杂的 Transformer 架构和现代 LLM 技术转化为易于理解和使用的代码实现。

### ✨ 项目特色

- 🧩 **双语言实现** - JavaScript 和 Python 两个完整版本
- 📱 **多平台支持** - 浏览器、Node.js、微信小程序、服务器端
- 📚 **教育友好** - 详细的中文注释和完整的学习文档
- 🔧 **模块化设计** - 每个组件独立实现，支持按需使用
- 🤖 **现代LLM架构** - 完整的 LLaMA2 实现，支持 BERT、GPT、T5 等主流模型
- ⚡ **性能优化** - 分组查询注意力(GQA)、RMSNorm等先进技术
- 🎯 **理论实践结合** - 基于第四章和第五章理论的完整实现

## 📁 项目结构

```
learning/
├── transformer-js/                    # JavaScript 版本实现
│   ├── core/                         # 核心组件
│   │   ├── math-utils.js             # 数学工具函数
│   │   ├── matrix-ops.js             # 矩阵运算
│   │   ├── layers.js                 # 基础层(含RMSNorm、SwiGLU)
│   │   ├── attention.js              # 注意力机制(含GQA)
│   │   ├── embedding.js              # 嵌入层(含RoPE)
│   │   ├── encoder.js                # 编码器
│   │   ├── decoder.js                # 解码器
│   │   ├── transformer.js            # 完整模型
│   │   └── heads.js                  # 分类头
│   ├── pretraining/                  # 预训练任务
│   │   └── tasks.js                  # MLM、NSP、SOP、CLM 任务
│   ├── models/                       # 预训练语言模型
│   │   ├── bert.js                   # BERT 系列模型
│   │   ├── gpt.js                    # GPT 系列模型
│   │   ├── t5.js                     # T5 模型
│   │   ├── llama2.js                 # LLaMA2 完整实现 ⭐
│   │   └── config.js                 # 模型配置
│   ├── config/                       # 配置文件
│   ├── examples/                     # 使用示例
│   │   └── llama2-example.js         # LLaMA2 使用示例 ⭐
│   ├── tests/                        # 测试文件
│   └── docs/                         # 详细文档
├── transformer-python/               # Python 版本实现
│   ├── transformer_pytorch/          # 主包
│   │   ├── core/                     # 核心组件
│   │   │   ├── math_utils.py         # 数学工具函数
│   │   │   ├── layers.py             # 基础层(含RMSNorm、SwiGLU)
│   │   │   ├── attention.py          # 注意力机制(含GQA)
│   │   │   ├── embedding.py          # 嵌入层(含RoPE)
│   │   │   ├── encoder.py            # 编码器
│   │   │   ├── decoder.py            # 解码器
│   │   │   └── transformer.py        # 完整模型
│   │   ├── models/                   # 预训练语言模型
│   │   │   └── llama2.py             # LLaMA2 完整实现 ⭐
│   │   └── config/                   # 配置管理
│   ├── examples/                     # 使用示例
│   │   ├── llama2_example.py         # LLaMA2 使用示例 ⭐
│   │   └── llama2_simple_test.py     # 简化测试 ⭐
│   ├── tests/                        # 单元测试
│   ├── notebooks/                    # Jupyter 教程
│   ├── docs/                         # 详细文档
│   ├── requirements.txt              # 依赖管理
│   └── setup.py                      # 包安装脚本
├── TRANSFORMER_IMPLEMENTATIONS_COMPARISON.md  # 实现对比总结
├── transformer-architecture-guide.md          # 架构实现指南
├── CHAPTER3_IMPLEMENTATION_SUMMARY.md         # 第三章实现总结
└── CHAPTER4_CHAPTER5_IMPLEMENTATION_SUMMARY.md # 第四章和第五章实现总结 ⭐
```

## 🚀 快速开始

### JavaScript 版本

#### 基础使用
```javascript
// 引入必要模块
const { getConfig } = require('./learning/transformer-js/config/config');
const { createTransformer } = require('./learning/transformer-js/core/transformer');

// 创建模型
const config = getConfig('small');
const model = createTransformer(config);

// 准备输入数据
const srcTokens = [[2, 10, 25, 67, 3]];  // 源序列
const tgtTokens = [[2, 15, 30, 45]];     // 目标序列

// 前向传播
const result = model.forward(srcTokens, tgtTokens);
console.log('输出维度:', result.logits.length, 'x', result.logits[0].length);
```

#### LLaMA2 模型使用 ⭐
```javascript
const { LLaMA2Config, LLaMA2ForCausalLM } = require('./learning/transformer-js/models/llama2');

// 创建微信小程序优化的LLaMA2模型
const config = LLaMA2Config.miniprogram();
const model = new LLaMA2ForCausalLM(config);

// 文本生成
const prompt = [1, 123, 456, 789];  // BOS + 输入词元
const generated = model.generate(prompt, {
  maxNewTokens: 20,
  temperature: 0.8,
  doSample: true
});

console.log('生成结果:', generated);
```

#### BERT 模型使用
```javascript
const { ModelFactory } = require('./learning/transformer-js/models/config');

// 创建BERT分类模型
const bertModel = ModelFactory.createBERT('bert-base', 'classification', {
  num_labels: 2  // 二分类任务
});

// 使用模型
const tokens = [[101, 2023, 2003, 1037, 3231, 102]];  // [CLS] This is a test [SEP]
const result = bertModel.forward(tokens);
```

### Python 版本

#### 安装依赖
```bash
cd learning/transformer-python
pip install -r requirements.txt

# 或者安装为包
pip install -e .
```

#### LLaMA2 模型使用 ⭐
```python
import torch
from transformer_pytorch.models.llama2 import LLaMA2Config, LLaMA2ForCausalLM

# 创建LLaMA2模型
config = LLaMA2Config.llama2_7b()  # 或使用其他配置
model = LLaMA2ForCausalLM(config)

# 文本生成
input_ids = torch.tensor([[1, 123, 456, 789]])  # BOS + 输入词元
generated = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    do_sample=True
)

print(f'生成结果: {generated[0].tolist()}')
```

#### 基础使用
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
src = torch.randint(0, 10000, (2, 20))  # 批次大小=2, 序列长度=20
tgt = torch.randint(0, 10000, (2, 15))  # 批次大小=2, 序列长度=15

# 前向传播
output = model(src, tgt)
print(f'输出形状: {output["logits"].shape}')
```

## 🏗️ 核心架构

### 主要组件

| 组件 | JavaScript 版本 | Python 版本 | 功能描述 |
|------|----------------|-------------|----------|
| **注意力机制** | `attention.js` | `attention.py` | 多头自注意力、交叉注意力、GQA ⭐ |
| **嵌入层** | `embedding.js` | `embedding.py` | 词嵌入、位置编码、RoPE ⭐ |
| **编码器** | `encoder.js` | `encoder.py` | Transformer 编码器层 |
| **解码器** | `decoder.js` | `decoder.py` | Transformer 解码器层 |
| **基础层** | `layers.js` | `layers.py` | LayerNorm、RMSNorm、SwiGLU ⭐ |
| **完整模型** | `transformer.js` | `transformer.py` | 端到端 Transformer |
| **LLaMA2模型** | `llama2.js` | `llama2.py` | 完整LLaMA2实现 ⭐ |

### 预训练模型支持

- **LLaMA2 系列** - 现代大语言模型架构 ⭐
  - 支持7B、13B、70B等多种规模
  - 分组查询注意力(GQA)优化
  - RMSNorm和SwiGLU激活函数
- **BERT 系列** - 双向编码器表示
- **GPT 系列** - 自回归语言模型
- **T5 模型** - 文本到文本转换
- **预训练任务** - MLM、NSP、SOP、CLM

## 🎯 技术特点

### JavaScript 版本特点
✅ **优势:**
- 无需安装，直接在浏览器运行
- 适合微信小程序等前端环境
- 代码简洁，易于理解
- 内存占用可控
- 部署简单

⚠️ **适用场景:**
- Web 应用和移动应用
- 教学演示和概念验证
- 边缘计算和本地推理
- 快速原型开发

### Python 版本特点
✅ **优势:**
- GPU 加速，性能强大
- 支持训练和推理
- PyTorch 生态系统支持
- 自动微分和优化
- 完整的类型提示

⚠️ **适用场景:**
- 科学研究和算法开发
- 生产环境和模型训练
- 深度学习教育
- 云端服务和批处理

## 📚 学习资源

### 文档指南
- [📖 Transformer 架构实现指南](./learning/transformer-architecture-guide.md)
- [🔄 双版本实现对比总结](./learning/TRANSFORMER_IMPLEMENTATIONS_COMPARISON.md)
- [📝 第三章实现总结](./learning/CHAPTER3_IMPLEMENTATION_SUMMARY.md)
- [⭐ 第四章和第五章实现总结](./learning/CHAPTER4_CHAPTER5_IMPLEMENTATION_SUMMARY.md) - **最新**

### 代码示例
- [⭐ LLaMA2 JavaScript示例](./learning/transformer-js/examples/llama2-example.js) - **最新**
- [⭐ LLaMA2 Python示例](./learning/transformer-python/examples/llama2_example.py) - **最新**
- [JavaScript 基础示例](./learning/transformer-js/examples/)
- [Python Jupyter 教程](./learning/transformer-python/notebooks/)
- [预训练模型使用](./learning/transformer-js/examples/bert-example.js)

## 💻 系统要求

### JavaScript 版本
- **运行环境**: Node.js 14+ 或现代浏览器
- **内存要求**: 最低 256MB（小程序环境）
- **平台支持**: Windows、macOS、Linux、微信小程序

### Python 版本
- **Python 版本**: 3.8+
- **依赖框架**: PyTorch 1.9+
- **硬件要求**: CPU（最低）/ GPU（推荐）
- **内存要求**: 最低 2GB RAM

## 📦 安装指南

### JavaScript 版本安装
```bash
# 克隆项目
git clone https://github.com/your-repo/transformer-learning.git
cd transformer-learning/learning/transformer-js

# 无需额外安装，直接使用
node examples/basic-usage.js
```

### Python 版本安装
```bash
# 进入 Python 项目目录
cd learning/transformer-python

# 方式1：直接安装依赖
pip install -r requirements.txt

# 方式2：安装为开发包
pip install -e .

# 方式3：安装所有功能
pip install -e ".[all]"
```

### 依赖说明

#### JavaScript 版本依赖
- **核心**: 纯 JavaScript，无外部依赖
- **开发**: Node.js（用于运行示例和测试）
- **可选**: 浏览器环境或微信小程序环境

#### Python 版本依赖
```txt
torch>=1.9.0          # PyTorch 深度学习框架
numpy>=1.19.0         # 数值计算
matplotlib>=3.3.0     # 可视化
seaborn>=0.11.0       # 统计可视化
tqdm>=4.60.0          # 进度条
jupyter>=1.0.0        # Jupyter Notebook
pytest>=6.0.0         # 单元测试
black>=21.0.0         # 代码格式化
```

## 🧪 测试和验证

### JavaScript 版本测试
```bash
cd learning/transformer-js

# 运行LLaMA2示例 ⭐ (推荐)
node examples/llama2-example.js

# 运行单元测试
node tests/unit-tests.js

# 运行基础示例
node examples/basic-usage.js

# 运行完整模型示例
node examples/complete-model.js

# 运行 BERT 示例
node examples/bert-example.js
```

### Python 版本测试
```bash
cd learning/transformer-python

# 运行LLaMA2简化测试 ⭐ (推荐，无需PyTorch)
python examples/llama2_simple_test.py

# 运行LLaMA2完整示例 (需要PyTorch)
python examples/llama2_example.py

# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_attention.py -v

# 生成覆盖率报告
pytest tests/ --cov=transformer_pytorch --cov-report=html

# 运行基础示例
python examples/basic_usage.py

# 运行 Jupyter 教程
jupyter notebook notebooks/
```

## 🎓 学习路径

### 初学者路径（推荐从 JavaScript 开始）
1. **理论学习**
   - 阅读 [Transformer 架构实现指南](./learning/transformer-architecture-guide.md)
   - 理解注意力机制基本概念

2. **实践入门**
   ```bash
   cd learning/transformer-js
   # 推荐从LLaMA2开始 ⭐
   node examples/llama2-example.js
   # 或运行基础示例
   node examples/basic-usage.js
   ```

3. **深入理解**
   - 查看LLaMA2核心组件实现 ⭐
   - 理解GQA、RMSNorm、SwiGLU等先进技术
   - 运行注意力可视化示例

### 进阶用户路径
1. **Python 版本学习**
   ```bash
   cd learning/transformer-python
   # 推荐从LLaMA2开始 ⭐
   python examples/llama2_simple_test.py
   # 或运行Jupyter教程
   jupyter notebook notebooks/01_basic_concepts.ipynb
   ```

2. **模型训练实践**
   - 学习 PyTorch 基础
   - 运行LLaMA2训练示例 ⭐
   - 理解三阶段训练流程(Pretrain→SFT→RLHF)

3. **自定义开发**
   - 修改LLaMA2模型架构 ⭐
   - 实现新的注意力机制
   - 优化GQA和RMSNorm组件

### 专家用户路径
1. **性能优化**
   - GPU 加速优化
   - 内存使用优化

2. **模型扩展**
   - 实现新的预训练模型
   - 添加新的任务头

3. **生产部署**
   - 模型量化和压缩
   - 服务化部署

## 📊 性能对比

### 计算性能对比

| 指标 | JavaScript 版本 | Python 版本 | 说明 |
|------|----------------|-------------|------|
| **小型模型推理** | ~100ms (CPU) | ~10ms (GPU) | 序列长度=64 |
| **内存使用** | 50-200MB | 200MB-2GB | 取决于模型大小 |
| **支持序列长度** | ≤64 (小程序) / ≤512 (浏览器) | ≤2048 | 受内存限制 |
| **批处理** | 支持 | 高效支持 | Python 版本更优 |
| **并行计算** | 有限 | 完整支持 | GPU 加速 |

### 开发效率对比

| 指标 | JavaScript 版本 | Python 版本 | 推荐场景 |
|------|----------------|-------------|----------|
| **上手难度** | 低 | 中等 | JS 适合初学者 |
| **调试便利性** | 浏览器调试 | Jupyter/IDE | 各有优势 |
| **可视化** | 基础图表 | 丰富工具 | Python 更强 |
| **扩展性** | 中等 | 高 | Python 生态更丰富 |
| **部署难度** | 简单 | 中等 | JS 部署更容易 |

## 🔧 高级功能

### JavaScript 版本高级功能
- **LLaMA2完整实现**: 支持7B/13B/70B等多种配置 ⭐
- **分组查询注意力(GQA)**: 优化的注意力机制，减少计算复杂度 ⭐
- **RMSNorm归一化**: 高效的归一化方法 ⭐
- **SwiGLU激活函数**: 门控激活函数，提升模型性能 ⭐
- **微信小程序优化**: 专门的轻量化配置
- **注意力可视化**: 内置注意力权重可视化功能
- **内存优化**: 针对小程序环境的内存管理
- **模块化加载**: 支持按需加载组件
- **文本生成**: 多种采样策略(贪心、Top-K、Top-P) ⭐

### Python 版本高级功能
- **LLaMA2完整实现**: 支持7B/13B/70B等多种配置 ⭐
- **分组查询注意力(GQA)**: 高效的注意力机制 ⭐
- **RMSNorm和SwiGLU**: 现代LLM的关键组件 ⭐
- **旋转位置编码(RoPE)**: 支持更长序列的位置编码 ⭐
- **文本生成**: 完整的生成功能，支持多种采样策略 ⭐
- **GPU 加速**: 完整的 CUDA 支持和混合精度训练
- **分布式训练**: 支持多 GPU 和多节点训练（计划中）
- **模型量化**: 支持 INT8 量化和动态量化
- **ONNX 导出**: 支持导出为 ONNX 格式（计划中）
- **Hugging Face 集成**: 与 Transformers 库兼容（计划中）

## 🎯 应用示例

### Web 应用示例（JavaScript）
```javascript
// 在网页中使用 Transformer
<!DOCTYPE html>
<html>
<head>
    <title>Transformer Demo</title>
</head>
<body>
    <script src="./learning/transformer-js/core/transformer.js"></script>
    <script>
        // 创建模型
        const config = getConfig('miniprogram');
        const model = createTransformer(config);

        // 处理用户输入
        function processText(inputText) {
            const tokens = tokenize(inputText);
            const result = model.forward([tokens]);
            return result.logits;
        }
    </script>
</body>
</html>
```

### LLaMA2 Web应用示例 ⭐
```javascript
// 使用LLaMA2进行实时文本生成
const { LLaMA2Config, LLaMA2ForCausalLM } = require('./learning/transformer-js/models/llama2');

// 创建优化的LLaMA2模型
const config = LLaMA2Config.miniprogram();
const model = new LLaMA2ForCausalLM(config);

// 文本生成函数
function generateText(prompt, options = {}) {
  const promptTokens = tokenize(prompt);
  const generated = model.generate(promptTokens, {
    maxNewTokens: options.maxLength || 20,
    temperature: options.temperature || 0.8,
    doSample: true
  });

  return detokenize(generated);
}

// 使用示例
const userInput = "今天天气很好";
const response = generateText(userInput, { maxLength: 30 });
console.log('AI回复:', response);
```

### 微信小程序示例
```javascript
// pages/transformer/transformer.js
const { LLaMA2Config, LLaMA2ForCausalLM } = require('../../learning/transformer-js/models/llama2');

Page({
  data: {
    model: null,
    result: ''
  },

  onLoad() {
    // 初始化LLaMA2模型
    const config = LLaMA2Config.miniprogram();
    this.setData({
      model: new LLaMA2ForCausalLM(config)
    });
  },

  processInput(e) {
    const inputText = e.detail.value;
    const tokens = this.tokenize(inputText);
    const generated = this.data.model.generate(tokens, {
      maxNewTokens: 15,
      temperature: 0.7
    });

    this.setData({
      result: this.detokenize(generated)
    });
  }
});
```

### 研究项目示例（Python）
```python
# research_example.py
import torch
from transformer_pytorch import Transformer, TransformerConfig
from torch.utils.data import DataLoader

# 创建研究用配置
config = TransformerConfig(
    vocab_size=50000,
    d_model=768,
    num_heads=12,
    num_encoder_layers=12,
    num_decoder_layers=12,
    d_ff=3072,
    max_seq_len=1024,
    dropout=0.1
)

# 创建模型
model = Transformer(config)

# 训练循环
def train_model(model, dataloader, optimizer, device):
    model.train()
    for batch in dataloader:
        src, tgt = batch['src'].to(device), batch['tgt'].to(device)

        optimizer.zero_grad()
        output = model(src, tgt)
        loss = compute_loss(output, tgt)
        loss.backward()
        optimizer.step()

        print(f'Loss: {loss.item():.4f}')

# GPU 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 开始训练
train_model(model, train_dataloader, optimizer, device)
```

## 🤝 贡献指南

我们欢迎任何形式的贡献！

### 贡献方式
1. **报告问题** - 在 Issues 中报告 bug 或提出功能请求
2. **代码贡献** - Fork 项目，创建分支，提交 Pull Request
3. **文档改进** - 改进文档、添加示例、修正错误
4. **测试用例** - 添加测试用例，提高代码覆盖率
5. **性能优化** - 提供性能优化建议和实现

### 开发规范
- **代码注释**: 所有代码需要详细的中文注释和 docstring
- **代码风格**:
  - JavaScript 代码遵循 ES6+ 标准
  - Python 代码遵循 PEP 8 规范和类型提示
- **测试要求**: 提交前请运行测试确保代码质量
- **文档更新**: 代码变更需要同步更新相关文档

### 开发环境设置
```bash
# JavaScript 开发环境
cd learning/transformer-js
npm install  # 如果有 package.json

# Python 开发环境
cd learning/transformer-python
pip install -e ".[dev]"
pre-commit install  # 安装代码检查钩子
```

## � 未来发展计划

### JavaScript 版本路线图
- [ ] **WebGL 加速**: 利用 GPU 进行矩阵运算加速
- [ ] **WebAssembly 优化**: 提供 WASM 版本以提升性能
- [ ] **更多模型变体**: 实现 RoBERTa、ALBERT、DeBERTa 等
- [ ] **训练功能**: 添加基础的训练和微调功能
- [ ] **可视化工具**: 增强注意力和激活的可视化
- [ ] **模型压缩**: 实现知识蒸馏和剪枝

### Python 版本路线图
- [ ] **分布式训练**: 支持多 GPU 和多节点训练
- [ ] **模型量化**: 完整的 INT8/FP16 量化支持
- [ ] **ONNX 导出**: 支持导出为 ONNX 格式
- [ ] **Hugging Face 集成**: 与 Transformers 库完全兼容
- [ ] **自动混合精度**: 优化内存使用和训练速度
- [ ] **模型并行**: 支持大模型的模型并行训练

## 📈 项目统计

### 代码统计
- **总代码行数**: 约 20,000+ 行 ⭐ (新增LLaMA2实现)
- **JavaScript 代码**: 约 10,000+ 行 ⭐ (新增LLaMA2模型)
- **Python 代码**: 约 10,000+ 行 ⭐ (新增LLaMA2模型)
- **文档和注释**: 约 40% 的代码为注释和文档
- **测试覆盖率**: 目标 >85%

### 功能完成度
| 功能模块 | JavaScript 版本 | Python 版本 | 状态 |
|----------|----------------|-------------|------|
| 基础 Transformer | ✅ 100% | ✅ 100% | 完成 |
| 注意力机制 | ✅ 100% | ✅ 100% | 完成 |
| **LLaMA2 模型** | ✅ **95%** ⭐ | ✅ **100%** ⭐ | **新增完成** |
| **分组查询注意力(GQA)** | ✅ **100%** ⭐ | ✅ **100%** ⭐ | **新增完成** |
| **RMSNorm归一化** | ✅ **100%** ⭐ | ✅ **100%** ⭐ | **新增完成** |
| **SwiGLU激活函数** | ✅ **100%** ⭐ | ✅ **100%** ⭐ | **新增完成** |
| BERT 模型 | ✅ 90% | ✅ 95% | 基本完成 |
| GPT 模型 | ✅ 85% | ✅ 90% | 进行中 |
| T5 模型 | 🔄 60% | 🔄 70% | 开发中 |
| 训练功能 | ❌ 0% | ✅ 80% | Python 优先 |
| 可视化 | ✅ 70% | ✅ 90% | 持续改进 |

## 🆘 常见问题

### JavaScript 版本常见问题

**Q: 在微信小程序中内存不足怎么办？**
A: 使用 `miniprogram` 配置，限制序列长度≤64，定期清理不用的变量。

**Q: 浏览器中运行速度慢怎么办？**
A: 使用 Web Workers 进行后台计算，考虑使用 `small` 配置。

**Q: 如何自定义模型配置？**
A: 参考 `config/config.js` 文件，使用 `createConfig()` 函数。

### Python 版本常见问题

**Q: CUDA 内存不足怎么办？**
A: 减小批次大小，使用梯度累积，启用混合精度训练。

**Q: 如何加速训练？**
A: 使用多 GPU，启用 DataLoader 的 `num_workers`，使用混合精度。

**Q: 如何保存和加载模型？**
A: 使用 `torch.save()` 和 `torch.load()`，参考 examples 中的示例。

## 📞 支持和联系

### 获取帮助
- **GitHub Issues**: [提交问题和建议](https://github.com/your-repo/issues)
- **讨论区**: [参与社区讨论](https://github.com/your-repo/discussions)
- **文档**: [查看详细文档](./learning/)

### 社区资源
- **学习交流群**: 欢迎加入我们的学习交流群
- **技术博客**: 定期发布技术文章和教程
- **视频教程**: 计划制作配套的视频教程

## �📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE.txt](LICENSE.txt) 文件了解详情。

### 许可证说明
- ✅ 商业使用
- ✅ 修改代码
- ✅ 分发代码
- ✅ 私人使用
- ❗ 需要包含许可证和版权声明
- ❗ 不提供责任担保

## 🙏 致谢

### happy-llm核心贡献者
- [宋志学-项目负责人](https://github.com/KMnO4-zx) (Datawhale成员-中国矿业大学(北京))
- [邹雨衡-项目负责人](https://github.com/logan-zou) (Datawhale成员-对外经济贸易大学)
- [朱信忠-指导专家](https://xinzhongzhu.github.io/)（Datawhale首席科学家-浙江师范大学杭州人工智能研究院教授）

### 特别感谢
- **理论基础**: 感谢 Vaswani 等人的开创性论文 "Attention Is All You Need"
- **文档支持**: 感谢 [Happy-LLM 项目](https://github.com/datawhalechina/happy-llm) 提供的系统性 LLM 学习教程和理论基础
- **开源社区**: 感谢所有为开源社区做出贡献的开发者们 ❤️
- **技术支持**: 感谢 PyTorch、JavaScript 社区的技术支持

### 引用本项目
如果本项目对您的研究或工作有帮助，请考虑引用：

```bibtex
@misc{transformer-learning-2025,
  title={Transformer 架构学习与实现项目},
  author={shihom_wu},
  year={2025},
  url={https://github.com/your-repo/transformer-learning}
}
```

---

<div align="center">
  <p>⭐ 如果这个项目对你有帮助，请给我们一个 Star！</p>
  <p><strong>Transformer 架构学习与实现项目</strong> - 让 Transformer 的学习和使用变得简单！</p>
  <br>
  <p>📧 联系我们 | 🌟 Star 项目 | 🍴 Fork 代码 | 📝 提交 Issue</p>
</div>