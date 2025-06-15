# Transformer-JS

> 基于 JavaScript 的 Transformer 架构实现，专为微信小程序环境优化
>
> **🆕 第三章扩展：预训练语言模型支持**

## 📖 项目简介

Transformer-JS 是一个完全使用 JavaScript 实现的 Transformer 神经网络架构库。本项目基于《第二章 Transformer架构》和《第三章 预训练语言模型》文档中的理论知识，提供了模块化、易于理解和使用的 Transformer 实现，现已扩展支持 BERT、GPT、T5 等主流预训练语言模型。

### ✨ 主要特性

- 🧩 **模块化设计** - 每个组件独立实现，支持按需使用
- 📱 **小程序优化** - 针对微信小程序环境的内存和性能限制进行优化
- 📚 **详细注释** - 所有代码都有中文注释和详细文档
- 🔧 **灵活配置** - 支持多种预设配置和自定义参数
- 🎯 **教育友好** - 代码结构清晰，便于学习和理解
- ⚡ **纯 JavaScript** - 无需额外依赖，可直接在浏览器和小程序中运行
- 🤖 **预训练模型** - 支持 BERT、GPT、T5 等主流预训练语言模型
- 📝 **预训练任务** - 实现 MLM、NSP、SOP、CLM 等预训练任务

### 🏗️ 架构组件

#### 核心组件
- **数学工具** - Softmax、ReLU、GELU 等激活函数
- **矩阵运算** - 矩阵乘法、转置、重塑等基础运算
- **基础层** - 线性层、层归一化、RMSNorm、MLP、Dropout
- **注意力机制** - 基础注意力、自注意力、多头注意力、因果掩码
- **嵌入层** - 词嵌入、正弦位置编码、可学习位置编码
- **编码器** - Transformer 编码器层和编码器块
- **解码器** - Transformer 解码器层和解码器块
- **完整模型** - 端到端的 Transformer 模型

#### 预训练语言模型
- **BERT 系列** - BERT、RoBERTa、ALBERT 等双向语言模型
- **GPT 系列** - GPT、GPT-2 等自回归语言模型
- **T5 模型** - 文本到文本转换的 Encoder-Decoder 模型
- **分类头** - 序列分类、Token分类、语言建模等任务头
- **预训练任务** - MLM、NSP、SOP、CLM 等预训练任务实现

## 🚀 快速开始

### 基础 Transformer 使用

```javascript
// 引入必要模块
const { getConfig } = require('./config/config');
const { createTransformer } = require('./core/transformer');

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

### BERT 模型使用

```javascript
const { ModelFactory } = require('./models/config');

// 创建BERT分类模型
const bertModel = ModelFactory.createBERT('bert-base', 'classification', {
  num_labels: 2  // 二分类任务
});

// 准备输入数据
const inputIds = [101, 2023, 2003, 1037, 3231, 102];  // [CLS] This is a test [SEP]
const tokenTypeIds = [0, 0, 0, 0, 0, 0];
const attentionMask = [1, 1, 1, 1, 1, 1];

// 前向传播
const result = bertModel.forward(inputIds, tokenTypeIds, attentionMask);
console.log('分类logits:', result.logits);
```

### GPT 文本生成

```javascript
const { ModelFactory } = require('./models/config');

// 创建GPT语言模型
const gptModel = ModelFactory.createGPT('gpt2-small', 'causal_lm');

// 生成文本
const inputIds = [464, 318, 257];  // "It is a"
const generated = gptModel.generate(inputIds, {
  maxLength: 20,
  temperature: 0.8,
  doSample: true
});

console.log('生成的token序列:', generated);
```

### T5 文本转换

```javascript
const { ModelFactory } = require('./models/config');

// 创建T5模型
const t5Model = ModelFactory.createT5('t5-small');

// 文本到文本转换
const inputIds = [13959, 10, 363, 19, 8, 1784, 13];  // "translate: What is the weather"
const decoderInputIds = [0];  // 开始token

const result = t5Model.forward(inputIds, decoderInputIds);
console.log('输出logits:', result.logits);
```

### 自定义配置

```javascript
const { createConfig, getConfig } = require('./config/config');

// 基于默认配置创建自定义配置
const customConfig = createConfig(getConfig('default'), {
  vocab_size: 8000,
  n_embd: 256,
  n_layers: 4,
  n_heads: 8,
  max_seq_len: 128,
  activation: 'gelu'
});

const model = createTransformer(customConfig);
```

### 单独使用组件

```javascript
const { MultiHeadAttention } = require('./core/attention');
const { MLP } = require('./core/layers');

// 使用多头注意力
const attention = new MultiHeadAttention(512, 8, 0.1);
const result = attention.forward(Q, K, V);

// 使用前馈网络
const mlp = new MLP(512, 2048, 512, 'relu', 0.1);
const output = mlp.forward(input);
```

## 📁 项目结构

```text
transformer-js/
├── core/                   # 核心组件
│   ├── math-utils.js       # 数学工具函数
│   ├── matrix-ops.js       # 矩阵运算
│   ├── layers.js           # 基础层实现（LayerNorm、RMSNorm、GELU等）
│   ├── attention.js        # 注意力机制（多头注意力、掩码生成）
│   ├── embedding.js        # 嵌入层
│   ├── encoder.js          # 编码器
│   ├── decoder.js          # 解码器
│   ├── transformer.js      # 完整模型
│   └── heads.js            # 分类头（序列分类、Token分类、语言建模）
├── pretraining/            # 预训练任务
│   └── tasks.js            # MLM、NSP、SOP、CLM任务实现
├── models/                 # 预训练语言模型
│   ├── bert.js             # BERT系列模型
│   ├── gpt.js              # GPT系列模型
│   ├── t5.js               # T5模型
│   └── config.js           # 模型配置和工厂函数
├── config/                 # 配置文件
│   ├── config.js           # 基础模型配置
│   └── constants.js        # 常量定义
├── examples/               # 使用示例
│   ├── basic-usage.js      # 基础使用示例
│   ├── complete-model.js   # 完整模型示例
│   ├── bert-example.js     # BERT使用示例
│   ├── gpt-example.js      # GPT使用示例
│   └── custom-config.js    # 自定义配置示例
├── tests/                  # 测试文件
├── docs/                   # 详细文档
└── README.md              # 项目说明
```

## 🔧 配置选项

### 预设配置

- **`small`** - 小型模型，适合快速测试
- **`default`** - 标准配置，基于原始 Transformer 论文
- **`large`** - 大型模型，适合生产环境
- **`miniprogram`** - 微信小程序优化配置

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `vocab_size` | 词汇表大小 | 30000 |
| `n_embd` | 嵌入维度 | 512 |
| `n_layers` | 层数 | 6 |
| `n_heads` | 注意力头数 | 8 |
| `max_seq_len` | 最大序列长度 | 512 |
| `dropout` | Dropout 概率 | 0.1 |
| `activation` | 激活函数 | 'relu' |

## 📚 详细文档

- [API 参考文档](./docs/api-reference.md)
- [使用教程](./docs/tutorial.md)
- [性能优化指南](./docs/performance.md)
- [架构设计说明](./docs/architecture.md)

## 🎯 使用场景

### 教育学习
- 理解 Transformer 架构原理
- 学习注意力机制实现
- 深度学习概念实践

### 原型开发
- 快速验证模型想法
- 小规模实验和测试
- 算法研究和改进

### 微信小程序
- 轻量级 NLP 应用
- 本地推理和预测
- 离线文本处理

## ⚡ 性能特点

### 内存优化
- 避免不必要的数组复制
- 使用高效的矩阵运算
- 支持梯度检查点（计划中）

### 计算优化
- 纯 JavaScript 实现，无需编译
- 支持批处理计算
- 可配置的精度控制

### 小程序适配
- 内存使用限制在 256MB 以内
- 计算时间控制在 10 秒以内
- 推荐序列长度不超过 64

## 🧪 示例和测试

### 运行基础示例
```bash
node examples/basic-usage.js
```

### 运行完整模型示例
```bash
node examples/complete-model.js
```

### 运行测试
```bash
node tests/unit-tests.js
```

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢 Vaswani 等人的开创性论文 "Attention Is All You Need"
- 感谢《第二章 Transformer架构》文档提供的理论基础
- 感谢所有为开源社区做出贡献的开发者

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 参与讨论

---

**Transformer-JS** - 让 Transformer 架构的学习和使用变得简单！
