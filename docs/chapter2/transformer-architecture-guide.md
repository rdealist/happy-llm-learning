# Transformer 架构实现指南

> 基于《第二章 Transformer架构》文档的完整 JavaScript 实现

## 📖 概述

本指南详细介绍了基于文档《第二章 Transformer架构》的完整 JavaScript 实现。我们将理论知识转化为可运行的代码，提供了模块化、易于理解的 Transformer 架构实现。

## 🏗️ 架构设计

### 整体架构图

```
Transformer 模型
├── 嵌入层 (Embedding Layer)
│   ├── 词嵌入 (Token Embedding)
│   └── 位置编码 (Positional Encoding)
├── 编码器 (Encoder)
│   └── N × 编码器层
│       ├── 多头自注意力 (Multi-Head Self-Attention)
│       ├── 残差连接 + 层归一化
│       ├── 前馈网络 (Feed Forward Network)
│       └── 残差连接 + 层归一化
├── 解码器 (Decoder)
│   └── N × 解码器层
│       ├── 掩码多头自注意力 (Masked Multi-Head Self-Attention)
│       ├── 残差连接 + 层归一化
│       ├── 多头交叉注意力 (Multi-Head Cross-Attention)
│       ├── 残差连接 + 层归一化
│       ├── 前馈网络 (Feed Forward Network)
│       └── 残差连接 + 层归一化
└── 输出层 (Output Layer)
    └── 线性投影 + Softmax
```

### 模块依赖关系

```
transformer.js
├── embedding.js
├── encoder.js
│   ├── attention.js
│   └── layers.js
├── decoder.js
│   ├── attention.js
│   └── layers.js
├── attention.js
│   ├── math-utils.js
│   └── matrix-ops.js
├── layers.js
│   ├── math-utils.js
│   └── matrix-ops.js
└── config/
    ├── config.js
    └── constants.js
```

## 🔧 核心组件实现

### 1. 注意力机制

#### 基础注意力计算
```javascript
// 实现公式: Attention(Q,K,V) = softmax(QK^T/√d_k)V
function attention(query, key, value, mask = null) {
  const dK = key[0].length;
  const scores = matmul(query, transpose(key));
  const scaledScores = scalarMultiply(scores, 1.0 / Math.sqrt(dK));
  
  if (mask) {
    // 应用掩码
    scaledScores = applyMask(scaledScores, mask);
  }
  
  const attentionWeights = scores.map(row => softmax(row));
  const output = matmul(attentionWeights, value);
  
  return { output, attention: attentionWeights };
}
```

#### 多头注意力
```javascript
class MultiHeadAttention {
  constructor(dModel, nHeads) {
    this.dModel = dModel;
    this.nHeads = nHeads;
    this.dK = dModel / nHeads;
    
    // Q, K, V 投影矩阵
    this.wQ = new Linear(dModel, dModel);
    this.wK = new Linear(dModel, dModel);
    this.wV = new Linear(dModel, dModel);
    this.wO = new Linear(dModel, dModel);
  }
  
  forward(query, key, value, mask = null) {
    // 1. 线性投影
    const Q = this.wQ.forward(query);
    const K = this.wK.forward(key);
    const V = this.wV.forward(value);
    
    // 2. 分割为多头
    const QHeads = this.splitHeads(Q);
    const KHeads = this.splitHeads(K);
    const VHeads = this.splitHeads(V);
    
    // 3. 并行计算注意力
    const headOutputs = [];
    for (let h = 0; h < this.nHeads; h++) {
      const result = attention(QHeads[h], KHeads[h], VHeads[h], mask);
      headOutputs.push(result.output);
    }
    
    // 4. 拼接头输出
    const concatenated = this.concatHeads(headOutputs);
    
    // 5. 最终投影
    return this.wO.forward(concatenated);
  }
}
```

### 2. 位置编码

#### 正弦余弦位置编码
```javascript
// 实现公式:
// PE(pos, 2i) = sin(pos/10000^(2i/d_model))
// PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
function computePositionEncodings(maxLen, embedDim) {
  const pe = zeros([maxLen, embedDim]);
  
  for (let pos = 0; pos < maxLen; pos++) {
    for (let i = 0; i < embedDim; i++) {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / embedDim);
      
      if (i % 2 === 0) {
        pe[pos][i] = Math.sin(angle);  // 偶数位置
      } else {
        pe[pos][i] = Math.cos(angle);  // 奇数位置
      }
    }
  }
  
  return pe;
}
```

### 3. 层归一化

```javascript
// 实现公式: LayerNorm(x) = γ * (x - μ) / σ + β
class LayerNorm {
  constructor(normalizedShape, eps = 1e-6) {
    this.normalizedShape = normalizedShape;
    this.eps = eps;
    this.gamma = new Array(normalizedShape).fill(1.0);
    this.beta = new Array(normalizedShape).fill(0.0);
  }
  
  forward(x) {
    return x.map(row => {
      const meanVal = mean(row);
      const stdVal = std(row, meanVal);
      
      return row.map((val, j) => {
        const normalized = (val - meanVal) / (stdVal + this.eps);
        return this.gamma[j] * normalized + this.beta[j];
      });
    });
  }
}
```

### 4. 前馈网络

```javascript
// 实现公式: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
class MLP {
  constructor(inputDim, hiddenDim, outputDim, activation = 'relu') {
    this.linear1 = new Linear(inputDim, hiddenDim);
    this.linear2 = new Linear(hiddenDim, outputDim);
    this.activation = activation;
  }
  
  forward(x) {
    let hidden = this.linear1.forward(x);
    hidden = hidden.map(row => this.activationFn(row));
    return this.linear2.forward(hidden);
  }
}
```

## 📊 配置系统

### 预设配置

| 配置名称 | 词汇表 | 嵌入维度 | 层数 | 注意力头数 | 最大序列长度 | 用途 |
|----------|--------|----------|------|------------|--------------|------|
| `small` | 10,000 | 256 | 4 | 4 | 128 | 快速测试 |
| `default` | 30,000 | 512 | 6 | 8 | 512 | 标准配置 |
| `large` | 50,000 | 1024 | 12 | 16 | 1024 | 生产环境 |
| `miniprogram` | 8,000 | 128 | 3 | 4 | 64 | 小程序优化 |

### 自定义配置

```javascript
const customConfig = createConfig(getConfig('default'), {
  vocab_size: 8000,
  n_embd: 256,
  n_layers: 4,
  n_heads: 8,
  max_seq_len: 128,
  activation: 'gelu',
  dropout: 0.05
});
```

## 🚀 使用示例

### 基础使用

```javascript
// 1. 创建模型
const config = getConfig('small');
const model = createTransformer(config);

// 2. 准备数据
const srcTokens = [[2, 10, 25, 67, 3]];  // 源序列
const tgtTokens = [[2, 15, 30, 45]];     // 目标序列

// 3. 前向传播
const result = model.forward(srcTokens, tgtTokens);
console.log('输出维度:', result.logits.length, 'x', result.logits[0].length);
```

### 序列到序列翻译

```javascript
// 编码源序列
const encoderResult = model.encode(srcTokens);

// 自回归解码
let generated = [2]; // 从 BOS 开始
for (let step = 0; step < maxLength; step++) {
  const probs = model.predictNext(srcTokens[0], generated);
  const nextToken = argmax(probs);
  generated.push(nextToken);
  
  if (nextToken === 3) break; // 遇到 EOS 停止
}
```

### 注意力可视化

```javascript
const result = model.forward(srcTokens, tgtTokens);

// 分析编码器注意力
result.encoderAttentions.forEach((layerAttentions, layerIdx) => {
  console.log(`编码器第 ${layerIdx + 1} 层注意力:`);
  layerAttentions.forEach((headAttention, headIdx) => {
    console.log(`  头 ${headIdx + 1}:`, headAttention);
  });
});
```

## 🎯 性能优化

### 内存优化

1. **避免不必要的数组复制**
   ```javascript
   // 好的做法：原地操作
   matrix[i][j] = newValue;
   
   // 避免：创建新数组
   matrix = matrix.map(row => row.map(val => newValue));
   ```

2. **使用 TypedArray**（计划中）
   ```javascript
   const weights = new Float32Array(inputDim * outputDim);
   ```

### 计算优化

1. **批处理计算**
   ```javascript
   // 同时处理多个序列
   const batchResult = model.forward(batchSrcTokens, batchTgtTokens);
   ```

2. **缓存机制**（计划中）
   ```javascript
   // 缓存注意力计算结果
   const cache = new Map();
   ```

### 小程序适配

1. **内存限制**: 控制在 256MB 以内
2. **计算时间**: 单次推理不超过 10 秒
3. **序列长度**: 推荐不超过 64

## 🧪 测试和验证

### 单元测试

```javascript
// 运行所有测试
node tests/unit-tests.js

// 运行特定测试
const { testAttention } = require('./tests/unit-tests');
testAttention();
```

### 性能基准

```javascript
// 性能测试
node examples/complete-model.js
```

## 📚 学习路径

### 初学者
1. 阅读《第二章 Transformer架构》文档
2. 运行 `examples/basic-usage.js`
3. 理解注意力机制实现
4. 学习各个组件的作用

### 进阶用户
1. 自定义模型配置
2. 实现新的激活函数
3. 优化矩阵运算
4. 添加新的注意力机制

### 专家用户
1. 实现 GPT/BERT 变体
2. 添加训练功能
3. 优化推理性能
4. 扩展到其他任务

## 🔮 未来计划

- [ ] 添加训练功能
- [ ] 实现 GPT 和 BERT 变体
- [ ] 支持更多激活函数
- [ ] 添加量化支持
- [ ] 实现键值缓存
- [ ] 支持流式推理
- [ ] 添加可视化工具

## 📞 支持和贡献

如有问题或建议，欢迎：
- 提交 Issue
- 发起 Pull Request
- 参与讨论

---

**Transformer-JS** - 让 Transformer 架构的学习和实现变得简单！
