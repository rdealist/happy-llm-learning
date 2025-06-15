# API 参考文档

本文档详细介绍了 Transformer-JS 的所有 API 接口和使用方法。

## 📋 目录

- [配置模块](#配置模块)
- [数学工具](#数学工具)
- [矩阵运算](#矩阵运算)
- [基础层](#基础层)
- [注意力机制](#注意力机制)
- [嵌入层](#嵌入层)
- [编码器](#编码器)
- [解码器](#解码器)
- [完整模型](#完整模型)

## 配置模块

### `getConfig(configName)`

获取预定义的模型配置。

**参数:**
- `configName` (string): 配置名称，可选值：'default', 'small', 'large', 'miniprogram'

**返回值:**
- `Object`: 配置对象

**示例:**
```javascript
const config = getConfig('small');
console.log(config.n_embd); // 256
```

### `createConfig(baseConfig, overrides)`

创建自定义配置。

**参数:**
- `baseConfig` (Object): 基础配置对象
- `overrides` (Object): 覆盖参数

**返回值:**
- `Object`: 合并后的配置对象

**示例:**
```javascript
const customConfig = createConfig(getConfig('default'), {
  n_embd: 256,
  n_layers: 4
});
```

### `estimateParameters(config)`

估算模型参数量。

**参数:**
- `config` (Object): 模型配置

**返回值:**
- `Object`: 参数统计信息

## 数学工具

### `softmax(x, dim)`

计算 Softmax 激活函数。

**参数:**
- `x` (Array<number>): 输入向量
- `dim` (number, 可选): 计算维度，默认为 -1

**返回值:**
- `Array<number>`: Softmax 输出

**示例:**
```javascript
const probs = softmax([2.0, 1.0, 0.1]);
// 输出: [0.659, 0.242, 0.099]
```

### `relu(x)`

ReLU 激活函数。

**参数:**
- `x` (number|Array<number>): 输入值或向量

**返回值:**
- `number|Array<number>`: ReLU 输出

### `gelu(x)`

GELU 激活函数。

**参数:**
- `x` (number|Array<number>): 输入值或向量

**返回值:**
- `number|Array<number>`: GELU 输出

### `randomNormalMatrix(shape, mean, stdDev)`

生成正态分布随机矩阵。

**参数:**
- `shape` (Array<number>): 矩阵形状 [rows, cols]
- `mean` (number, 可选): 均值，默认为 0
- `stdDev` (number, 可选): 标准差，默认为 0.02

**返回值:**
- `Array<Array<number>>`: 随机矩阵

## 矩阵运算

### `matmul(A, B)`

矩阵乘法。

**参数:**
- `A` (Array<Array<number>>): 左矩阵 [m, k]
- `B` (Array<Array<number>>): 右矩阵 [k, n]

**返回值:**
- `Array<Array<number>>`: 结果矩阵 [m, n]

**示例:**
```javascript
const A = [[1, 2], [3, 4]];
const B = [[5, 6], [7, 8]];
const C = matmul(A, B);
// C = [[19, 22], [43, 50]]
```

### `transpose(matrix)`

矩阵转置。

**参数:**
- `matrix` (Array<Array<number>>): 输入矩阵

**返回值:**
- `Array<Array<number>>`: 转置矩阵

### `add(A, B)`

矩阵加法。

**参数:**
- `A` (Array<Array<number>>): 矩阵 A
- `B` (Array<Array<number>>): 矩阵 B

**返回值:**
- `Array<Array<number>>`: 结果矩阵

## 基础层

### `Linear`

线性层（全连接层）。

#### 构造函数

```javascript
new Linear(inputDim, outputDim, useBias, initStd)
```

**参数:**
- `inputDim` (number): 输入维度
- `outputDim` (number): 输出维度
- `useBias` (boolean, 可选): 是否使用偏置，默认为 true
- `initStd` (number, 可选): 权重初始化标准差，默认为 0.02

#### 方法

##### `forward(x)`

前向传播。

**参数:**
- `x` (Array<Array<number>>): 输入矩阵 [batchSize, inputDim]

**返回值:**
- `Array<Array<number>>`: 输出矩阵 [batchSize, outputDim]

##### `getParameterCount()`

获取参数数量。

**返回值:**
- `number`: 参数总数

### `LayerNorm`

层归一化。

#### 构造函数

```javascript
new LayerNorm(normalizedShape, eps)
```

**参数:**
- `normalizedShape` (number): 归一化的维度大小
- `eps` (number, 可选): 防止除零的小值，默认为 1e-6

#### 方法

##### `forward(x)`

前向传播。

**参数:**
- `x` (Array<Array<number>>): 输入矩阵

**返回值:**
- `Array<Array<number>>`: 归一化后的矩阵

### `MLP`

多层感知机。

#### 构造函数

```javascript
new MLP(inputDim, hiddenDim, outputDim, activation, dropout, useBias)
```

**参数:**
- `inputDim` (number): 输入维度
- `hiddenDim` (number): 隐藏层维度
- `outputDim` (number, 可选): 输出维度，默认等于输入维度
- `activation` (string, 可选): 激活函数类型，默认为 'relu'
- `dropout` (number, 可选): Dropout 概率，默认为 0.1
- `useBias` (boolean, 可选): 是否使用偏置，默认为 false

#### 方法

##### `forward(x)`

前向传播。

**参数:**
- `x` (Array<Array<number>>): 输入矩阵

**返回值:**
- `Array<Array<number>>`: 输出矩阵

##### `setTraining(training)`

设置训练模式。

**参数:**
- `training` (boolean): 是否为训练模式

## 注意力机制

### `Attention`

基础注意力机制。

#### 构造函数

```javascript
new Attention(dropout)
```

**参数:**
- `dropout` (number, 可选): 注意力 dropout 概率，默认为 0.1

#### 方法

##### `forward(query, key, value, mask)`

计算注意力。

**参数:**
- `query` (Array<Array<number>>): 查询矩阵 Q
- `key` (Array<Array<number>>): 键矩阵 K
- `value` (Array<Array<number>>): 值矩阵 V
- `mask` (Array<Array<number>>, 可选): 注意力掩码

**返回值:**
- `Object`: {output: 注意力输出, attention: 注意力权重}

### `MultiHeadAttention`

多头注意力机制。

#### 构造函数

```javascript
new MultiHeadAttention(dModel, nHeads, dropout, useBias)
```

**参数:**
- `dModel` (number): 模型维度
- `nHeads` (number): 注意力头数
- `dropout` (number, 可选): Dropout 概率，默认为 0.1
- `useBias` (boolean, 可选): 是否使用偏置，默认为 false

#### 方法

##### `forward(query, key, value, mask)`

前向传播。

**参数:**
- `query` (Array<Array<number>>): 查询矩阵
- `key` (Array<Array<number>>): 键矩阵
- `value` (Array<Array<number>>): 值矩阵
- `mask` (Array<Array<number>>, 可选): 注意力掩码

**返回值:**
- `Object`: {output: 多头注意力输出, attention: 注意力权重}

### `MaskGenerator`

掩码生成工具。

#### 静态方法

##### `createCausalMask(seqLen)`

生成因果掩码。

**参数:**
- `seqLen` (number): 序列长度

**返回值:**
- `Array<Array<number>>`: 因果掩码矩阵

##### `createPaddingMask(tokenIds, padId)`

生成填充掩码。

**参数:**
- `tokenIds` (Array<number>): 词元ID数组
- `padId` (number, 可选): 填充词元的ID，默认为0

**返回值:**
- `Array<Array<number>>`: 填充掩码矩阵

## 嵌入层

### `TransformerEmbedding`

完整的嵌入层。

#### 构造函数

```javascript
new TransformerEmbedding(vocabSize, embedDim, maxLen, positionType, dropout, initStd)
```

**参数:**
- `vocabSize` (number): 词汇表大小
- `embedDim` (number): 嵌入维度
- `maxLen` (number): 最大序列长度
- `positionType` (string, 可选): 位置编码类型，默认为 'sinusoidal'
- `dropout` (number, 可选): Dropout 概率，默认为 0.1
- `initStd` (number, 可选): 初始化标准差，默认为 0.02

#### 方法

##### `forward(tokenIds, scaleEmbedding)`

前向传播。

**参数:**
- `tokenIds` (Array<Array<number>>): 词元ID矩阵
- `scaleEmbedding` (boolean, 可选): 是否缩放嵌入，默认为 true

**返回值:**
- `Array<Array<Array<number>>>`: 最终嵌入

## 完整模型

### `Transformer`

完整的 Transformer 模型。

#### 构造函数

```javascript
new Transformer(config)
```

**参数:**
- `config` (Object): 模型配置

#### 方法

##### `forward(srcTokens, tgtTokens, srcMask, tgtMask)`

前向传播。

**参数:**
- `srcTokens` (Array<Array<number>>): 源序列词元
- `tgtTokens` (Array<Array<number>>): 目标序列词元
- `srcMask` (Array<Array<number>>, 可选): 源序列掩码
- `tgtMask` (Array<Array<number>>, 可选): 目标序列掩码

**返回值:**
- `Object`: 模型输出

##### `encode(srcTokens, srcMask)`

编码源序列。

**参数:**
- `srcTokens` (Array<Array<number>>): 源序列词元
- `srcMask` (Array<Array<number>>, 可选): 源序列掩码

**返回值:**
- `Object`: 编码器输出

##### `predictNext(srcTokens, tgtTokens)`

预测下一个词元。

**参数:**
- `srcTokens` (Array<number>): 源序列
- `tgtTokens` (Array<number>): 目标序列（到当前位置）

**返回值:**
- `Array<number>`: 下一个词元的概率分布

##### `setTraining(training)`

设置训练模式。

**参数:**
- `training` (boolean): 是否为训练模式

##### `getParameterCount()`

获取模型参数数量。

**返回值:**
- `Object`: 参数统计信息

##### `summary()`

获取模型信息摘要。

**返回值:**
- `Object`: 模型信息

### `createTransformer(config)`

创建 Transformer 模型的工厂函数。

**参数:**
- `config` (Object): 模型配置

**返回值:**
- `Transformer`: Transformer 模型实例

**示例:**
```javascript
const config = getConfig('small');
const model = createTransformer(config);
```
