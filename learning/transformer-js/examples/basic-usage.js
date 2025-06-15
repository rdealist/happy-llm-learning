/**
 * Transformer-JS 基础使用示例
 * 演示如何使用各个组件和完整模型
 * 
 * @author Transformer-JS
 * @version 1.0.0
 */

// 引入必要的模块
const { getConfig } = require('../config/config');
const { softmax, relu } = require('../core/math-utils');
const { matmul, transpose } = require('../core/matrix-ops');
const { Linear, LayerNorm, MLP } = require('../core/layers');
const { Attention, MultiHeadAttention } = require('../core/attention');
const { TransformerEmbedding } = require('../core/embedding');
const { createTransformer } = require('../core/transformer');

/**
 * 示例1：基础数学运算
 */
function example1_basicMath() {
  console.log('=== 示例1：基础数学运算 ===');
  
  // Softmax 示例
  const logits = [2.0, 1.0, 0.1];
  const probs = softmax(logits);
  console.log('Softmax 输入:', logits);
  console.log('Softmax 输出:', probs);
  console.log('概率和:', probs.reduce((sum, p) => sum + p, 0));
  
  // ReLU 示例
  const values = [-1, 0, 1, 2, -0.5];
  const reluOutput = relu(values);
  console.log('ReLU 输入:', values);
  console.log('ReLU 输出:', reluOutput);
  
  console.log('');
}

/**
 * 示例2：矩阵运算
 */
function example2_matrixOps() {
  console.log('=== 示例2：矩阵运算 ===');
  
  // 矩阵乘法示例
  const A = [[1, 2], [3, 4]];
  const B = [[5, 6], [7, 8]];
  const C = matmul(A, B);
  
  console.log('矩阵 A:', A);
  console.log('矩阵 B:', B);
  console.log('A × B =', C);
  
  // 矩阵转置示例
  const matrix = [[1, 2, 3], [4, 5, 6]];
  const transposed = transpose(matrix);
  console.log('原矩阵:', matrix);
  console.log('转置矩阵:', transposed);
  
  console.log('');
}

/**
 * 示例3：基础层使用
 */
function example3_basicLayers() {
  console.log('=== 示例3：基础层使用 ===');
  
  // 线性层示例
  const linear = new Linear(3, 2, true); // 输入维度3，输出维度2，使用偏置
  const input = [[1, 2, 3], [4, 5, 6]]; // 批大小2，输入维度3
  const output = linear.forward(input);
  
  console.log('线性层输入:', input);
  console.log('线性层输出:', output);
  console.log('线性层参数数量:', linear.getParameterCount());
  
  // 层归一化示例
  const layerNorm = new LayerNorm(3);
  const normInput = [[1, 2, 3], [4, 5, 6]];
  const normOutput = layerNorm.forward(normInput);
  
  console.log('层归一化输入:', normInput);
  console.log('层归一化输出:', normOutput);
  
  // MLP 示例
  const mlp = new MLP(3, 6, 3, 'relu', 0.1); // 输入3，隐藏6，输出3
  const mlpOutput = mlp.forward(input);
  
  console.log('MLP 输出:', mlpOutput);
  console.log('MLP 参数数量:', mlp.getParameterCount());
  
  console.log('');
}

/**
 * 示例4：注意力机制
 */
function example4_attention() {
  console.log('=== 示例4：注意力机制 ===');
  
  // 基础注意力示例
  const attention = new Attention(0.0); // 不使用 dropout 便于观察结果
  
  // 创建简单的 Q, K, V 矩阵
  const seqLen = 3;
  const dModel = 4;
  
  const Q = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]; // [seqLen, dModel]
  const K = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]; // [seqLen, dModel]
  const V = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]; // [seqLen, dModel]
  
  const attentionResult = attention.forward(Q, K, V);
  
  console.log('Query:', Q);
  console.log('Key:', K);
  console.log('Value:', V);
  console.log('注意力输出:', attentionResult.output);
  console.log('注意力权重:', attentionResult.attention);
  
  // 多头注意力示例
  const multiHeadAttention = new MultiHeadAttention(dModel, 2, 0.0); // 4维，2个头
  const multiHeadResult = multiHeadAttention.forward(Q, K, V);
  
  console.log('多头注意力输出:', multiHeadResult.output);
  console.log('多头注意力参数数量:', multiHeadAttention.getParameterCount());
  
  console.log('');
}

/**
 * 示例5：嵌入层
 */
function example5_embedding() {
  console.log('=== 示例5：嵌入层 ===');
  
  // 创建嵌入层
  const vocabSize = 100;
  const embedDim = 8;
  const maxLen = 10;
  
  const embedding = new TransformerEmbedding(
    vocabSize, 
    embedDim, 
    maxLen, 
    'sinusoidal', 
    0.0
  );
  
  // 示例词元序列
  const tokenIds = [[1, 5, 10, 3]]; // 批大小1，序列长度4
  const embedded = embedding.forward(tokenIds);
  
  console.log('词元 ID:', tokenIds);
  console.log('嵌入维度:', embedded[0].length, 'x', embedded[0][0].length);
  console.log('嵌入输出（前两个位置）:');
  console.log('位置0:', embedded[0][0]);
  console.log('位置1:', embedded[0][1]);
  console.log('嵌入层参数数量:', embedding.getParameterCount());
  
  console.log('');
}

/**
 * 示例6：完整的 Transformer 模型
 */
function example6_fullTransformer() {
  console.log('=== 示例6：完整的 Transformer 模型 ===');
  
  // 获取小型配置
  const config = getConfig('small');
  console.log('使用配置:', config);
  
  // 创建模型
  const model = createTransformer(config);
  
  // 模型摘要
  const summary = model.summary();
  console.log('模型摘要:', summary);
  
  // 创建示例输入
  const batchSize = 1;
  const srcSeqLen = 5;
  const tgtSeqLen = 4;
  
  // 随机生成词元 ID（实际应用中来自分词器）
  const srcTokens = [Array.from({length: srcSeqLen}, () => Math.floor(Math.random() * 100))];
  const tgtTokens = [Array.from({length: tgtSeqLen}, () => Math.floor(Math.random() * 100))];
  
  console.log('源序列词元:', srcTokens[0]);
  console.log('目标序列词元:', tgtTokens[0]);
  
  // 设置为推理模式
  model.setTraining(false);
  
  try {
    // 前向传播
    const result = model.forward(srcTokens, tgtTokens);
    
    console.log('模型输出维度:', result.logits.length, 'x', result.logits[0].length);
    console.log('最后一个位置的 logits（前10个）:', result.logits[result.logits.length - 1].slice(0, 10));
    
    // 预测下一个词元
    const nextProbs = model.predictNext(srcTokens[0], tgtTokens[0]);
    const topProbs = nextProbs
      .map((prob, idx) => ({idx, prob}))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 5);
    
    console.log('下一个词元的 Top-5 预测:');
    topProbs.forEach((item, rank) => {
      console.log(`  ${rank + 1}. 词元 ${item.idx}: ${(item.prob * 100).toFixed(2)}%`);
    });
    
  } catch (error) {
    console.error('模型运行出错:', error.message);
  }
  
  console.log('');
}

/**
 * 示例7：编码器单独使用
 */
function example7_encoderOnly() {
  console.log('=== 示例7：编码器单独使用 ===');
  
  const config = getConfig('miniprogram'); // 使用小程序优化配置
  const model = createTransformer(config);
  
  // 只使用编码器进行编码
  const srcTokens = [[1, 5, 10, 3, 2]]; // 包含 EOS 标记
  
  console.log('输入序列:', srcTokens[0]);
  
  try {
    const encoderResult = model.encode(srcTokens);
    const encoderOutput = encoderResult.outputs[0];
    
    console.log('编码器输出维度:', encoderOutput.length, 'x', encoderOutput[0].length);
    console.log('第一个位置的编码（前5维）:', encoderOutput[0].slice(0, 5));
    console.log('最后一个位置的编码（前5维）:', encoderOutput[encoderOutput.length - 1].slice(0, 5));
    
  } catch (error) {
    console.error('编码器运行出错:', error.message);
  }
  
  console.log('');
}

/**
 * 运行所有示例
 */
function runAllExamples() {
  console.log('🚀 Transformer-JS 基础使用示例\n');
  
  try {
    example1_basicMath();
    example2_matrixOps();
    example3_basicLayers();
    example4_attention();
    example5_embedding();
    example6_fullTransformer();
    example7_encoderOnly();
    
    console.log('✅ 所有示例运行完成！');
    
  } catch (error) {
    console.error('❌ 示例运行出错:', error);
    console.error('错误堆栈:', error.stack);
  }
}

// 如果直接运行此文件，则执行所有示例
if (require.main === module) {
  runAllExamples();
}

// 导出示例函数
module.exports = {
  example1_basicMath,
  example2_matrixOps,
  example3_basicLayers,
  example4_attention,
  example5_embedding,
  example6_fullTransformer,
  example7_encoderOnly,
  runAllExamples
};
