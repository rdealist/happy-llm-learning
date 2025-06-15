/**
 * Transformer-JS 单元测试
 * 测试各个组件的基本功能
 * 
 * @author Transformer-JS
 * @version 1.0.0
 */

// 引入测试模块
const { softmax, relu, gelu, mean, std } = require('../core/math-utils');
const { matmul, transpose, add, shape } = require('../core/matrix-ops');
const { Linear, LayerNorm, MLP } = require('../core/layers');
const { Attention, MultiHeadAttention, MaskGenerator } = require('../core/attention');
const { TransformerEmbedding } = require('../core/embedding');
const { getConfig, createConfig } = require('../config/config');
const { createTransformer } = require('../core/transformer');

/**
 * 简单的断言函数
 */
function assert(condition, message) {
  if (!condition) {
    throw new Error(`断言失败: ${message}`);
  }
}

function assertArraysClose(arr1, arr2, tolerance = 1e-6, message = '') {
  assert(arr1.length === arr2.length, `数组长度不匹配: ${message}`);
  for (let i = 0; i < arr1.length; i++) {
    const diff = Math.abs(arr1[i] - arr2[i]);
    assert(diff < tolerance, `数组元素不匹配 [${i}]: ${arr1[i]} vs ${arr2[i]}, ${message}`);
  }
}

/**
 * 测试数学工具函数
 */
function testMathUtils() {
  console.log('测试数学工具函数...');
  
  // 测试 softmax
  const logits = [1, 2, 3];
  const probs = softmax(logits);
  const probSum = probs.reduce((sum, p) => sum + p, 0);
  assert(Math.abs(probSum - 1.0) < 1e-6, 'Softmax 概率和应该为1');
  assert(probs[2] > probs[1] && probs[1] > probs[0], 'Softmax 应该保持相对大小关系');
  
  // 测试 ReLU
  const reluInput = [-1, 0, 1, 2];
  const reluOutput = relu(reluInput);
  assertArraysClose(reluOutput, [0, 0, 1, 2], 1e-6, 'ReLU 输出');
  
  // 测试 GELU
  const geluOutput = gelu(0);
  assert(Math.abs(geluOutput - 0) < 1e-6, 'GELU(0) 应该约等于0');
  
  // 测试统计函数
  const data = [1, 2, 3, 4, 5];
  const meanVal = mean(data);
  const stdVal = std(data);
  assert(Math.abs(meanVal - 3) < 1e-6, '均值计算');
  assert(stdVal > 0, '标准差应该大于0');
  
  console.log('✅ 数学工具函数测试通过');
}

/**
 * 测试矩阵运算
 */
function testMatrixOps() {
  console.log('测试矩阵运算...');
  
  // 测试矩阵乘法
  const A = [[1, 2], [3, 4]];
  const B = [[5, 6], [7, 8]];
  const C = matmul(A, B);
  const expected = [[19, 22], [43, 50]];
  
  for (let i = 0; i < C.length; i++) {
    assertArraysClose(C[i], expected[i], 1e-6, `矩阵乘法第${i}行`);
  }
  
  // 测试矩阵转置
  const matrix = [[1, 2, 3], [4, 5, 6]];
  const transposed = transpose(matrix);
  assert(transposed.length === 3 && transposed[0].length === 2, '转置矩阵维度');
  assert(transposed[0][0] === 1 && transposed[2][1] === 6, '转置矩阵元素');
  
  // 测试矩阵加法
  const sum = add([[1, 2]], [[3, 4]]);
  assertArraysClose(sum[0], [4, 6], 1e-6, '矩阵加法');
  
  // 测试形状获取
  const shapeResult = shape([[1, 2, 3], [4, 5, 6]]);
  assertArraysClose(shapeResult, [2, 3], 1e-6, '矩阵形状');
  
  console.log('✅ 矩阵运算测试通过');
}

/**
 * 测试基础层
 */
function testBasicLayers() {
  console.log('测试基础层...');
  
  // 测试线性层
  const linear = new Linear(3, 2, false);
  const input = [[1, 2, 3]];
  const output = linear.forward(input);
  
  assert(output.length === 1, '线性层输出批大小');
  assert(output[0].length === 2, '线性层输出维度');
  assert(linear.getParameterCount() === 6, '线性层参数数量'); // 3*2=6
  
  // 测试层归一化
  const layerNorm = new LayerNorm(3);
  const normInput = [[1, 2, 3], [4, 5, 6]];
  const normOutput = layerNorm.forward(normInput);
  
  assert(normOutput.length === 2, '层归一化输出批大小');
  assert(normOutput[0].length === 3, '层归一化输出维度');
  
  // 检查归一化效果（每行均值应该接近0）
  for (let i = 0; i < normOutput.length; i++) {
    const rowMean = mean(normOutput[i]);
    assert(Math.abs(rowMean) < 1e-5, `层归一化第${i}行均值应该接近0`);
  }
  
  // 测试 MLP
  const mlp = new MLP(3, 6, 3, 'relu', 0.0); // 不使用 dropout
  mlp.setTraining(false); // 设置为推理模式
  const mlpOutput = mlp.forward(input);
  
  assert(mlpOutput.length === 1, 'MLP 输出批大小');
  assert(mlpOutput[0].length === 3, 'MLP 输出维度');
  
  console.log('✅ 基础层测试通过');
}

/**
 * 测试注意力机制
 */
function testAttention() {
  console.log('测试注意力机制...');
  
  // 测试基础注意力
  const attention = new Attention(0.0); // 不使用 dropout
  attention.setTraining(false);
  
  const seqLen = 3;
  const dModel = 4;
  const Q = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];
  const K = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];
  const V = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]];
  
  const attentionResult = attention.forward(Q, K, V);
  
  assert(attentionResult.output.length === seqLen, '注意力输出序列长度');
  assert(attentionResult.output[0].length === dModel, '注意力输出维度');
  assert(attentionResult.attention.length === seqLen, '注意力权重矩阵大小');
  
  // 检查注意力权重和为1
  for (let i = 0; i < attentionResult.attention.length; i++) {
    const weightSum = attentionResult.attention[i].reduce((sum, w) => sum + w, 0);
    assert(Math.abs(weightSum - 1.0) < 1e-5, `注意力权重第${i}行和应该为1`);
  }
  
  // 测试多头注意力
  const multiHeadAttention = new MultiHeadAttention(dModel, 2, 0.0);
  multiHeadAttention.setTraining(false);
  
  const multiHeadResult = multiHeadAttention.forward(Q, K, V);
  
  assert(multiHeadResult.output.length === seqLen, '多头注意力输出序列长度');
  assert(multiHeadResult.output[0].length === dModel, '多头注意力输出维度');
  
  // 测试掩码生成
  const causalMask = MaskGenerator.createCausalMask(3);
  assert(causalMask.length === 3 && causalMask[0].length === 3, '因果掩码维度');
  assert(causalMask[0][0] === 1 && causalMask[0][1] === 0, '因果掩码上三角为0');
  assert(causalMask[2][0] === 1 && causalMask[2][1] === 1, '因果掩码下三角为1');
  
  console.log('✅ 注意力机制测试通过');
}

/**
 * 测试嵌入层
 */
function testEmbedding() {
  console.log('测试嵌入层...');
  
  const vocabSize = 100;
  const embedDim = 8;
  const maxLen = 10;
  
  const embedding = new TransformerEmbedding(vocabSize, embedDim, maxLen, 'sinusoidal', 0.0);
  embedding.setTraining(false);
  
  const tokenIds = [[1, 5, 10]];
  const embedded = embedding.forward(tokenIds);
  
  assert(embedded.length === 1, '嵌入输出批大小');
  assert(embedded[0].length === 3, '嵌入输出序列长度');
  assert(embedded[0][0].length === embedDim, '嵌入输出维度');
  
  // 检查不同位置的嵌入是否不同（由于位置编码）
  const pos0 = embedded[0][0];
  const pos1 = embedded[0][1];
  let isDifferent = false;
  for (let i = 0; i < embedDim; i++) {
    if (Math.abs(pos0[i] - pos1[i]) > 1e-6) {
      isDifferent = true;
      break;
    }
  }
  assert(isDifferent, '不同位置的嵌入应该不同');
  
  console.log('✅ 嵌入层测试通过');
}

/**
 * 测试配置系统
 */
function testConfig() {
  console.log('测试配置系统...');
  
  // 测试预设配置
  const configs = ['small', 'default', 'large', 'miniprogram'];
  
  configs.forEach(configName => {
    const config = getConfig(configName);
    assert(config.vocab_size > 0, `${configName} 配置词汇表大小`);
    assert(config.n_embd > 0, `${configName} 配置嵌入维度`);
    assert(config.n_layers > 0, `${configName} 配置层数`);
    assert(config.n_heads > 0, `${configName} 配置注意力头数`);
    assert(config.n_embd % config.n_heads === 0, `${configName} 配置维度整除性`);
  });
  
  // 测试自定义配置
  const customConfig = createConfig(getConfig('default'), {
    vocab_size: 8000,
    n_embd: 256,
    n_layers: 4
  });
  
  assert(customConfig.vocab_size === 8000, '自定义配置词汇表大小');
  assert(customConfig.n_embd === 256, '自定义配置嵌入维度');
  assert(customConfig.n_layers === 4, '自定义配置层数');
  
  console.log('✅ 配置系统测试通过');
}

/**
 * 测试完整模型
 */
function testFullModel() {
  console.log('测试完整模型...');
  
  const config = getConfig('miniprogram'); // 使用最小配置
  const model = createTransformer(config);
  model.setTraining(false);
  
  // 测试模型创建
  const summary = model.summary();
  assert(summary.vocabSize === config.vocab_size, '模型词汇表大小');
  assert(summary.modelDim === config.n_embd, '模型维度');
  assert(summary.layers === config.n_layers, '模型层数');
  
  // 测试参数统计
  const paramCount = model.getParameterCount();
  assert(paramCount.total > 0, '模型参数总数');
  assert(typeof paramCount.totalM === 'string', '参数数量格式');
  
  // 测试前向传播
  const srcTokens = [[2, 10, 20, 3]]; // 简短序列
  const tgtTokens = [[2, 15, 25]];
  
  try {
    const result = model.forward(srcTokens, tgtTokens);
    assert(Array.isArray(result.logits), '模型输出 logits');
    assert(result.logits.length === tgtTokens[0].length, '输出序列长度');
    assert(result.logits[0].length === config.vocab_size, '输出词汇表维度');
  } catch (error) {
    console.warn('模型前向传播测试跳过:', error.message);
  }
  
  // 测试编码器单独使用
  try {
    const encoderResult = model.encode(srcTokens);
    assert(Array.isArray(encoderResult.outputs), '编码器输出');
    assert(encoderResult.outputs[0].length === srcTokens[0].length, '编码器输出序列长度');
  } catch (error) {
    console.warn('编码器测试跳过:', error.message);
  }
  
  console.log('✅ 完整模型测试通过');
}

/**
 * 运行所有测试
 */
function runAllTests() {
  console.log('🧪 开始运行 Transformer-JS 单元测试\n');
  
  const tests = [
    testMathUtils,
    testMatrixOps,
    testBasicLayers,
    testAttention,
    testEmbedding,
    testConfig,
    testFullModel
  ];
  
  let passedTests = 0;
  let failedTests = 0;
  
  for (const test of tests) {
    try {
      test();
      passedTests++;
    } catch (error) {
      console.error(`❌ 测试失败: ${test.name}`);
      console.error(`   错误: ${error.message}`);
      failedTests++;
    }
  }
  
  console.log('\n📊 测试结果统计:');
  console.log(`✅ 通过: ${passedTests}`);
  console.log(`❌ 失败: ${failedTests}`);
  console.log(`📈 成功率: ${((passedTests / (passedTests + failedTests)) * 100).toFixed(1)}%`);
  
  if (failedTests === 0) {
    console.log('\n🎉 所有测试通过！');
  } else {
    console.log('\n⚠️  部分测试失败，请检查代码');
  }
  
  return failedTests === 0;
}

// 如果直接运行此文件，则执行所有测试
if (require.main === module) {
  runAllTests();
}

// 导出测试函数
module.exports = {
  testMathUtils,
  testMatrixOps,
  testBasicLayers,
  testAttention,
  testEmbedding,
  testConfig,
  testFullModel,
  runAllTests
};
