/**
 * 完整 Transformer 模型使用示例
 * 演示如何构建、配置和使用完整的 Transformer 模型
 * 
 * @author Transformer-JS
 * @version 1.0.0
 */

const { getConfig, createConfig, printConfig, estimateParameters } = require('../config/config');
const { createTransformer } = require('../core/transformer');
const { MaskGenerator } = require('../core/attention');
const { softmax } = require('../core/math-utils');

/**
 * 示例1：创建不同规模的模型
 */
function example1_differentModelSizes() {
  console.log('=== 示例1：不同规模的模型 ===');
  
  const configs = ['small', 'default', 'large', 'miniprogram'];
  
  configs.forEach(configName => {
    console.log(`\n--- ${configName.toUpperCase()} 配置 ---`);
    
    const config = getConfig(configName);
    const paramEstimate = estimateParameters(config);
    
    console.log(`词汇表大小: ${config.vocab_size}`);
    console.log(`模型维度: ${config.n_embd}`);
    console.log(`层数: ${config.n_layers}`);
    console.log(`注意力头数: ${config.n_heads}`);
    console.log(`参数量估计: ${paramEstimate.totalParamsM}`);
    
    // 创建模型并获取实际参数量
    try {
      const model = createTransformer(config);
      const actualParams = model.getParameterCount();
      console.log(`实际参数量: ${actualParams.totalM}`);
    } catch (error) {
      console.log(`模型创建失败: ${error.message}`);
    }
  });
  
  console.log('');
}

/**
 * 示例2：自定义模型配置
 */
function example2_customConfig() {
  console.log('=== 示例2：自定义模型配置 ===');
  
  // 基于默认配置创建自定义配置
  const customConfig = createConfig(getConfig('default'), {
    vocab_size: 8000,      // 较小的词汇表
    n_embd: 256,           // 较小的模型维度
    n_layers: 4,           // 较少的层数
    n_heads: 8,            // 保持注意力头数
    max_seq_len: 128,      // 较短的序列长度
    dropout: 0.05,         // 较小的 dropout
    activation: 'gelu',    // 使用 GELU 激活函数
    tie_word_embeddings: true  // 共享嵌入权重
  });
  
  console.log('自定义配置:');
  printConfig(customConfig);
  
  // 创建自定义模型
  const customModel = createTransformer(customConfig);
  const summary = customModel.summary();
  
  console.log('自定义模型摘要:');
  console.log(`架构: ${summary.architecture}`);
  console.log(`参数量: ${summary.parameters.totalM}`);
  console.log(`配置验证: ✅ 成功`);
  
  console.log('');
}

/**
 * 示例3：序列到序列翻译任务
 */
function example3_seq2seqTranslation() {
  console.log('=== 示例3：序列到序列翻译任务 ===');
  
  // 使用小型配置进行演示
  const config = getConfig('small');
  const model = createTransformer(config);
  
  // 模拟翻译任务的词元序列
  // 假设: 0=PAD, 1=UNK, 2=BOS, 3=EOS
  const srcSentence = [2, 10, 25, 67, 89, 3];  // "BOS hello world how are EOS"
  const tgtSentence = [2, 15, 30, 45, 60, 75]; // "BOS 你好 世界 怎么 样 ..."
  
  console.log('源语言句子 (词元ID):', srcSentence);
  console.log('目标语言句子 (词元ID):', tgtSentence);
  
  // 设置为推理模式
  model.setTraining(false);
  
  try {
    // 完整的前向传播
    const result = model.forward([srcSentence], [tgtSentence]);
    
    console.log('模型输出维度:', result.logits.length, 'x', result.logits[0].length);
    
    // 分析每个位置的预测
    console.log('\n各位置的预测分析:');
    for (let pos = 0; pos < result.logits.length; pos++) {
      const logits = result.logits[pos];
      const probs = softmax(logits);
      
      // 找到概率最高的前3个词元
      const topPredictions = probs
        .map((prob, idx) => ({tokenId: idx, prob}))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 3);
      
      console.log(`位置 ${pos}:`);
      console.log(`  实际词元: ${tgtSentence[pos]}`);
      console.log(`  Top-3 预测:`);
      topPredictions.forEach((pred, rank) => {
        console.log(`    ${rank + 1}. 词元 ${pred.tokenId}: ${(pred.prob * 100).toFixed(1)}%`);
      });
    }
    
  } catch (error) {
    console.error('翻译任务执行失败:', error.message);
  }
  
  console.log('');
}

/**
 * 示例4：自回归文本生成
 */
function example4_autoregressiveGeneration() {
  console.log('=== 示例4：自回归文本生成 ===');
  
  const config = getConfig('miniprogram');
  const model = createTransformer(config);
  model.setTraining(false);
  
  // 输入上下文
  const context = [2, 20, 35, 50]; // "BOS 今天 天气 很"
  const maxLength = 8;
  const eosToken = 3;
  
  console.log('输入上下文:', context);
  console.log('开始自回归生成...\n');
  
  try {
    let generated = [...context];
    
    for (let step = 0; step < maxLength - context.length; step++) {
      console.log(`生成步骤 ${step + 1}:`);
      console.log(`当前序列: [${generated.join(', ')}]`);
      
      // 预测下一个词元
      const nextProbs = model.predictNext(context, generated);
      
      // 简单的贪心解码：选择概率最高的词元
      let nextToken = 0;
      let maxProb = 0;
      for (let i = 0; i < nextProbs.length; i++) {
        if (nextProbs[i] > maxProb) {
          maxProb = nextProbs[i];
          nextToken = i;
        }
      }
      
      console.log(`预测下一个词元: ${nextToken} (概率: ${(maxProb * 100).toFixed(1)}%)`);
      
      // 添加到序列
      generated.push(nextToken);
      
      // 如果生成了结束标记，停止生成
      if (nextToken === eosToken) {
        console.log('遇到结束标记，停止生成');
        break;
      }
      
      console.log('');
    }
    
    console.log('最终生成序列:', generated);
    console.log('生成的新词元:', generated.slice(context.length));
    
  } catch (error) {
    console.error('文本生成失败:', error.message);
  }
  
  console.log('');
}

/**
 * 示例5：注意力权重可视化
 */
function example5_attentionVisualization() {
  console.log('=== 示例5：注意力权重可视化 ===');
  
  const config = createConfig(getConfig('small'), {
    n_layers: 2,  // 减少层数便于观察
    n_heads: 2    // 减少头数便于观察
  });
  
  const model = createTransformer(config);
  model.setTraining(false);
  
  const srcTokens = [2, 10, 20, 30, 3]; // 长度为5的源序列
  const tgtTokens = [2, 15, 25];        // 长度为3的目标序列
  
  console.log('源序列:', srcTokens);
  console.log('目标序列:', tgtTokens);
  
  try {
    const result = model.forward([srcTokens], [tgtTokens]);
    
    // 分析编码器注意力
    console.log('\n编码器注意力权重:');
    result.encoderAttentions.forEach((layerAttentions, layerIdx) => {
      console.log(`\n第 ${layerIdx + 1} 层编码器:`);
      layerAttentions.forEach((headAttention, headIdx) => {
        console.log(`  头 ${headIdx + 1}:`);
        headAttention.forEach((row, i) => {
          const formattedRow = row.map(val => val.toFixed(3)).join(' ');
          console.log(`    位置 ${i}: [${formattedRow}]`);
        });
      });
    });
    
    // 分析解码器自注意力
    console.log('\n解码器自注意力权重:');
    result.decoderSelfAttentions.forEach((layerAttentions, layerIdx) => {
      console.log(`\n第 ${layerIdx + 1} 层解码器自注意力:`);
      layerAttentions.forEach((headAttention, headIdx) => {
        console.log(`  头 ${headIdx + 1}:`);
        headAttention.forEach((row, i) => {
          const formattedRow = row.map(val => val.toFixed(3)).join(' ');
          console.log(`    位置 ${i}: [${formattedRow}]`);
        });
      });
    });
    
    // 分析解码器交叉注意力
    console.log('\n解码器交叉注意力权重:');
    result.decoderCrossAttentions.forEach((layerAttentions, layerIdx) => {
      console.log(`\n第 ${layerIdx + 1} 层解码器交叉注意力:`);
      layerAttentions.forEach((headAttention, headIdx) => {
        console.log(`  头 ${headIdx + 1} (解码器位置 -> 编码器位置):`);
        headAttention.forEach((row, i) => {
          const formattedRow = row.map(val => val.toFixed(3)).join(' ');
          console.log(`    解码器位置 ${i}: [${formattedRow}]`);
        });
      });
    });
    
  } catch (error) {
    console.error('注意力分析失败:', error.message);
  }
  
  console.log('');
}

/**
 * 示例6：性能基准测试
 */
function example6_performanceBenchmark() {
  console.log('=== 示例6：性能基准测试 ===');
  
  const configs = ['miniprogram', 'small'];
  
  configs.forEach(configName => {
    console.log(`\n--- ${configName.toUpperCase()} 配置性能测试 ---`);
    
    const config = getConfig(configName);
    const model = createTransformer(config);
    model.setTraining(false);
    
    // 测试数据
    const srcTokens = Array.from({length: config.max_seq_len}, (_, i) => i % 100);
    const tgtTokens = Array.from({length: Math.floor(config.max_seq_len / 2)}, (_, i) => i % 100);
    
    console.log(`序列长度: 源=${srcTokens.length}, 目标=${tgtTokens.length}`);
    
    // 预热
    try {
      model.forward([srcTokens.slice(0, 10)], [tgtTokens.slice(0, 5)]);
    } catch (error) {
      console.log('预热失败，跳过性能测试');
      return;
    }
    
    // 性能测试
    const iterations = 3;
    const times = [];
    
    for (let i = 0; i < iterations; i++) {
      const startTime = Date.now();
      
      try {
        model.forward([srcTokens], [tgtTokens]);
        const endTime = Date.now();
        times.push(endTime - startTime);
      } catch (error) {
        console.log(`第 ${i + 1} 次测试失败:`, error.message);
        break;
      }
    }
    
    if (times.length > 0) {
      const avgTime = times.reduce((sum, t) => sum + t, 0) / times.length;
      const minTime = Math.min(...times);
      const maxTime = Math.max(...times);
      
      console.log(`平均推理时间: ${avgTime.toFixed(1)}ms`);
      console.log(`最快推理时间: ${minTime}ms`);
      console.log(`最慢推理时间: ${maxTime}ms`);
      console.log(`参数量: ${model.getParameterCount().totalM}`);
    }
  });
  
  console.log('');
}

/**
 * 运行所有示例
 */
function runAllExamples() {
  console.log('🚀 完整 Transformer 模型使用示例\n');
  
  try {
    example1_differentModelSizes();
    example2_customConfig();
    example3_seq2seqTranslation();
    example4_autoregressiveGeneration();
    example5_attentionVisualization();
    example6_performanceBenchmark();
    
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
  example1_differentModelSizes,
  example2_customConfig,
  example3_seq2seqTranslation,
  example4_autoregressiveGeneration,
  example5_attentionVisualization,
  example6_performanceBenchmark,
  runAllExamples
};
