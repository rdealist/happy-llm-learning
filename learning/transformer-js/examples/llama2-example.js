/**
 * LLaMA2 模型使用示例 (JavaScript版本)
 * 
 * 展示如何使用基于第四章和第五章理论实现的 LLaMA2 模型进行：
 * 1. 模型创建和配置
 * 2. 前向传播
 * 3. 文本生成
 * 4. 性能测试
 * 
 * @author shihom_wu
 * @version 1.0.0
 * 基于: Happy-LLM 项目第四章和第五章理论
 */

const { 
  LLaMA2Config, 
  LLaMA2ForCausalLM 
} = require('../models/llama2');
const { MaskGenerator } = require('../core/attention');

/**
 * 示例1: 模型创建和配置
 */
function exampleModelCreation() {
  console.log('='.repeat(60));
  console.log('示例1: LLaMA2 模型创建和配置');
  console.log('='.repeat(60));
  
  // 创建不同规模的模型配置
  const configs = {
    '微信小程序优化': LLaMA2Config.miniprogram(),
    '7B模型配置': LLaMA2Config.llama2_7b(),
    '13B模型配置': LLaMA2Config.llama2_13b(),
    '70B模型配置': LLaMA2Config.llama2_70b()
  };
  
  for (const [name, config] of Object.entries(configs)) {
    console.log(`\n${name}:`);
    console.log(`  词汇表大小: ${config.vocabSize}`);
    console.log(`  模型维度: ${config.dModel}`);
    console.log(`  层数: ${config.numLayers}`);
    console.log(`  注意力头数: ${config.numHeads}`);
    console.log(`  键值头数: ${config.numKVHeads}`);
    console.log(`  前馈维度: ${config.dFF}`);
    console.log(`  最大序列长度: ${config.maxSeqLen}`);
    
    if (name === '微信小程序优化') {
      // 只为小程序配置创建实际的模型实例
      const model = new LLaMA2ForCausalLM(config);
      const totalParams = model.getParameterCount();
      console.log(`  总参数量: ${totalParams.toLocaleString()}`);
    }
  }
}

/**
 * 示例2: 前向传播
 */
function exampleForwardPass() {
  console.log('\n' + '='.repeat(60));
  console.log('示例2: LLaMA2 前向传播');
  console.log('='.repeat(60));
  
  // 创建小型模型
  const config = new LLaMA2Config({
    vocabSize: 1000,
    dModel: 256,
    numLayers: 4,
    numHeads: 8,
    numKVHeads: 2, // 使用分组查询注意力
    dFF: 1024,
    maxSeqLen: 64
  });
  
  const model = new LLaMA2ForCausalLM(config);
  model.setTraining(false);
  
  // 创建示例输入
  const seqLen = 16;
  const inputIds = Array.from({ length: seqLen }, () => 
    Math.floor(Math.random() * config.vocabSize)
  );
  
  console.log(`输入长度: ${inputIds.length}`);
  console.log(`输入内容: [${inputIds.slice(0, 10).join(', ')}...]`);
  
  // 前向传播
  const startTime = performance.now();
  const outputs = model.forward(inputIds);
  const endTime = performance.now();
  
  const logits = outputs.logits;
  console.log(`输出logits形状: [${logits.length}, ${logits[0].length}]`);
  console.log(`前向传播耗时: ${(endTime - startTime).toFixed(2)} ms`);
  
  // 计算下一个词元的概率
  const lastLogits = logits[logits.length - 1];
  const maxLogit = Math.max(...lastLogits);
  const expLogits = lastLogits.map(x => Math.exp(x - maxLogit));
  const sumExp = expLogits.reduce((sum, x) => sum + x, 0);
  const probs = expLogits.map(x => x / sumExp);
  
  // 找到Top-5预测
  const probsWithIndices = probs.map((prob, idx) => ({ prob, idx }));
  probsWithIndices.sort((a, b) => b.prob - a.prob);
  
  console.log('下一个词元的Top-5预测:');
  for (let i = 0; i < 5; i++) {
    const { prob, idx } = probsWithIndices[i];
    console.log(`  ${i + 1}. 词元ID ${idx}: 概率 ${prob.toFixed(4)}`);
  }
}

/**
 * 示例3: 文本生成
 */
function exampleTextGeneration() {
  console.log('\n' + '='.repeat(60));
  console.log('示例3: LLaMA2 文本生成');
  console.log('='.repeat(60));
  
  // 创建小型模型
  const config = new LLaMA2Config({
    vocabSize: 1000,
    dModel: 256,
    numLayers: 4,
    numHeads: 8,
    numKVHeads: 2,
    dFF: 1024,
    maxSeqLen: 64
  });
  
  const model = new LLaMA2ForCausalLM(config);
  model.setTraining(false);
  
  // 输入提示
  const prompt = [1, 123, 456, 789]; // BOS + 一些词元
  console.log(`输入提示: [${prompt.join(', ')}]`);
  
  // 不同的生成策略
  const generationConfigs = [
    { maxNewTokens: 10, temperature: 1.0, doSample: false }, // 贪心
    { maxNewTokens: 10, temperature: 0.8, doSample: true },  // 采样
    { maxNewTokens: 10, temperature: 1.2, doSample: true },  // 高温度采样
  ];
  
  generationConfigs.forEach((genConfig, i) => {
    console.log(`\n生成策略 ${i + 1}: ${JSON.stringify(genConfig)}`);
    
    const startTime = performance.now();
    const generated = model.generate(prompt, genConfig);
    const endTime = performance.now();
    
    console.log(`生成结果: [${generated.join(', ')}]`);
    console.log(`生成耗时: ${(endTime - startTime).toFixed(2)} ms`);
  });
}

/**
 * 示例4: 性能测试
 */
function examplePerformanceTest() {
  console.log('\n' + '='.repeat(60));
  console.log('示例4: LLaMA2 性能测试');
  console.log('='.repeat(60));
  
  // 测试不同配置的性能
  const testConfigs = [
    {
      name: '微信小程序配置',
      config: LLaMA2Config.miniprogram()
    },
    {
      name: '小型配置',
      config: new LLaMA2Config({
        vocabSize: 1000,
        dModel: 512,
        numLayers: 6,
        numHeads: 8,
        numKVHeads: 4,
        dFF: 2048,
        maxSeqLen: 128
      })
    }
  ];
  
  testConfigs.forEach(({ name, config }) => {
    console.log(`\n测试配置: ${name}`);
    console.log(`参数: d_model=${config.dModel}, layers=${config.numLayers}, heads=${config.numHeads}/${config.numKVHeads}`);
    
    const model = new LLaMA2ForCausalLM(config);
    model.setTraining(false);
    
    // 测试不同序列长度的性能
    const seqLengths = [8, 16, 32];
    
    seqLengths.forEach(seqLen => {
      if (seqLen <= config.maxSeqLen) {
        const inputIds = Array.from({ length: seqLen }, () => 
          Math.floor(Math.random() * config.vocabSize)
        );
        
        // 预热
        model.forward(inputIds);
        
        // 性能测试
        const numRuns = 5;
        const times = [];
        
        for (let i = 0; i < numRuns; i++) {
          const startTime = performance.now();
          model.forward(inputIds);
          const endTime = performance.now();
          times.push(endTime - startTime);
        }
        
        const avgTime = times.reduce((sum, t) => sum + t, 0) / numRuns;
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);
        
        console.log(`  序列长度 ${seqLen}: 平均 ${avgTime.toFixed(2)}ms, 最小 ${minTime.toFixed(2)}ms, 最大 ${maxTime.toFixed(2)}ms`);
      }
    });
    
    // 内存使用估算
    const totalParams = model.getParameterCount();
    const memoryMB = (totalParams * 4) / (1024 * 1024); // 假设每个参数4字节
    console.log(`  估算内存使用: ${memoryMB.toFixed(2)} MB`);
  });
}

/**
 * 示例5: 注意力掩码使用
 */
function exampleAttentionMasks() {
  console.log('\n' + '='.repeat(60));
  console.log('示例5: 注意力掩码使用');
  console.log('='.repeat(60));
  
  const seqLen = 8;
  
  // 创建不同类型的掩码
  const causalMask = MaskGenerator.createCausalMask(seqLen);
  const bidirectionalMask = MaskGenerator.createBidirectionalMask(seqLen);
  const tokenIds = [1, 123, 456, 0, 0, 0, 789, 2]; // 包含填充的序列
  const paddingMask = MaskGenerator.createPaddingMask(tokenIds, 0);
  
  console.log('因果掩码 (下三角):');
  causalMask.forEach(row => {
    console.log('  ' + row.map(x => x ? '1' : '0').join(' '));
  });
  
  console.log('\n双向掩码 (全1):');
  console.log(`  ${bidirectionalMask[0].length}x${bidirectionalMask.length} 全1矩阵`);
  
  console.log('\n填充掩码:');
  console.log(`  输入序列: [${tokenIds.join(', ')}]`);
  paddingMask.forEach(row => {
    console.log('  ' + row.map(x => x ? '1' : '0').join(' '));
  });
  
  // 组合掩码
  const combinedMask = MaskGenerator.combineMasks([causalMask, paddingMask]);
  console.log('\n组合掩码 (因果 + 填充):');
  combinedMask.forEach(row => {
    console.log('  ' + row.map(x => x ? '1' : '0').join(' '));
  });
}

/**
 * 主函数：运行所有示例
 */
function main() {
  console.log('LLaMA2 模型示例 (JavaScript版本)');
  console.log('基于 Happy-LLM 项目第四章和第五章理论实现');
  console.log('作者: shihom_wu');
  
  try {
    exampleModelCreation();
    exampleForwardPass();
    exampleTextGeneration();
    examplePerformanceTest();
    exampleAttentionMasks();
    
    console.log('\n' + '='.repeat(60));
    console.log('所有示例运行完成！');
    console.log('='.repeat(60));
    
  } catch (error) {
    console.error('运行示例时出错:', error);
    console.error(error.stack);
  }
}

// 运行示例
if (require.main === module) {
  main();
}

module.exports = {
  exampleModelCreation,
  exampleForwardPass,
  exampleTextGeneration,
  examplePerformanceTest,
  exampleAttentionMasks
};
