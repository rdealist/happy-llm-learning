/**
 * 简单演示脚本
 * 展示 Transformer-JS 的核心功能，避免内存问题
 * 
 * @author Transformer-JS
 * @version 1.0.0
 */

const { getConfig } = require('../config/config');
const { createTransformer } = require('../core/transformer');
const { MultiHeadAttention } = require('../core/attention');
const { TransformerEmbedding } = require('../core/embedding');

/**
 * 演示1：基础组件功能
 */
function demo1_basicComponents() {
  console.log('=== 演示1：基础组件功能 ===');
  
  // 测试多头注意力
  console.log('测试多头注意力机制...');
  const attention = new MultiHeadAttention(64, 4, 0.0); // 小维度，4个头
  attention.setTraining(false);
  
  // 创建简单的输入
  const seqLen = 3;
  const dModel = 64;
  const Q = Array.from({length: seqLen}, () => 
    Array.from({length: dModel}, () => Math.random() * 0.1)
  );
  const K = Array.from({length: seqLen}, () => 
    Array.from({length: dModel}, () => Math.random() * 0.1)
  );
  const V = Array.from({length: seqLen}, () => 
    Array.from({length: dModel}, () => Math.random() * 0.1)
  );
  
  const result = attention.forward(Q, K, V);
  console.log(`✅ 多头注意力输出维度: ${result.output.length} x ${result.output[0].length}`);
  console.log(`✅ 参数数量: ${attention.getParameterCount()}`);
  
  // 测试嵌入层
  console.log('\n测试嵌入层...');
  const embedding = new TransformerEmbedding(1000, 64, 32, 'sinusoidal', 0.0);
  embedding.setTraining(false);
  
  const tokenIds = [[1, 5, 10, 3]];
  const embedded = embedding.forward(tokenIds);
  console.log(`✅ 嵌入输出维度: ${embedded.length} x ${embedded[0].length} x ${embedded[0][0].length}`);
  console.log(`✅ 嵌入层参数数量: ${embedding.getParameterCount()}`);
  
  console.log('');
}

/**
 * 演示2：小型 Transformer 模型
 */
function demo2_smallTransformer() {
  console.log('=== 演示2：小型 Transformer 模型 ===');
  
  // 创建超小型配置
  const config = {
    vocab_size: 1000,
    n_embd: 64,
    n_layers: 2,
    n_heads: 4,
    max_seq_len: 16,
    ffn_hidden_dim: 128,
    dropout: 0.0,
    activation: 'relu',
    use_bias: false,
    tie_word_embeddings: true
  };
  
  console.log('创建超小型 Transformer 模型...');
  console.log(`配置: 词汇表=${config.vocab_size}, 维度=${config.n_embd}, 层数=${config.n_layers}`);
  
  try {
    const model = createTransformer(config);
    model.setTraining(false);
    
    const summary = model.summary();
    console.log(`✅ 模型创建成功`);
    console.log(`✅ 参数量: ${summary.parameters.totalM}`);
    
    // 测试编码器
    console.log('\n测试编码器...');
    const srcTokens = [[2, 10, 20, 30, 3]]; // 短序列
    
    try {
      const encoderResult = model.encode(srcTokens);
      console.log(`✅ 编码器输出维度: ${encoderResult.outputs[0].length} x ${encoderResult.outputs[0][0].length}`);
    } catch (error) {
      console.log(`⚠️ 编码器测试跳过: ${error.message}`);
    }
    
    // 测试完整前向传播（非常短的序列）
    console.log('\n测试完整模型...');
    const tgtTokens = [[2, 15, 25]]; // 很短的目标序列
    
    try {
      const result = model.forward(srcTokens, tgtTokens);
      console.log(`✅ 模型输出维度: ${result.logits.length} x ${result.logits[0].length}`);
      
      // 显示最后一个位置的前5个 logits
      const lastLogits = result.logits[result.logits.length - 1];
      console.log(`✅ 最后位置前5个 logits: [${lastLogits.slice(0, 5).map(x => x.toFixed(3)).join(', ')}]`);
      
    } catch (error) {
      console.log(`⚠️ 完整模型测试跳过: ${error.message}`);
    }
    
  } catch (error) {
    console.log(`❌ 模型创建失败: ${error.message}`);
  }
  
  console.log('');
}

/**
 * 演示3：配置对比
 */
function demo3_configComparison() {
  console.log('=== 演示3：配置对比 ===');
  
  const configs = [
    {
      name: '微型',
      config: {
        vocab_size: 500,
        n_embd: 32,
        n_layers: 1,
        n_heads: 2,
        max_seq_len: 8,
        ffn_hidden_dim: 64
      }
    },
    {
      name: '小型',
      config: {
        vocab_size: 1000,
        n_embd: 64,
        n_layers: 2,
        n_heads: 4,
        max_seq_len: 16,
        ffn_hidden_dim: 128
      }
    },
    {
      name: '中型',
      config: {
        vocab_size: 2000,
        n_embd: 128,
        n_layers: 3,
        n_heads: 4,
        max_seq_len: 32,
        ffn_hidden_dim: 256
      }
    }
  ];
  
  configs.forEach(({name, config}) => {
    console.log(`\n--- ${name}配置 ---`);
    console.log(`词汇表: ${config.vocab_size}, 维度: ${config.n_embd}, 层数: ${config.n_layers}`);
    
    try {
      const fullConfig = {
        ...config,
        dropout: 0.0,
        activation: 'relu',
        use_bias: false,
        tie_word_embeddings: true
      };
      
      const model = createTransformer(fullConfig);
      const params = model.getParameterCount();
      console.log(`✅ 参数量: ${params.totalM}`);
      
      // 估算内存使用（简化计算）
      const memoryMB = (params.total * 4) / (1024 * 1024); // 假设每个参数4字节
      console.log(`📊 估算内存: ${memoryMB.toFixed(1)}MB`);
      
    } catch (error) {
      console.log(`❌ 创建失败: ${error.message}`);
    }
  });
  
  console.log('');
}

/**
 * 演示4：注意力权重可视化
 */
function demo4_attentionVisualization() {
  console.log('=== 演示4：注意力权重可视化 ===');
  
  // 创建最小配置用于注意力分析
  const config = {
    vocab_size: 100,
    n_embd: 32,
    n_layers: 1,
    n_heads: 2,
    max_seq_len: 8,
    ffn_hidden_dim: 64,
    dropout: 0.0,
    activation: 'relu',
    use_bias: false,
    tie_word_embeddings: true
  };
  
  try {
    const model = createTransformer(config);
    model.setTraining(false);
    
    const srcTokens = [[2, 10, 20, 3]]; // 4个词元
    const tgtTokens = [[2, 15]];        // 2个词元
    
    console.log(`输入序列: [${srcTokens[0].join(', ')}]`);
    console.log(`目标序列: [${tgtTokens[0].join(', ')}]`);
    
    const result = model.forward(srcTokens, tgtTokens);
    
    // 显示编码器注意力
    if (result.encoderAttentions && result.encoderAttentions.length > 0) {
      console.log('\n📊 编码器注意力权重 (第1层, 第1头):');
      const attention = result.encoderAttentions[0][0];
      attention.forEach((row, i) => {
        const formattedRow = row.map(val => val.toFixed(3)).join('  ');
        console.log(`  位置${i}: [${formattedRow}]`);
      });
    }
    
    // 显示解码器自注意力
    if (result.decoderSelfAttentions && result.decoderSelfAttentions.length > 0) {
      console.log('\n📊 解码器自注意力权重 (第1层, 第1头):');
      const attention = result.decoderSelfAttentions[0][0];
      attention.forEach((row, i) => {
        const formattedRow = row.map(val => val.toFixed(3)).join('  ');
        console.log(`  位置${i}: [${formattedRow}]`);
      });
    }
    
    console.log(`\n✅ 注意力分析完成`);
    
  } catch (error) {
    console.log(`❌ 注意力分析失败: ${error.message}`);
  }
  
  console.log('');
}

/**
 * 运行所有演示
 */
function runAllDemos() {
  console.log('🚀 Transformer-JS 简单演示\n');
  
  try {
    demo1_basicComponents();
    demo2_smallTransformer();
    demo3_configComparison();
    demo4_attentionVisualization();
    
    console.log('✅ 所有演示完成！');
    console.log('\n💡 提示:');
    console.log('- 这些演示使用了非常小的模型配置以避免内存问题');
    console.log('- 在实际应用中，可以根据需要调整模型大小');
    console.log('- 微信小程序环境建议使用 miniprogram 配置');
    
  } catch (error) {
    console.error('❌ 演示运行出错:', error);
    console.error('错误堆栈:', error.stack);
  }
}

// 如果直接运行此文件，则执行所有演示
if (require.main === module) {
  runAllDemos();
}

// 导出演示函数
module.exports = {
  demo1_basicComponents,
  demo2_smallTransformer,
  demo3_configComparison,
  demo4_attentionVisualization,
  runAllDemos
};
