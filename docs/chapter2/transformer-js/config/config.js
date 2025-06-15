/**
 * Transformer 模型配置文件
 * 定义模型的各种超参数和配置选项
 * 
 * @author Transformer-JS
 * @version 1.0.0
 */

/**
 * 默认模型配置
 * 基于原始 Transformer 论文的标准配置
 */
const DEFAULT_CONFIG = {
  // 模型基础参数
  vocab_size: 30000,        // 词汇表大小
  max_seq_len: 512,         // 最大序列长度
  n_embd: 512,              // 嵌入维度
  n_heads: 8,               // 注意力头数
  n_layers: 6,              // 编码器/解码器层数
  
  // 前馈网络参数
  ffn_hidden_dim: 2048,     // 前馈网络隐藏层维度
  
  // 正则化参数
  dropout: 0.1,             // Dropout 概率
  layer_norm_eps: 1e-6,     // 层归一化的 epsilon
  
  // 训练参数
  learning_rate: 1e-4,      // 学习率
  weight_decay: 0.01,       // 权重衰减
  warmup_steps: 4000,       // 预热步数
  
  // 初始化参数
  init_std: 0.02,           // 权重初始化标准差
  
  // 位置编码参数
  max_position_embeddings: 512,  // 最大位置编码长度
  position_encoding_base: 10000, // 位置编码基数
  
  // 注意力机制参数
  attention_dropout: 0.1,   // 注意力 dropout
  use_causal_mask: true,    // 是否使用因果掩码（解码器）
  
  // 激活函数
  activation: 'relu',       // 激活函数类型: 'relu', 'gelu'
  
  // 其他配置
  use_bias: false,          // 线性层是否使用偏置
  tie_word_embeddings: true, // 是否共享输入输出嵌入权重
};

/**
 * 小型模型配置
 * 适用于测试和快速原型开发
 */
const SMALL_CONFIG = {
  ...DEFAULT_CONFIG,
  vocab_size: 10000,
  max_seq_len: 128,
  n_embd: 256,
  n_heads: 4,
  n_layers: 4,
  ffn_hidden_dim: 1024,
  max_position_embeddings: 128,
};

/**
 * 大型模型配置
 * 适用于生产环境和高性能需求
 */
const LARGE_CONFIG = {
  ...DEFAULT_CONFIG,
  vocab_size: 50000,
  max_seq_len: 1024,
  n_embd: 1024,
  n_heads: 16,
  n_layers: 12,
  ffn_hidden_dim: 4096,
  max_position_embeddings: 1024,
};

/**
 * 微信小程序优化配置
 * 针对小程序环境的内存和性能限制进行优化
 */
const MINIPROGRAM_CONFIG = {
  ...DEFAULT_CONFIG,
  vocab_size: 8000,
  max_seq_len: 64,
  n_embd: 128,
  n_heads: 4,
  n_layers: 3,
  ffn_hidden_dim: 512,
  max_position_embeddings: 64,
  dropout: 0.05,
  attention_dropout: 0.05,
};

/**
 * 配置验证函数
 * 检查配置参数的有效性
 * 
 * @param {Object} config - 配置对象
 * @returns {Object} 验证后的配置对象
 * @throws {Error} 配置无效时抛出错误
 */
function validateConfig(config) {
  const validatedConfig = { ...config };
  
  // 检查必需参数
  const requiredParams = [
    'vocab_size', 'max_seq_len', 'n_embd', 'n_heads', 'n_layers'
  ];
  
  for (const param of requiredParams) {
    if (!(param in validatedConfig) || validatedConfig[param] <= 0) {
      throw new Error(`配置参数 ${param} 必须是正数`);
    }
  }
  
  // 检查嵌入维度是否能被注意力头数整除
  if (validatedConfig.n_embd % validatedConfig.n_heads !== 0) {
    throw new Error(`嵌入维度 (${validatedConfig.n_embd}) 必须能被注意力头数 (${validatedConfig.n_heads}) 整除`);
  }
  
  // 检查 dropout 概率范围
  if (validatedConfig.dropout < 0 || validatedConfig.dropout > 1) {
    throw new Error('Dropout 概率必须在 0 到 1 之间');
  }
  
  if (validatedConfig.attention_dropout < 0 || validatedConfig.attention_dropout > 1) {
    throw new Error('注意力 Dropout 概率必须在 0 到 1 之间');
  }
  
  // 检查激活函数类型
  const validActivations = ['relu', 'gelu'];
  if (!validActivations.includes(validatedConfig.activation)) {
    throw new Error(`激活函数必须是以下之一: ${validActivations.join(', ')}`);
  }
  
  // 设置默认值
  if (!validatedConfig.ffn_hidden_dim) {
    validatedConfig.ffn_hidden_dim = validatedConfig.n_embd * 4;
  }
  
  if (!validatedConfig.max_position_embeddings) {
    validatedConfig.max_position_embeddings = validatedConfig.max_seq_len;
  }
  
  return validatedConfig;
}

/**
 * 创建自定义配置
 * 基于基础配置创建自定义配置
 * 
 * @param {Object} baseConfig - 基础配置
 * @param {Object} overrides - 覆盖参数
 * @returns {Object} 合并后的配置
 */
function createConfig(baseConfig = DEFAULT_CONFIG, overrides = {}) {
  const mergedConfig = {
    ...baseConfig,
    ...overrides
  };
  
  return validateConfig(mergedConfig);
}

/**
 * 获取预定义配置
 * 
 * @param {string} configName - 配置名称: 'default', 'small', 'large', 'miniprogram'
 * @returns {Object} 配置对象
 */
function getConfig(configName = 'default') {
  const configs = {
    'default': DEFAULT_CONFIG,
    'small': SMALL_CONFIG,
    'large': LARGE_CONFIG,
    'miniprogram': MINIPROGRAM_CONFIG
  };
  
  if (!(configName in configs)) {
    throw new Error(`未知的配置名称: ${configName}. 可用配置: ${Object.keys(configs).join(', ')}`);
  }
  
  return validateConfig(configs[configName]);
}

/**
 * 打印配置信息
 * 
 * @param {Object} config - 配置对象
 */
function printConfig(config) {
  console.log('=== Transformer 模型配置 ===');
  console.log(`词汇表大小: ${config.vocab_size}`);
  console.log(`最大序列长度: ${config.max_seq_len}`);
  console.log(`嵌入维度: ${config.n_embd}`);
  console.log(`注意力头数: ${config.n_heads}`);
  console.log(`层数: ${config.n_layers}`);
  console.log(`前馈网络隐藏维度: ${config.ffn_hidden_dim}`);
  console.log(`Dropout: ${config.dropout}`);
  console.log(`激活函数: ${config.activation}`);
  console.log('========================');
}

/**
 * 估算模型参数量
 * 
 * @param {Object} config - 配置对象
 * @returns {Object} 参数量统计
 */
function estimateParameters(config) {
  const {
    vocab_size,
    n_embd,
    n_heads,
    n_layers,
    ffn_hidden_dim,
    max_position_embeddings
  } = config;
  
  // 嵌入层参数
  const tokenEmbedding = vocab_size * n_embd;
  const positionEmbedding = max_position_embeddings * n_embd;
  
  // 每个注意力层的参数
  const attentionParams = 4 * n_embd * n_embd; // Q, K, V, O 矩阵
  
  // 每个前馈网络的参数
  const ffnParams = 2 * n_embd * ffn_hidden_dim; // 两个线性层
  
  // 层归一化参数
  const layerNormParams = 2 * n_embd; // gamma 和 beta
  
  // 每层的总参数
  const perLayerParams = attentionParams + ffnParams + 2 * layerNormParams;
  
  // 总参数量
  const totalParams = tokenEmbedding + positionEmbedding + n_layers * perLayerParams;
  
  return {
    tokenEmbedding,
    positionEmbedding,
    attentionParams: attentionParams * n_layers,
    ffnParams: ffnParams * n_layers,
    layerNormParams: layerNormParams * n_layers * 2,
    totalParams,
    totalParamsM: (totalParams / 1e6).toFixed(2) + 'M'
  };
}

// 导出配置和函数
module.exports = {
  DEFAULT_CONFIG,
  SMALL_CONFIG,
  LARGE_CONFIG,
  MINIPROGRAM_CONFIG,
  validateConfig,
  createConfig,
  getConfig,
  printConfig,
  estimateParameters
};
