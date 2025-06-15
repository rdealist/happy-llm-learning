/**
 * 解码器模块
 * 实现 Transformer 解码器层和解码器块
 * 
 * @author Transformer-JS
 * @version 1.0.0
 */

const { MultiHeadAttention, MaskGenerator } = require('./attention');
const { LayerNorm, MLP } = require('./layers');
const { add } = require('./matrix-ops');

/**
 * 解码器层
 * 实现单个 Transformer 解码器层
 * 包含掩码自注意力、交叉注意力和前馈网络，以及残差连接和层归一化
 */
class DecoderLayer {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置对象
   * @param {number} config.d_model - 模型维度
   * @param {number} config.n_heads - 注意力头数
   * @param {number} config.d_ff - 前馈网络隐藏层维度
   * @param {number} config.dropout - Dropout 概率
   * @param {string} config.activation - 激活函数类型
   * @param {boolean} config.use_bias - 是否使用偏置
   */
  constructor(config) {
    this.dModel = config.d_model || config.n_embd;
    this.nHeads = config.n_heads;
    this.dFF = config.d_ff || config.ffn_hidden_dim;
    this.dropout = config.dropout || 0.1;
    this.activation = config.activation || 'relu';
    this.useBias = config.use_bias || false;
    
    // 掩码自注意力层
    this.maskedSelfAttention = new MultiHeadAttention(
      this.dModel,
      this.nHeads,
      this.dropout,
      this.useBias
    );
    
    // 交叉注意力层（编码器-解码器注意力）
    this.crossAttention = new MultiHeadAttention(
      this.dModel,
      this.nHeads,
      this.dropout,
      this.useBias
    );
    
    // 前馈网络
    this.feedForward = new MLP(
      this.dModel,
      this.dFF,
      this.dModel,
      this.activation,
      this.dropout,
      this.useBias
    );
    
    // 层归一化
    this.norm1 = new LayerNorm(this.dModel);
    this.norm2 = new LayerNorm(this.dModel);
    this.norm3 = new LayerNorm(this.dModel);
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} x - 解码器输入 [tgtSeqLen, dModel]
   * @param {Array<Array<number>>} encoderOutput - 编码器输出 [srcSeqLen, dModel]
   * @param {Array<Array<number>>|null} tgtMask - 目标序列掩码（因果掩码）
   * @param {Array<Array<number>>|null} srcMask - 源序列掩码，可选
   * @returns {Object} {output: 解码器层输出, selfAttention: 自注意力权重, crossAttention: 交叉注意力权重}
   */
  forward(x, encoderOutput, tgtMask = null, srcMask = null) {
    if (!Array.isArray(x) || x.length === 0) {
      throw new Error('解码器输入必须是非空二维数组');
    }
    
    if (!Array.isArray(encoderOutput) || encoderOutput.length === 0) {
      throw new Error('编码器输出必须是非空二维数组');
    }
    
    // 第一个子层：掩码自注意力 + 残差连接 + 层归一化
    const norm1Output = this.norm1.forward(x);
    const selfAttentionResult = this.maskedSelfAttention.forward(
      norm1Output, 
      norm1Output, 
      norm1Output, 
      tgtMask
    );
    const afterSelfAttention = add(x, selfAttentionResult.output);
    
    // 第二个子层：交叉注意力 + 残差连接 + 层归一化
    const norm2Output = this.norm2.forward(afterSelfAttention);
    const crossAttentionResult = this.crossAttention.forward(
      norm2Output,      // Query 来自解码器
      encoderOutput,    // Key 来自编码器
      encoderOutput,    // Value 来自编码器
      srcMask
    );
    const afterCrossAttention = add(afterSelfAttention, crossAttentionResult.output);
    
    // 第三个子层：前馈网络 + 残差连接 + 层归一化
    const norm3Output = this.norm3.forward(afterCrossAttention);
    const ffnOutput = this.feedForward.forward(norm3Output);
    const output = add(afterCrossAttention, ffnOutput);
    
    return {
      output: output,
      selfAttention: selfAttentionResult.attention,
      crossAttention: crossAttentionResult.attention
    };
  }
  
  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.maskedSelfAttention.setTraining(training);
    this.crossAttention.setTraining(training);
    this.feedForward.setTraining(training);
  }
  
  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return (
      this.maskedSelfAttention.getParameterCount() +
      this.crossAttention.getParameterCount() +
      this.feedForward.getParameterCount() +
      this.norm1.getParameterCount() +
      this.norm2.getParameterCount() +
      this.norm3.getParameterCount()
    );
  }
}

/**
 * Transformer 解码器
 * 由多个解码器层堆叠而成
 */
class TransformerDecoder {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置对象
   * @param {number} config.n_layers - 解码器层数
   * @param {number} config.d_model - 模型维度
   * @param {number} config.n_heads - 注意力头数
   * @param {number} config.d_ff - 前馈网络隐藏层维度
   * @param {number} config.dropout - Dropout 概率
   * @param {string} config.activation - 激活函数类型
   * @param {boolean} config.use_bias - 是否使用偏置
   */
  constructor(config) {
    this.nLayers = config.n_layers;
    this.dModel = config.d_model || config.n_embd;
    
    // 创建多个解码器层
    this.layers = [];
    for (let i = 0; i < this.nLayers; i++) {
      this.layers.push(new DecoderLayer(config));
    }
    
    // 最终的层归一化
    this.finalNorm = new LayerNorm(this.dModel);
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} x - 解码器输入 [tgtSeqLen, dModel]
   * @param {Array<Array<number>>} encoderOutput - 编码器输出 [srcSeqLen, dModel]
   * @param {Array<Array<number>>|null} tgtMask - 目标序列掩码（因果掩码）
   * @param {Array<Array<number>>|null} srcMask - 源序列掩码，可选
   * @returns {Object} {output: 解码器输出, selfAttentions: 各层自注意力权重, crossAttentions: 各层交叉注意力权重}
   */
  forward(x, encoderOutput, tgtMask = null, srcMask = null) {
    if (!Array.isArray(x) || x.length === 0) {
      throw new Error('解码器输入必须是非空二维数组');
    }
    
    // 如果没有提供目标掩码，自动生成因果掩码
    if (tgtMask === null) {
      tgtMask = MaskGenerator.createCausalMask(x.length);
    }
    
    let output = x;
    const selfAttentions = [];
    const crossAttentions = [];
    
    // 逐层前向传播
    for (let i = 0; i < this.nLayers; i++) {
      const layerResult = this.layers[i].forward(output, encoderOutput, tgtMask, srcMask);
      output = layerResult.output;
      selfAttentions.push(layerResult.selfAttention);
      crossAttentions.push(layerResult.crossAttention);
    }
    
    // 最终层归一化
    output = this.finalNorm.forward(output);
    
    return {
      output: output,
      selfAttentions: selfAttentions,
      crossAttentions: crossAttentions
    };
  }
  
  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    for (const layer of this.layers) {
      layer.setTraining(training);
    }
  }
  
  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    let totalParams = 0;
    
    // 各解码器层的参数
    for (const layer of this.layers) {
      totalParams += layer.getParameterCount();
    }
    
    // 最终层归一化的参数
    totalParams += this.finalNorm.getParameterCount();
    
    return totalParams;
  }
  
  /**
   * 增量解码
   * 用于自回归生成，只计算新添加的位置
   * 
   * @param {Array<Array<number>>} x - 当前解码器输入
   * @param {Array<Array<number>>} encoderOutput - 编码器输出
   * @param {Object|null} cache - 缓存的键值对，用于加速
   * @param {Array<Array<number>>|null} srcMask - 源序列掩码
   * @returns {Object} {output: 解码器输出, cache: 更新后的缓存}
   */
  incrementalForward(x, encoderOutput, cache = null, srcMask = null) {
    // 简化实现：直接调用完整的前向传播
    // 在实际应用中，可以实现键值缓存来加速推理
    const seqLen = x.length;
    const tgtMask = MaskGenerator.createCausalMask(seqLen);
    
    const result = this.forward(x, encoderOutput, tgtMask, srcMask);
    
    return {
      output: result.output,
      cache: null // 简化实现，不使用缓存
    };
  }
}

/**
 * 解码器工厂函数
 * 根据配置创建解码器实例
 * 
 * @param {Object} config - 配置对象
 * @returns {TransformerDecoder} 解码器实例
 */
function createDecoder(config) {
  // 验证必需的配置参数
  const requiredParams = ['n_layers', 'n_heads'];
  for (const param of requiredParams) {
    if (!(param in config)) {
      throw new Error(`缺少必需的配置参数: ${param}`);
    }
  }
  
  // 设置默认值
  const defaultConfig = {
    d_model: config.n_embd || 512,
    d_ff: config.ffn_hidden_dim || (config.n_embd || 512) * 4,
    dropout: 0.1,
    activation: 'relu',
    use_bias: false
  };
  
  const mergedConfig = { ...defaultConfig, ...config };
  
  return new TransformerDecoder(mergedConfig);
}

// 导出类和函数
module.exports = {
  DecoderLayer,
  TransformerDecoder,
  createDecoder
};
