/**
 * 编码器模块
 * 实现 Transformer 编码器层和编码器块
 * 
 * @author Transformer-JS
 * @version 1.0.0
 */

const { MultiHeadAttention } = require('./attention');
const { LayerNorm, MLP, ResidualConnection } = require('./layers');
const { add } = require('./matrix-ops');

/**
 * 编码器层
 * 实现单个 Transformer 编码器层
 * 包含多头自注意力和前馈网络，以及残差连接和层归一化
 */
class EncoderLayer {
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
    
    // 多头自注意力层
    this.selfAttention = new MultiHeadAttention(
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
    
    // 残差连接
    this.residual1 = new ResidualConnection(this.dModel, this.dropout);
    this.residual2 = new ResidualConnection(this.dModel, this.dropout);
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} x - 输入矩阵 [seqLen, dModel]
   * @param {Array<Array<number>>|null} mask - 注意力掩码，可选
   * @returns {Object} {output: 编码器层输出, attention: 注意力权重}
   */
  forward(x, mask = null) {
    if (!Array.isArray(x) || x.length === 0) {
      throw new Error('输入必须是非空二维数组');
    }
    
    // 第一个子层：多头自注意力 + 残差连接 + 层归一化
    // Pre-LN 结构: x + attention(norm(x))
    const norm1Output = this.norm1.forward(x);
    const attentionResult = this.selfAttention.forward(norm1Output, norm1Output, norm1Output, mask);
    const afterAttention = add(x, attentionResult.output);
    
    // 第二个子层：前馈网络 + 残差连接 + 层归一化
    // Pre-LN 结构: x + ffn(norm(x))
    const norm2Output = this.norm2.forward(afterAttention);
    const ffnOutput = this.feedForward.forward(norm2Output);
    const output = add(afterAttention, ffnOutput);
    
    return {
      output: output,
      attention: attentionResult.attention
    };
  }
  
  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.selfAttention.setTraining(training);
    this.feedForward.setTraining(training);
    this.residual1.setTraining(training);
    this.residual2.setTraining(training);
  }
  
  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return (
      this.selfAttention.getParameterCount() +
      this.feedForward.getParameterCount() +
      this.norm1.getParameterCount() +
      this.norm2.getParameterCount()
    );
  }
}

/**
 * Transformer 编码器
 * 由多个编码器层堆叠而成
 */
class TransformerEncoder {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置对象
   * @param {number} config.n_layers - 编码器层数
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
    
    // 创建多个编码器层
    this.layers = [];
    for (let i = 0; i < this.nLayers; i++) {
      this.layers.push(new EncoderLayer(config));
    }
    
    // 最终的层归一化
    this.finalNorm = new LayerNorm(this.dModel);
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} x - 输入矩阵 [seqLen, dModel]
   * @param {Array<Array<number>>|null} mask - 注意力掩码，可选
   * @returns {Object} {output: 编码器输出, attentions: 各层注意力权重}
   */
  forward(x, mask = null) {
    if (!Array.isArray(x) || x.length === 0) {
      throw new Error('输入必须是非空二维数组');
    }
    
    let output = x;
    const attentions = [];
    
    // 逐层前向传播
    for (let i = 0; i < this.nLayers; i++) {
      const layerResult = this.layers[i].forward(output, mask);
      output = layerResult.output;
      attentions.push(layerResult.attention);
    }
    
    // 最终层归一化
    output = this.finalNorm.forward(output);
    
    return {
      output: output,
      attentions: attentions
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
    
    // 各编码器层的参数
    for (const layer of this.layers) {
      totalParams += layer.getParameterCount();
    }
    
    // 最终层归一化的参数
    totalParams += this.finalNorm.getParameterCount();
    
    return totalParams;
  }
  
  /**
   * 获取指定层的输出
   * 用于分析中间层的表示
   * 
   * @param {Array<Array<number>>} x - 输入矩阵
   * @param {number} layerIndex - 层索引（0-based）
   * @param {Array<Array<number>>|null} mask - 注意力掩码，可选
   * @returns {Object} 指定层的输出和注意力权重
   */
  getLayerOutput(x, layerIndex, mask = null) {
    if (layerIndex < 0 || layerIndex >= this.nLayers) {
      throw new Error(`层索引 ${layerIndex} 超出范围 [0, ${this.nLayers - 1}]`);
    }
    
    let output = x;
    
    // 前向传播到指定层
    for (let i = 0; i <= layerIndex; i++) {
      const layerResult = this.layers[i].forward(output, mask);
      output = layerResult.output;
      
      if (i === layerIndex) {
        return {
          output: output,
          attention: layerResult.attention
        };
      }
    }
  }
  
  /**
   * 获取所有层的输出
   * 用于分析模型的层级表示
   * 
   * @param {Array<Array<number>>} x - 输入矩阵
   * @param {Array<Array<number>>|null} mask - 注意力掩码，可选
   * @returns {Array<Object>} 各层的输出和注意力权重
   */
  getAllLayerOutputs(x, mask = null) {
    let output = x;
    const layerOutputs = [];
    
    for (let i = 0; i < this.nLayers; i++) {
      const layerResult = this.layers[i].forward(output, mask);
      output = layerResult.output;
      
      layerOutputs.push({
        layerIndex: i,
        output: output,
        attention: layerResult.attention
      });
    }
    
    // 添加最终层归一化后的输出
    const finalOutput = this.finalNorm.forward(output);
    layerOutputs.push({
      layerIndex: 'final',
      output: finalOutput,
      attention: null
    });
    
    return layerOutputs;
  }
}

/**
 * 编码器工厂函数
 * 根据配置创建编码器实例
 * 
 * @param {Object} config - 配置对象
 * @returns {TransformerEncoder} 编码器实例
 */
function createEncoder(config) {
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
  
  return new TransformerEncoder(mergedConfig);
}

// 导出类和函数
module.exports = {
  EncoderLayer,
  TransformerEncoder,
  createEncoder
};
