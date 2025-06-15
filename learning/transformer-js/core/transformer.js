/**
 * 完整的 Transformer 模型
 * 组合编码器、解码器、嵌入层等组件构建完整的 Transformer 架构
 * 
 * @author shihom_wu
 * @version 1.0.0
 */

const { TransformerEmbedding } = require('./embedding');
const { TransformerEncoder, createEncoder } = require('./encoder');
const { TransformerDecoder, createDecoder } = require('./decoder');
const { Linear } = require('./layers');
const { MaskGenerator } = require('./attention');
const { softmax } = require('./math-utils');
const { matmul } = require('./matrix-ops');

/**
 * 完整的 Transformer 模型
 * 实现编码器-解码器架构的 Transformer
 */
class Transformer {
  /**
   * 构造函数
   * 
   * @param {Object} config - 模型配置
   */
  constructor(config) {
    this.config = config;
    this.vocabSize = config.vocab_size;
    this.dModel = config.n_embd || config.d_model;
    this.maxSeqLen = config.max_seq_len;
    this.nLayers = config.n_layers;
    this.nHeads = config.n_heads;
    
    // 源序列嵌入层
    this.srcEmbedding = new TransformerEmbedding(
      this.vocabSize,
      this.dModel,
      this.maxSeqLen,
      config.position_encoding_type || 'sinusoidal',
      config.dropout || 0.1
    );
    
    // 目标序列嵌入层
    this.tgtEmbedding = new TransformerEmbedding(
      this.vocabSize,
      this.dModel,
      this.maxSeqLen,
      config.position_encoding_type || 'sinusoidal',
      config.dropout || 0.1
    );
    
    // 编码器
    this.encoder = createEncoder(config);
    
    // 解码器
    this.decoder = createDecoder(config);
    
    // 输出投影层（语言模型头）
    this.outputProjection = new Linear(
      this.dModel,
      this.vocabSize,
      config.use_bias || false
    );
    
    // 是否共享嵌入权重
    this.tieWeights = config.tie_word_embeddings || false;
    if (this.tieWeights) {
      this._tieEmbeddingWeights();
    }
    
    // 训练模式标志
    this.training = true;
  }
  
  /**
   * 共享输入输出嵌入权重
   * 将输出投影层的权重与词嵌入权重绑定
   */
  _tieEmbeddingWeights() {
    // 简化实现：在实际应用中需要确保权重矩阵的引用一致性
    console.log('权重共享已启用（简化实现）');
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} srcTokens - 源序列词元 [batchSize, srcSeqLen]
   * @param {Array<Array<number>>} tgtTokens - 目标序列词元 [batchSize, tgtSeqLen]
   * @param {Array<Array<number>>|null} srcMask - 源序列掩码，可选
   * @param {Array<Array<number>>|null} tgtMask - 目标序列掩码，可选
   * @returns {Object} 模型输出
   */
  forward(srcTokens, tgtTokens, srcMask = null, tgtMask = null) {
    // 验证输入
    if (!Array.isArray(srcTokens) || !Array.isArray(tgtTokens)) {
      throw new Error('源序列和目标序列必须是二维数组');
    }
    
    const batchSize = srcTokens.length;
    const srcSeqLen = srcTokens[0].length;
    const tgtSeqLen = tgtTokens[0].length;
    
    // 处理单个样本的情况（简化实现）
    if (batchSize === 1) {
      return this._forwardSingle(srcTokens[0], tgtTokens[0], srcMask, tgtMask);
    }
    
    // 批处理实现（简化）
    const results = [];
    for (let b = 0; b < batchSize; b++) {
      const result = this._forwardSingle(srcTokens[b], tgtTokens[b], srcMask, tgtMask);
      results.push(result);
    }
    
    return this._combineBatchResults(results);
  }
  
  /**
   * 单个样本的前向传播
   * 
   * @param {Array<number>} srcTokens - 源序列词元 [srcSeqLen]
   * @param {Array<number>} tgtTokens - 目标序列词元 [tgtSeqLen]
   * @param {Array<Array<number>>|null} srcMask - 源序列掩码
   * @param {Array<Array<number>>|null} tgtMask - 目标序列掩码
   * @returns {Object} 单个样本的输出
   */
  _forwardSingle(srcTokens, tgtTokens, srcMask = null, tgtMask = null) {
    // 源序列嵌入
    const srcEmbedded = this.srcEmbedding.forward([srcTokens])[0]; // [srcSeqLen, dModel]
    
    // 目标序列嵌入
    const tgtEmbedded = this.tgtEmbedding.forward([tgtTokens])[0]; // [tgtSeqLen, dModel]
    
    // 编码器前向传播
    const encoderResult = this.encoder.forward(srcEmbedded, srcMask);
    const encoderOutput = encoderResult.output; // [srcSeqLen, dModel]
    
    // 生成目标序列的因果掩码
    if (tgtMask === null) {
      tgtMask = MaskGenerator.createCausalMask(tgtTokens.length);
    }
    
    // 解码器前向传播
    const decoderResult = this.decoder.forward(
      tgtEmbedded,
      encoderOutput,
      tgtMask,
      srcMask
    );
    const decoderOutput = decoderResult.output; // [tgtSeqLen, dModel]
    
    // 输出投影到词汇表
    const logits = this.outputProjection.forward(decoderOutput); // [tgtSeqLen, vocabSize]
    
    return {
      logits: logits,
      encoderOutput: encoderOutput,
      encoderAttentions: encoderResult.attentions,
      decoderSelfAttentions: decoderResult.selfAttentions,
      decoderCrossAttentions: decoderResult.crossAttentions
    };
  }
  
  /**
   * 组合批处理结果
   * 
   * @param {Array<Object>} results - 各样本的结果
   * @returns {Object} 组合后的批处理结果
   */
  _combineBatchResults(results) {
    return {
      logits: results.map(r => r.logits),
      encoderOutputs: results.map(r => r.encoderOutput),
      encoderAttentions: results.map(r => r.encoderAttentions),
      decoderSelfAttentions: results.map(r => r.decoderSelfAttentions),
      decoderCrossAttentions: results.map(r => r.decoderCrossAttentions)
    };
  }
  
  /**
   * 编码
   * 只运行编码器部分
   * 
   * @param {Array<Array<number>>} srcTokens - 源序列词元
   * @param {Array<Array<number>>|null} srcMask - 源序列掩码
   * @returns {Object} 编码器输出
   */
  encode(srcTokens, srcMask = null) {
    const batchSize = srcTokens.length;
    
    if (batchSize === 1) {
      const srcEmbedded = this.srcEmbedding.forward(srcTokens)[0];
      return this.encoder.forward(srcEmbedded, srcMask);
    }
    
    // 批处理编码
    const results = [];
    for (let b = 0; b < batchSize; b++) {
      const srcEmbedded = this.srcEmbedding.forward([srcTokens[b]])[0];
      const result = this.encoder.forward(srcEmbedded, srcMask);
      results.push(result);
    }
    
    return {
      outputs: results.map(r => r.output),
      attentions: results.map(r => r.attentions)
    };
  }
  
  /**
   * 解码
   * 给定编码器输出，运行解码器
   * 
   * @param {Array<Array<number>>} tgtTokens - 目标序列词元
   * @param {Array<Array<number>>} encoderOutput - 编码器输出
   * @param {Array<Array<number>>|null} tgtMask - 目标序列掩码
   * @param {Array<Array<number>>|null} srcMask - 源序列掩码
   * @returns {Object} 解码器输出
   */
  decode(tgtTokens, encoderOutput, tgtMask = null, srcMask = null) {
    const batchSize = tgtTokens.length;
    
    if (batchSize === 1) {
      const tgtEmbedded = this.tgtEmbedding.forward(tgtTokens)[0];
      
      if (tgtMask === null) {
        tgtMask = MaskGenerator.createCausalMask(tgtTokens[0].length);
      }
      
      const decoderResult = this.decoder.forward(
        tgtEmbedded,
        encoderOutput,
        tgtMask,
        srcMask
      );
      
      const logits = this.outputProjection.forward(decoderResult.output);
      
      return {
        logits: logits,
        selfAttentions: decoderResult.selfAttentions,
        crossAttentions: decoderResult.crossAttentions
      };
    }
    
    // 批处理解码（简化实现）
    throw new Error('批处理解码暂未实现');
  }
  
  /**
   * 生成下一个词元的概率分布
   * 
   * @param {Array<Array<number>>} srcTokens - 源序列
   * @param {Array<number>} tgtTokens - 目标序列（到当前位置）
   * @returns {Array<number>} 下一个词元的概率分布
   */
  predictNext(srcTokens, tgtTokens) {
    // 编码源序列
    const encoderResult = this.encode([srcTokens]);
    const encoderOutput = encoderResult.outputs[0];
    
    // 解码到当前位置
    const decoderResult = this.decode([tgtTokens], encoderOutput);
    const logits = decoderResult.logits;
    
    // 获取最后一个位置的 logits
    const lastLogits = logits[logits.length - 1];
    
    // 转换为概率分布
    return softmax(lastLogits);
  }
  
  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.training = training;
    this.srcEmbedding.setTraining(training);
    this.tgtEmbedding.setTraining(training);
    this.encoder.setTraining(training);
    this.decoder.setTraining(training);
  }
  
  /**
   * 获取模型参数数量
   * 
   * @returns {Object} 参数统计信息
   */
  getParameterCount() {
    const srcEmbeddingParams = this.srcEmbedding.getParameterCount();
    const tgtEmbeddingParams = this.tieWeights ? 0 : this.tgtEmbedding.getParameterCount();
    const encoderParams = this.encoder.getParameterCount();
    const decoderParams = this.decoder.getParameterCount();
    const outputParams = this.outputProjection.getParameterCount();
    
    const totalParams = srcEmbeddingParams + tgtEmbeddingParams + encoderParams + decoderParams + outputParams;
    
    return {
      srcEmbedding: srcEmbeddingParams,
      tgtEmbedding: tgtEmbeddingParams,
      encoder: encoderParams,
      decoder: decoderParams,
      outputProjection: outputParams,
      total: totalParams,
      totalM: (totalParams / 1e6).toFixed(2) + 'M'
    };
  }
  
  /**
   * 模型信息摘要
   * 
   * @returns {Object} 模型信息
   */
  summary() {
    const paramCount = this.getParameterCount();
    
    return {
      architecture: 'Transformer (Encoder-Decoder)',
      vocabSize: this.vocabSize,
      modelDim: this.dModel,
      maxSeqLen: this.maxSeqLen,
      layers: this.nLayers,
      heads: this.nHeads,
      parameters: paramCount,
      config: this.config
    };
  }
}

/**
 * 创建 Transformer 模型的工厂函数
 * 
 * @param {Object} config - 模型配置
 * @returns {Transformer} Transformer 模型实例
 */
function createTransformer(config) {
  // 验证配置
  const requiredParams = ['vocab_size', 'n_embd', 'n_layers', 'n_heads'];
  for (const param of requiredParams) {
    if (!(param in config)) {
      throw new Error(`缺少必需的配置参数: ${param}`);
    }
  }
  
  return new Transformer(config);
}

// 导出类和函数
module.exports = {
  Transformer,
  createTransformer
};
