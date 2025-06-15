/**
 * T5模型实现
 * 基于Transformer Encoder-Decoder的文本到文本转换模型
 * 
 * @author shihom_wu
 * @version 1.0.0
 */

const { TransformerEmbedding } = require('../core/embedding');
const { TransformerEncoder, createEncoder } = require('../core/encoder');
const { TransformerDecoder, createDecoder } = require('../core/decoder');
const { RMSNorm } = require('../core/layers');
const { MaskGenerator } = require('../core/attention');

/**
 * T5模型
 * 实现完整的Encoder-Decoder架构，支持文本到文本的转换
 */
class T5Model {
  /**
   * 构造函数
   * 
   * @param {Object} config - 模型配置
   * @param {number} config.vocab_size - 词汇表大小
   * @param {number} config.d_model - 模型维度
   * @param {number} config.num_layers - 编码器和解码器层数
   * @param {number} config.num_heads - 注意力头数
   * @param {number} config.d_ff - 前馈网络维度
   * @param {number} config.max_length - 最大序列长度
   * @param {number} config.dropout_rate - dropout概率
   */
  constructor(config) {
    this.config = config;
    this.vocabSize = config.vocab_size;
    this.dModel = config.d_model;
    this.numLayers = config.num_layers;
    this.numHeads = config.num_heads;
    this.dFF = config.d_ff || config.d_model * 4;
    this.maxLength = config.max_length || 512;

    // 共享的词嵌入
    this.sharedEmbedding = new T5Embedding(config);

    // 编码器
    this.encoder = createEncoder({
      n_layers: this.numLayers,
      n_embd: this.dModel,
      n_heads: this.numHeads,
      ffn_hidden_dim: this.dFF,
      dropout: config.dropout_rate || 0.1,
      activation: 'relu',
      use_bias: false,
      use_rms_norm: true
    });

    // 解码器
    this.decoder = createDecoder({
      n_layers: this.numLayers,
      n_embd: this.dModel,
      n_heads: this.numHeads,
      ffn_hidden_dim: this.dFF,
      dropout: config.dropout_rate || 0.1,
      activation: 'relu',
      use_bias: false,
      use_rms_norm: true
    });

    // 最终的RMS归一化层
    this.finalLayerNorm = new RMSNorm(this.dModel);

    // 语言模型头（共享嵌入权重）
    this.lmHead = new T5LMHead(config);
  }

  /**
   * 前向传播
   * 
   * @param {Array<number>} inputIds - 编码器输入token序列
   * @param {Array<number>} decoderInputIds - 解码器输入token序列
   * @param {Array<number>|null} attentionMask - 编码器注意力掩码
   * @param {Array<number>|null} decoderAttentionMask - 解码器注意力掩码
   * @returns {Object} 模型输出
   */
  forward(inputIds, decoderInputIds, attentionMask = null, decoderAttentionMask = null) {
    // 编码器嵌入
    const encoderEmbeddings = this.sharedEmbedding.forward(inputIds);
    
    // 解码器嵌入
    const decoderEmbeddings = this.sharedEmbedding.forward(decoderInputIds);

    // 创建注意力掩码
    const encoderMask = attentionMask || this._createPaddingMask(inputIds);
    const decoderCausalMask = MaskGenerator.createCausalMask(decoderInputIds.length);
    const decoderMask = decoderAttentionMask || this._createPaddingMask(decoderInputIds);
    
    // 组合解码器掩码（因果掩码 + 填充掩码）
    const combinedDecoderMask = MaskGenerator.combineMasks([decoderCausalMask, decoderMask]);

    // 编码器前向传播
    const encoderOutput = this.encoder.forward(encoderEmbeddings, encoderMask);

    // 解码器前向传播
    const decoderOutput = this.decoder.forward(
      decoderEmbeddings,
      encoderOutput.output,
      combinedDecoderMask,
      encoderMask
    );

    // 最终层归一化
    const normalizedOutput = this.finalLayerNorm.forward(decoderOutput.output);

    // 语言模型头
    const logits = this.lmHead.forward(normalizedOutput, this.sharedEmbedding.weight);

    return {
      logits: logits,
      encoderLastHiddenState: encoderOutput.output,
      decoderLastHiddenState: normalizedOutput,
      encoderAttentions: encoderOutput.attentions,
      decoderSelfAttentions: decoderOutput.selfAttentions,
      decoderCrossAttentions: decoderOutput.crossAttentions
    };
  }

  /**
   * 生成文本
   * 
   * @param {Array<number>} inputIds - 输入token序列
   * @param {Object} generateConfig - 生成配置
   * @returns {Array<number>} 生成的token序列
   */
  generate(inputIds, generateConfig = {}) {
    const {
      maxLength = 50,
      temperature = 1.0,
      doSample = false,
      eosTokenId = 1
    } = generateConfig;

    // 编码输入
    const encoderEmbeddings = this.sharedEmbedding.forward(inputIds);
    const encoderMask = this._createPaddingMask(inputIds);
    const encoderOutput = this.encoder.forward(encoderEmbeddings, encoderMask);

    // 初始化解码器输入（通常以特殊token开始）
    const generatedIds = [0]; // 假设0是开始token

    for (let i = 0; i < maxLength; i++) {
      // 解码器前向传播
      const decoderEmbeddings = this.sharedEmbedding.forward(generatedIds);
      const decoderMask = MaskGenerator.createCausalMask(generatedIds.length);
      
      const decoderOutput = this.decoder.forward(
        decoderEmbeddings,
        encoderOutput.output,
        decoderMask,
        encoderMask
      );

      const normalizedOutput = this.finalLayerNorm.forward(decoderOutput.output);
      const logits = this.lmHead.forward(normalizedOutput, this.sharedEmbedding.weight);

      // 获取最后一个位置的logits
      const lastLogits = logits[logits.length - 1];

      // 选择下一个token
      const nextTokenId = doSample ? 
        this._sampleToken(lastLogits, temperature) :
        this._greedySelect(lastLogits);

      generatedIds.push(nextTokenId);

      // 如果生成了结束符，停止生成
      if (nextTokenId === eosTokenId) {
        break;
      }
    }

    return generatedIds.slice(1); // 移除开始token
  }

  /**
   * 创建填充掩码
   * 
   * @param {Array<number>} tokenIds - token序列
   * @param {number} padTokenId - 填充token ID
   * @returns {Array<Array<number>>} 填充掩码
   */
  _createPaddingMask(tokenIds, padTokenId = 0) {
    return MaskGenerator.createPaddingMask(tokenIds, padTokenId);
  }

  /**
   * 贪婪选择
   * 
   * @param {Array<number>} logits - logits数组
   * @returns {number} 选择的token ID
   */
  _greedySelect(logits) {
    let maxIndex = 0;
    let maxValue = logits[0];
    
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > maxValue) {
        maxValue = logits[i];
        maxIndex = i;
      }
    }
    
    return maxIndex;
  }

  /**
   * 采样token
   * 
   * @param {Array<number>} logits - logits数组
   * @param {number} temperature - 温度参数
   * @returns {number} 采样的token ID
   */
  _sampleToken(logits, temperature) {
    const { softmax } = require('../core/math-utils');
    const scaledLogits = logits.map(logit => logit / temperature);
    const probs = softmax(scaledLogits);
    
    const rand = Math.random();
    let cumSum = 0;
    
    for (let i = 0; i < probs.length; i++) {
      cumSum += probs[i];
      if (rand <= cumSum) {
        return i;
      }
    }
    
    return probs.length - 1;
  }

  /**
   * 获取参数数量
   * 
   * @returns {Object} 参数统计信息
   */
  getParameterCount() {
    const embeddingParams = this.sharedEmbedding.getParameterCount();
    const encoderParams = this.encoder.getParameterCount();
    const decoderParams = this.decoder.getParameterCount();
    const lnParams = this.finalLayerNorm.getParameterCount();
    const totalParams = embeddingParams + encoderParams + decoderParams + lnParams;

    return {
      embeddings: embeddingParams,
      encoder: encoderParams,
      decoder: decoderParams,
      layerNorm: lnParams,
      total: totalParams,
      totalM: (totalParams / 1e6).toFixed(2) + 'M'
    };
  }

  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.sharedEmbedding.setTraining(training);
    this.encoder.setTraining(training);
    this.decoder.setTraining(training);
    this.finalLayerNorm.setTraining(training);
    this.lmHead.setTraining(training);
  }
}

/**
 * T5嵌入层
 * 只包含词嵌入，不使用位置嵌入（T5使用相对位置编码）
 */
class T5Embedding {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置对象
   */
  constructor(config) {
    this.vocabSize = config.vocab_size;
    this.dModel = config.d_model;
    this.weight = this._initEmbedding(this.vocabSize, this.dModel);
  }

  /**
   * 初始化嵌入权重
   * 
   * @param {number} vocabSize - 词汇表大小
   * @param {number} dModel - 模型维度
   * @returns {Array<Array<number>>} 嵌入矩阵
   */
  _initEmbedding(vocabSize, dModel) {
    const embedding = [];
    const std = 1.0 / Math.sqrt(dModel);
    
    for (let i = 0; i < vocabSize; i++) {
      const row = [];
      for (let j = 0; j < dModel; j++) {
        row.push((Math.random() - 0.5) * 2 * std);
      }
      embedding.push(row);
    }
    return embedding;
  }

  /**
   * 前向传播
   * 
   * @param {Array<number>} inputIds - 输入token ID
   * @returns {Array<Array<number>>} 嵌入输出
   */
  forward(inputIds) {
    return inputIds.map(id => [...this.weight[id]]);
  }

  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return this.vocabSize * this.dModel;
  }

  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    // 嵌入层没有需要设置训练模式的组件
  }
}

/**
 * T5语言模型头
 * 使用共享的嵌入权重
 */
class T5LMHead {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置对象
   */
  constructor(config) {
    this.dModel = config.d_model;
    this.vocabSize = config.vocab_size;
  }

  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} hiddenStates - 隐藏状态
   * @param {Array<Array<number>>} embeddingWeights - 共享的嵌入权重
   * @returns {Array<Array<number>>} logits
   */
  forward(hiddenStates, embeddingWeights) {
    const results = [];
    
    for (let i = 0; i < hiddenStates.length; i++) {
      const logits = [];
      const hidden = hiddenStates[i];
      
      // 计算 hidden @ embedding_weights.T
      for (let j = 0; j < embeddingWeights.length; j++) {
        let score = 0;
        for (let k = 0; k < hidden.length; k++) {
          score += hidden[k] * embeddingWeights[j][k];
        }
        logits.push(score);
      }
      
      results.push(logits);
    }
    
    return results;
  }

  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数（共享权重，所以为0）
   */
  getParameterCount() {
    return 0;
  }

  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    // 语言模型头没有需要设置训练模式的组件
  }
}

// 导出T5模型和相关类
module.exports = {
  T5Model,
  T5Embedding,
  T5LMHead
};
