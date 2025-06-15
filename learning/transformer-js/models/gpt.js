/**
 * GPT模型实现
 * 基于Transformer Decoder的自回归语言模型
 * 
 * @author shihom_wu
 * @version 1.0.0
 */

const { TransformerEmbedding } = require('../core/embedding');
const { TransformerDecoder, createDecoder } = require('../core/decoder');
const { LanguageModelingHead } = require('../core/heads');
const { MaskGenerator } = require('../core/attention');

/**
 * GPT模型
 * 实现Decoder-only的自回归语言模型
 */
class GPTModel {
  /**
   * 构造函数
   * 
   * @param {Object} config - 模型配置
   * @param {number} config.vocab_size - 词汇表大小
   * @param {number} config.n_embd - 嵌入维度
   * @param {number} config.n_layer - 解码器层数
   * @param {number} config.n_head - 注意力头数
   * @param {number} config.n_positions - 最大位置数
   * @param {number} config.dropout - dropout概率
   * @param {string} config.activation_function - 激活函数
   */
  constructor(config) {
    this.config = config;
    this.vocabSize = config.vocab_size;
    this.nEmbd = config.n_embd;
    this.nLayer = config.n_layer;
    this.nHead = config.n_head;
    this.nPositions = config.n_positions || 1024;

    // 词嵌入
    this.wte = new GPTEmbedding(config.vocab_size, config.n_embd);
    
    // 位置嵌入
    this.wpe = new GPTPositionalEmbedding(config.n_positions, config.n_embd);
    
    // Dropout
    this.dropout = new (require('../core/layers').Dropout)(config.dropout || 0.1);
    
    // 解码器层
    this.decoder = createDecoder({
      n_layers: this.nLayer,
      n_embd: this.nEmbd,
      n_heads: this.nHead,
      ffn_hidden_dim: config.n_inner || this.nEmbd * 4,
      dropout: config.dropout || 0.1,
      activation: config.activation_function || 'gelu',
      use_bias: true
    });

    // 最终层归一化
    this.lnF = new (require('../core/layers').LayerNorm)(this.nEmbd);
  }

  /**
   * 前向传播
   * 
   * @param {Array<number>} inputIds - 输入token ID序列
   * @param {Array<number>|null} positionIds - 位置ID序列
   * @param {Array<Array<number>>|null} attentionMask - 注意力掩码
   * @param {boolean} useCausalMask - 是否使用因果掩码
   * @returns {Object} 模型输出
   */
  forward(inputIds, positionIds = null, attentionMask = null, useCausalMask = true) {
    const seqLength = inputIds.length;

    // 默认位置ID
    if (positionIds === null) {
      positionIds = Array.from({length: seqLength}, (_, i) => i);
    }

    // 词嵌入和位置嵌入
    const tokenEmbeddings = this.wte.forward(inputIds);
    const positionEmbeddings = this.wpe.forward(positionIds);

    // 嵌入相加
    const hiddenStates = [];
    for (let i = 0; i < seqLength; i++) {
      const combined = [];
      for (let j = 0; j < this.nEmbd; j++) {
        combined.push(tokenEmbeddings[i][j] + positionEmbeddings[i][j]);
      }
      hiddenStates.push(combined);
    }

    // 应用dropout
    const droppedStates = this.dropout.forward(hiddenStates);

    // 创建因果掩码
    let mask = attentionMask;
    if (useCausalMask) {
      const causalMask = MaskGenerator.createCausalMask(seqLength);
      if (attentionMask) {
        mask = MaskGenerator.combineMasks([causalMask, attentionMask]);
      } else {
        mask = causalMask;
      }
    }

    // 通过解码器
    const decoderOutput = this.decoder.forward(droppedStates, null, mask);

    // 最终层归一化
    const normalizedOutput = this.lnF.forward(decoderOutput.output);

    return {
      lastHiddenState: normalizedOutput,
      hiddenStates: [droppedStates, ...decoderOutput.hiddenStates || []],
      attentions: decoderOutput.selfAttentions
    };
  }

  /**
   * 生成文本
   * 
   * @param {Array<number>} inputIds - 输入token序列
   * @param {number} maxLength - 最大生成长度
   * @param {number} temperature - 温度参数
   * @param {number} topK - top-k采样
   * @param {number} topP - top-p采样
   * @returns {Array<number>} 生成的token序列
   */
  generate(inputIds, maxLength = 50, temperature = 1.0, topK = 0, topP = 1.0) {
    const generatedIds = [...inputIds];
    
    for (let i = 0; i < maxLength; i++) {
      // 前向传播
      const output = this.forward(generatedIds);
      
      // 获取最后一个位置的logits
      const lastLogits = output.lastHiddenState[output.lastHiddenState.length - 1];
      
      // 应用温度
      const scaledLogits = lastLogits.map(logit => logit / temperature);
      
      // 采样下一个token
      const nextTokenId = this._sample(scaledLogits, topK, topP);
      
      generatedIds.push(nextTokenId);
      
      // 如果生成了结束符，停止生成
      if (nextTokenId === this.config.eos_token_id) {
        break;
      }
    }
    
    return generatedIds;
  }

  /**
   * 采样函数
   * 
   * @param {Array<number>} logits - logits数组
   * @param {number} topK - top-k参数
   * @param {number} topP - top-p参数
   * @returns {number} 采样的token ID
   */
  _sample(logits, topK, topP) {
    // 简化实现：使用贪婪采样
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
   * 获取参数数量
   * 
   * @returns {Object} 参数统计信息
   */
  getParameterCount() {
    const embeddingParams = this.wte.getParameterCount() + this.wpe.getParameterCount();
    const decoderParams = this.decoder.getParameterCount();
    const lnParams = this.lnF.getParameterCount();
    const totalParams = embeddingParams + decoderParams + lnParams;

    return {
      embeddings: embeddingParams,
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
    this.wte.setTraining(training);
    this.wpe.setTraining(training);
    this.dropout.setTraining(training);
    this.decoder.setTraining(training);
  }
}

/**
 * GPT词嵌入
 */
class GPTEmbedding {
  /**
   * 构造函数
   * 
   * @param {number} vocabSize - 词汇表大小
   * @param {number} embeddingDim - 嵌入维度
   */
  constructor(vocabSize, embeddingDim) {
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
    this.weight = this._initEmbedding(vocabSize, embeddingDim);
  }

  /**
   * 初始化嵌入权重
   * 
   * @param {number} vocabSize - 词汇表大小
   * @param {number} embeddingDim - 嵌入维度
   * @returns {Array<Array<number>>} 嵌入矩阵
   */
  _initEmbedding(vocabSize, embeddingDim) {
    const embedding = [];
    const std = 0.02;
    
    for (let i = 0; i < vocabSize; i++) {
      const row = [];
      for (let j = 0; j < embeddingDim; j++) {
        // 正态分布初始化
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
    return this.vocabSize * this.embeddingDim;
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
 * GPT位置嵌入
 */
class GPTPositionalEmbedding {
  /**
   * 构造函数
   * 
   * @param {number} maxPositions - 最大位置数
   * @param {number} embeddingDim - 嵌入维度
   */
  constructor(maxPositions, embeddingDim) {
    this.maxPositions = maxPositions;
    this.embeddingDim = embeddingDim;
    this.weight = this._initEmbedding(maxPositions, embeddingDim);
  }

  /**
   * 初始化位置嵌入权重
   * 
   * @param {number} maxPositions - 最大位置数
   * @param {number} embeddingDim - 嵌入维度
   * @returns {Array<Array<number>>} 位置嵌入矩阵
   */
  _initEmbedding(maxPositions, embeddingDim) {
    const embedding = [];
    const std = 0.01;
    
    for (let i = 0; i < maxPositions; i++) {
      const row = [];
      for (let j = 0; j < embeddingDim; j++) {
        row.push((Math.random() - 0.5) * 2 * std);
      }
      embedding.push(row);
    }
    return embedding;
  }

  /**
   * 前向传播
   * 
   * @param {Array<number>} positionIds - 位置ID
   * @returns {Array<Array<number>>} 位置嵌入输出
   */
  forward(positionIds) {
    return positionIds.map(id => [...this.weight[id]]);
  }

  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return this.maxPositions * this.embeddingDim;
  }

  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    // 位置嵌入层没有需要设置训练模式的组件
  }
}

/**
 * GPT语言建模模型
 * 用于因果语言建模任务
 */
class GPTForCausalLM {
  /**
   * 构造函数
   *
   * @param {Object} config - 配置对象
   */
  constructor(config) {
    this.gpt = new GPTModel(config);
    this.lmHead = new LanguageModelingHead({
      hidden_size: config.n_embd,
      vocab_size: config.vocab_size,
      tie_word_embeddings: config.tie_word_embeddings || true
    });
  }

  /**
   * 前向传播
   *
   * @param {Array<number>} inputIds - 输入token ID序列
   * @param {Array<number>|null} positionIds - 位置ID序列
   * @param {Array<Array<number>>|null} attentionMask - 注意力掩码
   * @returns {Object} 模型输出
   */
  forward(inputIds, positionIds = null, attentionMask = null) {
    const gptOutput = this.gpt.forward(inputIds, positionIds, attentionMask);

    // 获取词嵌入权重（如果共享权重）
    const embeddingWeights = this.lmHead.tieWordEmbeddings ?
      this.gpt.wte.weight : null;

    const logits = this.lmHead.forward(gptOutput.lastHiddenState, embeddingWeights);

    return {
      logits: logits,
      hiddenStates: gptOutput.hiddenStates,
      attentions: gptOutput.attentions
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
      topK = 0,
      topP = 1.0,
      doSample = false
    } = generateConfig;

    const generatedIds = [...inputIds];

    for (let i = 0; i < maxLength; i++) {
      // 前向传播
      const output = this.forward(generatedIds);

      // 获取最后一个位置的logits
      const lastLogits = output.logits[output.logits.length - 1];

      // 采样或贪婪选择下一个token
      const nextTokenId = doSample ?
        this._sampleToken(lastLogits, temperature, topK, topP) :
        this._greedySelect(lastLogits);

      generatedIds.push(nextTokenId);

      // 如果生成了结束符，停止生成
      if (nextTokenId === this.gpt.config.eos_token_id) {
        break;
      }
    }

    return generatedIds;
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
   * @param {number} topK - top-k参数
   * @param {number} topP - top-p参数
   * @returns {number} 采样的token ID
   */
  _sampleToken(logits, temperature, topK, topP) {
    // 应用温度
    const scaledLogits = logits.map(logit => logit / temperature);

    // 计算概率分布
    const { softmax } = require('../core/math-utils');
    const probs = softmax(scaledLogits);

    // 简化实现：随机采样
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
   * 设置训练模式
   *
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.gpt.setTraining(training);
    this.lmHead.setTraining(training);
  }

  /**
   * 获取参数数量
   *
   * @returns {Object} 参数统计信息
   */
  getParameterCount() {
    const gptParams = this.gpt.getParameterCount();
    const lmHeadParams = this.lmHead.getParameterCount();

    return {
      gpt: gptParams.total,
      lmHead: lmHeadParams,
      total: gptParams.total + lmHeadParams,
      totalM: ((gptParams.total + lmHeadParams) / 1e6).toFixed(2) + 'M'
    };
  }
}

// 导出GPT模型和相关类
module.exports = {
  GPTModel,
  GPTEmbedding,
  GPTPositionalEmbedding,
  GPTForCausalLM
};
