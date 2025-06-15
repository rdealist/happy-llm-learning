/**
 * 分类头模块
 * 实现各种下游任务的分类器和预测头
 * 
 * @author shihom_wu
 * @version 1.0.0
 */

const { Linear, Dropout } = require('./layers');
const { softmax } = require('./math-utils');

/**
 * 序列分类头
 * 用于文本分类、情感分析等任务
 */
class SequenceClassificationHead {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置对象
   * @param {number} config.hidden_size - 隐藏层大小
   * @param {number} config.num_labels - 标签数量
   * @param {number} config.dropout - Dropout概率，默认0.1
   */
  constructor(config) {
    this.hiddenSize = config.hidden_size;
    this.numLabels = config.num_labels;
    this.dropout = new Dropout(config.dropout || 0.1);
    this.classifier = new Linear(this.hiddenSize, this.numLabels);
  }

  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} hiddenStates - 隐藏状态 [seqLen, hiddenSize]
   * @param {number} poolingStrategy - 池化策略: 0=CLS token, 1=mean pooling, 2=max pooling
   * @returns {Array<number>} 分类logits [numLabels]
   */
  forward(hiddenStates, poolingStrategy = 0) {
    let pooledOutput;

    switch (poolingStrategy) {
      case 0: // CLS token pooling
        pooledOutput = hiddenStates[0]; // 使用第一个token ([CLS])
        break;
      case 1: // Mean pooling
        pooledOutput = this._meanPooling(hiddenStates);
        break;
      case 2: // Max pooling
        pooledOutput = this._maxPooling(hiddenStates);
        break;
      default:
        throw new Error(`不支持的池化策略: ${poolingStrategy}`);
    }

    // 应用dropout和分类器
    const droppedOutput = this.dropout.forward(pooledOutput);
    return this.classifier.forward(droppedOutput);
  }

  /**
   * 平均池化
   * 
   * @param {Array<Array<number>>} hiddenStates - 隐藏状态
   * @returns {Array<number>} 池化后的向量
   */
  _meanPooling(hiddenStates) {
    const seqLen = hiddenStates.length;
    const hiddenSize = hiddenStates[0].length;
    const pooled = new Array(hiddenSize).fill(0);

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < hiddenSize; j++) {
        pooled[j] += hiddenStates[i][j];
      }
    }

    return pooled.map(val => val / seqLen);
  }

  /**
   * 最大池化
   * 
   * @param {Array<Array<number>>} hiddenStates - 隐藏状态
   * @returns {Array<number>} 池化后的向量
   */
  _maxPooling(hiddenStates) {
    const hiddenSize = hiddenStates[0].length;
    const pooled = new Array(hiddenSize).fill(-Infinity);

    for (let i = 0; i < hiddenStates.length; i++) {
      for (let j = 0; j < hiddenSize; j++) {
        pooled[j] = Math.max(pooled[j], hiddenStates[i][j]);
      }
    }

    return pooled;
  }

  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return this.classifier.getParameterCount();
  }

  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.dropout.setTraining(training);
  }
}

/**
 * Token分类头
 * 用于命名实体识别、词性标注等token级别任务
 */
class TokenClassificationHead {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置对象
   * @param {number} config.hidden_size - 隐藏层大小
   * @param {number} config.num_labels - 标签数量
   * @param {number} config.dropout - Dropout概率，默认0.1
   */
  constructor(config) {
    this.hiddenSize = config.hidden_size;
    this.numLabels = config.num_labels;
    this.dropout = new Dropout(config.dropout || 0.1);
    this.classifier = new Linear(this.hiddenSize, this.numLabels);
  }

  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} hiddenStates - 隐藏状态 [seqLen, hiddenSize]
   * @returns {Array<Array<number>>} 每个token的分类logits [seqLen, numLabels]
   */
  forward(hiddenStates) {
    const results = [];

    for (let i = 0; i < hiddenStates.length; i++) {
      const droppedOutput = this.dropout.forward(hiddenStates[i]);
      const logits = this.classifier.forward(droppedOutput);
      results.push(logits);
    }

    return results;
  }

  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return this.classifier.getParameterCount();
  }

  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.dropout.setTraining(training);
  }
}

/**
 * 语言模型头
 * 用于MLM、CLM等语言建模任务
 */
class LanguageModelingHead {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置对象
   * @param {number} config.hidden_size - 隐藏层大小
   * @param {number} config.vocab_size - 词汇表大小
   * @param {boolean} config.tie_word_embeddings - 是否共享词嵌入权重
   */
  constructor(config) {
    this.hiddenSize = config.hidden_size;
    this.vocabSize = config.vocab_size;
    this.tieWordEmbeddings = config.tie_word_embeddings || false;
    
    if (!this.tieWordEmbeddings) {
      this.lmHead = new Linear(this.hiddenSize, this.vocabSize, false);
    }
  }

  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} hiddenStates - 隐藏状态 [seqLen, hiddenSize]
   * @param {Array<Array<number>>|null} embeddingWeights - 词嵌入权重（如果共享权重）
   * @returns {Array<Array<number>>} 词汇表上的logits [seqLen, vocabSize]
   */
  forward(hiddenStates, embeddingWeights = null) {
    if (this.tieWordEmbeddings && embeddingWeights) {
      // 使用共享的嵌入权重
      return this._computeWithSharedWeights(hiddenStates, embeddingWeights);
    } else {
      // 使用独立的语言模型头
      const results = [];
      for (let i = 0; i < hiddenStates.length; i++) {
        const logits = this.lmHead.forward(hiddenStates[i]);
        results.push(logits);
      }
      return results;
    }
  }

  /**
   * 使用共享权重计算logits
   * 
   * @param {Array<Array<number>>} hiddenStates - 隐藏状态
   * @param {Array<Array<number>>} embeddingWeights - 嵌入权重 [vocabSize, hiddenSize]
   * @returns {Array<Array<number>>} logits
   */
  _computeWithSharedWeights(hiddenStates, embeddingWeights) {
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
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return this.tieWordEmbeddings ? 0 : this.lmHead.getParameterCount();
  }

  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    if (!this.tieWordEmbeddings) {
      // 语言模型头没有dropout等需要设置训练模式的组件
    }
  }
}

// 导出所有分类头
module.exports = {
  SequenceClassificationHead,
  TokenClassificationHead,
  LanguageModelingHead
};
