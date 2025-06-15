/**
 * BERT模型实现
 * 基于Transformer Encoder的双向语言模型
 * 
 * @author shihom_wu
 * @version 1.0.0
 */

const { TransformerEmbedding } = require('../core/embedding');
const { TransformerEncoder, createEncoder } = require('../core/encoder');
const { SequenceClassificationHead, TokenClassificationHead, LanguageModelingHead } = require('../core/heads');
const { MaskGenerator } = require('../core/attention');

/**
 * BERT模型
 * 实现Encoder-only的双向语言模型
 */
class BERTModel {
  /**
   * 构造函数
   * 
   * @param {Object} config - 模型配置
   * @param {number} config.vocab_size - 词汇表大小
   * @param {number} config.hidden_size - 隐藏层大小
   * @param {number} config.num_hidden_layers - 编码器层数
   * @param {number} config.num_attention_heads - 注意力头数
   * @param {number} config.intermediate_size - 中间层大小
   * @param {number} config.max_position_embeddings - 最大位置编码长度
   * @param {number} config.type_vocab_size - token类型词汇表大小
   * @param {number} config.hidden_dropout_prob - 隐藏层dropout概率
   * @param {number} config.attention_probs_dropout_prob - 注意力dropout概率
   */
  constructor(config) {
    this.config = config;
    this.vocabSize = config.vocab_size;
    this.hiddenSize = config.hidden_size;
    this.numLayers = config.num_hidden_layers;
    this.numHeads = config.num_attention_heads;
    this.maxPositionEmbeddings = config.max_position_embeddings || 512;
    this.typeVocabSize = config.type_vocab_size || 2;

    // 嵌入层
    this.embeddings = new BERTEmbeddings(config);
    
    // 编码器
    this.encoder = createEncoder({
      n_layers: this.numLayers,
      n_embd: this.hiddenSize,
      n_heads: this.numHeads,
      ffn_hidden_dim: config.intermediate_size || this.hiddenSize * 4,
      dropout: config.hidden_dropout_prob || 0.1,
      activation: 'gelu',
      use_bias: true
    });

    // 池化层（用于获取句子级表示）
    this.pooler = new BERTPooler(config);
  }

  /**
   * 前向传播
   * 
   * @param {Array<number>} inputIds - 输入token ID序列
   * @param {Array<number>|null} tokenTypeIds - token类型ID序列
   * @param {Array<number>|null} attentionMask - 注意力掩码
   * @param {Array<number>|null} positionIds - 位置ID序列
   * @returns {Object} 模型输出
   */
  forward(inputIds, tokenTypeIds = null, attentionMask = null, positionIds = null) {
    const seqLength = inputIds.length;

    // 默认值处理
    if (tokenTypeIds === null) {
      tokenTypeIds = new Array(seqLength).fill(0);
    }
    if (positionIds === null) {
      positionIds = Array.from({length: seqLength}, (_, i) => i);
    }
    if (attentionMask === null) {
      attentionMask = new Array(seqLength).fill(1);
    }

    // 嵌入层
    const embeddingOutput = this.embeddings.forward(inputIds, tokenTypeIds, positionIds);

    // 创建扩展的注意力掩码
    const extendedAttentionMask = this._getExtendedAttentionMask(attentionMask);

    // 编码器
    const encoderOutput = this.encoder.forward(embeddingOutput, extendedAttentionMask);

    // 池化层
    const pooledOutput = this.pooler.forward(encoderOutput.output);

    return {
      lastHiddenState: encoderOutput.output,
      poolerOutput: pooledOutput,
      attentions: encoderOutput.attentions
    };
  }

  /**
   * 创建扩展的注意力掩码
   * 
   * @param {Array<number>} attentionMask - 原始注意力掩码
   * @returns {Array<Array<number>>} 扩展的注意力掩码
   */
  _getExtendedAttentionMask(attentionMask) {
    const seqLength = attentionMask.length;
    const extendedMask = [];

    for (let i = 0; i < seqLength; i++) {
      const row = [];
      for (let j = 0; j < seqLength; j++) {
        // BERT使用双向注意力，但需要考虑padding掩码
        row.push(attentionMask[j] === 1 ? 0 : -10000);
      }
      extendedMask.push(row);
    }

    return extendedMask;
  }

  /**
   * 获取参数数量
   * 
   * @returns {Object} 参数统计信息
   */
  getParameterCount() {
    const embeddingParams = this.embeddings.getParameterCount();
    const encoderParams = this.encoder.getParameterCount();
    const poolerParams = this.pooler.getParameterCount();
    const totalParams = embeddingParams + encoderParams + poolerParams;

    return {
      embeddings: embeddingParams,
      encoder: encoderParams,
      pooler: poolerParams,
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
    this.embeddings.setTraining(training);
    this.encoder.setTraining(training);
    this.pooler.setTraining(training);
  }
}

/**
 * BERT嵌入层
 * 包含词嵌入、位置嵌入和token类型嵌入
 */
class BERTEmbeddings {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置对象
   */
  constructor(config) {
    this.vocabSize = config.vocab_size;
    this.hiddenSize = config.hidden_size;
    this.maxPositionEmbeddings = config.max_position_embeddings || 512;
    this.typeVocabSize = config.type_vocab_size || 2;

    // 词嵌入
    this.wordEmbeddings = this._initEmbedding(this.vocabSize, this.hiddenSize);
    
    // 位置嵌入
    this.positionEmbeddings = this._initEmbedding(this.maxPositionEmbeddings, this.hiddenSize);
    
    // token类型嵌入
    this.tokenTypeEmbeddings = this._initEmbedding(this.typeVocabSize, this.hiddenSize);

    // 层归一化和dropout
    this.layerNorm = new (require('../core/layers').LayerNorm)(this.hiddenSize);
    this.dropout = new (require('../core/layers').Dropout)(config.hidden_dropout_prob || 0.1);
  }

  /**
   * 初始化嵌入矩阵
   * 
   * @param {number} vocabSize - 词汇表大小
   * @param {number} embeddingDim - 嵌入维度
   * @returns {Array<Array<number>>} 嵌入矩阵
   */
  _initEmbedding(vocabSize, embeddingDim) {
    const embedding = [];
    for (let i = 0; i < vocabSize; i++) {
      const row = [];
      for (let j = 0; j < embeddingDim; j++) {
        row.push((Math.random() - 0.5) * 0.02); // 小的随机初始化
      }
      embedding.push(row);
    }
    return embedding;
  }

  /**
   * 前向传播
   * 
   * @param {Array<number>} inputIds - 输入token ID
   * @param {Array<number>} tokenTypeIds - token类型ID
   * @param {Array<number>} positionIds - 位置ID
   * @returns {Array<Array<number>>} 嵌入输出
   */
  forward(inputIds, tokenTypeIds, positionIds) {
    const seqLength = inputIds.length;
    const embeddings = [];

    for (let i = 0; i < seqLength; i++) {
      const wordEmb = this.wordEmbeddings[inputIds[i]];
      const posEmb = this.positionEmbeddings[positionIds[i]];
      const typeEmb = this.tokenTypeEmbeddings[tokenTypeIds[i]];

      // 三种嵌入相加
      const combined = [];
      for (let j = 0; j < this.hiddenSize; j++) {
        combined.push(wordEmb[j] + posEmb[j] + typeEmb[j]);
      }
      embeddings.push(combined);
    }

    // 层归一化和dropout
    const normalized = this.layerNorm.forward(embeddings);
    return this.dropout.forward(normalized);
  }

  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return (this.vocabSize + this.maxPositionEmbeddings + this.typeVocabSize) * this.hiddenSize +
           this.layerNorm.getParameterCount();
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
 * BERT池化层
 * 用于获取句子级别的表示
 */
class BERTPooler {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置对象
   */
  constructor(config) {
    this.hiddenSize = config.hidden_size;
    this.dense = new (require('../core/layers').Linear)(this.hiddenSize, this.hiddenSize);
  }

  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} hiddenStates - 隐藏状态
   * @returns {Array<number>} 池化输出
   */
  forward(hiddenStates) {
    // 取第一个token ([CLS]) 的隐藏状态
    const firstTokenTensor = hiddenStates[0];
    
    // 通过线性层和tanh激活
    const pooledOutput = this.dense.forward(firstTokenTensor);
    return pooledOutput.map(x => Math.tanh(x));
  }

  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return this.dense.getParameterCount();
  }

  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    // 池化层没有需要设置训练模式的组件
  }
}

/**
 * BERT序列分类模型
 * 用于文本分类、情感分析等任务
 */
class BERTForSequenceClassification {
  /**
   * 构造函数
   *
   * @param {Object} config - 配置对象
   */
  constructor(config) {
    this.numLabels = config.num_labels || 2;
    this.bert = new BERTModel(config);
    this.classifier = new SequenceClassificationHead({
      hidden_size: config.hidden_size,
      num_labels: this.numLabels,
      dropout: config.classifier_dropout || 0.1
    });
  }

  /**
   * 前向传播
   *
   * @param {Array<number>} inputIds - 输入token ID序列
   * @param {Array<number>|null} tokenTypeIds - token类型ID序列
   * @param {Array<number>|null} attentionMask - 注意力掩码
   * @returns {Object} 模型输出
   */
  forward(inputIds, tokenTypeIds = null, attentionMask = null) {
    const bertOutput = this.bert.forward(inputIds, tokenTypeIds, attentionMask);
    const logits = this.classifier.forward([bertOutput.poolerOutput], 0); // 使用pooler输出

    return {
      logits: logits,
      hiddenStates: bertOutput.lastHiddenState,
      attentions: bertOutput.attentions
    };
  }

  /**
   * 设置训练模式
   *
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.bert.setTraining(training);
    this.classifier.setTraining(training);
  }
}

/**
 * BERT Token分类模型
 * 用于命名实体识别、词性标注等任务
 */
class BERTForTokenClassification {
  /**
   * 构造函数
   *
   * @param {Object} config - 配置对象
   */
  constructor(config) {
    this.numLabels = config.num_labels;
    this.bert = new BERTModel(config);
    this.classifier = new TokenClassificationHead({
      hidden_size: config.hidden_size,
      num_labels: this.numLabels,
      dropout: config.classifier_dropout || 0.1
    });
  }

  /**
   * 前向传播
   *
   * @param {Array<number>} inputIds - 输入token ID序列
   * @param {Array<number>|null} tokenTypeIds - token类型ID序列
   * @param {Array<number>|null} attentionMask - 注意力掩码
   * @returns {Object} 模型输出
   */
  forward(inputIds, tokenTypeIds = null, attentionMask = null) {
    const bertOutput = this.bert.forward(inputIds, tokenTypeIds, attentionMask);
    const logits = this.classifier.forward(bertOutput.lastHiddenState);

    return {
      logits: logits,
      hiddenStates: bertOutput.lastHiddenState,
      attentions: bertOutput.attentions
    };
  }

  /**
   * 设置训练模式
   *
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.bert.setTraining(training);
    this.classifier.setTraining(training);
  }
}

/**
 * BERT掩码语言模型
 * 用于MLM预训练任务
 */
class BERTForMaskedLM {
  /**
   * 构造函数
   *
   * @param {Object} config - 配置对象
   */
  constructor(config) {
    this.bert = new BERTModel(config);
    this.lmHead = new LanguageModelingHead({
      hidden_size: config.hidden_size,
      vocab_size: config.vocab_size,
      tie_word_embeddings: config.tie_word_embeddings || false
    });
  }

  /**
   * 前向传播
   *
   * @param {Array<number>} inputIds - 输入token ID序列
   * @param {Array<number>|null} tokenTypeIds - token类型ID序列
   * @param {Array<number>|null} attentionMask - 注意力掩码
   * @returns {Object} 模型输出
   */
  forward(inputIds, tokenTypeIds = null, attentionMask = null) {
    const bertOutput = this.bert.forward(inputIds, tokenTypeIds, attentionMask);

    // 获取词嵌入权重（如果共享权重）
    const embeddingWeights = this.lmHead.tieWordEmbeddings ?
      this.bert.embeddings.wordEmbeddings : null;

    const logits = this.lmHead.forward(bertOutput.lastHiddenState, embeddingWeights);

    return {
      logits: logits,
      hiddenStates: bertOutput.lastHiddenState,
      attentions: bertOutput.attentions
    };
  }

  /**
   * 设置训练模式
   *
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.bert.setTraining(training);
    this.lmHead.setTraining(training);
  }
}

// 导出BERT模型和相关类
module.exports = {
  BERTModel,
  BERTEmbeddings,
  BERTPooler,
  BERTForSequenceClassification,
  BERTForTokenClassification,
  BERTForMaskedLM
};
