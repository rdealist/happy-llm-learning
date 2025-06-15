/**
 * 预训练语言模型配置
 * 定义各种模型的默认配置和工厂函数
 * 
 * @author shihom_wu
 * @version 1.0.0
 */

/**
 * BERT模型配置
 */
const BERT_CONFIGS = {
  'bert-base': {
    vocab_size: 30522,
    hidden_size: 768,
    num_hidden_layers: 12,
    num_attention_heads: 12,
    intermediate_size: 3072,
    max_position_embeddings: 512,
    type_vocab_size: 2,
    hidden_dropout_prob: 0.1,
    attention_probs_dropout_prob: 0.1,
    classifier_dropout: 0.1,
    tie_word_embeddings: false
  },
  'bert-large': {
    vocab_size: 30522,
    hidden_size: 1024,
    num_hidden_layers: 24,
    num_attention_heads: 16,
    intermediate_size: 4096,
    max_position_embeddings: 512,
    type_vocab_size: 2,
    hidden_dropout_prob: 0.1,
    attention_probs_dropout_prob: 0.1,
    classifier_dropout: 0.1,
    tie_word_embeddings: false
  },
  'bert-small': {
    vocab_size: 30522,
    hidden_size: 512,
    num_hidden_layers: 4,
    num_attention_heads: 8,
    intermediate_size: 2048,
    max_position_embeddings: 512,
    type_vocab_size: 2,
    hidden_dropout_prob: 0.1,
    attention_probs_dropout_prob: 0.1,
    classifier_dropout: 0.1,
    tie_word_embeddings: false
  }
};

/**
 * GPT模型配置
 */
const GPT_CONFIGS = {
  'gpt2': {
    vocab_size: 50257,
    n_embd: 768,
    n_layer: 12,
    n_head: 12,
    n_positions: 1024,
    n_inner: 3072,
    activation_function: 'gelu',
    dropout: 0.1,
    tie_word_embeddings: true,
    eos_token_id: 50256,
    bos_token_id: 50256,
    pad_token_id: 50256
  },
  'gpt2-medium': {
    vocab_size: 50257,
    n_embd: 1024,
    n_layer: 24,
    n_head: 16,
    n_positions: 1024,
    n_inner: 4096,
    activation_function: 'gelu',
    dropout: 0.1,
    tie_word_embeddings: true,
    eos_token_id: 50256,
    bos_token_id: 50256,
    pad_token_id: 50256
  },
  'gpt2-large': {
    vocab_size: 50257,
    n_embd: 1280,
    n_layer: 36,
    n_head: 20,
    n_positions: 1024,
    n_inner: 5120,
    activation_function: 'gelu',
    dropout: 0.1,
    tie_word_embeddings: true,
    eos_token_id: 50256,
    bos_token_id: 50256,
    pad_token_id: 50256
  },
  'gpt2-small': {
    vocab_size: 50257,
    n_embd: 512,
    n_layer: 6,
    n_head: 8,
    n_positions: 1024,
    n_inner: 2048,
    activation_function: 'gelu',
    dropout: 0.1,
    tie_word_embeddings: true,
    eos_token_id: 50256,
    bos_token_id: 50256,
    pad_token_id: 50256
  }
};

/**
 * T5模型配置
 */
const T5_CONFIGS = {
  't5-small': {
    vocab_size: 32128,
    d_model: 512,
    num_layers: 6,
    num_heads: 8,
    d_ff: 2048,
    max_length: 512,
    dropout_rate: 0.1,
    eos_token_id: 1,
    pad_token_id: 0,
    decoder_start_token_id: 0
  },
  't5-base': {
    vocab_size: 32128,
    d_model: 768,
    num_layers: 12,
    num_heads: 12,
    d_ff: 3072,
    max_length: 512,
    dropout_rate: 0.1,
    eos_token_id: 1,
    pad_token_id: 0,
    decoder_start_token_id: 0
  },
  't5-large': {
    vocab_size: 32128,
    d_model: 1024,
    num_layers: 24,
    num_heads: 16,
    d_ff: 4096,
    max_length: 512,
    dropout_rate: 0.1,
    eos_token_id: 1,
    pad_token_id: 0,
    decoder_start_token_id: 0
  }
};

/**
 * 模型工厂类
 * 提供创建各种预训练模型的统一接口
 */
class ModelFactory {
  /**
   * 创建BERT模型
   * 
   * @param {string} modelName - 模型名称
   * @param {string} taskType - 任务类型 ('base', 'classification', 'token_classification', 'masked_lm')
   * @param {Object} customConfig - 自定义配置
   * @returns {Object} BERT模型实例
   */
  static createBERT(modelName = 'bert-base', taskType = 'base', customConfig = {}) {
    const { 
      BERTModel, 
      BERTForSequenceClassification, 
      BERTForTokenClassification, 
      BERTForMaskedLM 
    } = require('./bert');

    const config = { ...BERT_CONFIGS[modelName], ...customConfig };

    switch (taskType) {
      case 'base':
        return new BERTModel(config);
      case 'classification':
        return new BERTForSequenceClassification(config);
      case 'token_classification':
        return new BERTForTokenClassification(config);
      case 'masked_lm':
        return new BERTForMaskedLM(config);
      default:
        throw new Error(`不支持的BERT任务类型: ${taskType}`);
    }
  }

  /**
   * 创建GPT模型
   * 
   * @param {string} modelName - 模型名称
   * @param {string} taskType - 任务类型 ('base', 'causal_lm')
   * @param {Object} customConfig - 自定义配置
   * @returns {Object} GPT模型实例
   */
  static createGPT(modelName = 'gpt2', taskType = 'causal_lm', customConfig = {}) {
    const { GPTModel, GPTForCausalLM } = require('./gpt');

    const config = { ...GPT_CONFIGS[modelName], ...customConfig };

    switch (taskType) {
      case 'base':
        return new GPTModel(config);
      case 'causal_lm':
        return new GPTForCausalLM(config);
      default:
        throw new Error(`不支持的GPT任务类型: ${taskType}`);
    }
  }

  /**
   * 创建T5模型
   * 
   * @param {string} modelName - 模型名称
   * @param {Object} customConfig - 自定义配置
   * @returns {Object} T5模型实例
   */
  static createT5(modelName = 't5-base', customConfig = {}) {
    const { T5Model } = require('./t5');

    const config = { ...T5_CONFIGS[modelName], ...customConfig };
    return new T5Model(config);
  }

  /**
   * 获取模型配置
   * 
   * @param {string} modelType - 模型类型 ('bert', 'gpt', 't5')
   * @param {string} modelName - 模型名称
   * @returns {Object} 模型配置
   */
  static getConfig(modelType, modelName) {
    switch (modelType.toLowerCase()) {
      case 'bert':
        return BERT_CONFIGS[modelName];
      case 'gpt':
        return GPT_CONFIGS[modelName];
      case 't5':
        return T5_CONFIGS[modelName];
      default:
        throw new Error(`不支持的模型类型: ${modelType}`);
    }
  }

  /**
   * 列出所有可用的模型
   * 
   * @returns {Object} 可用模型列表
   */
  static listAvailableModels() {
    return {
      bert: Object.keys(BERT_CONFIGS),
      gpt: Object.keys(GPT_CONFIGS),
      t5: Object.keys(T5_CONFIGS)
    };
  }

  /**
   * 创建自定义模型
   * 
   * @param {string} modelType - 模型类型
   * @param {Object} config - 完整配置
   * @param {string} taskType - 任务类型
   * @returns {Object} 模型实例
   */
  static createCustomModel(modelType, config, taskType = 'base') {
    switch (modelType.toLowerCase()) {
      case 'bert':
        return this.createBERT('bert-base', taskType, config);
      case 'gpt':
        return this.createGPT('gpt2', taskType, config);
      case 't5':
        return this.createT5('t5-base', config);
      default:
        throw new Error(`不支持的模型类型: ${modelType}`);
    }
  }
}

/**
 * 预训练任务配置
 */
const PRETRAINING_CONFIGS = {
  mlm: {
    mask_ratio: 0.15,
    mask_token_id: 103,
    cls_token_id: 101,
    sep_token_id: 102,
    pad_token_id: 0
  },
  nsp: {
    cls_token_id: 101,
    sep_token_id: 102,
    pad_token_id: 0
  },
  sop: {
    cls_token_id: 101,
    sep_token_id: 102,
    pad_token_id: 0
  },
  clm: {
    eos_token_id: 2,
    pad_token_id: 0
  }
};

// 导出配置和工厂类
module.exports = {
  BERT_CONFIGS,
  GPT_CONFIGS,
  T5_CONFIGS,
  PRETRAINING_CONFIGS,
  ModelFactory
};
