/**
 * 预训练任务模块
 * 实现各种预训练语言模型的训练任务
 * 
 * @author shihom_wu
 * @version 1.0.0
 */

const { MaskGenerator } = require('../core/attention');
const { softmax } = require('../core/math-utils');

/**
 * 掩码语言模型任务 (MLM)
 * 用于BERT等双向语言模型的预训练
 */
class MLMTask {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置对象
   * @param {number} config.vocab_size - 词汇表大小
   * @param {number} config.mask_ratio - 掩码比例，默认0.15
   * @param {number} config.mask_token_id - [MASK] token的ID
   * @param {number} config.cls_token_id - [CLS] token的ID
   * @param {number} config.sep_token_id - [SEP] token的ID
   * @param {number} config.pad_token_id - [PAD] token的ID
   */
  constructor(config) {
    this.vocabSize = config.vocab_size || 30000;
    this.maskRatio = config.mask_ratio || 0.15;
    this.maskTokenId = config.mask_token_id || 103;
    this.clsTokenId = config.cls_token_id || 101;
    this.sepTokenId = config.sep_token_id || 102;
    this.padTokenId = config.pad_token_id || 0;
  }

  /**
   * 准备MLM训练数据
   * 
   * @param {Array<number>} tokenIds - 输入token序列
   * @returns {Object} 处理后的训练数据
   */
  prepareData(tokenIds) {
    // 过滤掉特殊token，不对其进行掩码
    const specialTokens = new Set([
      this.clsTokenId, 
      this.sepTokenId, 
      this.padTokenId
    ]);

    const candidatePositions = [];
    for (let i = 0; i < tokenIds.length; i++) {
      if (!specialTokens.has(tokenIds[i])) {
        candidatePositions.push(i);
      }
    }

    // 从候选位置中选择要掩码的位置
    const numMask = Math.floor(candidatePositions.length * this.maskRatio);
    const maskResult = MaskGenerator.createMLMMask(
      tokenIds, 
      this.maskRatio, 
      this.maskTokenId, 
      this.vocabSize
    );

    return {
      input_ids: maskResult.maskedTokens,
      labels: maskResult.labels,
      mask_positions: maskResult.maskPositions,
      attention_mask: this._createAttentionMask(tokenIds)
    };
  }

  /**
   * 创建注意力掩码
   * 
   * @param {Array<number>} tokenIds - token序列
   * @returns {Array<number>} 注意力掩码
   */
  _createAttentionMask(tokenIds) {
    return tokenIds.map(tokenId => tokenId !== this.padTokenId ? 1 : 0);
  }

  /**
   * 计算MLM损失
   * 
   * @param {Array<Array<number>>} predictions - 模型预测结果 [seqLen, vocabSize]
   * @param {Array<number>} labels - 真实标签
   * @returns {number} 损失值
   */
  computeLoss(predictions, labels) {
    let totalLoss = 0;
    let validPositions = 0;

    for (let i = 0; i < labels.length; i++) {
      if (labels[i] !== -100) { // 只计算被掩码位置的损失
        const probs = softmax(predictions[i]);
        const targetProb = probs[labels[i]];
        totalLoss += -Math.log(Math.max(targetProb, 1e-8)); // 避免log(0)
        validPositions++;
      }
    }

    return validPositions > 0 ? totalLoss / validPositions : 0;
  }
}

/**
 * 下一句预测任务 (NSP)
 * 用于BERT的句子级别理解预训练
 */
class NSPTask {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置对象
   */
  constructor(config) {
    this.clsTokenId = config.cls_token_id || 101;
    this.sepTokenId = config.sep_token_id || 102;
    this.padTokenId = config.pad_token_id || 0;
  }

  /**
   * 准备NSP训练数据
   * 
   * @param {Array<number>} sentenceA - 第一个句子的token序列
   * @param {Array<number>} sentenceB - 第二个句子的token序列
   * @param {boolean} isNext - 是否为连续句子
   * @returns {Object} 处理后的训练数据
   */
  prepareData(sentenceA, sentenceB, isNext) {
    // 构建输入序列: [CLS] sentenceA [SEP] sentenceB [SEP]
    const inputIds = [
      this.clsTokenId,
      ...sentenceA,
      this.sepTokenId,
      ...sentenceB,
      this.sepTokenId
    ];

    // 创建token类型ID (0表示第一个句子，1表示第二个句子)
    const tokenTypeIds = [
      0, // [CLS]
      ...new Array(sentenceA.length).fill(0),
      0, // [SEP]
      ...new Array(sentenceB.length).fill(1),
      1  // [SEP]
    ];

    return {
      input_ids: inputIds,
      token_type_ids: tokenTypeIds,
      attention_mask: this._createAttentionMask(inputIds),
      next_sentence_label: isNext ? 1 : 0
    };
  }

  /**
   * 创建注意力掩码
   * 
   * @param {Array<number>} tokenIds - token序列
   * @returns {Array<number>} 注意力掩码
   */
  _createAttentionMask(tokenIds) {
    return tokenIds.map(tokenId => tokenId !== this.padTokenId ? 1 : 0);
  }

  /**
   * 计算NSP损失
   * 
   * @param {Array<number>} predictions - 模型预测结果 [2] (是/否的概率)
   * @param {number} label - 真实标签 (0或1)
   * @returns {number} 损失值
   */
  computeLoss(predictions, label) {
    const probs = softmax(predictions);
    const targetProb = probs[label];
    return -Math.log(Math.max(targetProb, 1e-8));
  }
}

/**
 * 因果语言模型任务 (CLM)
 * 用于GPT等自回归语言模型的预训练
 */
class CLMTask {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置对象
   */
  constructor(config) {
    this.vocabSize = config.vocab_size || 30000;
    this.padTokenId = config.pad_token_id || 0;
    this.eosTokenId = config.eos_token_id || 2;
  }

  /**
   * 准备CLM训练数据
   * 
   * @param {Array<number>} tokenIds - 输入token序列
   * @returns {Object} 处理后的训练数据
   */
  prepareData(tokenIds) {
    // CLM任务中，输入是前n-1个token，标签是后n-1个token
    const inputIds = tokenIds.slice(0, -1);
    const labels = tokenIds.slice(1);

    return {
      input_ids: inputIds,
      labels: labels,
      attention_mask: this._createCausalAttentionMask(inputIds.length)
    };
  }

  /**
   * 创建因果注意力掩码
   * 
   * @param {number} seqLen - 序列长度
   * @returns {Array<Array<number>>} 因果掩码矩阵
   */
  _createCausalAttentionMask(seqLen) {
    return MaskGenerator.createCausalMask(seqLen);
  }

  /**
   * 计算CLM损失
   * 
   * @param {Array<Array<number>>} predictions - 模型预测结果 [seqLen, vocabSize]
   * @param {Array<number>} labels - 真实标签
   * @returns {number} 损失值
   */
  computeLoss(predictions, labels) {
    let totalLoss = 0;
    let validPositions = 0;

    for (let i = 0; i < labels.length; i++) {
      if (labels[i] !== this.padTokenId) { // 忽略填充位置
        const probs = softmax(predictions[i]);
        const targetProb = probs[labels[i]];
        totalLoss += -Math.log(Math.max(targetProb, 1e-8));
        validPositions++;
      }
    }

    return validPositions > 0 ? totalLoss / validPositions : 0;
  }
}

/**
 * 句子顺序预测任务 (SOP)
 * 用于ALBERT等模型的改进预训练任务
 */
class SOPTask {
  /**
   * 构造函数
   *
   * @param {Object} config - 配置对象
   */
  constructor(config) {
    this.clsTokenId = config.cls_token_id || 101;
    this.sepTokenId = config.sep_token_id || 102;
    this.padTokenId = config.pad_token_id || 0;
  }

  /**
   * 准备SOP训练数据
   *
   * @param {Array<number>} sentenceA - 第一个句子的token序列
   * @param {Array<number>} sentenceB - 第二个句子的token序列
   * @param {boolean} isCorrectOrder - 是否为正确顺序
   * @returns {Object} 处理后的训练数据
   */
  prepareData(sentenceA, sentenceB, isCorrectOrder) {
    // 如果不是正确顺序，交换两个句子
    const [firstSent, secondSent] = isCorrectOrder ?
      [sentenceA, sentenceB] : [sentenceB, sentenceA];

    // 构建输入序列: [CLS] firstSent [SEP] secondSent [SEP]
    const inputIds = [
      this.clsTokenId,
      ...firstSent,
      this.sepTokenId,
      ...secondSent,
      this.sepTokenId
    ];

    // 创建token类型ID
    const tokenTypeIds = [
      0, // [CLS]
      ...new Array(firstSent.length).fill(0),
      0, // [SEP]
      ...new Array(secondSent.length).fill(1),
      1  // [SEP]
    ];

    return {
      input_ids: inputIds,
      token_type_ids: tokenTypeIds,
      attention_mask: this._createAttentionMask(inputIds),
      sentence_order_label: isCorrectOrder ? 1 : 0
    };
  }

  /**
   * 创建注意力掩码
   *
   * @param {Array<number>} tokenIds - token序列
   * @returns {Array<number>} 注意力掩码
   */
  _createAttentionMask(tokenIds) {
    return tokenIds.map(tokenId => tokenId !== this.padTokenId ? 1 : 0);
  }

  /**
   * 计算SOP损失
   *
   * @param {Array<number>} predictions - 模型预测结果 [2]
   * @param {number} label - 真实标签 (0或1)
   * @returns {number} 损失值
   */
  computeLoss(predictions, label) {
    const probs = softmax(predictions);
    const targetProb = probs[label];
    return -Math.log(Math.max(targetProb, 1e-8));
  }
}

/**
 * 预训练数据处理器
 * 统一处理各种预训练任务的数据
 */
class PretrainingDataProcessor {
  /**
   * 构造函数
   *
   * @param {Object} config - 配置对象
   */
  constructor(config) {
    this.config = config;
    this.mlmTask = new MLMTask(config);
    this.nspTask = new NSPTask(config);
    this.sopTask = new SOPTask(config);
    this.clmTask = new CLMTask(config);
  }

  /**
   * 处理BERT风格的预训练数据 (MLM + NSP)
   *
   * @param {Array<number>} sentenceA - 第一个句子
   * @param {Array<number>} sentenceB - 第二个句子
   * @param {boolean} isNext - 是否为连续句子
   * @returns {Object} 处理后的数据
   */
  processBERTData(sentenceA, sentenceB, isNext) {
    // 准备NSP数据
    const nspData = this.nspTask.prepareData(sentenceA, sentenceB, isNext);

    // 对组合后的序列应用MLM
    const mlmData = this.mlmTask.prepareData(nspData.input_ids);

    return {
      input_ids: mlmData.input_ids,
      token_type_ids: nspData.token_type_ids,
      attention_mask: mlmData.attention_mask,
      mlm_labels: mlmData.labels,
      nsp_labels: nspData.next_sentence_label,
      mask_positions: mlmData.mask_positions
    };
  }

  /**
   * 处理ALBERT风格的预训练数据 (MLM + SOP)
   *
   * @param {Array<number>} sentenceA - 第一个句子
   * @param {Array<number>} sentenceB - 第二个句子
   * @param {boolean} isCorrectOrder - 是否为正确顺序
   * @returns {Object} 处理后的数据
   */
  processALBERTData(sentenceA, sentenceB, isCorrectOrder) {
    // 准备SOP数据
    const sopData = this.sopTask.prepareData(sentenceA, sentenceB, isCorrectOrder);

    // 对组合后的序列应用MLM
    const mlmData = this.mlmTask.prepareData(sopData.input_ids);

    return {
      input_ids: mlmData.input_ids,
      token_type_ids: sopData.token_type_ids,
      attention_mask: mlmData.attention_mask,
      mlm_labels: mlmData.labels,
      sop_labels: sopData.sentence_order_label,
      mask_positions: mlmData.mask_positions
    };
  }

  /**
   * 处理GPT风格的预训练数据 (CLM)
   *
   * @param {Array<number>} tokenIds - 输入token序列
   * @returns {Object} 处理后的数据
   */
  processGPTData(tokenIds) {
    return this.clmTask.prepareData(tokenIds);
  }

  /**
   * 批量处理数据
   *
   * @param {Array<Object>} batch - 批量数据
   * @param {string} taskType - 任务类型 ('bert', 'albert', 'gpt')
   * @returns {Array<Object>} 处理后的批量数据
   */
  processBatch(batch, taskType) {
    const processedBatch = [];

    for (const item of batch) {
      let processedItem;

      switch (taskType.toLowerCase()) {
        case 'bert':
          processedItem = this.processBERTData(
            item.sentence_a,
            item.sentence_b,
            item.is_next
          );
          break;
        case 'albert':
          processedItem = this.processALBERTData(
            item.sentence_a,
            item.sentence_b,
            item.is_correct_order
          );
          break;
        case 'gpt':
          processedItem = this.processGPTData(item.token_ids);
          break;
        default:
          throw new Error(`不支持的任务类型: ${taskType}`);
      }

      processedBatch.push(processedItem);
    }

    return processedBatch;
  }
}

// 导出所有任务类和处理器
module.exports = {
  MLMTask,
  NSPTask,
  CLMTask,
  SOPTask,
  PretrainingDataProcessor
};
