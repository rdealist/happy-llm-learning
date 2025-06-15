/**
 * 注意力机制模块
 * 实现各种注意力机制，包括基础注意力、自注意力、多头注意力等
 * 
 * @author shihom_wu
 * @version 1.0.0
 */

const { softmax } = require('./math-utils');
const { matmul, transpose, scalarDivide, add, zeros, split, concat } = require('./matrix-ops');
const { Linear, Dropout } = require('./layers');
const { MATH_CONSTANTS } = require('../config/constants');

/**
 * 基础注意力机制
 * 实现 Attention(Q, K, V) = softmax(QK^T / √d_k)V
 */
class Attention {
  /**
   * 构造函数
   * 
   * @param {number} dropout - 注意力 dropout 概率，默认为 0.1
   */
  constructor(dropout = 0.1) {
    this.dropout = new Dropout(dropout);
  }
  
  /**
   * 计算注意力
   * 
   * @param {Array<Array<number>>} query - 查询矩阵 Q [seqLen, dModel]
   * @param {Array<Array<number>>} key - 键矩阵 K [seqLen, dModel]
   * @param {Array<Array<number>>} value - 值矩阵 V [seqLen, dModel]
   * @param {Array<Array<number>>|null} mask - 注意力掩码，可选
   * @returns {Object} {output: 注意力输出, attention: 注意力权重}
   */
  forward(query, key, value, mask = null) {
    if (!Array.isArray(query) || !Array.isArray(key) || !Array.isArray(value)) {
      throw new Error('Q, K, V 必须是二维数组');
    }
    
    const dK = key[0].length; // 键的维度
    
    // 计算注意力分数: Q @ K^T
    const keyTransposed = transpose(key);
    let scores = matmul(query, keyTransposed);
    
    // 缩放: scores / √d_k
    const scale = 1.0 / Math.sqrt(dK);
    scores = scores.map(row => row.map(val => val * scale));
    
    // 应用掩码（如果提供）
    if (mask) {
      scores = this._applyMask(scores, mask);
    }
    
    // 计算注意力权重: softmax(scores)
    const attentionWeights = scores.map(row => softmax(row));
    
    // 应用 dropout
    const droppedWeights = this.dropout.forward(attentionWeights);
    
    // 计算输出: attention_weights @ V
    const output = matmul(droppedWeights, value);
    
    return {
      output: output,
      attention: attentionWeights
    };
  }
  
  /**
   * 应用注意力掩码
   * 
   * @param {Array<Array<number>>} scores - 注意力分数
   * @param {Array<Array<number>>} mask - 掩码矩阵
   * @returns {Array<Array<number>>} 应用掩码后的分数
   */
  _applyMask(scores, mask) {
    const maskedScores = [];
    
    for (let i = 0; i < scores.length; i++) {
      const row = [];
      for (let j = 0; j < scores[i].length; j++) {
        if (mask[i][j] === 0) {
          row.push(-Infinity); // 掩码位置设为负无穷
        } else {
          row.push(scores[i][j]);
        }
      }
      maskedScores.push(row);
    }
    
    return maskedScores;
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
 * 多头注意力机制
 * 实现 Multi-Head Attention
 */
class MultiHeadAttention {
  /**
   * 构造函数
   * 
   * @param {number} dModel - 模型维度
   * @param {number} nHeads - 注意力头数
   * @param {number} dropout - Dropout 概率，默认为 0.1
   * @param {boolean} useBias - 是否使用偏置，默认为 false
   */
  constructor(dModel, nHeads, dropout = 0.1, useBias = false) {
    if (dModel % nHeads !== 0) {
      throw new Error(`模型维度 ${dModel} 必须能被注意力头数 ${nHeads} 整除`);
    }
    
    this.dModel = dModel;
    this.nHeads = nHeads;
    this.dK = dModel / nHeads; // 每个头的维度
    
    // Q, K, V 的线性变换层
    this.wQ = new Linear(dModel, dModel, useBias);
    this.wK = new Linear(dModel, dModel, useBias);
    this.wV = new Linear(dModel, dModel, useBias);
    
    // 输出投影层
    this.wO = new Linear(dModel, dModel, useBias);
    
    // 基础注意力机制
    this.attention = new Attention(dropout);
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} query - 查询矩阵 [seqLen, dModel]
   * @param {Array<Array<number>>} key - 键矩阵 [seqLen, dModel]
   * @param {Array<Array<number>>} value - 值矩阵 [seqLen, dModel]
   * @param {Array<Array<number>>|null} mask - 注意力掩码，可选
   * @returns {Object} {output: 多头注意力输出, attention: 注意力权重}
   */
  forward(query, key, value, mask = null) {
    const seqLen = query.length;
    
    // 线性变换得到 Q, K, V
    const Q = this.wQ.forward(query);
    const K = this.wK.forward(key);
    const V = this.wV.forward(value);
    
    // 重塑为多头形状并分割
    const QHeads = this._splitHeads(Q);
    const KHeads = this._splitHeads(K);
    const VHeads = this._splitHeads(V);
    
    // 对每个头计算注意力
    const headOutputs = [];
    const headAttentions = [];
    
    for (let h = 0; h < this.nHeads; h++) {
      const result = this.attention.forward(
        QHeads[h], 
        KHeads[h], 
        VHeads[h], 
        mask
      );
      
      headOutputs.push(result.output);
      headAttentions.push(result.attention);
    }
    
    // 拼接所有头的输出
    const concatenated = this._concatHeads(headOutputs);
    
    // 最终的线性投影
    const output = this.wO.forward(concatenated);
    
    return {
      output: output,
      attention: headAttentions
    };
  }
  
  /**
   * 将输入分割为多个注意力头
   * 
   * @param {Array<Array<number>>} x - 输入矩阵 [seqLen, dModel]
   * @returns {Array<Array<Array<number>>>} 分割后的头 [nHeads][seqLen, dK]
   */
  _splitHeads(x) {
    const seqLen = x.length;
    const heads = [];
    
    for (let h = 0; h < this.nHeads; h++) {
      const head = [];
      for (let i = 0; i < seqLen; i++) {
        const start = h * this.dK;
        const end = start + this.dK;
        head.push(x[i].slice(start, end));
      }
      heads.push(head);
    }
    
    return heads;
  }
  
  /**
   * 拼接多个注意力头的输出
   * 
   * @param {Array<Array<Array<number>>>} heads - 多个头的输出 [nHeads][seqLen, dK]
   * @returns {Array<Array<number>>} 拼接后的矩阵 [seqLen, dModel]
   */
  _concatHeads(heads) {
    const seqLen = heads[0].length;
    const concatenated = [];
    
    for (let i = 0; i < seqLen; i++) {
      const row = [];
      for (let h = 0; h < this.nHeads; h++) {
        row.push(...heads[h][i]);
      }
      concatenated.push(row);
    }
    
    return concatenated;
  }
  
  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.attention.setTraining(training);
  }
  
  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return (
      this.wQ.getParameterCount() +
      this.wK.getParameterCount() +
      this.wV.getParameterCount() +
      this.wO.getParameterCount()
    );
  }
}

/**
 * 掩码生成工具
 * 生成各种类型的注意力掩码
 */
class MaskGenerator {
  /**
   * 生成因果掩码（下三角掩码）
   * 用于解码器的自注意力，确保只能看到当前位置之前的信息
   * 
   * @param {number} seqLen - 序列长度
   * @returns {Array<Array<number>>} 因果掩码矩阵
   */
  static createCausalMask(seqLen) {
    const mask = [];
    
    for (let i = 0; i < seqLen; i++) {
      const row = [];
      for (let j = 0; j < seqLen; j++) {
        row.push(j <= i ? 1 : 0); // 下三角为1，上三角为0
      }
      mask.push(row);
    }
    
    return mask;
  }
  
  /**
   * 生成填充掩码
   * 用于忽略填充位置的注意力计算
   * 
   * @param {Array<number>} tokenIds - 词元ID数组
   * @param {number} padId - 填充词元的ID，默认为0
   * @returns {Array<Array<number>>} 填充掩码矩阵
   */
  static createPaddingMask(tokenIds, padId = 0) {
    const seqLen = tokenIds.length;
    const mask = [];
    
    for (let i = 0; i < seqLen; i++) {
      const row = [];
      for (let j = 0; j < seqLen; j++) {
        // 如果目标位置是填充词元，则掩码为0
        row.push(tokenIds[j] !== padId ? 1 : 0);
      }
      mask.push(row);
    }
    
    return mask;
  }
  
  /**
   * 创建MLM掩码
   * 用于BERT等模型的掩码语言建模任务
   *
   * @param {Array<number>} tokenIds - 输入token序列
   * @param {number} maskRatio - 掩码比例，默认0.15
   * @param {number} maskTokenId - [MASK] token的ID，默认103
   * @param {number} vocabSize - 词汇表大小，用于随机替换
   * @returns {Object} {maskedTokens: 掩码后的序列, labels: 原始标签, maskPositions: 掩码位置}
   */
  static createMLMMask(tokenIds, maskRatio = 0.15, maskTokenId = 103, vocabSize = 30000) {
    const seqLen = tokenIds.length;
    const numMask = Math.floor(seqLen * maskRatio);
    const maskedTokens = [...tokenIds];
    const labels = new Array(seqLen).fill(-100); // -100表示不计算损失
    const maskPositions = [];

    // 随机选择要掩码的位置
    const positions = [];
    for (let i = 0; i < seqLen; i++) {
      positions.push(i);
    }

    // Fisher-Yates洗牌算法
    for (let i = positions.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [positions[i], positions[j]] = [positions[j], positions[i]];
    }

    // 选择前numMask个位置进行掩码
    for (let i = 0; i < numMask; i++) {
      const pos = positions[i];
      labels[pos] = tokenIds[pos]; // 保存原始token用于计算损失
      maskPositions.push(pos);

      const rand = Math.random();
      if (rand < 0.8) {
        // 80%的概率替换为[MASK]
        maskedTokens[pos] = maskTokenId;
      } else if (rand < 0.9) {
        // 10%的概率替换为随机token
        maskedTokens[pos] = Math.floor(Math.random() * vocabSize);
      }
      // 10%的概率保持不变
    }

    return {
      maskedTokens,
      labels,
      maskPositions
    };
  }

  /**
   * 创建双向注意力掩码
   * 用于BERT等双向模型，允许每个位置关注所有位置
   *
   * @param {number} seqLen - 序列长度
   * @returns {Array<Array<number>>} 全1的掩码矩阵
   */
  static createBidirectionalMask(seqLen) {
    const mask = [];

    for (let i = 0; i < seqLen; i++) {
      const row = new Array(seqLen).fill(1);
      mask.push(row);
    }

    return mask;
  }

  /**
   * 创建填充掩码
   * 用于忽略填充token的注意力计算
   *
   * @param {Array<number>} tokenIds - token序列
   * @param {number} padTokenId - 填充token ID，默认0
   * @returns {Array<Array<number>>} 填充掩码矩阵
   */
  static createPaddingMask(tokenIds, padTokenId = 0) {
    const seqLen = tokenIds.length;
    const mask = [];

    for (let i = 0; i < seqLen; i++) {
      const row = [];
      for (let j = 0; j < seqLen; j++) {
        // 如果目标位置是填充token，则掩码为0（不关注）
        row.push(tokenIds[j] !== padTokenId ? 1 : 0);
      }
      mask.push(row);
    }

    return mask;
  }

  /**
   * 组合多个掩码
   *
   * @param {Array<Array<Array<number>>>} masks - 掩码数组
   * @returns {Array<Array<number>>} 组合后的掩码
   */
  static combineMasks(masks) {
    if (!masks || masks.length === 0) {
      return null;
    }

    const seqLen = masks[0].length;
    const combinedMask = [];

    for (let i = 0; i < seqLen; i++) {
      const row = [];
      for (let j = 0; j < seqLen; j++) {
        // 所有掩码的逻辑与操作
        let value = 1;
        for (const mask of masks) {
          value = value && mask[i][j];
        }
        row.push(value);
      }
      combinedMask.push(row);
    }

    return combinedMask;
  }
}

// 导出所有类和函数
module.exports = {
  Attention,
  MultiHeadAttention,
  MaskGenerator
};
