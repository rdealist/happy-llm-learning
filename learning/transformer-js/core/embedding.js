/**
 * 嵌入层模块
 * 实现词嵌入和位置编码功能
 * 
 * @author shihom_wu
 * @version 1.0.0
 */

const { randomNormalMatrix, zeros } = require('./math-utils');
const { add } = require('./matrix-ops');
const { Dropout } = require('./layers');
const { MATH_CONSTANTS } = require('../config/constants');

/**
 * 词嵌入层
 * 将词汇索引转换为稠密向量表示
 */
class TokenEmbedding {
  /**
   * 构造函数
   * 
   * @param {number} vocabSize - 词汇表大小
   * @param {number} embedDim - 嵌入维度
   * @param {number} initStd - 初始化标准差，默认为 0.02
   */
  constructor(vocabSize, embedDim, initStd = 0.02) {
    this.vocabSize = vocabSize;
    this.embedDim = embedDim;
    
    // 初始化嵌入权重矩阵 [vocabSize, embedDim]
    this.weight = randomNormalMatrix([vocabSize, embedDim], 0, initStd);
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} tokenIds - 词元ID矩阵 [batchSize, seqLen]
   * @returns {Array<Array<Array<number>>>} 嵌入矩阵 [batchSize, seqLen, embedDim]
   */
  forward(tokenIds) {
    if (!Array.isArray(tokenIds) || tokenIds.length === 0) {
      throw new Error('输入必须是非空二维数组');
    }
    
    const batchSize = tokenIds.length;
    const seqLen = tokenIds[0].length;
    const embeddings = [];
    
    for (let b = 0; b < batchSize; b++) {
      const batchEmbeddings = [];
      
      for (let s = 0; s < seqLen; s++) {
        const tokenId = tokenIds[b][s];
        
        // 检查词元ID是否有效
        if (tokenId < 0 || tokenId >= this.vocabSize) {
          throw new Error(`无效的词元ID: ${tokenId}，应在 [0, ${this.vocabSize}) 范围内`);
        }
        
        // 查找对应的嵌入向量
        batchEmbeddings.push([...this.weight[tokenId]]);
      }
      
      embeddings.push(batchEmbeddings);
    }
    
    return embeddings;
  }
  
  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return this.vocabSize * this.embedDim;
  }
}

/**
 * 正弦余弦位置编码
 * 使用正弦和余弦函数生成位置编码
 */
class SinusoidalPositionalEncoding {
  /**
   * 构造函数
   * 
   * @param {number} maxLen - 最大序列长度
   * @param {number} embedDim - 嵌入维度
   * @param {number} base - 位置编码基数，默认为 10000
   */
  constructor(maxLen, embedDim, base = 10000) {
    this.maxLen = maxLen;
    this.embedDim = embedDim;
    this.base = base;
    
    // 预计算位置编码矩阵
    this.positionEncodings = this._computePositionEncodings();
  }
  
  /**
   * 计算位置编码矩阵
   * 
   * @returns {Array<Array<number>>} 位置编码矩阵 [maxLen, embedDim]
   */
  _computePositionEncodings() {
    const pe = zeros([this.maxLen, this.embedDim]);
    
    for (let pos = 0; pos < this.maxLen; pos++) {
      for (let i = 0; i < this.embedDim; i++) {
        const angle = pos / Math.pow(this.base, (2 * Math.floor(i / 2)) / this.embedDim);
        
        if (i % 2 === 0) {
          // 偶数位置使用正弦
          pe[pos][i] = Math.sin(angle);
        } else {
          // 奇数位置使用余弦
          pe[pos][i] = Math.cos(angle);
        }
      }
    }
    
    return pe;
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<Array<number>>>} x - 输入嵌入 [batchSize, seqLen, embedDim]
   * @returns {Array<Array<Array<number>>>} 添加位置编码后的嵌入
   */
  forward(x) {
    if (!Array.isArray(x) || x.length === 0) {
      throw new Error('输入必须是非空三维数组');
    }
    
    const batchSize = x.length;
    const seqLen = x[0].length;
    
    if (seqLen > this.maxLen) {
      throw new Error(`序列长度 ${seqLen} 超过最大长度 ${this.maxLen}`);
    }
    
    const result = [];
    
    for (let b = 0; b < batchSize; b++) {
      const batchResult = [];
      
      for (let s = 0; s < seqLen; s++) {
        const posEncoded = [];
        
        for (let d = 0; d < this.embedDim; d++) {
          posEncoded.push(x[b][s][d] + this.positionEncodings[s][d]);
        }
        
        batchResult.push(posEncoded);
      }
      
      result.push(batchResult);
    }
    
    return result;
  }
}

/**
 * 可学习位置编码
 * 使用可训练的参数矩阵作为位置编码
 */
class LearnedPositionalEncoding {
  /**
   * 构造函数
   * 
   * @param {number} maxLen - 最大序列长度
   * @param {number} embedDim - 嵌入维度
   * @param {number} initStd - 初始化标准差，默认为 0.02
   */
  constructor(maxLen, embedDim, initStd = 0.02) {
    this.maxLen = maxLen;
    this.embedDim = embedDim;
    
    // 初始化可学习的位置编码矩阵 [maxLen, embedDim]
    this.positionEmbeddings = randomNormalMatrix([maxLen, embedDim], 0, initStd);
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<Array<number>>>} x - 输入嵌入 [batchSize, seqLen, embedDim]
   * @returns {Array<Array<Array<number>>>} 添加位置编码后的嵌入
   */
  forward(x) {
    if (!Array.isArray(x) || x.length === 0) {
      throw new Error('输入必须是非空三维数组');
    }
    
    const batchSize = x.length;
    const seqLen = x[0].length;
    
    if (seqLen > this.maxLen) {
      throw new Error(`序列长度 ${seqLen} 超过最大长度 ${this.maxLen}`);
    }
    
    const result = [];
    
    for (let b = 0; b < batchSize; b++) {
      const batchResult = [];
      
      for (let s = 0; s < seqLen; s++) {
        const posEncoded = [];
        
        for (let d = 0; d < this.embedDim; d++) {
          posEncoded.push(x[b][s][d] + this.positionEmbeddings[s][d]);
        }
        
        batchResult.push(posEncoded);
      }
      
      result.push(batchResult);
    }
    
    return result;
  }
  
  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return this.maxLen * this.embedDim;
  }
}

/**
 * 完整的嵌入层
 * 组合词嵌入、位置编码和 dropout
 */
class TransformerEmbedding {
  /**
   * 构造函数
   * 
   * @param {number} vocabSize - 词汇表大小
   * @param {number} embedDim - 嵌入维度
   * @param {number} maxLen - 最大序列长度
   * @param {string} positionType - 位置编码类型: 'sinusoidal' 或 'learned'
   * @param {number} dropout - Dropout 概率，默认为 0.1
   * @param {number} initStd - 初始化标准差，默认为 0.02
   */
  constructor(vocabSize, embedDim, maxLen, positionType = 'sinusoidal', dropout = 0.1, initStd = 0.02) {
    this.vocabSize = vocabSize;
    this.embedDim = embedDim;
    this.maxLen = maxLen;
    this.positionType = positionType;
    
    // 词嵌入层
    this.tokenEmbedding = new TokenEmbedding(vocabSize, embedDim, initStd);
    
    // 位置编码层
    if (positionType === 'sinusoidal') {
      this.positionEncoding = new SinusoidalPositionalEncoding(maxLen, embedDim);
    } else if (positionType === 'learned') {
      this.positionEncoding = new LearnedPositionalEncoding(maxLen, embedDim, initStd);
    } else {
      throw new Error(`不支持的位置编码类型: ${positionType}`);
    }
    
    // Dropout 层
    this.dropout = new Dropout(dropout);
    
    // 嵌入缩放因子（可选）
    this.embedScale = Math.sqrt(embedDim);
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} tokenIds - 词元ID矩阵 [batchSize, seqLen]
   * @param {boolean} scaleEmbedding - 是否缩放嵌入，默认为 true
   * @returns {Array<Array<Array<number>>>} 最终嵌入 [batchSize, seqLen, embedDim]
   */
  forward(tokenIds, scaleEmbedding = true) {
    // 获取词嵌入
    let embeddings = this.tokenEmbedding.forward(tokenIds);
    
    // 可选的嵌入缩放
    if (scaleEmbedding) {
      embeddings = embeddings.map(batch =>
        batch.map(seq =>
          seq.map(val => val * this.embedScale)
        )
      );
    }
    
    // 添加位置编码
    embeddings = this.positionEncoding.forward(embeddings);
    
    // 应用 dropout
    const result = [];
    for (let b = 0; b < embeddings.length; b++) {
      result.push(this.dropout.forward(embeddings[b]));
    }
    
    return result;
  }
  
  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.dropout.setTraining(training);
  }
  
  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    let totalParams = this.tokenEmbedding.getParameterCount();
    
    if (this.positionType === 'learned') {
      totalParams += this.positionEncoding.getParameterCount();
    }
    
    return totalParams;
  }
}

// 导出所有类
module.exports = {
  TokenEmbedding,
  SinusoidalPositionalEncoding,
  LearnedPositionalEncoding,
  TransformerEmbedding
};
