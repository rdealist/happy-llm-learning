/**
 * LLaMA2 模型实现 (JavaScript版本)
 * 
 * 基于第五章《动手搭建大模型》的理论，实现简化版的 LLaMA2 架构。
 * 包含以下关键组件：
 * - RMSNorm 归一化
 * - 分组查询注意力 (GQA)
 * - SwiGLU 激活函数
 * - 完整的 LLaMA2 Transformer 块
 * 
 * @author shihom_wu
 * @version 1.0.0
 * 基于: Happy-LLM 项目第五章理论
 */

const { RMSNorm, SwiGLU, Linear } = require('../core/layers');
const { GroupedQueryAttention } = require('../core/attention');
const { TokenEmbedding } = require('../core/embedding');
const { softmax } = require('../core/math-utils');
const { matmul } = require('../core/matrix-ops');

/**
 * LLaMA2 模型配置
 */
class LLaMA2Config {
  /**
   * 构造函数
   * 
   * @param {Object} config - 配置参数
   */
  constructor(config = {}) {
    this.vocabSize = config.vocabSize || 32000;
    this.dModel = config.dModel || 4096;
    this.numLayers = config.numLayers || 32;
    this.numHeads = config.numHeads || 32;
    this.numKVHeads = config.numKVHeads || 32;
    this.dFF = config.dFF || 11008;
    this.maxSeqLen = config.maxSeqLen || 2048;
    this.dropout = config.dropout || 0.0;
    this.normEps = config.normEps || 1e-6;
    this.padTokenId = config.padTokenId || 0;
    this.bosTokenId = config.bosTokenId || 1;
    this.eosTokenId = config.eosTokenId || 2;
  }
  
  /**
   * LLaMA2 7B 模型配置
   * 
   * @returns {LLaMA2Config} 7B模型配置
   */
  static llama2_7b() {
    return new LLaMA2Config({
      vocabSize: 32000,
      dModel: 4096,
      numLayers: 32,
      numHeads: 32,
      numKVHeads: 32,
      dFF: 11008,
      maxSeqLen: 2048
    });
  }
  
  /**
   * LLaMA2 13B 模型配置
   * 
   * @returns {LLaMA2Config} 13B模型配置
   */
  static llama2_13b() {
    return new LLaMA2Config({
      vocabSize: 32000,
      dModel: 5120,
      numLayers: 40,
      numHeads: 40,
      numKVHeads: 40,
      dFF: 13824,
      maxSeqLen: 2048
    });
  }
  
  /**
   * LLaMA2 70B 模型配置（使用 GQA）
   * 
   * @returns {LLaMA2Config} 70B模型配置
   */
  static llama2_70b() {
    return new LLaMA2Config({
      vocabSize: 32000,
      dModel: 8192,
      numLayers: 80,
      numHeads: 64,
      numKVHeads: 8, // 使用分组查询注意力
      dFF: 28672,
      maxSeqLen: 2048
    });
  }
  
  /**
   * 微信小程序优化配置
   * 
   * @returns {LLaMA2Config} 小程序优化配置
   */
  static miniprogram() {
    return new LLaMA2Config({
      vocabSize: 8000,
      dModel: 256,
      numLayers: 6,
      numHeads: 8,
      numKVHeads: 2, // 使用GQA减少计算量
      dFF: 1024,
      maxSeqLen: 64,
      dropout: 0.1
    });
  }
}

/**
 * LLaMA2 前馈网络
 * 使用 SwiGLU 激活函数的三层线性网络
 */
class LLaMA2MLP {
  /**
   * 构造函数
   * 
   * @param {LLaMA2Config} config - 模型配置
   */
  constructor(config) {
    this.config = config;
    
    // 三个线性层
    this.gateProj = new Linear(config.dModel, config.dFF, false);
    this.upProj = new Linear(config.dModel, config.dFF, false);
    this.downProj = new Linear(config.dFF, config.dModel, false);
  }
  
  /**
   * 前向传播
   * 实现 SwiGLU: SiLU(xW_gate) ⊙ (xW_up)W_down
   * 
   * @param {Array<Array<number>>} x - 输入矩阵 [seqLen, dModel]
   * @returns {Array<Array<number>>} 输出矩阵 [seqLen, dModel]
   */
  forward(x) {
    // 门控投影：应用SiLU激活函数
    const gateOutput = this.gateProj.forward(x);
    const gateActivated = gateOutput.map(row => 
      row.map(val => this._silu(val))
    );
    
    // 上投影
    const upOutput = this.upProj.forward(x);
    
    // 逐元素相乘（门控机制）
    const hidden = gateActivated.map((row, i) => 
      row.map((val, j) => val * upOutput[i][j])
    );
    
    // 下投影
    const output = this.downProj.forward(hidden);
    return output;
  }
  
  /**
   * SiLU (Swish) 激活函数
   * SiLU(x) = x * sigmoid(x)
   * 
   * @param {number} x - 输入值
   * @returns {number} 激活后的值
   */
  _silu(x) {
    return x / (1 + Math.exp(-x));
  }
  
  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return (
      this.gateProj.getParameterCount() +
      this.upProj.getParameterCount() +
      this.downProj.getParameterCount()
    );
  }
}

/**
 * LLaMA2 解码器层
 * 包含自注意力机制、前馈网络、RMSNorm归一化和残差连接
 */
class LLaMA2DecoderLayer {
  /**
   * 构造函数
   * 
   * @param {LLaMA2Config} config - 模型配置
   */
  constructor(config) {
    this.config = config;
    
    // 自注意力机制（使用GQA）
    this.selfAttn = new GroupedQueryAttention(
      config.dModel,
      config.numHeads,
      config.numKVHeads,
      config.dropout,
      false // 不使用偏置
    );
    
    // 前馈网络
    this.mlp = new LLaMA2MLP(config);
    
    // RMSNorm 归一化层
    this.inputLayernorm = new RMSNorm(config.dModel, config.normEps);
    this.postAttentionLayernorm = new RMSNorm(config.dModel, config.normEps);
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} hiddenStates - 输入隐藏状态 [seqLen, dModel]
   * @param {Array<Array<number>>|null} attentionMask - 注意力掩码
   * @returns {Object} {output: 输出隐藏状态, attention: 注意力权重}
   */
  forward(hiddenStates, attentionMask = null) {
    let residual = hiddenStates;
    
    // 1. 自注意力 + 残差连接
    hiddenStates = this.inputLayernorm.forward(hiddenStates);
    const attnResult = this.selfAttn.forward(
      hiddenStates,
      hiddenStates,
      hiddenStates,
      attentionMask
    );
    hiddenStates = this._addResidual(residual, attnResult.output);
    
    // 2. 前馈网络 + 残差连接
    residual = hiddenStates;
    hiddenStates = this.postAttentionLayernorm.forward(hiddenStates);
    hiddenStates = this.mlp.forward(hiddenStates);
    hiddenStates = this._addResidual(residual, hiddenStates);
    
    return {
      output: hiddenStates,
      attention: attnResult.attention
    };
  }
  
  /**
   * 残差连接
   * 
   * @param {Array<Array<number>>} x - 原始输入
   * @param {Array<Array<number>>} y - 子层输出
   * @returns {Array<Array<number>>} 残差连接结果
   */
  _addResidual(x, y) {
    return x.map((row, i) => 
      row.map((val, j) => val + y[i][j])
    );
  }
  
  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.selfAttn.setTraining(training);
  }
  
  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return (
      this.selfAttn.getParameterCount() +
      this.mlp.getParameterCount() +
      this.inputLayernorm.getParameterCount() +
      this.postAttentionLayernorm.getParameterCount()
    );
  }
}

/**
 * LLaMA2 基础模型
 * 实现完整的 LLaMA2 Transformer 架构
 */
class LLaMA2Model {
  /**
   * 构造函数
   *
   * @param {LLaMA2Config} config - 模型配置
   */
  constructor(config) {
    this.config = config;

    // 词嵌入层
    this.embedTokens = new TokenEmbedding(config.vocabSize, config.dModel);

    // Transformer 解码器层
    this.layers = [];
    for (let i = 0; i < config.numLayers; i++) {
      this.layers.push(new LLaMA2DecoderLayer(config));
    }

    // 最终归一化层
    this.norm = new RMSNorm(config.dModel, config.normEps);
  }

  /**
   * 前向传播
   *
   * @param {Array<number>} inputIds - 输入词元 ID [seqLen]
   * @param {Array<Array<number>>|null} attentionMask - 注意力掩码
   * @returns {Object} {lastHiddenState: 最终隐藏状态, attentions: 注意力权重}
   */
  forward(inputIds, attentionMask = null) {
    // 1. 词嵌入
    let hiddenStates = this.embedTokens.forward(inputIds);

    // 2. 通过所有 Transformer 层
    const allAttentions = [];

    for (let i = 0; i < this.layers.length; i++) {
      const layerResult = this.layers[i].forward(hiddenStates, attentionMask);
      hiddenStates = layerResult.output;
      allAttentions.push(layerResult.attention);
    }

    // 3. 最终归一化
    hiddenStates = this.norm.forward(hiddenStates);

    return {
      lastHiddenState: hiddenStates,
      attentions: allAttentions
    };
  }

  /**
   * 设置训练模式
   *
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.layers.forEach(layer => layer.setTraining(training));
  }

  /**
   * 获取参数数量
   *
   * @returns {number} 参数总数
   */
  getParameterCount() {
    let totalParams = this.embedTokens.getParameterCount();
    totalParams += this.layers.reduce((sum, layer) => sum + layer.getParameterCount(), 0);
    totalParams += this.norm.getParameterCount();
    return totalParams;
  }
}

/**
 * LLaMA2 因果语言模型
 * 在基础模型上添加语言模型头，用于文本生成
 */
class LLaMA2ForCausalLM {
  /**
   * 构造函数
   *
   * @param {LLaMA2Config} config - 模型配置
   */
  constructor(config) {
    this.config = config;

    // 基础模型
    this.model = new LLaMA2Model(config);

    // 语言模型头（输出层）
    this.lmHead = new Linear(config.dModel, config.vocabSize, false);
  }

  /**
   * 前向传播
   *
   * @param {Array<number>} inputIds - 输入词元 ID [seqLen]
   * @param {Array<Array<number>>|null} attentionMask - 注意力掩码
   * @returns {Object} {logits: 输出logits, attentions: 注意力权重}
   */
  forward(inputIds, attentionMask = null) {
    // 获取基础模型输出
    const outputs = this.model.forward(inputIds, attentionMask);
    const hiddenStates = outputs.lastHiddenState;

    // 计算 logits
    const logits = this.lmHead.forward(hiddenStates);

    return {
      logits: logits,
      attentions: outputs.attentions
    };
  }

  /**
   * 文本生成
   *
   * @param {Array<number>} inputIds - 输入词元 ID
   * @param {Object} options - 生成选项
   * @returns {Array<number>} 生成的词元 ID
   */
  generate(inputIds, options = {}) {
    const {
      maxNewTokens = 50,
      temperature = 1.0,
      topK = null,
      topP = null,
      doSample = true
    } = options;

    let generated = [...inputIds];

    // 设置为推理模式
    this.setTraining(false);

    for (let i = 0; i < maxNewTokens; i++) {
      // 前向传播
      const outputs = this.forward(generated);
      const logits = outputs.logits;

      // 获取最后一个位置的 logits
      const lastLogits = logits[logits.length - 1];

      // 应用温度
      let nextTokenLogits = lastLogits;
      if (temperature !== 1.0) {
        nextTokenLogits = lastLogits.map(val => val / temperature);
      }

      // 生成下一个词元
      let nextToken;
      if (doSample) {
        nextToken = this._sampleToken(nextTokenLogits, topK, topP);
      } else {
        nextToken = this._greedyToken(nextTokenLogits);
      }

      // 添加到生成序列
      generated.push(nextToken);

      // 检查是否生成了结束词元
      if (nextToken === this.config.eosTokenId) {
        break;
      }

      // 限制序列长度以避免内存问题
      if (generated.length > this.config.maxSeqLen) {
        generated = generated.slice(-this.config.maxSeqLen);
      }
    }

    return generated;
  }

  /**
   * 贪心解码
   *
   * @param {Array<number>} logits - 输出logits
   * @returns {number} 选择的词元ID
   */
  _greedyToken(logits) {
    let maxIdx = 0;
    let maxVal = logits[0];

    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        maxIdx = i;
      }
    }

    return maxIdx;
  }

  /**
   * 采样解码
   *
   * @param {Array<number>} logits - 输出logits
   * @param {number|null} topK - Top-K采样
   * @param {number|null} topP - Top-P采样
   * @returns {number} 选择的词元ID
   */
  _sampleToken(logits, topK = null, topP = null) {
    let filteredLogits = [...logits];

    // Top-K 过滤
    if (topK !== null) {
      const sorted = logits.map((val, idx) => ({ val, idx }))
                           .sort((a, b) => b.val - a.val);
      const threshold = sorted[Math.min(topK - 1, sorted.length - 1)].val;

      filteredLogits = logits.map(val => val >= threshold ? val : -Infinity);
    }

    // Top-P 过滤
    if (topP !== null) {
      const sorted = filteredLogits.map((val, idx) => ({ val, idx }))
                                   .sort((a, b) => b.val - a.val);

      const probs = softmax(sorted.map(item => item.val));
      let cumulativeProb = 0;

      for (let i = 0; i < probs.length; i++) {
        cumulativeProb += probs[i];
        if (cumulativeProb > topP) {
          const threshold = sorted[i].val;
          filteredLogits = logits.map(val => val >= threshold ? val : -Infinity);
          break;
        }
      }
    }

    // 计算概率并采样
    const probs = softmax(filteredLogits);
    const rand = Math.random();
    let cumulativeProb = 0;

    for (let i = 0; i < probs.length; i++) {
      cumulativeProb += probs[i];
      if (rand <= cumulativeProb) {
        return i;
      }
    }

    return probs.length - 1; // fallback
  }

  /**
   * 设置训练模式
   *
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.model.setTraining(training);
  }

  /**
   * 获取参数数量
   *
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return this.model.getParameterCount() + this.lmHead.getParameterCount();
  }
}

// 导出所有类
module.exports = {
  LLaMA2Config,
  LLaMA2MLP,
  LLaMA2DecoderLayer,
  LLaMA2Model,
  LLaMA2ForCausalLM
};
