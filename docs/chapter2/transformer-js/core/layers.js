/**
 * 基础层模块
 * 实现 Transformer 模型的基础组件层
 * 
 * @author Transformer-JS
 * @version 1.0.0
 */

const { mean, std, rsqrt, relu, gelu, randomNormalMatrix, zeros } = require('./math-utils');
const { matmul, add, scalarMultiply } = require('./matrix-ops');
const { MATH_CONSTANTS, ACTIVATION_FUNCTIONS } = require('../config/constants');

/**
 * 线性层（全连接层）
 * 实现线性变换 y = xW + b
 */
class Linear {
  /**
   * 构造函数
   * 
   * @param {number} inputDim - 输入维度
   * @param {number} outputDim - 输出维度
   * @param {boolean} useBias - 是否使用偏置，默认为 true
   * @param {number} initStd - 权重初始化标准差，默认为 0.02
   */
  constructor(inputDim, outputDim, useBias = true, initStd = 0.02) {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.useBias = useBias;
    
    // 初始化权重矩阵 [inputDim, outputDim]
    this.weight = randomNormalMatrix([inputDim, outputDim], 0, initStd);
    
    // 初始化偏置向量 [outputDim]
    this.bias = useBias ? new Array(outputDim).fill(0) : null;
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} x - 输入矩阵 [batchSize, inputDim]
   * @returns {Array<Array<number>>} 输出矩阵 [batchSize, outputDim]
   */
  forward(x) {
    if (!Array.isArray(x) || x.length === 0) {
      throw new Error('输入必须是非空二维数组');
    }
    
    // 矩阵乘法: x @ weight
    let output = matmul(x, this.weight);
    
    // 添加偏置
    if (this.useBias && this.bias) {
      output = output.map(row => 
        row.map((val, idx) => val + this.bias[idx])
      );
    }
    
    return output;
  }
  
  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    const weightParams = this.inputDim * this.outputDim;
    const biasParams = this.useBias ? this.outputDim : 0;
    return weightParams + biasParams;
  }
}

/**
 * 层归一化（Layer Normalization）
 * 对每个样本的特征维度进行归一化
 */
class LayerNorm {
  /**
   * 构造函数
   * 
   * @param {number} normalizedShape - 归一化的维度大小
   * @param {number} eps - 防止除零的小值，默认为 1e-6
   */
  constructor(normalizedShape, eps = MATH_CONSTANTS.LAYER_NORM_EPS) {
    this.normalizedShape = normalizedShape;
    this.eps = eps;
    
    // 可学习参数：缩放因子和偏移因子
    this.gamma = new Array(normalizedShape).fill(1.0);  // 缩放参数
    this.beta = new Array(normalizedShape).fill(0.0);   // 偏移参数
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} x - 输入矩阵 [batchSize, normalizedShape]
   * @returns {Array<Array<number>>} 归一化后的矩阵
   */
  forward(x) {
    if (!Array.isArray(x) || x.length === 0) {
      throw new Error('输入必须是非空二维数组');
    }
    
    const batchSize = x.length;
    const output = new Array(batchSize);
    
    for (let i = 0; i < batchSize; i++) {
      const row = x[i];
      
      // 计算均值和标准差
      const meanVal = mean(row);
      const stdVal = std(row, meanVal);
      
      // 归一化并应用可学习参数
      output[i] = row.map((val, j) => {
        const normalized = (val - meanVal) / (stdVal + this.eps);
        return this.gamma[j] * normalized + this.beta[j];
      });
    }
    
    return output;
  }
  
  /**
   * 获取参数数量
   * 
   * @returns {number} 参数总数
   */
  getParameterCount() {
    return 2 * this.normalizedShape; // gamma 和 beta
  }
}

/**
 * Dropout 层
 * 在训练时随机将部分神经元输出置为零，防止过拟合
 */
class Dropout {
  /**
   * 构造函数
   * 
   * @param {number} p - Dropout 概率，默认为 0.1
   */
  constructor(p = 0.1) {
    if (p < 0 || p > 1) {
      throw new Error('Dropout 概率必须在 0 到 1 之间');
    }
    
    this.p = p;
    this.training = true; // 训练模式标志
  }
  
  /**
   * 设置训练模式
   * 
   * @param {boolean} training - 是否为训练模式
   */
  setTraining(training) {
    this.training = training;
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} x - 输入矩阵
   * @returns {Array<Array<number>>} 处理后的矩阵
   */
  forward(x) {
    if (!Array.isArray(x) || x.length === 0) {
      throw new Error('输入必须是非空二维数组');
    }
    
    // 推理模式下直接返回输入
    if (!this.training) {
      return x;
    }
    
    // 训练模式下应用 dropout
    const scale = 1.0 / (1.0 - this.p); // 缩放因子，保持期望不变
    
    return x.map(row => 
      row.map(val => {
        if (Math.random() < this.p) {
          return 0; // 随机置零
        } else {
          return val * scale; // 缩放保持期望
        }
      })
    );
  }
}

/**
 * 多层感知机（MLP）/ 前馈神经网络（FFN）
 * 实现两层全连接网络，中间使用激活函数
 */
class MLP {
  /**
   * 构造函数
   * 
   * @param {number} inputDim - 输入维度
   * @param {number} hiddenDim - 隐藏层维度
   * @param {number} outputDim - 输出维度，默认等于输入维度
   * @param {string} activation - 激活函数类型，默认为 'relu'
   * @param {number} dropout - Dropout 概率，默认为 0.1
   * @param {boolean} useBias - 是否使用偏置，默认为 false
   */
  constructor(inputDim, hiddenDim, outputDim = null, activation = 'relu', dropout = 0.1, useBias = false) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.outputDim = outputDim || inputDim;
    this.activation = activation;
    
    // 第一层：输入 -> 隐藏层
    this.linear1 = new Linear(this.inputDim, this.hiddenDim, useBias);
    
    // 第二层：隐藏层 -> 输出
    this.linear2 = new Linear(this.hiddenDim, this.outputDim, useBias);
    
    // Dropout 层
    this.dropout = new Dropout(dropout);
    
    // 激活函数
    this.activationFn = this._getActivationFunction(activation);
  }
  
  /**
   * 获取激活函数
   * 
   * @param {string} activation - 激活函数名称
   * @returns {Function} 激活函数
   */
  _getActivationFunction(activation) {
    switch (activation.toLowerCase()) {
      case ACTIVATION_FUNCTIONS.RELU:
        return relu;
      case ACTIVATION_FUNCTIONS.GELU:
        return gelu;
      default:
        throw new Error(`不支持的激活函数: ${activation}`);
    }
  }
  
  /**
   * 前向传播
   * 
   * @param {Array<Array<number>>} x - 输入矩阵 [batchSize, inputDim]
   * @returns {Array<Array<number>>} 输出矩阵 [batchSize, outputDim]
   */
  forward(x) {
    // 第一层线性变换
    let hidden = this.linear1.forward(x);
    
    // 激活函数
    hidden = hidden.map(row => this.activationFn(row));
    
    // Dropout
    hidden = this.dropout.forward(hidden);
    
    // 第二层线性变换
    const output = this.linear2.forward(hidden);
    
    return output;
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
    return this.linear1.getParameterCount() + this.linear2.getParameterCount();
  }
}

/**
 * 残差连接（Residual Connection）
 * 实现 y = x + F(x) 的残差连接
 */
class ResidualConnection {
  /**
   * 构造函数
   * 
   * @param {number} dim - 特征维度
   * @param {number} dropout - Dropout 概率，默认为 0.1
   */
  constructor(dim, dropout = 0.1) {
    this.dim = dim;
    this.layerNorm = new LayerNorm(dim);
    this.dropout = new Dropout(dropout);
  }
  
  /**
   * 前向传播
   * 实现 Pre-LN 结构: x + dropout(sublayer(layernorm(x)))
   * 
   * @param {Array<Array<number>>} x - 输入矩阵
   * @param {Function} sublayer - 子层函数
   * @returns {Array<Array<number>>} 输出矩阵
   */
  forward(x, sublayer) {
    // 层归一化
    const normalized = this.layerNorm.forward(x);
    
    // 子层计算
    const sublayerOutput = sublayer(normalized);
    
    // Dropout
    const dropped = this.dropout.forward(sublayerOutput);
    
    // 残差连接
    return add(x, dropped);
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
    return this.layerNorm.getParameterCount();
  }
}

// 导出所有类
module.exports = {
  Linear,
  LayerNorm,
  Dropout,
  MLP,
  ResidualConnection
};
