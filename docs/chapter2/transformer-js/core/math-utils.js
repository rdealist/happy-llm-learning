/**
 * 数学工具函数模块
 * 提供 Transformer 模型所需的基础数学运算函数
 * 
 * @author Transformer-JS
 * @version 1.0.0
 */

/**
 * Softmax 激活函数
 * 将输入向量转换为概率分布，所有元素和为1
 * 
 * @param {Array<number>} x - 输入向量
 * @param {number} dim - 计算维度，默认为最后一维
 * @returns {Array<number>} 经过 softmax 处理的概率分布
 */
function softmax(x, dim = -1) {
  if (!Array.isArray(x) || x.length === 0) {
    throw new Error('输入必须是非空数组');
  }
  
  // 处理负数维度索引
  if (dim < 0) {
    dim = x.length + dim;
  }
  
  // 为数值稳定性，减去最大值
  const maxVal = Math.max(...x);
  const expValues = x.map(val => Math.exp(val - maxVal));
  const sumExp = expValues.reduce((sum, val) => sum + val, 0);
  
  return expValues.map(val => val / sumExp);
}

/**
 * ReLU 激活函数
 * 线性整流函数，将负值置为0，正值保持不变
 * 
 * @param {number|Array<number>} x - 输入值或向量
 * @returns {number|Array<number>} 经过 ReLU 处理的结果
 */
function relu(x) {
  if (typeof x === 'number') {
    return Math.max(0, x);
  }
  
  if (Array.isArray(x)) {
    return x.map(val => Math.max(0, val));
  }
  
  throw new Error('输入必须是数字或数字数组');
}

/**
 * GELU 激活函数
 * 高斯误差线性单元，平滑的 ReLU 变体
 * 
 * @param {number|Array<number>} x - 输入值或向量
 * @returns {number|Array<number>} 经过 GELU 处理的结果
 */
function gelu(x) {
  const geluSingle = (val) => {
    return 0.5 * val * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (val + 0.044715 * Math.pow(val, 3))));
  };
  
  if (typeof x === 'number') {
    return geluSingle(x);
  }
  
  if (Array.isArray(x)) {
    return x.map(geluSingle);
  }
  
  throw new Error('输入必须是数字或数字数组');
}

/**
 * 计算向量的均值
 * 
 * @param {Array<number>} x - 输入向量
 * @returns {number} 均值
 */
function mean(x) {
  if (!Array.isArray(x) || x.length === 0) {
    throw new Error('输入必须是非空数组');
  }
  
  return x.reduce((sum, val) => sum + val, 0) / x.length;
}

/**
 * 计算向量的标准差
 * 
 * @param {Array<number>} x - 输入向量
 * @param {number} meanVal - 预计算的均值（可选）
 * @returns {number} 标准差
 */
function std(x, meanVal = null) {
  if (!Array.isArray(x) || x.length === 0) {
    throw new Error('输入必须是非空数组');
  }
  
  const m = meanVal !== null ? meanVal : mean(x);
  const variance = x.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / x.length;
  
  return Math.sqrt(variance);
}

/**
 * 生成正态分布随机数
 * 使用 Box-Muller 变换生成标准正态分布随机数
 * 
 * @param {number} mean - 均值，默认为0
 * @param {number} stdDev - 标准差，默认为1
 * @returns {number} 正态分布随机数
 */
function randomNormal(mean = 0, stdDev = 1) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random(); // 避免 log(0)
  while (v === 0) v = Math.random();
  
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * stdDev + mean;
}

/**
 * 生成指定形状的随机正态分布矩阵
 * 
 * @param {Array<number>} shape - 矩阵形状 [rows, cols]
 * @param {number} mean - 均值，默认为0
 * @param {number} stdDev - 标准差，默认为0.02
 * @returns {Array<Array<number>>} 随机矩阵
 */
function randomNormalMatrix(shape, mean = 0, stdDev = 0.02) {
  if (!Array.isArray(shape) || shape.length !== 2) {
    throw new Error('形状必须是包含两个元素的数组 [rows, cols]');
  }
  
  const [rows, cols] = shape;
  const matrix = [];
  
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      row.push(randomNormal(mean, stdDev));
    }
    matrix.push(row);
  }
  
  return matrix;
}

/**
 * 生成零矩阵
 * 
 * @param {Array<number>} shape - 矩阵形状 [rows, cols]
 * @returns {Array<Array<number>>} 零矩阵
 */
function zeros(shape) {
  if (!Array.isArray(shape) || shape.length !== 2) {
    throw new Error('形状必须是包含两个元素的数组 [rows, cols]');
  }
  
  const [rows, cols] = shape;
  const matrix = [];
  
  for (let i = 0; i < rows; i++) {
    matrix.push(new Array(cols).fill(0));
  }
  
  return matrix;
}

/**
 * 生成单位矩阵
 * 
 * @param {number} size - 矩阵大小
 * @returns {Array<Array<number>>} 单位矩阵
 */
function eye(size) {
  const matrix = zeros([size, size]);
  
  for (let i = 0; i < size; i++) {
    matrix[i][i] = 1;
  }
  
  return matrix;
}

/**
 * 计算平方根倒数
 * 用于层归一化等操作
 * 
 * @param {number} x - 输入值
 * @param {number} eps - 防止除零的小值，默认为1e-6
 * @returns {number} 平方根倒数
 */
function rsqrt(x, eps = 1e-6) {
  return 1.0 / Math.sqrt(x + eps);
}

/**
 * 裁剪函数
 * 将值限制在指定范围内
 * 
 * @param {number} x - 输入值
 * @param {number} min - 最小值
 * @param {number} max - 最大值
 * @returns {number} 裁剪后的值
 */
function clip(x, min, max) {
  return Math.min(Math.max(x, min), max);
}

// 导出所有函数
module.exports = {
  softmax,
  relu,
  gelu,
  mean,
  std,
  randomNormal,
  randomNormalMatrix,
  zeros,
  eye,
  rsqrt,
  clip
};
