/**
 * 矩阵运算模块
 * 提供 Transformer 模型所需的矩阵操作函数
 * 
 * @author shihom_wu
 * @version 1.0.0
 */

/**
 * 矩阵乘法
 * 计算两个矩阵的乘积 A × B
 * 
 * @param {Array<Array<number>>} A - 左矩阵 [m, k]
 * @param {Array<Array<number>>} B - 右矩阵 [k, n]
 * @returns {Array<Array<number>>} 结果矩阵 [m, n]
 */
function matmul(A, B) {
  if (!Array.isArray(A) || !Array.isArray(B)) {
    throw new Error('输入必须是二维数组');
  }
  
  if (A.length === 0 || B.length === 0) {
    throw new Error('矩阵不能为空');
  }
  
  const m = A.length;        // A 的行数
  const k = A[0].length;     // A 的列数 / B 的行数
  const n = B[0].length;     // B 的列数
  
  // 检查矩阵维度是否匹配
  if (B.length !== k) {
    throw new Error(`矩阵维度不匹配: A[${m}, ${k}] × B[${B.length}, ${n}]`);
  }
  
  // 初始化结果矩阵
  const result = new Array(m);
  for (let i = 0; i < m; i++) {
    result[i] = new Array(n).fill(0);
  }
  
  // 执行矩阵乘法
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      for (let l = 0; l < k; l++) {
        result[i][j] += A[i][l] * B[l][j];
      }
    }
  }
  
  return result;
}

/**
 * 矩阵转置
 * 交换矩阵的行和列
 * 
 * @param {Array<Array<number>>} matrix - 输入矩阵 [m, n]
 * @returns {Array<Array<number>>} 转置矩阵 [n, m]
 */
function transpose(matrix) {
  if (!Array.isArray(matrix) || matrix.length === 0) {
    throw new Error('输入必须是非空二维数组');
  }
  
  const rows = matrix.length;
  const cols = matrix[0].length;
  
  const result = new Array(cols);
  for (let i = 0; i < cols; i++) {
    result[i] = new Array(rows);
    for (let j = 0; j < rows; j++) {
      result[i][j] = matrix[j][i];
    }
  }
  
  return result;
}

/**
 * 矩阵重塑
 * 改变矩阵的形状但保持元素总数不变
 * 
 * @param {Array<Array<number>>} matrix - 输入矩阵
 * @param {Array<number>} newShape - 新形状 [rows, cols]
 * @returns {Array<Array<number>>} 重塑后的矩阵
 */
function reshape(matrix, newShape) {
  if (!Array.isArray(matrix) || !Array.isArray(newShape)) {
    throw new Error('输入参数类型错误');
  }
  
  // 将矩阵展平为一维数组
  const flattened = matrix.flat();
  const [newRows, newCols] = newShape;
  
  // 检查元素总数是否匹配
  if (flattened.length !== newRows * newCols) {
    throw new Error(`元素总数不匹配: ${flattened.length} !== ${newRows * newCols}`);
  }
  
  // 重新组织为新形状
  const result = new Array(newRows);
  for (let i = 0; i < newRows; i++) {
    result[i] = flattened.slice(i * newCols, (i + 1) * newCols);
  }
  
  return result;
}

/**
 * 矩阵加法
 * 对应元素相加
 * 
 * @param {Array<Array<number>>} A - 矩阵A
 * @param {Array<Array<number>>} B - 矩阵B
 * @returns {Array<Array<number>>} 结果矩阵
 */
function add(A, B) {
  if (!Array.isArray(A) || !Array.isArray(B)) {
    throw new Error('输入必须是二维数组');
  }
  
  if (A.length !== B.length || A[0].length !== B[0].length) {
    throw new Error('矩阵形状必须相同');
  }
  
  const rows = A.length;
  const cols = A[0].length;
  const result = new Array(rows);
  
  for (let i = 0; i < rows; i++) {
    result[i] = new Array(cols);
    for (let j = 0; j < cols; j++) {
      result[i][j] = A[i][j] + B[i][j];
    }
  }
  
  return result;
}

/**
 * 矩阵减法
 * 对应元素相减
 * 
 * @param {Array<Array<number>>} A - 矩阵A
 * @param {Array<Array<number>>} B - 矩阵B
 * @returns {Array<Array<number>>} 结果矩阵
 */
function subtract(A, B) {
  if (!Array.isArray(A) || !Array.isArray(B)) {
    throw new Error('输入必须是二维数组');
  }
  
  if (A.length !== B.length || A[0].length !== B[0].length) {
    throw new Error('矩阵形状必须相同');
  }
  
  const rows = A.length;
  const cols = A[0].length;
  const result = new Array(rows);
  
  for (let i = 0; i < rows; i++) {
    result[i] = new Array(cols);
    for (let j = 0; j < cols; j++) {
      result[i][j] = A[i][j] - B[i][j];
    }
  }
  
  return result;
}

/**
 * 标量乘法
 * 矩阵的每个元素乘以标量
 * 
 * @param {Array<Array<number>>} matrix - 输入矩阵
 * @param {number} scalar - 标量值
 * @returns {Array<Array<number>>} 结果矩阵
 */
function scalarMultiply(matrix, scalar) {
  if (!Array.isArray(matrix)) {
    throw new Error('输入必须是二维数组');
  }
  
  return matrix.map(row => row.map(val => val * scalar));
}

/**
 * 矩阵除法（标量）
 * 矩阵的每个元素除以标量
 * 
 * @param {Array<Array<number>>} matrix - 输入矩阵
 * @param {number} scalar - 标量值
 * @returns {Array<Array<number>>} 结果矩阵
 */
function scalarDivide(matrix, scalar) {
  if (scalar === 0) {
    throw new Error('除数不能为零');
  }
  
  return scalarMultiply(matrix, 1 / scalar);
}

/**
 * 获取矩阵形状
 * 
 * @param {Array<Array<number>>} matrix - 输入矩阵
 * @returns {Array<number>} 形状数组 [rows, cols]
 */
function shape(matrix) {
  if (!Array.isArray(matrix)) {
    throw new Error('输入必须是数组');
  }
  
  if (matrix.length === 0) {
    return [0, 0];
  }
  
  return [matrix.length, matrix[0].length];
}

/**
 * 矩阵拼接（沿指定轴）
 * 
 * @param {Array<Array<Array<number>>>} matrices - 矩阵数组
 * @param {number} axis - 拼接轴，0为行方向，1为列方向
 * @returns {Array<Array<number>>} 拼接后的矩阵
 */
function concat(matrices, axis = 0) {
  if (!Array.isArray(matrices) || matrices.length === 0) {
    throw new Error('输入必须是非空矩阵数组');
  }
  
  if (axis === 0) {
    // 按行拼接
    return matrices.reduce((result, matrix) => result.concat(matrix), []);
  } else if (axis === 1) {
    // 按列拼接
    const rows = matrices[0].length;
    const result = new Array(rows);
    
    for (let i = 0; i < rows; i++) {
      result[i] = [];
      for (const matrix of matrices) {
        result[i] = result[i].concat(matrix[i]);
      }
    }
    
    return result;
  } else {
    throw new Error('轴参数必须是 0 或 1');
  }
}

/**
 * 矩阵分割
 * 将矩阵沿指定轴分割为多个子矩阵
 * 
 * @param {Array<Array<number>>} matrix - 输入矩阵
 * @param {number} numSplits - 分割数量
 * @param {number} axis - 分割轴，0为行方向，1为列方向
 * @returns {Array<Array<Array<number>>>} 分割后的矩阵数组
 */
function split(matrix, numSplits, axis = 0) {
  if (!Array.isArray(matrix)) {
    throw new Error('输入必须是二维数组');
  }
  
  const [rows, cols] = shape(matrix);
  const result = [];
  
  if (axis === 0) {
    // 按行分割
    const rowsPerSplit = Math.floor(rows / numSplits);
    for (let i = 0; i < numSplits; i++) {
      const start = i * rowsPerSplit;
      const end = i === numSplits - 1 ? rows : start + rowsPerSplit;
      result.push(matrix.slice(start, end));
    }
  } else if (axis === 1) {
    // 按列分割
    const colsPerSplit = Math.floor(cols / numSplits);
    for (let i = 0; i < numSplits; i++) {
      const start = i * colsPerSplit;
      const end = i === numSplits - 1 ? cols : start + colsPerSplit;
      const subMatrix = matrix.map(row => row.slice(start, end));
      result.push(subMatrix);
    }
  } else {
    throw new Error('轴参数必须是 0 或 1');
  }
  
  return result;
}

// 导出所有函数
module.exports = {
  matmul,
  transpose,
  reshape,
  add,
  subtract,
  scalarMultiply,
  scalarDivide,
  shape,
  concat,
  split
};
