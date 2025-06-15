/**
 * 常量定义文件
 * 定义 Transformer 模型中使用的各种常量
 * 
 * @author Transformer-JS
 * @version 1.0.0
 */

/**
 * 特殊 Token 常量
 * 用于序列处理的特殊标记
 */
const SPECIAL_TOKENS = {
  PAD: '<PAD>',           // 填充标记
  UNK: '<UNK>',           // 未知词标记
  BOS: '<BOS>',           // 序列开始标记
  EOS: '<EOS>',           // 序列结束标记
  MASK: '<MASK>',         // 掩码标记
  SEP: '<SEP>',           // 分隔符标记
  CLS: '<CLS>'            // 分类标记
};

/**
 * 特殊 Token ID
 * 对应特殊标记的数字 ID
 */
const SPECIAL_TOKEN_IDS = {
  PAD_ID: 0,
  UNK_ID: 1,
  BOS_ID: 2,
  EOS_ID: 3,
  MASK_ID: 4,
  SEP_ID: 5,
  CLS_ID: 6
};

/**
 * 激活函数类型
 */
const ACTIVATION_FUNCTIONS = {
  RELU: 'relu',
  GELU: 'gelu',
  SWISH: 'swish',
  TANH: 'tanh',
  SIGMOID: 'sigmoid'
};

/**
 * 注意力机制类型
 */
const ATTENTION_TYPES = {
  SELF_ATTENTION: 'self_attention',
  CROSS_ATTENTION: 'cross_attention',
  MASKED_ATTENTION: 'masked_attention',
  MULTI_HEAD_ATTENTION: 'multi_head_attention'
};

/**
 * 位置编码类型
 */
const POSITION_ENCODING_TYPES = {
  SINUSOIDAL: 'sinusoidal',     // 正弦余弦位置编码
  LEARNED: 'learned',           // 可学习位置编码
  RELATIVE: 'relative',         // 相对位置编码
  ROTARY: 'rotary'             // 旋转位置编码
};

/**
 * 层归一化类型
 */
const NORMALIZATION_TYPES = {
  LAYER_NORM: 'layer_norm',
  BATCH_NORM: 'batch_norm',
  RMS_NORM: 'rms_norm',
  GROUP_NORM: 'group_norm'
};

/**
 * 数学常量
 */
const MATH_CONSTANTS = {
  PI: Math.PI,
  E: Math.E,
  SQRT_2: Math.sqrt(2),
  SQRT_PI: Math.sqrt(Math.PI),
  LOG_2: Math.log(2),
  LOG_10: Math.log(10),
  EPSILON: 1e-8,              // 防止除零的小值
  LAYER_NORM_EPS: 1e-6,       // 层归一化 epsilon
  ATTENTION_SCALE_EPS: 1e-4   // 注意力缩放 epsilon
};

/**
 * 模型架构类型
 */
const MODEL_ARCHITECTURES = {
  TRANSFORMER: 'transformer',
  ENCODER_ONLY: 'encoder_only',     // 如 BERT
  DECODER_ONLY: 'decoder_only',     // 如 GPT
  ENCODER_DECODER: 'encoder_decoder' // 如原始 Transformer
};

/**
 * 训练模式
 */
const TRAINING_MODES = {
  TRAIN: 'train',
  EVAL: 'eval',
  INFERENCE: 'inference'
};

/**
 * 损失函数类型
 */
const LOSS_FUNCTIONS = {
  CROSS_ENTROPY: 'cross_entropy',
  MSE: 'mse',
  MAE: 'mae',
  HUBER: 'huber'
};

/**
 * 优化器类型
 */
const OPTIMIZERS = {
  SGD: 'sgd',
  ADAM: 'adam',
  ADAMW: 'adamw',
  RMSPROP: 'rmsprop'
};

/**
 * 学习率调度器类型
 */
const LR_SCHEDULERS = {
  CONSTANT: 'constant',
  LINEAR: 'linear',
  COSINE: 'cosine',
  EXPONENTIAL: 'exponential',
  STEP: 'step',
  WARMUP: 'warmup'
};

/**
 * 数据类型
 */
const DATA_TYPES = {
  FLOAT32: 'float32',
  FLOAT64: 'float64',
  INT32: 'int32',
  INT64: 'int64'
};

/**
 * 设备类型
 */
const DEVICE_TYPES = {
  CPU: 'cpu',
  GPU: 'gpu',
  TPU: 'tpu'
};

/**
 * 模型状态
 */
const MODEL_STATES = {
  INITIALIZED: 'initialized',
  TRAINING: 'training',
  TRAINED: 'trained',
  LOADING: 'loading',
  LOADED: 'loaded',
  ERROR: 'error'
};

/**
 * 错误类型
 */
const ERROR_TYPES = {
  DIMENSION_MISMATCH: 'dimension_mismatch',
  INVALID_CONFIG: 'invalid_config',
  INVALID_INPUT: 'invalid_input',
  MEMORY_ERROR: 'memory_error',
  COMPUTATION_ERROR: 'computation_error',
  IO_ERROR: 'io_error'
};

/**
 * 日志级别
 */
const LOG_LEVELS = {
  DEBUG: 'debug',
  INFO: 'info',
  WARN: 'warn',
  ERROR: 'error',
  FATAL: 'fatal'
};

/**
 * 文件格式
 */
const FILE_FORMATS = {
  JSON: 'json',
  BINARY: 'binary',
  TEXT: 'text',
  CSV: 'csv'
};

/**
 * 编码格式
 */
const ENCODINGS = {
  UTF8: 'utf8',
  ASCII: 'ascii',
  BASE64: 'base64',
  HEX: 'hex'
};

/**
 * 默认超参数
 */
const DEFAULT_HYPERPARAMS = {
  LEARNING_RATE: 1e-4,
  BATCH_SIZE: 32,
  EPOCHS: 100,
  WARMUP_STEPS: 4000,
  WEIGHT_DECAY: 0.01,
  GRADIENT_CLIP: 1.0,
  DROPOUT: 0.1,
  ATTENTION_DROPOUT: 0.1
};

/**
 * 性能配置
 */
const PERFORMANCE_CONFIG = {
  MAX_BATCH_SIZE: 64,
  MAX_SEQUENCE_LENGTH: 1024,
  MEMORY_LIMIT_MB: 512,
  COMPUTATION_TIMEOUT_MS: 30000,
  CACHE_SIZE_LIMIT: 100
};

/**
 * 微信小程序特定配置
 */
const MINIPROGRAM_CONFIG = {
  MAX_MEMORY_MB: 256,
  MAX_COMPUTATION_TIME_MS: 10000,
  RECOMMENDED_BATCH_SIZE: 8,
  RECOMMENDED_SEQ_LENGTH: 64,
  STORAGE_LIMIT_MB: 10
};

/**
 * 版本信息
 */
const VERSION_INFO = {
  MAJOR: 1,
  MINOR: 0,
  PATCH: 0,
  BUILD: '20240101',
  VERSION_STRING: '1.0.0'
};

// 导出所有常量
module.exports = {
  SPECIAL_TOKENS,
  SPECIAL_TOKEN_IDS,
  ACTIVATION_FUNCTIONS,
  ATTENTION_TYPES,
  POSITION_ENCODING_TYPES,
  NORMALIZATION_TYPES,
  MATH_CONSTANTS,
  MODEL_ARCHITECTURES,
  TRAINING_MODES,
  LOSS_FUNCTIONS,
  OPTIMIZERS,
  LR_SCHEDULERS,
  DATA_TYPES,
  DEVICE_TYPES,
  MODEL_STATES,
  ERROR_TYPES,
  LOG_LEVELS,
  FILE_FORMATS,
  ENCODINGS,
  DEFAULT_HYPERPARAMS,
  PERFORMANCE_CONFIG,
  MINIPROGRAM_CONFIG,
  VERSION_INFO
};
