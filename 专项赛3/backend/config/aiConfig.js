// AI模型服务配置
const AI_CONFIG = {
  BASE_URL: process.env.AI_CHATBOT_URL || 'http://localhost:8080',
  TIMEOUT: parseInt(process.env.AI_CHATBOT_TIMEOUT) || 30000,
  RETRY_COUNT: 3,
  RETRY_DELAY: 1000, // 重试间隔1秒
  
  // 接口路径
  ENDPOINTS: {
    SEARCH: '/api/search',
    WATCH: '/api/watch',
    WATCH_COMPLETE: '/api/watch/complete',
    INQUIRY: '/api/inquiry', 
    INQUIRY_COMPLETE: '/api/inquiry/complete',
    RECORD: '/api/record',
    IMPORT: '/api/import',
    CHAT: '/api/chat',
    ANALYZE: '/api/analyze',
    GENERATE_RECORD: '/api/generate-record',
    EXTRACT_RECORD: '/api/extract-record'
  },
  
  // 请求头配置
  HEADERS: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'User-Agent': 'Medical-System/1.0'
  },
  
  // 文件上传配置
  UPLOAD: {
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_TYPES: [
      'image/jpeg',
      'image/png', 
      'image/gif',
      'image/webp',
      'application/pdf'
    ]
  },
  
  // 错误消息配置
  ERROR_MESSAGES: {
    TIMEOUT: 'AI服务响应超时，请稍后重试',
    CONNECTION_ERROR: 'AI服务连接失败，请检查网络连接',
    INVALID_RESPONSE: 'AI服务返回数据格式错误',
    FILE_TOO_LARGE: '文件大小超出限制（最大10MB）',
    UNSUPPORTED_FILE: '不支持的文件格式',
    RATE_LIMIT: 'AI服务请求频率过高，请稍后重试',
    SERVER_ERROR: 'AI服务内部错误，请稍后重试'
  },
  
  // 默认参数配置
  DEFAULTS: {
    CONFIDENCE_THRESHOLD: 0.6,
    MAX_RESULTS: 5,
    ANALYSIS_DEPTH: 'standard'
  }
};

module.exports = AI_CONFIG;
