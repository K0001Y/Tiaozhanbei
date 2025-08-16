// 环境配置
export const config = {
  // API基础URL
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001/api',
  
  // 应用配置
  APP_NAME: 'AI辅助诊疗系统',
  APP_VERSION: '1.0.0',
  
  // 开发模式
  IS_DEVELOPMENT: import.meta.env.DEV,
  IS_PRODUCTION: import.meta.env.PROD,
  
  // 认证配置
  TOKEN_STORAGE_KEY: 'medical_system_token',
  USER_STORAGE_KEY: 'medical_system_user',
  
  // 分页配置
  DEFAULT_PAGE_SIZE: 10,
  MAX_PAGE_SIZE: 100,
  
  // 文件上传配置
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  ALLOWED_FILE_TYPES: ['image/jpeg', 'image/png', 'image/gif', 'application/pdf'],
  
  // 密码验证规则
  PASSWORD_MIN_LENGTH: 6,
  PASSWORD_REGEX: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
  
  // 用户名验证规则
  USERNAME_MIN_LENGTH: 3,
  USERNAME_REGEX: /^[a-zA-Z0-9_]+$/,
};
