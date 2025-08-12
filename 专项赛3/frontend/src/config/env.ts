// 环境配置文件 - 集中管理所有环境变量
export const config = {
  // API配置
  API_BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:5001',
  
  // 端口配置
  FRONTEND_PORT: parseInt(import.meta.env.VITE_FRONTEND_PORT || '3001'),
  BACKEND_PORT: parseInt(import.meta.env.VITE_BACKEND_PORT || '5001'),
  
  // 环境配置
  NODE_ENV: import.meta.env.VITE_NODE_ENV || 'development',
  
  // 开发模式检查
  isDevelopment: import.meta.env.VITE_NODE_ENV === 'development',
  isProduction: import.meta.env.VITE_NODE_ENV === 'production',
};

// 导出完整的API URL
export const API_ENDPOINTS = {
  BASE: `${config.API_BASE_URL}/api`,
  AUTH: `${config.API_BASE_URL}/api/auth`,
  LIBRARY: `${config.API_BASE_URL}/api/library`,
  SEARCH: `${config.API_BASE_URL}/api/search`,
  WATCH: `${config.API_BASE_URL}/api/watch`,
  INQUIRY: `${config.API_BASE_URL}/api/inquiry`,
  RECORD: `${config.API_BASE_URL}/api/record`,
};

export default config;
