// 路由常量
export const ROUTES = {
  HOME: '/',
  DASHBOARD: '/dashboard',
  MEDICAL: '/medical',
  LOGIN: '/login',
  REGISTER: '/register',
  AUTH_LOGIN: '/auth/login',
  AUTH_REGISTER: '/auth/register',
} as const;

// 医疗系统模块常量
export const MEDICAL_MODULES = {
  DASHBOARD: 'dashboard',
  KNOWLEDGE: 'knowledge',
  SEARCH: 'search',
  DIAGNOSIS: 'diagnosis',
  RECORD: 'record',
  INQUIRY: 'inquiry',
  AI_ASSIST: 'ai-assist',
} as const;

// API端点常量
export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    REGISTER: '/auth/register',
    LOGOUT: '/auth/logout',
    REFRESH: '/auth/refresh',
    VERIFY: '/auth/verify',
  },
  USER: {
    PROFILE: '/user/profile',
    UPDATE: '/user/update',
  },
  MEDICAL: {
    SEARCH: '/medical/search',
    DIAGNOSIS: '/medical/diagnosis',
    RECORDS: '/medical/records',
  },
} as const;

// 存储键常量
export const STORAGE_KEYS = {
  TOKEN: 'medical_system_token',
  USER: 'medical_system_user',
  THEME: 'medical_system_theme',
  LANGUAGE: 'medical_system_language',
} as const;

// 错误消息常量
export const ERROR_MESSAGES = {
  NETWORK_ERROR: '网络连接错误，请检查网络设置',
  UNAUTHORIZED: '登录已过期，请重新登录',
  FORBIDDEN: '权限不足，无法执行此操作',
  NOT_FOUND: '请求的资源不存在',
  SERVER_ERROR: '服务器内部错误，请稍后重试',
  VALIDATION_ERROR: '输入数据格式错误',
  UNKNOWN_ERROR: '未知错误，请联系管理员',
} as const;

// 成功消息常量
export const SUCCESS_MESSAGES = {
  LOGIN_SUCCESS: '登录成功',
  REGISTER_SUCCESS: '注册成功',
  LOGOUT_SUCCESS: '退出登录成功',
  SAVE_SUCCESS: '保存成功',
  UPDATE_SUCCESS: '更新成功',
  DELETE_SUCCESS: '删除成功',
} as const;
