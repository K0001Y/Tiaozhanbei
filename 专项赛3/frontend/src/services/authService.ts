import axios from 'axios';
import { LoginRequest, RegisterRequest, AuthResponse } from '../types/auth';

const API_BASE_URL = import.meta.env.REACT_APP_API_URL || 'http://localhost:3000/api';

const authAPI = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

// 请求拦截器 - 添加 token
authAPI.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// 响应拦截器 - 处理错误
authAPI.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const authService = {
  // 用户登录
  login: async (data: LoginRequest): Promise<AuthResponse> => {
    const response = await authAPI.post('/auth/login', data);
    return response.data;
  },

  // 用户注册
  register: async (data: RegisterRequest): Promise<AuthResponse> => {
    const response = await authAPI.post('/auth/register', data);
    return response.data;
  },

  // 验证 token
  verifyToken: async (): Promise<AuthResponse> => {
    const response = await authAPI.get('/auth/verify');
    return response.data;
  },

  // 获取用户信息
  getProfile: async (): Promise<AuthResponse> => {
    const response = await authAPI.get('/auth/profile');
    return response.data;
  }
};