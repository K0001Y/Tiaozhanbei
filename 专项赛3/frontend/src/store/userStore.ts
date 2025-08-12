import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authService } from '../services/authService';
import { User, LoginRequest, RegisterRequest } from '../types/auth';

// 错误响应类型
interface ErrorResponse {
  response?: {
    data?: {
      message?: string;
      errors?: Array<{
        param?: string;
        msg?: string;
      }>;
    };
  };
}

interface UserState {
  // 状态
  user: User | null;
  token: string | null;
  isLoading: boolean;
  error: string | null;
  fieldErrors: Record<string, string>;
  isAuthenticated: boolean;

  // 登录相关方法
  login: (data: LoginRequest) => Promise<boolean>;
  
  // 注册相关方法
  register: (data: RegisterRequest) => Promise<boolean>;
  
  // 用户信息相关方法
  getUserProfile: () => Promise<boolean>;
  
  // 通用方法
  logout: () => Promise<void>;
  clearError: () => void;
  clearFieldErrors: () => void;
  setFieldError: (field: string, message: string) => void;
  verifyToken: () => Promise<boolean>;
  
  // 重置状态
  reset: () => void;
}

const initialState = {
  user: null,
  token: null,
  isLoading: false,
  error: null,
  fieldErrors: {},
  isAuthenticated: false,
};

export const useUserStore = create<UserState>()(
  persist(
    (set, get) => ({
      ...initialState,

      // 用户登录
      login: async (data: LoginRequest): Promise<boolean> => {
        set({ isLoading: true, error: null, fieldErrors: {} });

        try {
          const response = await authService.login(data);
          
          if (response.success) {
            const { token, user } = response.data;
            
            set({
              user,
              token,
              isAuthenticated: true,
              isLoading: false,
              error: null
            });
            
            return true;
          }
          
          return false;
        } catch (err: unknown) {
          const errorData = (err as ErrorResponse)?.response?.data;
          let errorMessage = '登录失败，请重试';
          let fieldErrors = {};

          if (errorData?.errors && Array.isArray(errorData.errors)) {
            // 处理字段级错误
            const newFieldErrors: Record<string, string> = {};
            errorData.errors.forEach((error: { param?: string; msg?: string }) => {
              if (error.param) {
                newFieldErrors[error.param] = error.msg || '输入有误';
              }
            });
            fieldErrors = newFieldErrors;
          } else if (errorData?.message) {
            errorMessage = errorData.message;
          }

          set({
            isLoading: false,
            error: errorMessage,
            fieldErrors,
            isAuthenticated: false
          });

          return false;
        }
      },

      // 用户注册
      register: async (data: RegisterRequest): Promise<boolean> => {
        set({ isLoading: true, error: null, fieldErrors: {} });

        try {
          const response = await authService.register(data);
          
          if (response.success) {
            set({
              isLoading: false,
              error: null
            });
            
            return true;
          }
          
          return false;
        } catch (err: unknown) {
          const errorData = (err as ErrorResponse)?.response?.data;
          let errorMessage = '注册失败，请重试';
          let fieldErrors = {};

          if (errorData?.errors && Array.isArray(errorData.errors)) {
            // 处理字段级错误
            const newFieldErrors: Record<string, string> = {};
            errorData.errors.forEach((error: { param?: string; msg?: string }) => {
              if (error.param) {
                newFieldErrors[error.param] = error.msg || '输入有误';
              }
            });
            fieldErrors = newFieldErrors;
          } else if (errorData?.message) {
            errorMessage = errorData.message;
          }

          set({
            isLoading: false,
            error: errorMessage,
            fieldErrors,
            isAuthenticated: false
          });

          return false;
        }
      },

      // 验证 Token
      verifyToken: async (): Promise<boolean> => {
        const { token } = get();
        
        if (!token) {
          set({ isAuthenticated: false });
          return false;
        }

        try {
          const response = await authService.verifyToken();
          
          if (response.success) {
            set({
              user: response.data.user,
              isAuthenticated: true,
              error: null
            });
            return true;
          }
          
          // Token 无效，清除状态
          await get().logout();
          return false;
        } catch (err: unknown) {
          // Token 验证失败，清除状态
          console.error('Token验证失败:', err);
          await get().logout();
          return false;
        }
      },

      // 获取用户资料
      getUserProfile: async (): Promise<boolean> => {
        const { token } = get();
        
        if (!token) {
          return false;
        }

        try {
          const response = await authService.getProfile();
          
          if (response.success) {
            set({
              user: response.data.user,
              error: null
            });
            return true;
          }
          
          return false;
        } catch (err: unknown) {
          console.error('获取用户资料失败:', err);
          return false;
        }
      },

      // 用户登出
      logout: async (): Promise<void> => {
        // 重置状态
        set(initialState);
        
        // 手动清理localStorage中的持久化数据
        localStorage.removeItem('user-storage');
        
        // 跳转到登录页
        window.location.href = '/login';
      },

      // 清除错误信息
      clearError: () => {
        set({ error: null });
      },

      // 清除字段错误
      clearFieldErrors: () => {
        set({ fieldErrors: {} });
      },

      // 设置字段错误
      setFieldError: (field: string, message: string) => {
        set(state => ({
          fieldErrors: {
            ...state.fieldErrors,
            [field]: message
          }
        }));
      },

      // 重置所有状态
      reset: () => {
        set(initialState);
      }
    }),
    {
      name: 'user-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated
      })
    }
  )
);