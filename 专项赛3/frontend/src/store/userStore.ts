import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authService } from '../services/authService';
import { User, LoginRequest, RegisterRequest } from '../types/auth';

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
  
  // 通用方法
  logout: () => void;
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

            // 保存到 localStorage
            localStorage.setItem('token', token);
            localStorage.setItem('user', JSON.stringify(user));
            
            return true;
          }
          
          return false;
        } catch (err: any) {
          const errorData = err.response?.data;
          let errorMessage = '登录失败，请重试';
          let fieldErrors = {};

          if (errorData?.errors && Array.isArray(errorData.errors)) {
            // 处理字段级错误
            const newFieldErrors: Record<string, string> = {};
            errorData.errors.forEach((error: any) => {
              if (error.param) {
                newFieldErrors[error.param] = error.msg;
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
            const { token, user } = response.data;
            
            set({
              user,
              token,
              isAuthenticated: true,
              isLoading: false,
              error: null
            });

            // 保存到 localStorage
            localStorage.setItem('token', token);
            localStorage.setItem('user', JSON.stringify(user));
            
            return true;
          }
          
          return false;
        } catch (err: any) {
          const errorData = err.response?.data;
          let errorMessage = '注册失败，请重试';
          let fieldErrors = {};

          if (errorData?.errors && Array.isArray(errorData.errors)) {
            // 处理字段级错误
            const newFieldErrors: Record<string, string> = {};
            errorData.errors.forEach((error: any) => {
              if (error.param) {
                newFieldErrors[error.param] = error.msg;
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
          get().logout();
          return false;
        } catch (err) {
          // Token 验证失败，清除状态
          get().logout();
          return false;
        }
      },

      // 用户登出
      logout: () => {
        // 清除 localStorage
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        
        // 重置状态
        set(initialState);
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