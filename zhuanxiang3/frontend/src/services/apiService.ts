// API服务类 - 处理所有API调用
import { API_ENDPOINTS } from '../config/env';

const API_BASE_URL = API_ENDPOINTS.BASE;

// Zustand persist存储类型定义
interface UserStorageData {
  state?: {
    token?: string | null;
    user?: Record<string, unknown> | null;
    isAuthenticated?: boolean;
  };
  token?: string | null; // 向后兼容
}

interface UserProfile {
  userId: number;
  username: string;
  name: string;
  age: number;
  gender: string;
  phone: string;
}

interface MedicalRecord {
  recordId: number;
  symptoms: string;
  disease: string;
  prescription: string;
  date?: string;
}

interface ApiResponse<T> {
  success: boolean;
  message: string;
  data?: T;
}

class ApiService {
  private getAuthToken(): string | null {
    try {
      // 从Zustand persist存储中获取token
      const userStorage = localStorage.getItem('user-storage');
      if (userStorage) {
        const parsedStorage: UserStorageData = JSON.parse(userStorage);
        
        // 支持Zustand persist的标准格式
        const token = parsedStorage.state?.token || parsedStorage.token;
        
        return token || null;
      }
      return null;
    } catch (error) {
      console.error('获取认证token失败:', error);
      return null;
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const token = this.getAuthToken();
    
    const defaultHeaders: HeadersInit = {};
    
    // 只有当body不是FormData时才设置Content-Type
    if (!(options.body instanceof FormData)) {
      defaultHeaders['Content-Type'] = 'application/json';
    }

    if (token) {
      defaultHeaders['Authorization'] = `Bearer ${token}`;
    }

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        headers: {
          ...defaultHeaders,
          ...options.headers,
        },
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || '请求失败');
      }

      return data;
    } catch (error) {
      console.error('API请求错误:', error);
      throw error;
    }
  }

  // 获取用户资料
  async getUserProfile(): Promise<ApiResponse<{ user: UserProfile }>> {
    return this.request<{ user: UserProfile }>('/auth/profile');
  }

  // 用户登录
  async login(username: string, password: string): Promise<ApiResponse<{
    user: UserProfile & { records: MedicalRecord[] };
    token: string;
  }>> {
    return this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
  }

  // 用户注册
  async register(userData: {
    username: string;
    password: string;
    name: string;
    age: number;
    gender: string;
    phone?: string;
  }): Promise<ApiResponse<void>> {
    return this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  // 疾病搜索
  async searchDiseases(keyword: string): Promise<ApiResponse<{
    diseases: Array<{
      diseaseId: number;
      diseaseName: string;
      description: string;
      source: string;
      relevance: string;
    }>;
  }>> {
    return this.request(`/search?search=${encodeURIComponent(keyword)}`);
  }

  // 图片望诊
  async analyzeImage(formData: FormData): Promise<ApiResponse<{
    results: string;
    imageId: string;
    imagePath: string;
  }>> {
    const token = this.getAuthToken();
    
    try {
      const response = await fetch(`${API_BASE_URL}/watch`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || '请求失败');
      }

      return data;
    } catch (error) {
      console.error('图片分析错误:', error);
      throw error;
    }
  }

  // 望诊补充
  async completeWatchAnalysis(formData: FormData): Promise<ApiResponse<{
    results: string;
    updatedAt: string;
  }>> {
    const token = this.getAuthToken();
    
    try {
      const response = await fetch(`${API_BASE_URL}/watch/complete`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || '请求失败');
      }

      return data;
    } catch (error) {
      console.error('望诊补充错误:', error);
      throw error;
    }
  }

  // 问诊分析
  async analyzeInquiry(inquiryData: {
    age: number;
    gender: string;
    symptoms: string;
  }): Promise<ApiResponse<{
    results: string;
    analysisId: string;
  }>> {
    return this.request('/inquiry', {
      method: 'POST',
      body: JSON.stringify(inquiryData),
    });
  }

  // 问诊补充
  async completeInquiry(formData: FormData): Promise<ApiResponse<{
    results: string;
    updatedAt: string;
  }>> {
    const token = this.getAuthToken();
    
    try {
      const response = await fetch(`${API_BASE_URL}/inquiry/complete`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || '请求失败');
      }

      return data;
    } catch (error) {
      console.error('问诊补充错误:', error);
      throw error;
    }
  }

  // 从病历图片导入生成 - 6.2接口
  async importRecord(imageFile: File): Promise<ApiResponse<{
    symptoms: string;
    disease: string;
    prescription: string;
  }>> {
    try {
      const formData = new FormData();
      formData.append('recordImage', imageFile);
      
      console.log('导入病历图片，文件:', imageFile.name);
      return await this.request('/record/import', {
        method: 'POST',
        body: formData
      });
    } catch (error) {
      console.error('导入病历图片错误:', error);
      throw error;
    }
  }

  // 生成病历 - 6.1接口
  async generateRecord(requestData: { watchResults?: string; inquiryResults?: string }): Promise<ApiResponse<{
    symptoms: string;
    disease: string;
    prescription: string;
  }>> {
    try {
      console.log('生成病历，参数:', requestData);
      return await this.request('/record', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });
    } catch (error) {
      console.error('生成病历错误:', error);
      throw error;
    }
  }

  // 保存病历
  async saveRecord(recordData: {
    symptoms: string;
    diagnosis?: string;
    prescription?: string;
  }): Promise<ApiResponse<{ message: string }>> {
    try {
      console.log('保存病历，参数:', recordData);
      return await this.request('/record/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(recordData),
      });
    } catch (error) {
      console.error('保存病历错误:', error);
      throw error;
    }
  }

  // 获取知识库列表
  async getLibraryList(page: number = 1, limit: number = 20): Promise<ApiResponse<{
    page: number;
    limit: number;
    total: number;
    libraries: Array<{
      libraryId: number;
      filePath: string;
      fileName: string;
      uploadTime: string;
    }>;
  }>> {
    return this.request(`/library?page=${page}&limit=${limit}`);
  }

  // 上传资料到知识库
  async uploadLibraryFile(formData: FormData): Promise<ApiResponse<void>> {
    const token = this.getAuthToken();
    
    try {
      const response = await fetch(`${API_BASE_URL}/library`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || '请求失败');
      }

      return data;
    } catch (error) {
      console.error('文件上传错误:', error);
      throw error;
    }
  }

  // 删除知识库资料
  async deleteLibraryFile(libraryId: number): Promise<ApiResponse<void>> {
    return this.request(`/library/${libraryId}`, {
      method: 'DELETE',
    });
  }

  // 获取病历历史
  async getRecordHistory(page: number = 1, limit: number = 10): Promise<ApiResponse<any>> {
    try {
      return await this.request(`/record?page=${page}&limit=${limit}`, {
        method: 'GET',
      });
    } catch (error) {
      console.error('获取病历历史错误:', error);
      throw error;
    }
  }

  // 获取病历详情
  async getRecordDetail(recordId: number): Promise<ApiResponse<any>> {
    try {
      return await this.request(`/record/${recordId}`, {
        method: 'GET',
      });
    } catch (error) {
      console.error('获取病历详情错误:', error);
      throw error;
    }
  }

  // 删除病历
  async deleteRecord(recordId: number): Promise<ApiResponse<void>> {
    try {
      return await this.request(`/record/${recordId}`, {
        method: 'DELETE',
      });
    } catch (error) {
      console.error('删除病历错误:', error);
      throw error;
    }
  }

  // 7.1 AI智能分析
  async aiAnalyze(formData: FormData): Promise<ApiResponse<{ solution: string }>> {
    try {
      return await this.request('/ai/analyze', {
        method: 'POST',
        body: formData,
        // 不设置Content-Type，让浏览器自动设置multipart/form-data边界
      });
    } catch (error) {
      console.error('AI智能分析错误:', error);
      throw error;
    }
  }
}

export const apiService = new ApiService();
export type { UserProfile, MedicalRecord, ApiResponse };
