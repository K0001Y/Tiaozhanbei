// API服务类 - 处理所有API调用
const API_BASE_URL = 'http://localhost:3000/api';

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
    return localStorage.getItem('token');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const token = this.getAuthToken();
    
    const defaultHeaders: HeadersInit = {
      'Content-Type': 'application/json',
    };

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
    analysisId: string;
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

  // 生成病历
  async generateRecord(recordData: {
    patientInfo: string;
    watchResults?: string;
    inquiryResults?: string;
  }): Promise<ApiResponse<{
    symptoms: string;
    disease: string;
    prescription: string;
  }>> {
    return this.request('/record', {
      method: 'POST',
      body: JSON.stringify(recordData),
    });
  }

  // 保存病历
  async saveRecord(formData: FormData): Promise<ApiResponse<void>> {
    const token = this.getAuthToken();
    
    try {
      const response = await fetch(`${API_BASE_URL}/record/save`, {
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
}

export const apiService = new ApiService();
export type { UserProfile, MedicalRecord, ApiResponse };
