// AI服务接口 - 严格按照API文档
import { API_BASE_URL } from './authService';

interface SearchRequest {
  search: string;
}

interface InquiryRequest {
  symptoms: string;
  duration?: string;
  severity?: string;
  additional_info?: string;
}

interface WatchRequest {
  image: File;
  description?: string;
}

interface RecordGenerateRequest {
  symptoms: string;
  diagnosis_info?: string;
  treatment_history?: string;
  current_medications?: string;
  vital_signs?: any;
}

interface AIAnalyzeRequest {
  query?: string;
  file?: File;
}

class AIService {
  private static instance: AIService;
  private baseURL: string;

  private constructor() {
    this.baseURL = API_BASE_URL;
  }

  static getInstance(): AIService {
    if (!AIService.instance) {
      AIService.instance = new AIService();
    }
    return AIService.instance;
  }

  private async makeRequest(url: string, options: RequestInit = {}) {
    const token = localStorage.getItem('token');
    
    const defaultHeaders = {
      'Authorization': `Bearer ${token}`,
      ...options.headers,
    };

    // 只有在不是FormData时才设置Content-Type
    if (!(options.body instanceof FormData)) {
      defaultHeaders['Content-Type'] = 'application/json';
    }

    try {
      const response = await fetch(`${this.baseURL}${url}`, {
        ...options,
        headers: defaultHeaders,
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || '请求失败');
      }

      return data;
    } catch (error) {
      console.error('AI服务请求错误:', error);
      throw error instanceof Error ? error : new Error('网络请求失败');
    }
  }

  // 3.1 疾病搜索 - GET /api/search
  async searchDiseases(request: SearchRequest) {
    const params = new URLSearchParams({ search: request.search });
    return this.makeRequest(`/api/search?${params}`, {
      method: 'GET'
    });
  }

  // 4.1 图片望诊 - POST /api/watch
  async analyzeImage(request: WatchRequest) {
    const formData = new FormData();
    formData.append('image', request.image);
    if (request.description) {
      formData.append('description', request.description);
    }

    return this.makeRequest('/api/watch', {
      method: 'POST',
      body: formData
    });
  }

  // 4.2 望诊补充 - POST /api/watch/complete
  async completeWatchAnalysis(originalAnalysisId: string, additionalDescription?: string, additionalFile?: File) {
    const formData = new FormData();
    formData.append('originalAnalysisId', originalAnalysisId);
    if (additionalDescription) {
      formData.append('additionalDescription', additionalDescription);
    }
    if (additionalFile) {
      formData.append('additionalFile', additionalFile);
    }

    return this.makeRequest('/api/watch/complete', {
      method: 'POST',
      body: formData
    });
  }

  // 5.1 初步问诊 - POST /api/inquiry
  async initialInquiry(request: InquiryRequest) {
    return this.makeRequest('/api/inquiry', {
      method: 'POST',
      body: JSON.stringify(request)
    });
  }

  // 5.2 问诊补充 - POST /api/inquiry/complete
  async completeInquiry(originalInquiryId: string, additionalSymptoms?: string, additionalInfo?: string, additionalFile?: File) {
    const formData = new FormData();
    formData.append('originalInquiryId', originalInquiryId);
    if (additionalSymptoms) {
      formData.append('additionalSymptoms', additionalSymptoms);
    }
    if (additionalInfo) {
      formData.append('additionalInfo', additionalInfo);
    }
    if (additionalFile) {
      formData.append('additionalFile', additionalFile);
    }

    return this.makeRequest('/api/inquiry/complete', {
      method: 'POST',
      body: formData
    });
  }

  // 6.1 生成病历 - POST /api/record/generate
  async generateRecord(request: RecordGenerateRequest) {
    return this.makeRequest('/api/record/generate', {
      method: 'POST',
      body: JSON.stringify(request)
    });
  }

  // 6.2 导入病历 - POST /api/record/import
  async importRecord(recordImage: File) {
    const formData = new FormData();
    formData.append('recordImage', recordImage);

    return this.makeRequest('/api/record/import', {
      method: 'POST',
      body: formData
    });
  }

  // 保存病历 - POST /api/record
  async saveRecord(recordData: Record<string, unknown>) {
    return this.makeRequest('/api/record', {
      method: 'POST',
      body: JSON.stringify(recordData)
    });
  }

  // 获取病历历史 - GET /api/record
  async getRecordHistory() {
    return this.makeRequest('/api/record', {
      method: 'GET'
    });
  }

  // 获取病历详情 - GET /api/record/:id
  async getRecordDetail(id: string) {
    return this.makeRequest(`/api/record/${id}`, {
      method: 'GET'
    });
  }

  // 删除病历 - DELETE /api/record/:id
  async deleteRecord(id: string) {
    return this.makeRequest(`/api/record/${id}`, {
      method: 'DELETE'
    });
  }

  // 7.1 AI智能分析 - POST /api/ai/analyze
  async intelligentAnalyze(request: AIAnalyzeRequest) {
    // 如果有文件，使用FormData
    if (request.file) {
      const formData = new FormData();
      if (request.query) {
        formData.append('query', request.query);
      }
      formData.append('file', request.file);

      return this.makeRequest('/api/ai/analyze', {
        method: 'POST',
        body: formData
      });
    } else if (request.query) {
      // 纯文本查询使用JSON
      return this.makeRequest('/api/ai/analyze', {
        method: 'POST',
        body: JSON.stringify({ query: request.query })
      });
    } else {
      throw new Error('query 和 file 参数至少需要提供一项');
    }
  }
}

export default AIService.getInstance();

// 导出接口类型
export type {
  SearchRequest,
  InquiryRequest,
  WatchRequest,
  RecordGenerateRequest,
  AIAnalyzeRequest
};
