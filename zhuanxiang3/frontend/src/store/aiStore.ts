// AI相关数据状态管理
import { create } from 'zustand';

interface AIAnalysisResult {
  id: string;
  type: 'search' | 'inquiry' | 'watch' | 'record' | 'analyze';
  timestamp: string;
  data: any;
  confidence?: number;
}

interface AIState {
  // 分析历史记录
  analysisHistory: AIAnalysisResult[];
  
  // 当前分析状态
  isAnalyzing: boolean;
  currentAnalysis: AIAnalysisResult | null;
  
  // 错误状态
  error: string | null;
  
  // 搜索相关
  searchResults: any[];
  searchQuery: string;
  
  // 问诊相关
  inquiryData: {
    symptoms: string;
    followUpQuestions: string[];
    diagnosis: string[];
    recommendations: string[];
  };
  
  // 望诊相关
  watchAnalysis: {
    analysis: string;
    suggestions: string[];
    confidence: number;
  } | null;
  
  // 病历相关
  recordData: {
    generated: any;
    extracted: any;
  };

  // Actions
  setAnalyzing: (analyzing: boolean) => void;
  setError: (error: string | null) => void;
  addAnalysisResult: (result: AIAnalysisResult) => void;
  setCurrentAnalysis: (analysis: AIAnalysisResult | null) => void;
  
  // 搜索相关操作
  setSearchResults: (results: any[]) => void;
  setSearchQuery: (query: string) => void;
  clearSearchResults: () => void;
  
  // 问诊相关操作
  setInquiryData: (data: Partial<AIState['inquiryData']>) => void;
  clearInquiryData: () => void;
  
  // 望诊相关操作
  setWatchAnalysis: (analysis: AIState['watchAnalysis']) => void;
  clearWatchAnalysis: () => void;
  
  // 病历相关操作
  setRecordData: (data: Partial<AIState['recordData']>) => void;
  clearRecordData: () => void;
  
  // 清理所有数据
  clearAllData: () => void;
}

export const useAIStore = create<AIState>((set, get) => ({
  // 初始状态
  analysisHistory: [],
  isAnalyzing: false,
  currentAnalysis: null,
  error: null,
  searchResults: [],
  searchQuery: '',
  inquiryData: {
    symptoms: '',
    followUpQuestions: [],
    diagnosis: [],
    recommendations: []
  },
  watchAnalysis: null,
  recordData: {
    generated: null,
    extracted: null
  },

  // 基础操作
  setAnalyzing: (analyzing) => set({ isAnalyzing: analyzing }),
  
  setError: (error) => set({ error }),
  
  addAnalysisResult: (result) => set((state) => ({
    analysisHistory: [result, ...state.analysisHistory.slice(0, 49)] // 保留最近50条记录
  })),
  
  setCurrentAnalysis: (analysis) => set({ currentAnalysis: analysis }),

  // 搜索相关操作
  setSearchResults: (results) => set({ searchResults: results }),
  
  setSearchQuery: (query) => set({ searchQuery: query }),
  
  clearSearchResults: () => set({ 
    searchResults: [], 
    searchQuery: '' 
  }),

  // 问诊相关操作
  setInquiryData: (data) => set((state) => ({
    inquiryData: { ...state.inquiryData, ...data }
  })),
  
  clearInquiryData: () => set({
    inquiryData: {
      symptoms: '',
      followUpQuestions: [],
      diagnosis: [],
      recommendations: []
    }
  }),

  // 望诊相关操作
  setWatchAnalysis: (analysis) => set({ watchAnalysis: analysis }),
  
  clearWatchAnalysis: () => set({ watchAnalysis: null }),

  // 病历相关操作
  setRecordData: (data) => set((state) => ({
    recordData: { ...state.recordData, ...data }
  })),
  
  clearRecordData: () => set({
    recordData: {
      generated: null,
      extracted: null
    }
  }),

  // 清理所有数据
  clearAllData: () => set({
    analysisHistory: [],
    isAnalyzing: false,
    currentAnalysis: null,
    error: null,
    searchResults: [],
    searchQuery: '',
    inquiryData: {
      symptoms: '',
      followUpQuestions: [],
      diagnosis: [],
      recommendations: []
    },
    watchAnalysis: null,
    recordData: {
      generated: null,
      extracted: null
    }
  })
}));

// 导出类型
export type { AIAnalysisResult, AIState };
