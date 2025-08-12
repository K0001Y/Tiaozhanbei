import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// 望诊结果类型
interface DiagnosisResult {
  imageUrl?: string;
  description: string;
  analysisReport: string;
  supplements: Array<{
    description: string;
    analysis: string;
    timestamp: string;
  }>;
  timestamp: string;
}

// 问诊结果类型
interface InquiryResult {
  symptoms: string;
  analysisReport: string;
  supplements: Array<{
    additionalInfo: string;
    analysis: string;
    timestamp: string;
  }>;
  timestamp: string;
}

// 疾病搜索历史类型
interface SearchHistory {
  id: string;
  keyword: string;
  results: Array<{
    name: string;
    description: string;
    symptoms: string[];
    treatments: string[];
    relevance: number;
  }>;
  timestamp: string;
}

interface MedicalState {
  // 当前分析结果
  currentDiagnosis: DiagnosisResult | null;
  currentInquiry: InquiryResult | null;
  
  // 历史记录
  diagnosisHistory: DiagnosisResult[];
  inquiryHistory: InquiryResult[];
  searchHistory: SearchHistory[];
  
  // 状态管理
  isLoading: boolean;
  error: string | null;
  
  // 望诊相关方法
  setDiagnosisResult: (result: DiagnosisResult) => void;
  addDiagnosisSupplement: (supplement: DiagnosisResult['supplements'][0]) => void;
  clearCurrentDiagnosis: () => void;
  
  // 问诊相关方法
  setInquiryResult: (result: InquiryResult) => void;
  addInquirySupplement: (supplement: InquiryResult['supplements'][0]) => void;
  clearCurrentInquiry: () => void;
  
  // 搜索历史方法
  addSearchHistory: (search: SearchHistory) => void;
  clearSearchHistory: () => void;
  
  // 通用方法
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
  reset: () => void;
  
  // 获取当前完整医疗记录（用于生成病历）
  getCurrentMedicalRecord: () => {
    diagnosis: DiagnosisResult | null;
    inquiry: InquiryResult | null;
    searchHistory: SearchHistory[];
  };
}

const initialState = {
  currentDiagnosis: null,
  currentInquiry: null,
  diagnosisHistory: [],
  inquiryHistory: [],
  searchHistory: [],
  isLoading: false,
  error: null,
};

export const useMedicalStore = create<MedicalState>()(
  persist(
    (set, get) => ({
      ...initialState,

      // 设置望诊结果
      setDiagnosisResult: (result: DiagnosisResult) => {
        set(state => ({
          currentDiagnosis: result,
          diagnosisHistory: [result, ...state.diagnosisHistory.slice(0, 9)] // 保留最近10条
        }));
      },

      // 添加望诊补充
      addDiagnosisSupplement: (supplement: DiagnosisResult['supplements'][0]) => {
        set(state => {
          if (!state.currentDiagnosis) return state;
          
          const updatedDiagnosis = {
            ...state.currentDiagnosis,
            supplements: [...state.currentDiagnosis.supplements, supplement]
          };
          
          return {
            currentDiagnosis: updatedDiagnosis,
            diagnosisHistory: [updatedDiagnosis, ...state.diagnosisHistory.slice(1)] // 替换第一条记录
          };
        });
      },

      // 清除当前望诊结果
      clearCurrentDiagnosis: () => {
        set({ currentDiagnosis: null });
      },

      // 设置问诊结果
      setInquiryResult: (result: InquiryResult) => {
        set(state => ({
          currentInquiry: result,
          inquiryHistory: [result, ...state.inquiryHistory.slice(0, 9)] // 保留最近10条
        }));
      },

      // 添加问诊补充
      addInquirySupplement: (supplement: InquiryResult['supplements'][0]) => {
        set(state => {
          if (!state.currentInquiry) return state;
          
          const updatedInquiry = {
            ...state.currentInquiry,
            supplements: [...state.currentInquiry.supplements, supplement]
          };
          
          return {
            currentInquiry: updatedInquiry,
            inquiryHistory: [updatedInquiry, ...state.inquiryHistory.slice(1)] // 替换第一条记录
          };
        });
      },

      // 清除当前问诊结果
      clearCurrentInquiry: () => {
        set({ currentInquiry: null });
      },

      // 添加搜索历史
      addSearchHistory: (search: SearchHistory) => {
        set(state => ({
          searchHistory: [search, ...state.searchHistory.slice(0, 19)] // 保留最近20条
        }));
      },

      // 清除搜索历史
      clearSearchHistory: () => {
        set({ searchHistory: [] });
      },

      // 设置加载状态
      setLoading: (loading: boolean) => {
        set({ isLoading: loading });
      },

      // 设置错误
      setError: (error: string | null) => {
        set({ error });
      },

      // 清除错误
      clearError: () => {
        set({ error: null });
      },

      // 重置所有状态
      reset: () => {
        set(initialState);
      },

      // 获取当前完整医疗记录
      getCurrentMedicalRecord: () => {
        const state = get();
        return {
          diagnosis: state.currentDiagnosis,
          inquiry: state.currentInquiry,
          searchHistory: state.searchHistory
        };
      }
    }),
    {
      name: 'medical-storage',
      partialize: (state) => ({
        currentDiagnosis: state.currentDiagnosis,
        currentInquiry: state.currentInquiry,
        diagnosisHistory: state.diagnosisHistory,
        inquiryHistory: state.inquiryHistory,
        searchHistory: state.searchHistory
      })
    }
  )
);
