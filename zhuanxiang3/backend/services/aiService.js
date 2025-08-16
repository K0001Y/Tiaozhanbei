const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));
const FormData = require('form-data');
const AI_CONFIG = require('../config/aiConfig');

class AIService {
  // 基础AI服务调用方法
  static async callAI(endpoint, data = null, options = {}) {
    const url = `${AI_CONFIG.BASE_URL}${endpoint}`;
    
    try {
      const config = {
        timeout: AI_CONFIG.TIMEOUT,
        ...options
      };

      let response;
      
      if (data instanceof FormData) {
        // 处理文件上传
        response = await fetch(url, {
          method: 'POST',
          body: data,
          ...config
        });
      } else if (data && options.method === 'POST') {
        // 处理JSON数据
        response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...config.headers
          },
          body: JSON.stringify(data),
          ...config
        });
      } else {
        // 处理GET请求
        response = await fetch(url, config);
      }

      if (!response.ok) {
        throw new Error(`AI服务响应错误: ${response.status} - ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('AI服务调用失败:', error);
      return {
        success: false,
        message: error.message.includes('timeout') ? 'AI服务响应超时，请稍后重试' : 'AI服务暂时不可用，请稍后重试',
        data: {}
      };
    }
  }

  // 重试机制
  static async callWithRetry(endpoint, data, options = {}, maxRetries = AI_CONFIG.RETRY_COUNT) {
    for (let i = 0; i < maxRetries; i++) {
      try {
        const result = await this.callAI(endpoint, data, options);
        if (result.success) {
          return result;
        }
        
        // 如果不是最后一次重试，等待后重试
        if (i < maxRetries - 1) {
          await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
        }
      } catch (error) {
        if (i === maxRetries - 1) {
          return {
            success: false,
            message: '服务多次尝试失败，请稍后重试',
            data: {}
          };
        }
        await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
      }
    }
  }

  // 数据格式转换方法
  static formatInquiryData(frontendData) {
    const { age, gender, symptoms, duration, severity, medicalHistory } = frontendData;
    
    return {
      symptoms: symptoms.trim(),
      duration: duration || '',
      severity: severity || 'medium',
      patientInfo: {
        age: parseInt(age) || 0,
        gender: gender || '',
        medicalHistory: medicalHistory || ''
      }
    };
  }

  static formatWatchData(description, imageBuffer, filename) {
    const formData = new FormData();
    formData.append('image', imageBuffer, {
      filename: filename,
      contentType: 'image/jpeg'
    });
    
    if (description && description.trim()) {
      formData.append('description', description.trim());
    }
    
    return formData;
  }

  static formatRecordData(frontendData) {
    const { patientInfo, watchResults, inquiryResults } = frontendData;
    
    return {
      patientInfo: patientInfo || '',
      watchResults: watchResults || '',
      inquiryResults: inquiryResults || ''
    };
  }

  // 响应格式标准化
  static standardizeResponse(aiResponse, successMessage = '操作成功') {
    if (aiResponse.success) {
      return {
        success: true,
        message: successMessage,
        data: aiResponse.data || {}
      };
    } else {
      return {
        success: false,
        message: aiResponse.message || '操作失败',
        data: {}
      };
    }
  }

  // 7.1 AI智能分析服务
  static async intelligentAnalyze(analysisRequest) {
    try {
      const { query, file } = analysisRequest;
      
      // 准备请求数据
      if (file) {
        // 如果有文件，使用FormData
        const fs = require('fs');
        const formData = new FormData();
        
        if (query) {
          formData.append('query', query);
        }
        
        // 添加文件
        formData.append('file', fs.createReadStream(file.path), {
          filename: file.originalname,
          contentType: file.mimetype
        });

        // 调用AI chatbot服务
        const aiResponse = await this.callAI('/api/ai/analyze', formData, {
          method: 'POST'
        });

        // 直接返回AI响应，不使用模拟数据
        if (!aiResponse.success) {
          throw new Error(aiResponse.message || 'AI分析服务调用失败');
        }

        return aiResponse.data;
      } else if (query) {
        // 纯文本查询
        const requestData = { query };
        
        const aiResponse = await this.callAI('/api/ai/analyze', requestData, {
          method: 'POST'
        });

        // 直接返回AI响应，不使用模拟数据
        if (!aiResponse.success) {
          throw new Error(aiResponse.message || 'AI分析服务调用失败');
        }

        return aiResponse.data;
      } else {
        throw new Error('query 和 file 参数至少需要提供一项');
      }
    } catch (error) {
      console.error('AI智能分析服务错误:', error);
      
      // 直接抛出错误，不使用模拟数据
      throw new Error(error.message || 'AI智能分析服务调用失败');
    }
  }

  // 生成模拟解决方案（用于测试和fallback）
  static generateMockSolution(query, fileName) {
    const timestamp = new Date().toLocaleString('zh-CN');
    
    if (fileName && query) {
      return `基于您的查询"${query}"和上传的文件"${fileName}"的分析结果：

🔍 **综合分析**：
经过AI智能分析，您提供的信息显示了明确的症状模式和特征。结合文件内容，我们可以提供以下专业建议。

📋 **初步评估**：
1. 症状特征分析 - 您描述的症状具有典型的表现特征
2. 文件内容解读 - 上传的文件提供了重要的补充信息
3. 综合判断 - 需要综合考虑多个因素

💡 **建议措施**：
1. **即时建议**：密切观察症状变化，记录详细情况
2. **生活调理**：注意休息，保持规律作息，均衡饮食
3. **医疗建议**：建议及时就医进行专业检查，获得准确诊断
4. **随访计划**：定期复查，监测病情变化

⚠️ **重要提醒**：本分析仅供参考，具体诊断和治疗方案请咨询专业医生。

📊 分析时间：${timestamp}`;
    } else if (query) {
      return `针对您的查询"${query}"的AI智能分析：

🔍 **症状分析**：
根据您描述的症状，可能涉及以下几个方面的问题。建议从多个角度进行综合评估。

📋 **可能原因**：
1. **生理因素** - 可能与身体机能变化相关
2. **环境因素** - 外部环境和生活习惯的影响
3. **心理因素** - 情绪和精神状态的作用

💡 **专业建议**：
1. **观察记录**：详细记录症状出现的时间、频率和诱发因素
2. **生活调整**：改善作息，保持健康的生活方式
3. **专业诊断**：建议尽快就医，进行必要的检查
4. **预防措施**：采取相应的预防和保健措施

⚠️ **注意事项**：
- 如症状持续或加重，请立即就医
- 本分析不能替代专业医疗诊断
- 请遵循医生的专业建议进行治疗

📊 分析时间：${timestamp}`;
    } else if (fileName) {
      return `对上传文件"${fileName}"的AI智能分析：

🔍 **文件解读**：
经过AI智能识别和分析，您上传的文件包含重要的医疗信息，以下是详细的解读结果。

📋 **内容分析**：
1. **关键信息提取** - 从文件中识别出的重要医疗数据
2. **数据解读** - 对各项指标和参数的专业解释
3. **趋势分析** - 数据变化趋势和意义

💡 **专业解读**：
1. **正常范围** - 哪些指标在正常范围内
2. **异常发现** - 需要关注的异常指标
3. **相关性分析** - 不同指标间的关联性
4. **建议措施** - 基于分析结果的专业建议

⚠️ **重要提醒**：
- 文件分析结果仅供参考
- 具体诊断需要专业医生综合判断
- 如有疑问请咨询主治医生

📊 分析时间：${timestamp}`;
    } else {
      return `AI智能分析服务当前可用，请提供查询内容或上传相关文件进行分析。

📋 **服务说明**：
1. **文本分析** - 支持症状描述、医疗问题咨询
2. **图像分析** - 支持医学图像、检查报告分析
3. **文档分析** - 支持PDF、TXT等医疗文档解读
4. **综合分析** - 支持文本+文件的组合分析

💡 **使用建议**：
- 详细描述您的症状或问题
- 上传清晰的图片或文档
- 提供相关的背景信息

📊 服务时间：${timestamp}`;
    }
  }
}

module.exports = AIService;

module.exports = AIService;
