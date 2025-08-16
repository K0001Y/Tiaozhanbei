const { pool } = require('../config/database');

// 5.1 初步问诊
const initialInquiry = async (req, res) => {
  try {
    console.log(`${new Date().toISOString()} - POST /api/inquiry`);
    
    const { age, gender, symptoms } = req.body;

    console.log('初步问诊请求数据:', { age, gender, symptoms });

    // 验证必需参数
    if (!symptoms || !symptoms.trim()) {
      return res.status(400).json({
        success: false,
        message: '请提供症状描述'
      });
    }

    // TODO: 调用大模型进行问诊分析
    // 以下是占位实现，实际应该调用AI模型
    /*
    const aiAnalysis = await callAIInquiryModel({
      patientInfo: { age, gender },
      symptoms: symptoms
    });
    */

    // 模拟AI分析结果
    const mockAnalysis = generateMockInquiryAnalysis(age, gender, symptoms);

    console.log('问诊分析完成');

    res.json({
      success: true,
      message: '问诊分析成功',
      data: {
        results: mockAnalysis,
        analysisId: `inquiry_${Date.now()}`, // 用于后续补充分析
        inquiryId: `inquiry_${Date.now()}`, // 保持兼容性
        analysisTime: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('问诊分析失败:', error);
    res.status(500).json({
      success: false,
      message: '问诊分析失败，请稍后重试',
      error: error.message
    });
  }
};

// 5.2 问诊补充
const completeInquiry = async (req, res) => {
  try {
    console.log(`${new Date().toISOString()} - POST /api/inquiry/complete`);
    
    const { analysisId, additionalInfo, prevInquiry } = req.body;
    const additionalFile = req.file;

    console.log('问诊补充请求数据:', { 
      analysisId,
      additionalInfo, 
      hasAdditionalFile: !!additionalFile 
    });

    // 验证必需参数
    if (!analysisId || !additionalInfo) {
      return res.status(400).json({
        success: false,
        message: '请提供问诊ID和补充信息'
      });
    }

    let additionalFilePath = null;
    if (additionalFile) {
      additionalFilePath = additionalFile.path;
    }

    // TODO: 调用大模型进行补充分析
    // 以下是占位实现，实际应该调用AI模型
    /*
    const updatedAnalysis = await callAISupplementInquiry({
      analysisId: analysisId,
      additionalInfo: additionalInfo,
      additionalFilePath: additionalFilePath
    });
    */

    // 模拟补充分析结果
    const mockUpdatedAnalysis = generateMockSupplementInquiry(analysisId, additionalInfo);

    console.log('问诊补充分析完成');

    res.json({
      success: true,
      message: '补充问诊信息成功',
      data: {
        results: mockUpdatedAnalysis,
        updatedAt: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('问诊补充失败:', error);
    res.status(500).json({
      success: false,
      message: '问诊补充失败，请稍后重试',
      error: error.message
    });
  }
};

// 生成模拟问诊分析结果
function generateMockInquiryAnalysis(age, gender, symptoms) {
  const baseAnalysis = `根据您提供的症状信息分析：

【基本信息】
年龄：${age || '未提供'}岁
性别：${gender || '未提供'}

【症状概述】
主要症状：${symptoms}

【初步分析】
1. 根据症状描述，结合年龄和性别特点进行分析
2. 建议关注症状的发展趋势和伴随症状
3. 从中医角度来看，此类症状可能与气血运行、脏腑功能等相关

【建议】
- 观察症状变化规律
- 注意休息和饮食调理
- 如症状持续或加重，建议及时就医
- 可考虑结合舌诊、脉诊等进行综合判断

【免责声明】
此分析仅供参考，不能替代专业医生的诊断，请及时就医获得专业诊疗。`;

  return baseAnalysis;
}

// 生成模拟补充问诊分析结果
function generateMockSupplementInquiry(analysisId, additionalInfo) {
  const supplementAnalysis = `【补充分析】

基于您提供的补充信息，更新的分析如下：

1. 结合补充信息，症状的具体特征更加明确
2. 建议注意观察症状与日常生活习惯的关联性
3. 推荐适当的调理方法和注意事项

【综合建议】
- 结合补充信息，建议采取综合性的调理方案
- 包括生活方式调整和必要的医疗干预
- 定期监测症状变化
- 保持良好的心理状态

更新时间：${new Date().toLocaleString()}`;

  return supplementAnalysis;
}

module.exports = {
  initialInquiry,
  completeInquiry
};
