const { pool } = require('../config/database');

// 从病历图片导入生成病历
const importFromImage = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: '请上传病历图片'
      });
    }

    const userId = req.user.id;
    const imagePath = req.file.path;

    // 处理上传的病历图片
    const extractedData = await processRecordImage(imagePath);

    res.json({
      success: true,
      message: '病历图片处理成功',
      data: extractedData
    });

  } catch (error) {
    console.error('处理病历图片失败:', error);
    res.status(500).json({
      success: false,
      message: '处理病历图片失败',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};
const generateRecord = async (req, res) => {
  try {
    const { diagnosis, inquiry } = req.body;
    const userId = req.user.id;

    // 验证必要数据
    if (!diagnosis && !inquiry) {
      return res.status(400).json({
        success: false,
        message: '请至少提供望诊或问诊结果'
      });
    }

    // 整合医疗信息生成病历
    const recordData = await generateMedicalRecord(diagnosis, inquiry, userId);

    res.json({
      success: true,
      message: '病历生成成功',
      data: recordData
    });

  } catch (error) {
    console.error('生成病历失败:', error);
    res.status(500).json({
      success: false,
      message: '生成病历失败',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// 保存病历到数据库
const saveRecord = async (req, res) => {
  try {
    const { symptoms, diagnosis, prescription, recordData } = req.body;
    const userId = req.user.id;

    // 验证必要字段
    if (!symptoms) {
      return res.status(400).json({
        success: false,
        message: '症状描述不能为空'
      });
    }

    // 保存到数据库
    const [result] = await pool.execute(
      'INSERT INTO records (user_id, symptoms, diagnosis, prescription, created_at) VALUES (?, ?, ?, ?, NOW())',
      [userId, symptoms, diagnosis || '', prescription || '']
    );

    res.json({
      success: true,
      message: '病历保存成功',
      data: {
        recordId: result.insertId,
        symptoms,
        diagnosis,
        prescription,
        createdAt: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('保存病历失败:', error);
    res.status(500).json({
      success: false,
      message: '保存病历失败',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// 获取用户病历历史
const getRecordHistory = async (req, res) => {
  try {
    const userId = req.user.id;
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const offset = (page - 1) * limit;

    // 获取病历总数
    const [countResult] = await pool.execute(
      'SELECT COUNT(*) as total FROM records WHERE user_id = ?',
      [userId]
    );
    const total = countResult[0].total;

    // 获取病历列表
    const [records] = await pool.execute(
      'SELECT id, symptoms, diagnosis, prescription, created_at FROM records WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?',
      [userId, limit, offset]
    );

    res.json({
      success: true,
      data: {
        records,
        pagination: {
          current: page,
          pageSize: limit,
          total,
          pages: Math.ceil(total / limit)
        }
      }
    });

  } catch (error) {
    console.error('获取病历历史失败:', error);
    res.status(500).json({
      success: false,
      message: '获取病历历史失败',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// 获取单个病历详情
const getRecordDetail = async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user.id;

    const [records] = await pool.execute(
      'SELECT * FROM records WHERE id = ? AND user_id = ?',
      [id, userId]
    );

    if (records.length === 0) {
      return res.status(404).json({
        success: false,
        message: '病历不存在'
      });
    }

    res.json({
      success: true,
      data: records[0]
    });

  } catch (error) {
    console.error('获取病历详情失败:', error);
    res.status(500).json({
      success: false,
      message: '获取病历详情失败',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// 删除病历
const deleteRecord = async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user.id;

    const [result] = await pool.execute(
      'DELETE FROM records WHERE id = ? AND user_id = ?',
      [id, userId]
    );

    if (result.affectedRows === 0) {
      return res.status(404).json({
        success: false,
        message: '病历不存在'
      });
    }

    res.json({
      success: true,
      message: '病历删除成功'
    });

  } catch (error) {
    console.error('删除病历失败:', error);
    res.status(500).json({
      success: false,
      message: '删除病历失败',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// 处理病历图片的核心逻辑
async function processRecordImage(imagePath) {
  // ================================
  // AI模型集成占位符 - 病历图片直接分析
  // ================================
  // 真实实现时，这里应该：
  // 1. 将上传的图片文件直接发送给大模型
  // 2. 大模型对图像进行端到端分析（OCR + 医疗理解 + 信息提取）
  // 3. 大模型直接返回结构化的病历数据
  // 4. 无需单独的OCR和NLP步骤，一站式处理
  
  // TODO: 集成大模型的代码示例（目前注释掉）:
  /*
  try {
    const fs = require('fs');
    
    // 1. 读取图片文件
    const imageBuffer = fs.readFileSync(imagePath);
    
    // 2. 直接发送给大模型进行图像分析
    const analysisResult = await callAIModelForImageAnalysis({
      image: imageBuffer,
      task: 'medical_record_extraction',
      format: 'structured_json'
    });
    
    // 3. 大模型返回的结构化病历数据
    return {
      recordId: `imported_${Date.now()}`,
      imagePath: imagePath,
      extractedContent: analysisResult.medicalData,
      confidence: analysisResult.confidence,
      aiModel: analysisResult.modelInfo,
      processedAt: new Date().toLocaleString('zh-CN')
    };
  } catch (error) {
    console.error('大模型分析失败:', error);
    throw new Error('病历图片分析失败');
  }
  */
  
  // 暂时使用mock数据模拟大模型病历图片分析结果
  return generateMockRecordFromImage(imagePath);
}

// 生成mock病历图片处理结果（模拟大模型的分析结果）
function generateMockRecordFromImage(imagePath) {
  const currentTime = new Date().toLocaleString('zh-CN');
  
  // 模拟大模型从病历图片中直接分析得出的结构化数据
  const mockExtractedContent = {
    // 基本信息
    patientName: '张三',
    gender: '男',
    age: '35岁',
    visitDate: '2025年8月12日',
    
    // 主诉
    chiefComplaint: '头痛、发热3天',
    
    // 现病史
    presentIllness: '患者3天前无明显诱因出现头痛，伴发热，体温最高38.5°C，无恶心呕吐，无畏寒，精神食欲尚可',
    
    // 既往史
    pastHistory: '既往体健，否认高血压、糖尿病等慢性病史',
    
    // 体格检查
    physicalExam: '体温38.2°C，血压120/80mmHg，神志清楚，精神可，咽部轻度充血，双肺呼吸音清，心率80次/分，律齐',
    
    // 辅助检查
    labTests: '血常规：白细胞计数6.8×10⁹/L，中性粒细胞70%，血红蛋白135g/L',
    
    // 诊断
    diagnosis: '上呼吸道感染',
    
    // 治疗方案
    treatment: '1. 休息，多饮水\n2. 对乙酰氨基酚 0.5g po tid\n3. 阿莫西林 0.5g po tid\n4. 复方氨酚烷胺胶囊 1粒 po tid',
    
    // 医嘱
    advice: '注意休息，多饮水，避免受凉，如症状加重及时复诊',
    
    // 医生签名
    doctorName: '李医生',
    
    // 处理时间
    processedAt: currentTime,
    
    // 置信度（模拟AI识别的准确度）
    confidence: 0.92
  };
  
  return {
    recordId: `imported_${Date.now()}`,
    imagePath: imagePath,
    extractedContent: mockExtractedContent,
    structuredData: {
      symptoms: mockExtractedContent.chiefComplaint + '；' + mockExtractedContent.presentIllness,
      diagnosis: mockExtractedContent.diagnosis,
      prescription: mockExtractedContent.treatment,
      recommendations: mockExtractedContent.advice.split('，'),
      doctorInfo: {
        name: mockExtractedContent.doctorName,
        visitDate: mockExtractedContent.visitDate
      }
    },
    confidence: mockExtractedContent.confidence,
    processedAt: currentTime
  };
}
async function generateMedicalRecord(diagnosis, inquiry, userId) {
  // ================================
  // AI模型集成占位符
  // ================================
  // 这里应该集成真实的AI模型来：
  // 1. 分析望诊和问诊结果
  // 2. 生成诊断建议
  // 3. 推荐治疗方案
  // 4. 生成格式化的病历文档
  
  // 暂时使用mock数据生成病历
  return generateMockMedicalRecord(diagnosis, inquiry);
}

// 生成mock病历数据
function generateMockMedicalRecord(diagnosis, inquiry) {
  const currentTime = new Date().toLocaleString('zh-CN');
  
  // 整合症状信息
  const symptoms = [];
  if (diagnosis) {
    symptoms.push(`望诊观察: ${diagnosis.description}`);
  }
  if (inquiry) {
    symptoms.push(`问诊信息: ${inquiry.symptoms}`);
  }
  
  // 整合分析结果
  const analysisResults = [];
  if (diagnosis) {
    analysisResults.push(`望诊分析: ${diagnosis.analysisReport}`);
    if (diagnosis.supplements && diagnosis.supplements.length > 0) {
      diagnosis.supplements.forEach((supplement, index) => {
        analysisResults.push(`望诊补充${index + 1}: ${supplement.analysis}`);
      });
    }
  }
  if (inquiry) {
    analysisResults.push(`问诊分析: ${inquiry.analysisReport}`);
    if (inquiry.supplements && inquiry.supplements.length > 0) {
      inquiry.supplements.forEach((supplement, index) => {
        analysisResults.push(`问诊补充${index + 1}: ${supplement.analysis}`);
      });
    }
  }
  
  // 生成综合诊断
  const possibleDiagnoses = [
    '风寒感冒', '风热感冒', '胃肠不适', '肝气郁结', 
    '脾胃虚弱', '肾阳虚', '肾阴虚', '心脾两虚',
    '湿热内蕴', '气血不足', '肝肾不足', '痰湿内阻'
  ];
  const diagnosisResult = possibleDiagnoses[Math.floor(Math.random() * possibleDiagnoses.length)];
  
  // 生成治疗建议
  const treatments = [
    '清热解毒，疏风散寒',
    '健脾和胃，理气消食',
    '疏肝理气，调和脾胃',
    '温阳补肾，健脾益气',
    '滋阴降火，清热润燥',
    '养心安神，健脾益气',
    '清热利湿，健脾和胃',
    '补气养血，调理脾胃'
  ];
  const treatmentPlan = treatments[Math.floor(Math.random() * treatments.length)];
  
  // 生成处方建议
  const prescriptions = [
    '银翘散加减：金银花15g，连翘15g，薄荷6g，桔梗9g，甘草6g',
    '四君子汤加减：人参9g，白术9g，茯苓9g，甘草6g，陈皮6g',
    '逍遥散加减：柴胡6g，当归9g，白芍9g，白术9g，茯苓9g，薄荷3g，甘草3g',
    '金匮肾气丸加减：熟地黄24g，山药12g，山茱萸12g，茯苓9g，牡丹皮9g，泽泻9g',
    '知柏地黄丸加减：知母9g，黄柏9g，熟地黄24g，山药12g，山茱萸12g，茯苓9g'
  ];
  const prescription = prescriptions[Math.floor(Math.random() * prescriptions.length)];
  
  return {
    recordId: `record_${Date.now()}`,
    patientInfo: {
      symptoms: symptoms.join('\n'),
      analysisResults: analysisResults.join('\n\n')
    },
    diagnosis: diagnosisResult,
    treatmentPlan: treatmentPlan,
    prescription: prescription,
    recommendations: [
      '注意休息，避免过度劳累',
      '饮食清淡，忌辛辣刺激食物',
      '保持情绪舒畅，避免过度焦虑',
      '按时服药，观察病情变化',
      '如症状加重，及时复诊'
    ],
    followUp: '建议一周后复诊，观察治疗效果',
    createdAt: currentTime,
    generatedSummary: `基于望诊和问诊的综合分析，患者主要表现为：${symptoms.join('；')}。经过中医辨证分析，初步诊断为${diagnosisResult}。治疗方针：${treatmentPlan}。处方：${prescription}。`
  };
}

module.exports = {
  importFromImage,
  generateRecord,
  saveRecord,
  getRecordHistory,
  getRecordDetail,
  deleteRecord
};
