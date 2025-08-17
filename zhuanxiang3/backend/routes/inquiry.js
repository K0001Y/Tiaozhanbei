const express = require('express');
const router = express.Router();
const { initialInquiry, completeInquiry } = require('../controllers/inquiryController');
const AIService = require('../services/aiService');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
const auth = require('../middleware/auth');
const AI_CONFIG = require('../config/aiConfig');  
// 配置multer用于文件上传（用于补充问诊的文件）
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, '..', 'uploads', 'inquiry');
    // 确保目录存在
    fs.mkdir(uploadDir, { recursive: true }).then(() => {
      cb(null, uploadDir);
    }).catch(err => {
      cb(err);
    });
  },
  filename: function (req, file, cb) {
    // 生成唯一文件名
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'inquiry-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB限制
  },
  fileFilter: (req, file, cb) => {
    // 允许图片和文档文件
    const allowedTypes = ['image/', 'application/pdf', 'text/', 'application/msword', 'application/vnd.openxmlformats-officedocument'];
    if (allowedTypes.some(type => file.mimetype.startsWith(type))) {
      cb(null, true);
    } else {
      cb(new Error('不支持的文件格式'));
    }
  }
});

// 所有路由都需要认证
router.use(auth);

// 5.1 初步问诊（集成AI）
router.post('/', async (req, res) => {
  try {
    const { symptoms, duration, severity, additional_info } = req.body;
    
    // 参数验证
    if (!symptoms || symptoms.trim().length === 0) {
      return res.status(400).json({
        success: false,
        message: '症状描述不能为空'
      });
    }

    // 解析前端拼接的症状字符串
    function parseSymptoms(symptomsText) {
      const parsed = {
        symptoms: '',
        duration: '',
        severity: '',
        additional_info: ''
      };

      if (symptomsText.includes('主要症状：')) {
        const lines = symptomsText.split('\n');
        for (const line of lines) {
          if (line.startsWith('主要症状：')) {
            parsed.symptoms = line.replace('主要症状：', '').trim();
          } else if (line.startsWith('持续时间：')) {
            const duration = line.replace('持续时间：', '').trim();
            if (duration !== '未指定') {
              parsed.duration = duration;
            }
          } else if (line.startsWith('严重程度：')) {
            const severity = line.replace('严重程度：', '').trim();
            if (severity !== '未指定') {
              parsed.severity = severity;
            }
          } else if (line.startsWith('其他信息：')) {
            const info = line.replace('其他信息：', '').trim();
            if (info !== '无') {
              parsed.additional_info = info;
            }
          }
        }
      } else {
        // 如果不是拼接格式，直接使用原症状描述
        parsed.symptoms = symptomsText;
      }

      return parsed;
    }

    // 解析症状信息
    const parsedSymptoms = parseSymptoms(symptoms.trim());
    
    // 使用解析后的数据，优先使用单独传递的字段
    const finalDuration = duration || parsedSymptoms.duration;
    const finalSeverity = severity || parsedSymptoms.severity;
    const finalAdditionalInfo = additional_info || parsedSymptoms.additional_info;

    // 构建AI请求数据
    const inquiryData = {
      user_id: req.user.id,
      symptoms: parsedSymptoms.symptoms,
      duration: finalDuration,
      severity: finalSeverity || 'unknown',
      additional_info: finalAdditionalInfo,
      user_profile: {
        name: req.user.name || '',
        age: req.user.age || 0,
        gender: req.user.gender || ''
      }
    };

    console.log('用户信息 req.user:', JSON.stringify(req.user, null, 2));
    console.log('解析后的症状信息:', JSON.stringify(parsedSymptoms, null, 2));
    console.log('原始inquiryData:', JSON.stringify(inquiryData, null, 2));
    
    const formattedData = AIService.formatInquiryData(inquiryData);
    console.log('格式化后的数据:', JSON.stringify(formattedData, null, 2));

    // 调用AI服务
    const aiResponse = await AIService.callWithRetry(
      AI_CONFIG.ENDPOINTS.INQUIRY, 
      formattedData,
      { headers: AI_CONFIG.HEADERS }
    );
    
    if (aiResponse.success) {
      console.log('AI服务原始响应:', JSON.stringify(aiResponse.data, null, 2));
      
      // 直接使用AI服务返回的 results 字段
      const analysisText = aiResponse.data.results || '';
      
      console.log('构建的分析文本:', analysisText);

      res.json({
        success: true,
        message: "问诊分析完成",
        data: {
          results: analysisText,
          analysisId: `analysis_${Date.now()}`
        }
      });
    } else {
      // AI服务失败时，回退到传统问诊
      await initialInquiry(req, res);
    }
  } catch (error) {
    console.error('AI问诊分析错误，回退到传统问诊:', error);
    // 发生错误时回退到传统问诊
    await initialInquiry(req, res);
  }
});

// 5.2 问诊补充（集成AI）
router.post('/complete', upload.single('additionalFile'), async (req, res) => {
  try {
    const { prevInquiry, additionalInfo } = req.body;

    // 参数验证
    if (!prevInquiry || !additionalInfo) {
      return res.status(400).json({
        success: false,
        message: '必须提供之前的问诊结果和补充信息'
      });
    }

    // 构建AI补充问诊请求，按照todolist文档要求
    const supplementData = {
      prevInquiry: prevInquiry,
      additionalInfo: additionalInfo
    };

    // 如果有文件，添加到请求中
    if (req.file) {
      supplementData.additionalFile = req.file;
    }

    console.log('问诊补充请求数据:', JSON.stringify(supplementData, null, 2));

    // 调用AI服务
    const aiResponse = await AIService.callWithRetry(
      AI_CONFIG.ENDPOINTS.INQUIRY_COMPLETE, 
      supplementData,
      { headers: AI_CONFIG.HEADERS }
    );
    
    if (aiResponse.success) {
      // 直接使用AI服务返回的results字段
      const analysisContent = aiResponse.data?.results || '补充分析完成';
      
      res.json({
        success: true,
        message: "补充问诊完成",
        data: {
          results: analysisContent
        }
      });
    } else {
      throw new Error(aiResponse.message || 'AI服务调用失败');
    }
  } catch (error) {
    console.error('AI补充问诊错误，回退到传统问诊:', error);
    // 发生错误时回退到传统问诊
    await completeInquiry(req, res);
  }
});

module.exports = router;
