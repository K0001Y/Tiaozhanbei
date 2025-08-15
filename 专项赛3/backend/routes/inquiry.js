const express = require('express');
const router = express.Router();
const { initialInquiry, completeInquiry } = require('../controllers/inquiryController');
const AIService = require('../services/aiService');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
const auth = require('../middleware/auth');

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

    // 构建AI请求数据
    const inquiryData = {
      user_id: req.user.id,
      symptoms: symptoms.trim(),
      duration: duration || '',
      severity: severity || 'unknown',
      additional_info: additional_info || '',
      user_profile: {
        name: req.user.name,
        age: req.user.age,
        gender: req.user.gender
      }
    };

    // 调用AI服务
    const aiResponse = await AIService.callWithRetry('inquiry', AIService.formatInquiryData(inquiryData));
    
    if (aiResponse.success) {
      res.json({
        success: true,
        message: "问诊分析完成",
        data: {
          analysis: aiResponse.data.analysis || '',
          follow_up_questions: aiResponse.data.follow_up_questions || [],
          preliminary_diagnosis: aiResponse.data.preliminary_diagnosis || [],
          recommendations: aiResponse.data.recommendations || []
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
    const { originalInquiryId, additionalSymptoms, additionalInfo } = req.body;

    // 构建AI补充问诊请求
    const supplementData = {
      user_id: req.user.id,
      original_inquiry_id: originalInquiryId,
      additional_symptoms: additionalSymptoms || '',
      additional_info: additionalInfo || '',
      additional_file: req.file || null
    };

    // 调用AI服务
    const aiResponse = await AIService.callWithRetry('inquiry-supplement', AIService.formatInquiryData(supplementData));
    
    if (aiResponse.success) {
      res.json({
        success: true,
        message: "补充问诊完成",
        data: {
          updated_analysis: aiResponse.data.updated_analysis || '',
          additional_questions: aiResponse.data.additional_questions || [],
          refined_diagnosis: aiResponse.data.refined_diagnosis || [],
          updated_recommendations: aiResponse.data.updated_recommendations || []
        }
      });
    } else {
      // AI服务失败时，回退到传统补充问诊
      await completeInquiry(req, res);
    }
  } catch (error) {
    console.error('AI补充问诊错误，回退到传统问诊:', error);
    // 发生错误时回退到传统问诊
    await completeInquiry(req, res);
  }
});

module.exports = router;
