const express = require('express');
const router = express.Router();
const { 
  importFromImage,
  generateRecord, 
  saveRecord, 
  getRecordHistory, 
  getRecordDetail, 
  deleteRecord 
} = require('../controllers/recordController');
const AIService = require('../services/aiService');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
const auth = require('../middleware/auth');

// 配置multer用于病历图片上传
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, '..', 'uploads', 'records');
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
    cb(null, 'record-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB限制
  },
  fileFilter: (req, file, cb) => {
    // 只允许图片文件
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('只支持图片格式的病历文件'));
    }
  }
});

// 所有路由都需要认证
router.use(auth);

// 6.2 从病历图片导入生成（集成AI）
router.post('/import', upload.single('recordImage'), async (req, res) => {
  try {
    // 验证图片文件
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: '请上传病历图片文件'
      });
    }

    // 构建AI请求数据
    const recordData = {
      user_id: req.user.id,
      image: req.file,
      analysis_type: 'record_ocr_extraction'
    };

    // 调用AI服务
    const aiResponse = await AIService.callWithRetry('/api/extract-record', AIService.formatRecordData(recordData));
    
    if (aiResponse.success) {
      res.json({
        success: true,
        message: "病历识别完成",
        data: {
          extracted_text: aiResponse.data.extracted_text || '',
          structured_data: aiResponse.data.structured_data || {},
          confidence: aiResponse.data.confidence || 0
        }
      });
    } else {
      // AI服务失败时，回退到传统导入
      await importFromImage(req, res);
    }
  } catch (error) {
    console.error('AI病历识别错误，回退到传统导入:', error);
    // 发生错误时回退到传统导入
    await importFromImage(req, res);
  }
});

// 6.1 生成病历（集成AI）
router.post('/generate', async (req, res) => {
  try {
    const { 
      symptoms, 
      diagnosis_info, 
      treatment_history, 
      current_medications,
      vital_signs 
    } = req.body;
    
    // 参数验证
    if (!symptoms || symptoms.trim().length === 0) {
      return res.status(400).json({
        success: false,
        message: '症状信息不能为空'
      });
    }

    // 构建AI请求数据
    const recordData = {
      user_id: req.user.id,
      patient_info: {
        name: req.user.name,
        age: req.user.age,
        gender: req.user.gender,
        phone: req.user.phone
      },
      symptoms: symptoms.trim(),
      diagnosis_info: diagnosis_info || '',
      treatment_history: treatment_history || '',
      current_medications: current_medications || '',
      vital_signs: vital_signs || {},
      generate_type: 'comprehensive_record'
    };

    // 调用AI服务
    const aiResponse = await AIService.callWithRetry('/api/generate-record', AIService.formatRecordData(recordData));
    
    if (aiResponse.success) {
      res.json({
        success: true,
        message: "病历生成完成",
        data: {
          record: aiResponse.data.record || {},
          summary: aiResponse.data.summary || '',
          recommendations: aiResponse.data.recommendations || [],
          follow_up: aiResponse.data.follow_up || ''
        }
      });
    } else {
      // AI服务失败时，回退到传统生成
      await generateRecord(req, res);
    }
  } catch (error) {
    console.error('AI病历生成错误，回退到传统生成:', error);
    // 发生错误时回退到传统生成
    await generateRecord(req, res);
  }
});

// 保存病历
router.post('/', saveRecord);

// 获取病历历史
router.get('/', getRecordHistory);

// 获取病历详情
router.get('/:id', getRecordDetail);

// 删除病历
router.delete('/:id', deleteRecord);

module.exports = router;
