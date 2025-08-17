const express = require('express');
const FormData = require('form-data');
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
const AI_CONFIG = require('../config/aiConfig');
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

    console.log('病历导入请求:', {
      fileName: req.file.originalname,
      fileSize: req.file.size,
      hasFile: !!req.file
    });

    // 构建AI请求数据，按照todolist文档格式
    const fs = require('fs');
    const formData = new FormData();
    
    // 添加病历图片文件，参数名为recordImage
    formData.append('recordImage', fs.createReadStream(req.file.path), {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    });

    // 调用AI服务 - 使用正确的端点
    const aiResponse = await AIService.callWithRetry(
      AI_CONFIG.ENDPOINTS.IMPORT, 
      formData,
      { method: 'POST' }
    );
    
    if (aiResponse.success) {
      // 根据API文档返回格式
      const analysisResult = aiResponse.data || {};
      
      res.json({
        success: true,
        message: "病历导入成功",
        data: {
          symptoms: analysisResult.symptoms || '',
          disease: analysisResult.disease || analysisResult.diagnosis || '',
          prescription: analysisResult.prescription || ''
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
router.post('/', async (req, res) => {
  try {
    const { watchResults, inquiryResults } = req.body;
    
    // 参数验证：至少需要提供一个分析结果
    if (!watchResults && !inquiryResults) {
      return res.status(400).json({
        success: false,
        message: 'watchResults 和 inquiryResults 至少需要提供一项'
      });
    }

    console.log('病历生成请求:', {
      hasWatchResults: !!watchResults,
      hasInquiryResults: !!inquiryResults,
      userId: req.user.id
    });

    // 构建AI请求数据，按照todolist文档格式
    const recordData = {
      watchResults: watchResults || '',
      inquiryResults: inquiryResults || ''
    };

    // 调用AI服务 - 使用正确的端点
    const aiResponse = await AIService.callWithRetry(
      AI_CONFIG.ENDPOINTS.RECORD, 
      recordData,
      { headers: AI_CONFIG.HEADERS }
    );
    
    if (aiResponse.success) {
      // 根据API文档返回格式
      const analysisResult = aiResponse.data || {};
      
      res.json({
        success: true,
        message: "病历生成成功",
        data: {
          symptoms: analysisResult.symptoms || '',
          disease: analysisResult.disease || analysisResult.diagnosis || '',
          prescription: analysisResult.prescription || ''
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
router.post('/save', saveRecord);

// 获取病历历史
router.get('/', getRecordHistory);

// 获取病历详情
router.get('/:id', getRecordDetail);

// 删除病历
router.delete('/:id', deleteRecord);

module.exports = router;
