const express = require('express');
const FormData = require('form-data');
const router = express.Router();
const { analyzeImage, completeAnalysis, upload } = require('../controllers/watchController');
const AIService = require('../services/aiService');
const AI_CONFIG = require('../config/aiConfig');
const auth = require('../middleware/auth');

// 所有路由都需要认证
router.use(auth);

// 4.1 图片望诊（集成AI）
router.post('/', upload.single('image'), async (req, res) => {
  try {
    // 验证图片文件
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: '请上传图片文件'
      });
    }

    const { description } = req.body;

    console.log('望诊分析请求数据:', {
      description: description,
      fileName: req.file.originalname,
      hasFile: !!req.file
    });

    // 构建AI请求数据，按照todolist文档格式
    const fs = require('fs');
    const formData = new FormData();
    
    // 添加图片文件
    formData.append('image', fs.createReadStream(req.file.path), {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    });
    
    // 添加描述信息（可选）
    if (description && description.trim()) {
      formData.append('description', description.trim());
    }

    // 调用AI服务 - 使用正确的端点和参数格式
    const aiResponse = await AIService.callWithRetry(
      AI_CONFIG.ENDPOINTS.WATCH, 
      formData,
      { method: 'POST' }
    );
    
    if (aiResponse.success) {
      // 直接使用AI服务返回的results字段
      const analysisResult = aiResponse.data?.results || '';
      
      res.json({
        success: true,
        message: "望诊分析完成",
        data: {
          results: analysisResult,
          analysisId: `watch_${Date.now()}`
        }
      });
    } else {
      // AI服务失败时，回退到传统分析
      await analyzeImage(req, res);
    }
  } catch (error) {
    console.error('AI望诊分析错误，回退到传统分析:', error);
    // 发生错误时回退到传统分析
    await analyzeImage(req, res);
  }
});

// 4.2 望诊补充（集成AI）
router.post('/complete', upload.single('additionalFile'), async (req, res) => {
  try {
    const { prevAnalysis, additionalInfo } = req.body;

    // 验证必选参数
    if (!prevAnalysis || !additionalInfo) {
      return res.status(400).json({
        success: false,
        message: 'prevAnalysis 和 additionalInfo 参数都是必选的'
      });
    }

    // 构建AI补充分析请求，按照todolist文档格式
    const fs = require('fs');
    const formData = new FormData();
    
    // 添加必选参数
    formData.append('prevAnalysis', prevAnalysis);
    formData.append('additionalInfo', additionalInfo);
    
    // 添加可选的附加文件
    if (req.file) {
      formData.append('additionalFile', fs.createReadStream(req.file.path), {
        filename: req.file.originalname,
        contentType: req.file.mimetype
      });
    }

    console.log('望诊补充请求数据:', {
      hasPrevAnalysis: !!prevAnalysis,
      hasAdditionalInfo: !!additionalInfo,
      hasFile: !!req.file,
      fileName: req.file?.originalname
    });

    // 调用AI服务 - 使用正确的端点和参数格式
    const aiResponse = await AIService.callWithRetry(
      AI_CONFIG.ENDPOINTS.WATCH_COMPLETE, 
      formData,
      { method: 'POST' }
    );
    
    if (aiResponse.success) {
      // 根据AI服务返回的数据构建响应
      const analysisResult = aiResponse.data?.results || aiResponse.data?.analysis || '';
      
      res.json({
        success: true,
        message: "补充分析完成",
        data: {
          results: analysisResult,
          analysisId: `supplement_${Date.now()}`
        }
      });
    } else {
      // AI服务失败时，回退到传统补充分析
      await completeAnalysis(req, res);
    }
  } catch (error) {
    console.error('AI补充分析错误，回退到传统分析:', error);
    // 发生错误时回退到传统分析
    await completeAnalysis(req, res);
  }
});

module.exports = router;
