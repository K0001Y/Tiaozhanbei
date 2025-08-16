const express = require('express');
const router = express.Router();
const { analyzeImage, completeAnalysis, upload } = require('../controllers/watchController');
const AIService = require('../services/aiService');
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

    // 构建AI请求数据
    const watchData = {
      user_id: req.user.id,
      image: req.file,
      description: description || '',
      analysis_type: 'visual_diagnosis'
    };

    // 调用AI服务
    const aiResponse = await AIService.callWithRetry('/api/analyze', AIService.formatWatchData(watchData));
    
    if (aiResponse.success) {
      res.json({
        success: true,
        message: "望诊分析完成",
        data: {
          analysis: aiResponse.data.analysis || '',
          suggestions: aiResponse.data.suggestions || [],
          confidence: aiResponse.data.confidence || 0
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
    const { originalAnalysisId, additionalDescription } = req.body;

    // 构建AI补充分析请求
    const supplementData = {
      user_id: req.user.id,
      original_analysis_id: originalAnalysisId,
      additional_description: additionalDescription || '',
      additional_file: req.file || null
    };

    // 调用AI服务
    const aiResponse = await AIService.callWithRetry('/api/analyze-supplement', AIService.formatWatchData(supplementData));
    
    if (aiResponse.success) {
      res.json({
        success: true,
        message: "补充分析完成",
        data: {
          updated_analysis: aiResponse.data.updated_analysis || '',
          additional_suggestions: aiResponse.data.additional_suggestions || [],
          confidence: aiResponse.data.confidence || 0
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
