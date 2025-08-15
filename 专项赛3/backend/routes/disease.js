const express = require('express');
const router = express.Router();
const { searchDiseases } = require('../controllers/diseaseController');
const AIService = require('../services/aiService');
const auth = require('../middleware/auth');

// 所有路由都需要认证
router.use(auth);

// 3.1 获取检索结果 - 疾病知识检索和匹配（集成AI）
router.get('/', async (req, res) => {
  try {
    const { search } = req.query;
    
    // 参数验证
    if (!search || search.trim().length === 0) {
      return res.status(400).json({
        success: false,
        message: '搜索关键词不能为空'
      });
    }

    // 调用AI服务进行智能搜索
    const requestData = {
      query: search.trim(),
      mode: 'disease_search',
      user_id: req.user.id
    };
    
    const aiResponse = await AIService.callWithRetry('search', requestData);
    
    if (aiResponse.success) {
      res.json({
        success: true,
        message: "检索成功",
        data: {
          diseases: aiResponse.data.diseases || []
        }
      });
    } else {
      // AI服务失败时，回退到传统搜索
      await searchDiseases(req, res);
    }
  } catch (error) {
    console.error('AI搜索错误，回退到传统搜索:', error);
    // 发生错误时回退到传统搜索
    await searchDiseases(req, res);
  }
});

module.exports = router;
