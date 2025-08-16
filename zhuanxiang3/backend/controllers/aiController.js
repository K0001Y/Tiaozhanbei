const aiService = require('../services/aiService');

const aiController = {
  // 7.1 AI智能分析 - POST /api/ai/analyze
  async intelligentAnalyze(req, res) {
    try {
      // 从请求中提取参数
      const { query } = req.body;
      const file = req.file;

      // 验证参数：query 和 file 至少需要一个
      if (!query && !file) {
        return res.status(400).json({
          success: false,
          message: 'query 和 file 参数至少需要提供一项'
        });
      }

      console.log('AI分析请求:', {
        hasQuery: !!query,
        hasFile: !!file,
        fileName: file?.originalname,
        fileSize: file?.size
      });

      // 调用AI服务进行分析
      const analysisRequest = {
        query: query || undefined,
        file: file || undefined
      };

      const result = await aiService.intelligentAnalyze(analysisRequest);

      res.json({
        success: true,
        message: 'AI分析完成',
        data: {
          solution: result.solution
        }
      });

    } catch (error) {
      console.error('AI智能分析错误:', error);
      res.status(500).json({
        success: false,
        message: error.message || 'AI分析失败，请稍后重试'
      });
    }
  }
};

module.exports = aiController;
