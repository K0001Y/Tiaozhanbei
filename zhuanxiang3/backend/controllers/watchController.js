const { pool } = require('../config/database');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;

// 配置multer用于图片上传
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, '..', 'uploads', 'watch');
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
    cb(null, 'watch-' + uniqueSuffix + path.extname(file.originalname));
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
      cb(new Error('只支持图片文件格式'));
    }
  }
});

// 4.1 图片望诊
const analyzeImage = async (req, res) => {
  try {
    console.log(`${new Date().toISOString()} - POST /api/watch`);
    
    const { description } = req.body;
    const imageFile = req.file;

    console.log('望诊分析请求数据:', { description, fileName: imageFile?.filename });

    // 验证图片文件
    if (!imageFile) {
      return res.status(400).json({
        success: false,
        message: '请上传图片文件'
      });
    }

    const imagePath = imageFile.path;
    
    // TODO: 调用大模型进行图片分析
    // 以下是占位实现，实际应该调用AI视觉模型
    /*
    const aiAnalysis = await callAIVisionModel({
      imagePath: imagePath,
      description: description || ''
    });
    */

    // 模拟AI分析结果
    const mockAnalysis = generateMockWatchAnalysis(description, imageFile.originalname);

    console.log('望诊分析完成');

    res.json({
      success: true,
      message: '望诊分析成功',
      data: {
        results: mockAnalysis,
        imageId: `watch_${Date.now()}`, // 用于后续补充分析
        imagePath: `/api/files/watch/${imageFile.filename}`
      }
    });

  } catch (error) {
    console.error('望诊分析失败:', error);
    
    // 清理上传的文件
    if (req.file && req.file.path) {
      try {
        await fs.unlink(req.file.path);
      } catch (unlinkError) {
        console.error('删除临时文件失败:', unlinkError);
      }
    }

    res.status(500).json({
      success: false,
      message: '望诊分析失败，请稍后重试',
      error: error.message
    });
  }
};

// 4.2 望诊补充
const completeAnalysis = async (req, res) => {
  try {
    console.log(`${new Date().toISOString()} - POST /api/watch/complete`);
    
    const { prevAnalysis, additionalInfo } = req.body;
    const additionalFile = req.file;

    console.log('望诊补充请求数据:', { 
      hasPrevAnalysis: !!prevAnalysis,
      additionalInfo, 
      hasAdditionalFile: !!additionalFile 
    });

    // 验证必需参数
    if (!prevAnalysis || !additionalInfo) {
      return res.status(400).json({
        success: false,
        message: '请提供之前的分析结果和补充信息'
      });
    }

    let additionalImagePath = null;
    if (additionalFile) {
      additionalImagePath = additionalFile.path;
    }

    // TODO: 调用大模型进行补充分析
    // 以下是占位实现，实际应该调用AI模型
    /*
    const updatedAnalysis = await callAISupplementAnalysis({
      previousAnalysis: prevAnalysis,
      additionalInfo: additionalInfo,
      additionalImagePath: additionalImagePath
    });
    */

    // 模拟补充分析结果
    const mockUpdatedAnalysis = generateMockSupplementAnalysis(prevAnalysis, additionalInfo);

    console.log('望诊补充分析完成');

    res.json({
      success: true,
      message: '补充望诊信息成功',
      data: {
        results: mockUpdatedAnalysis,
        updatedAt: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('望诊补充失败:', error);
    
    // 清理上传的文件
    if (req.file && req.file.path) {
      try {
        await fs.unlink(req.file.path);
      } catch (unlinkError) {
        console.error('删除临时文件失败:', unlinkError);
      }
    }

    res.status(500).json({
      success: false,
      message: '望诊补充失败，请稍后重试',
      error: error.message
    });
  }
};

// 生成模拟望诊分析结果
function generateMockWatchAnalysis(description, fileName) {
  const imageType = detectImageType(fileName, description);
  
  const analysisTemplates = {
    'ct': `CT影像分析结果：
    
影像特征：
- 密度分布：软组织密度正常，未见明显异常密度影
- 结构清晰：各组织结构边界清晰，层次分明
- 对比度：自然对比度良好，未使用造影剂

初步观察：
- 未发现明显占位性病变
- 组织结构基本正常
- 建议结合临床症状进一步评估

注意事项：
- 此分析仅供参考，最终诊断需结合临床症状
- 建议由专业影像科医生进行详细解读
- 如有疑问请及时就医咨询`,

    'xray': `X光片分析结果：

影像表现：
- 骨质结构：骨质密度正常，未见明显骨折征象
- 软组织：软组织轮廓清晰，未见异常阴影
- 透明度：透光度正常，对比度适中

观察要点：
- 心脏轮廓：心影大小在正常范围内
- 肺野：双肺纹理清晰，未见明显异常
- 膈肌：双侧膈肌位置正常，轮廓清晰

建议：
- 影像学表现基本正常
- 建议结合临床症状综合评估
- 如有不适症状请及时复查`,

    'skin': `皮肤病变分析结果：

外观特征：
- 颜色：病灶颜色较为均匀，边界相对清晰
- 形状：形状较规则，表面质地可见
- 大小：病灶范围适中，分布较为局限

初步评估：
- 病变性质：外观符合良性病变特征
- 炎症程度：轻度炎症反应，无明显渗出
- 发展趋势：病变相对稳定，边界清晰

处理建议：
- 建议保持局部清洁干燥
- 避免搔抓和刺激
- 如症状加重请及时就医
- 必要时可进行进一步检查`,

    'default': `医学图像分析结果：

图像质量：
- 清晰度：图像清晰度良好，细节可辨
- 光照条件：曝光适中，对比度合适
- 拍摄角度：角度适宜，重要结构可见

观察发现：
- 整体结构：各组织结构排列正常
- 异常征象：未发现明显异常征象
- 密度变化：密度分布基本均匀

专业建议：
- 图像显示基本正常征象
- 建议结合患者症状和体征综合判断
- 如有疑虑建议咨询专科医生
- 必要时可考虑进一步检查`
  };

  return analysisTemplates[imageType] || analysisTemplates['default'];
}

// 检测图像类型
function detectImageType(fileName, description) {
  const lowerFileName = fileName.toLowerCase();
  const lowerDescription = description ? description.toLowerCase() : '';
  
  if (lowerFileName.includes('ct') || lowerDescription.includes('ct') || 
      lowerDescription.includes('断层') || lowerDescription.includes('扫描')) {
    return 'ct';
  }
  
  if (lowerFileName.includes('xray') || lowerFileName.includes('x-ray') || 
      lowerDescription.includes('x光') || lowerDescription.includes('胸片')) {
    return 'xray';
  }
  
  if (lowerDescription.includes('皮肤') || lowerDescription.includes('皮疹') || 
      lowerDescription.includes('斑点') || lowerDescription.includes('痣')) {
    return 'skin';
  }
  
  return 'default';
}

// 生成模拟补充分析结果
function generateMockSupplementAnalysis(prevAnalysis, additionalInfo) {
  return `【补充分析】

基于您提供的补充信息，更新的分析如下：

1. 结合补充信息，症状的具体特征更加明确
2. 建议注意观察症状与日常生活习惯的关联性  
3. 推荐适当的调理方法和注意事项

【综合建议】
- 结合补充信息，建议采取综合性的调理方案
- 包括生活方式调整和必要的医疗干预
- 定期监测症状变化
- 保持良好的心理状态

更新时间：${new Date().toLocaleString('zh-CN')}`;
}

module.exports = {
  analyzeImage,
  completeAnalysis,
  upload
};
