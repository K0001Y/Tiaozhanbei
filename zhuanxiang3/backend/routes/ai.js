const express = require('express');
const multer = require('multer');
const path = require('path');
const aiController = require('../controllers/aiController');
const authMiddleware = require('../middleware/auth');

const router = express.Router();

// 配置multer用于文件上传
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/ai');
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const fileFilter = (req, file, cb) => {
  // 允许图片和文档类型
  const allowedTypes = [
    'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp',
    'application/pdf', 'text/plain', 'application/msword', 
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
  ];
  
  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('不支持的文件类型。支持的格式：图片（JPG、PNG、GIF等）、PDF、TXT、DOC、DOCX'), false);
  }
};

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 20 * 1024 * 1024 // 20MB限制
  },
  fileFilter: fileFilter
});

// 确保上传目录存在
const fs = require('fs');
const uploadDir = 'uploads/ai';
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// 7.1 AI智能分析路由
router.post('/analyze', authMiddleware, upload.single('file'), aiController.intelligentAnalyze);

module.exports = router;
