const express = require('express');
const router = express.Router();
const { initialInquiry, completeInquiry } = require('../controllers/inquiryController');
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

// 5.1 初步问诊
router.post('/', initialInquiry);

// 5.2 问诊补充
router.post('/complete', upload.single('additionalFile'), completeInquiry);

module.exports = router;
