const multer = require('multer');
const path = require('path');
const fs = require('fs');

// 确保上传目录存在
const ensureDirectoryExists = (dirPath) => {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
};

// 配置存储
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadPath = path.join(__dirname, '../uploads/library');
    ensureDirectoryExists(uploadPath);
    cb(null, uploadPath);
  },
  filename: (req, file, cb) => {
    // 生成唯一文件名：时间戳 + 随机数 + 原始扩展名
    const timestamp = Date.now();
    const randomString = Math.random().toString(36).substring(2, 15);
    const extension = path.extname(file.originalname);
    const fileName = `${timestamp}_${randomString}${extension}`;
    cb(null, fileName);
  }
});

// 文件过滤器
const fileFilter = (req, file, cb) => {
  // 允许的文件类型
  const allowedMimes = [
    'text/plain',
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/markdown'
  ];

  if (allowedMimes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('不支持的文件类型'), false);
  }
};

// 创建multer实例
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 限制文件大小为10MB
  },
  fileFilter: fileFilter
});

// 错误处理中间件
const handleUploadError = (error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        success: false,
        message: '文件大小超过限制，最大支持10MB'
      });
    }
    return res.status(400).json({
      success: false,
      message: '文件上传失败：' + error.message
    });
  }
  
  if (error) {
    return res.status(400).json({
      success: false,
      message: error.message || '文件上传失败'
    });
  }
  
  next();
};

module.exports = {
  upload,
  handleUploadError
};
