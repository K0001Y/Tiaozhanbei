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

// 从病历图片导入生成
router.post('/import', upload.single('recordImage'), importFromImage);

// 生成病历
router.post('/generate', generateRecord);

// 保存病历
router.post('/', saveRecord);

// 获取病历历史
router.get('/', getRecordHistory);

// 获取病历详情
router.get('/:id', getRecordDetail);

// 删除病历
router.delete('/:id', deleteRecord);

module.exports = router;
