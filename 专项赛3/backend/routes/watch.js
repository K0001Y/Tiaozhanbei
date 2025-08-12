const express = require('express');
const router = express.Router();
const { analyzeImage, completeAnalysis, upload } = require('../controllers/watchController');
const auth = require('../middleware/auth');

// 所有路由都需要认证
router.use(auth);

// 4.1 图片望诊
router.post('/', upload.single('image'), analyzeImage);

// 4.2 望诊补充
router.post('/complete', upload.single('additionalFile'), completeAnalysis);

module.exports = router;
