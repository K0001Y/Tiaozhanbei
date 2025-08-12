const express = require('express');
const router = express.Router();
const { searchDiseases } = require('../controllers/diseaseController');
const auth = require('../middleware/auth');

// 所有路由都需要认证
router.use(auth);

// 获取检索结果 - 疾病知识检索和匹配
router.get('/', searchDiseases);

module.exports = router;
