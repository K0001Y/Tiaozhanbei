const express = require('express');
const router = express.Router();
const authController = require('../controllers/authController');
const auth = require('../middleware/auth');
const { validateRegister, validateLogin } = require('../middleware/validation');

// 用户注册
router.post('/register', validateRegister, authController.register);

// 用户登录
router.post('/login', validateLogin, authController.login);

// 获取当前用户信息（需要认证）
router.get('/profile', auth, authController.getProfile);

// 验证token有效性
router.get('/verify', auth, authController.verifyToken);

module.exports = router;