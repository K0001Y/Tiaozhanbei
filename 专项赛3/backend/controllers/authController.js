const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { pool } = require('../config/database');
const jwtConfig = require('../config/jwt');
const { validationResult } = require('express-validator');

// 用户注册
const register = async (req, res) => {
  try {
    // 检查验证结果
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        message: '输入数据验证失败',
        errors: errors.array()
      });
    }

    const { username, email, password } = req.body;

    // 检查用户是否已存在
    const [existingUsers] = await pool.execute(
      'SELECT id FROM users WHERE email = ? OR username = ?',
      [email, username]
    );

    if (existingUsers.length > 0) {
      return res.status(400).json({
        success: false,
        message: '用户名或邮箱已被注册'
      });
    }

    // 加密密码
    const saltRounds = 12;
    const hashedPassword = await bcrypt.hash(password, saltRounds);

    // 创建用户
    const [result] = await pool.execute(
      'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
      [username, email, hashedPassword]
    );

    // 生成JWT token
    const token = jwt.sign(
      { userId: result.insertId },
      jwtConfig.secret,
      { expiresIn: jwtConfig.expiresIn }
    );

    res.status(201).json({
      success: true,
      message: '注册成功',
      data: {
        token,
        user: {
          id: result.insertId,
          username,
          email
        }
      }
    });
  } catch (error) {
    console.error('注册错误:', error);
    res.status(500).json({
      success: false,
      message: '服务器内部错误，注册失败'
    });
  }
};

// 用户登录
const login = async (req, res) => {
  try {
    // 检查验证结果
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        message: '输入数据验证失败',
        errors: errors.array()
      });
    }

    const { email, password } = req.body;

    // 查找用户
    const [users] = await pool.execute(
      'SELECT id, username, email, password FROM users WHERE email = ?',
      [email]
    );

    if (users.length === 0) {
      return res.status(400).json({
        success: false,
        message: '邮箱或密码错误'
      });
    }

    const user = users[0];

    // 验证密码
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return res.status(400).json({
        success: false,
        message: '邮箱或密码错误'
      });
    }

    // 生成JWT token
    const token = jwt.sign(
      { userId: user.id },
      jwtConfig.secret,
      { expiresIn: jwtConfig.expiresIn }
    );

    res.json({
      success: true,
      message: '登录成功',
      data: {
        token,
        user: {
          id: user.id,
          username: user.username,
          email: user.email
        }
      }
    });
  } catch (error) {
    console.error('登录错误:', error);
    res.status(500).json({
      success: false,
      message: '服务器内部错误，登录失败'
    });
  }
};

// 获取当前用户信息
const getProfile = async (req, res) => {
  try {
    const [users] = await pool.execute(
      'SELECT id, username, email, avatar, created_at FROM users WHERE id = ?',
      [req.user.id]
    );

    if (users.length === 0) {
      return res.status(404).json({
        success: false,
        message: '用户不存在'
      });
    }

    res.json({
      success: true,
      message: '获取用户信息成功',
      data: {
        user: users[0]
      }
    });
  } catch (error) {
    console.error('获取用户信息错误:', error);
    res.status(500).json({
      success: false,
      message: '服务器内部错误'
    });
  }
};

// 验证token有效性
const verifyToken = async (req, res) => {
  try {
    // 如果能到达这里，说明token有效（通过了auth中间件）
    res.json({
      success: true,
      message: 'Token有效',
      data: {
        user: req.user
      }
    });
  } catch (error) {
    console.error('验证token错误:', error);
    res.status(500).json({
      success: false,
      message: '服务器内部错误'
    });
  }
};

module.exports = {
  register,
  login,
  getProfile,
  verifyToken
};