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

    const { username, password, name, age, gender, phone } = req.body;

    // 检查用户是否已存在
    const [existingUsers] = await pool.execute(
      'SELECT id FROM users WHERE username = ?',
      [username]
    );

    if (existingUsers.length > 0) {
      return res.status(400).json({
        success: false,
        message: '用户名已存在'
      });
    }

    // 加密密码
    const saltRounds = 12;
    const hashedPassword = await bcrypt.hash(password, saltRounds);

    // 创建用户
    const [result] = await pool.execute(
      'INSERT INTO users (username, password, name, age, gender, phone) VALUES (?, ?, ?, ?, ?, ?)',
      [username, hashedPassword, name, age, gender, phone]
    );

    res.status(201).json({
      success: true,
      message: '注册成功'
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

    const { username, password } = req.body;

    // 查找用户
    const [users] = await pool.execute(
      'SELECT id, username, password, name, age, gender, phone FROM users WHERE username = ?',
      [username]
    );

    if (users.length === 0) {
      return res.status(401).json({
        success: false,
        message: '用户名或密码错误'
      });
    }

    const user = users[0];

    // 验证密码
    const isValidPassword = await bcrypt.compare(password, user.password);
    if (!isValidPassword) {
      return res.status(401).json({
        success: false,
        message: '用户名或密码错误'
      });
    }

    // 获取用户的病历记录
    const [records] = await pool.execute(
      'SELECT id as recordId, symptoms, diagnosis as disease, prescription, created_at as time FROM records WHERE user_id = ? ORDER BY created_at DESC',
      [user.id]
    );

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
        user: {
          userId: user.id,
          username: user.username,
          name: user.name,
          age: user.age,
          gender: user.gender,
          phone: user.phone,
          records: records
        },
        token
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
      'SELECT id, username, name, age, gender, phone FROM users WHERE id = ?',
      [req.user.id]
    );

    if (users.length === 0) {
      return res.status(404).json({
        success: false,
        message: '用户不存在'
      });
    }

    const user = users[0];

    // 获取用户的病历记录
    const [records] = await pool.execute(
      'SELECT id as recordId, symptoms, diagnosis as disease, prescription, created_at as time FROM records WHERE user_id = ? ORDER BY created_at DESC',
      [user.id]
    );

    res.json({
      success: true,
      message: '获取用户信息成功',
      data: {
        user: {
          userId: user.id,
          username: user.username,
          name: user.name,
          age: user.age,
          gender: user.gender,
          phone: user.phone,
          records: records
        }
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