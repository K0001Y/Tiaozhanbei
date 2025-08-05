const jwt = require('jsonwebtoken');
const { pool } = require('../config/database');
const jwtConfig = require('../config/jwt');

const auth = async (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(401).json({ 
        success: false,
        message: '访问被拒绝，需要提供认证token' 
      });
    }

    const decoded = jwt.verify(token, jwtConfig.secret);
    
    // 从数据库验证用户是否存在
    const [users] = await pool.execute(
      'SELECT id, username, email FROM users WHERE id = ?',
      [decoded.userId]
    );

    if (users.length === 0) {
      return res.status(401).json({ 
        success: false,
        message: '无效的token，用户不存在' 
      });
    }

    req.user = users[0];
    next();
  } catch (error) {
    if (error.name === 'TokenExpiredError') {
      return res.status(401).json({ 
        success: false,
        message: 'token已过期，请重新登录' 
      });
    }
    
    res.status(401).json({ 
      success: false,
      message: '无效的token' 
    });
  }
};

module.exports = auth;