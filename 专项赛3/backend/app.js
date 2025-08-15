const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
require('dotenv').config();

const { createTables } = require('./config/database');
const authRoutes = require('./routes/auth');
const libraryRoutes = require('./routes/library');
const diseaseRoutes = require('./routes/disease');
const watchRoutes = require('./routes/watch');
const inquiryRoutes = require('./routes/inquiry');
const recordRoutes = require('./routes/record');
const aiRoutes = require('./routes/ai');

const app = express();

// 安全中间件
app.use(helmet());
app.use(cors({
  origin: process.env.NODE_ENV === 'production' 
    ? ['https://yourdomain.com'] 
    : ['http://localhost:3000', 'http://localhost:3001'],
  credentials: true
}));

// 限流中间件
// const limiter = rateLimit({
//   windowMs: 15 * 60 * 1000, // 15分钟
//   max: 100, // 限制每个IP 15分钟内最多100个请求
//   message: {
//     success: false,
//     message: '请求过于频繁，请稍后再试'
//   }
// });

// 认证限流中间件 - 测试期间注释掉
// const authLimiter = rateLimit({
//   windowMs: 15 * 60 * 1000, // 15分钟
//   max: 5, // 登录注册限制更严格
//   message: {
//     success: false,
//     message: '登录/注册请求过于频繁，请15分钟后再试'
//   }
// });

// app.use('/api/', limiter);
// app.use('/api/auth', authLimiter); // 测试期间注释掉

// 解析中间件 - 对文件上传路径完全跳过JSON解析
app.use((req, res, next) => {
  // 完全跳过文件上传路径的JSON解析
  if (req.path === '/api/record/import') {
    return next();
  }
  // 对其他路径使用JSON解析
  express.json({ limit: '10mb' })(req, res, next);
});
app.use(express.urlencoded({ extended: true }));

// 请求日志中间件
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// 路由
app.use('/api/auth', authRoutes);
app.use('/api/library', libraryRoutes);
app.use('/api/search', diseaseRoutes);
app.use('/api/watch', watchRoutes);
app.use('/api/inquiry', inquiryRoutes);
app.use('/api/record', recordRoutes);
app.use('/api/ai', aiRoutes);

// 根路径
app.get('/', (req, res) => {
  res.json({
    success: true,
    message: 'AI助手后端API服务运行中',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  });
});

// 健康检查
app.get('/health', (req, res) => {
  res.json({
    success: true,
    status: 'OK',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// 全局错误处理中间件
app.use((err, req, res, next) => {
  console.error('全局错误:', err.stack);
  res.status(500).json({
    success: false,
    message: '服务器内部错误',
    ...(process.env.NODE_ENV === 'development' && { error: err.message })
  });
});

// 404处理
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    message: `路由 ${req.originalUrl} 不存在`
  });
});

const PORT = process.env.PORT || 5001;

// 启动服务器
const startServer = async () => {
  try {
    // 创建数据库表
    await createTables();
    
    app.listen(PORT, () => {
      console.log(`🚀 服务器运行在端口 ${PORT}`);
      console.log(`🏠 健康检查: http://localhost:${PORT}/health`);
      console.log(`📡 API根路径: http://localhost:${PORT}/api`);
      console.log(`🔐 认证路由: http://localhost:${PORT}/api/auth`);
    });
  } catch (error) {
    console.error('❌ 启动服务器失败:', error);
    process.exit(1);
  }
};

// 优雅关闭
process.on('SIGTERM', () => {
  console.log('收到SIGTERM信号，正在关闭服务器...');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('收到SIGINT信号，正在关闭服务器...');
  process.exit(0);
});

startServer();