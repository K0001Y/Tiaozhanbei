const express = require('express');
const path = require('path');
const app = express();

// 静态文件服务
app.use(express.static(path.join(__dirname, 'public')));

// 错误处理中间件
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send('出现了一些问题！');
});

// 所有路由指向index.html
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// 在8080端口运行前端服务器
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`前端服务器运行在 http://localhost:${PORT}`);
});