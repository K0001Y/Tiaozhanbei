# 中医诊断系统

基于 Node.js + Vue.js + AI 的智能中医诊断系统

## 项目结构

```
Tiaozhanbei/zhuanxiang3/
├── frontend/          # 前端项目 (Vue.js)
├── backend/           # 后端项目 (Node.js)
├── AIchatbot/         # AI 聊天机器人 (Python)
└── README.md          # 项目说明文档
```

## 环境要求

- Node.js >= 16.0.0
- Python >= 3.8
- MySQL >= 8.0
- npm 或 yarn

## 快速开始

### 1. 安装依赖

#### 前端依赖安装
```bash
cd frontend
npm install
```

#### 后端依赖安装
```bash
cd backend
npm install
```

#### Python AI 模块依赖安装
```bash
cd AIchatbot
pip install -r requirements.txt
```

### 2. 环境配置

#### 后端环境配置
复制并配置后端环境变量：
```bash
cd backend
cp .env.example .env
```

编辑 `.env` 文件，配置数据库连接等信息：
```bash
# 数据库配置
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=tcm_system

# JWT 配置
JWT_SECRET=your_jwt_secret_key
JWT_EXPIRE=7d

# 服务器配置
PORT=5001
NODE_ENV=development
```

#### 前端环境配置
复制并配置前端环境变量：
```bash
cd frontend
cp .env.example .env
```

编辑 `.env` 文件：
```bash
VITE_API_BASE_URL=http://localhost:5001/api
VITE_FRONTEND_PORT=3000
VITE_BACKEND_PORT=5001
VITE_NODE_ENV=development
```

### 3. 数据库初始化

确保 MySQL 服务已启动，创建数据库：
```sql
CREATE DATABASE tcm_system CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 4. 全局安装 nodemon

```bash
npm install -g nodemon
```

### 5. 启动服务

#### 启动后端服务
```bash
cd backend
nodemon app.js
```

后端服务将在 http://localhost:5001 启动

#### 启动前端服务
```bash
cd frontend
npm run dev
```

前端服务将在 http://localhost:3001 启动

#### 启动 AI 聊天机器人服务（可选）
```bash
cd AIchatbot
python server.py
```

AI 服务将在 http://localhost:8080 启动

## 开发说明

### 前端开发
- 使用 Vue 3 + Vite 构建
- UI 框架：Element Plus / Ant Design Vue
- 状态管理：Pinia
- 路由：Vue Router

### 后端开发
- 使用 Express.js 框架
- 数据库：MySQL + Sequelize ORM
- 身份验证：JWT
- API 文档：自动生成

### AI 模块
- 基于 Python Flask 框架
- 集成 PaddleOCR、LangChain 等 AI 库
- 支持医疗文档分析和智能问诊

## 常用命令

### 前端
```bash
npm run dev          # 开发模式启动
npm run build        # 生产构建
npm run preview      # 预览构建结果
```

### 后端
```bash
nodemon app.js       # 开发模式启动（热重载）
node app.js          # 生产模式启动
npm test             # 运行测试
```

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # Windows 查看端口占用
   netstat -ano | findstr :3000
   netstat -ano | findstr :5001
   
   # 杀死进程
   taskkill /PID <进程ID> /F
   ```

2. **数据库连接失败**
   - 检查 MySQL 服务是否启动
   - 确认 `.env` 中的数据库配置正确
   - 确保数据库 `tcm_system` 已创建

3. **模块安装失败**
   ```bash
   # 清除 npm 缓存
   npm cache clean --force
   
   # 删除 node_modules 重新安装
   rm -rf node_modules package-lock.json
   npm install
   ```

## 项目结构详细说明

```
frontend/
├── src/
│   ├── components/    # 公共组件
│   ├── views/         # 页面组件
│   ├── router/        # 路由配置
│   ├── store/         # 状态管理
│   ├── api/           # API 接口
│   ├── utils/         # 工具函数
│   └── assets/        # 静态资源

backend/
├── config/            # 配置文件
├── controllers/       # 控制器
├── models/            # 数据模型
├── routes/            # 路由定义
├── middleware/        # 中间件
├── utils/             # 工具函数
└── app.js             # 应用入口

AIchatbot/
├── routes/            # API 路由
├── models/            # AI 模型
├── utils/             # 工具函数
├── requirements.txt   # Python 依赖
└── server.py          # 服务入口
```

## 贡献指南

1. Fork 本仓库
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可