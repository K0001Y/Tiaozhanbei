# AI医疗助手集成说明

## 概述

本项目已成功集成AI医疗助手功能，提供智能诊断辅助服务。AI集成包括后端服务接口、前端组件和数据管理。

## 功能特性

### 1. AI疾病搜索 (AI Disease Search)
- **接口**: `GET /api/search/ai-search`
- **功能**: 基于症状关键词智能搜索相关疾病信息
- **输入**: 搜索关键词
- **输出**: 匹配的疾病列表及相关信息

### 2. AI智能问诊 (AI Inquiry)
- **接口**: `POST /api/inquiry/ai-inquiry`
- **功能**: 基于症状描述提供智能问诊分析
- **输入**: 症状描述、持续时间、严重程度等
- **输出**: 分析结果、追问建议、初步诊断方向、建议措施

### 3. AI望诊分析 (AI Visual Analysis)
- **接口**: `POST /api/watch/ai-analyze`
- **功能**: 图像识别和望诊分析
- **输入**: 图片文件 + 描述（可选）
- **输出**: 望诊分析结果、建议、置信度

### 4. AI病历生成 (AI Record Generation)
- **接口**: `POST /api/record/ai-generate`
- **功能**: 基于症状信息智能生成医疗病历
- **输入**: 症状、诊断信息、治疗历史等
- **输出**: 结构化病历、摘要、建议

### 5. AI病历识别 (AI Record OCR)
- **接口**: `POST /api/record/ai-import`
- **功能**: 病历图片OCR识别和结构化提取
- **输入**: 病历图片文件
- **输出**: 提取的文本、结构化数据、置信度

## 技术架构

### 后端架构
```
backend/
├── services/
│   └── aiService.js           # AI服务封装类
├── config/
│   └── aiConfig.js           # AI配置文件
└── routes/
    ├── disease.js            # 疾病搜索路由（含AI集成）
    ├── inquiry.js            # 问诊路由（含AI集成）
    ├── watch.js              # 望诊路由（含AI集成）
    └── record.js             # 病历路由（含AI集成）
```

### 前端架构
```
frontend/src/
├── services/
│   └── aiService.ts          # AI服务接口
├── store/
│   └── aiStore.ts            # AI数据状态管理
├── components/
│   └── AIAssistant/          # AI助手通用组件
└── pages/
    └── AICopilot/            # AI助手页面
```

## 配置说明

### 环境变量
```bash
# AI服务配置
AI_CHATBOT_URL=http://localhost:8080
AI_CHATBOT_TIMEOUT=30000
```

### AI服务配置
- **基础URL**: `http://localhost:8080`
- **超时时间**: 30秒
- **重试次数**: 3次
- **重试间隔**: 1秒
- **文件大小限制**: 10MB

## 使用指南

### 1. 启动AI服务
```bash
cd "AI chatbot/V2"
python -m pip install -r requirements.txt
python main.py
```

### 2. 启动后端服务
```bash
cd backend
npm install
npm start
```

### 3. 启动前端服务
```bash
cd frontend
npm install
npm run dev
```

### 4. 访问AI功能
- 登录系统后访问AI助手页面
- 选择所需的AI功能模式
- 输入相应的数据进行分析

## API接口详情

### 疾病搜索
```http
GET /api/search/ai-search?search=头痛
Authorization: Bearer <token>
```

### 智能问诊
```http
POST /api/inquiry/ai-inquiry
Authorization: Bearer <token>
Content-Type: application/json

{
  "symptoms": "头痛，伴有恶心",
  "duration": "3天",
  "severity": "中度"
}
```

### 望诊分析
```http
POST /api/watch/ai-analyze
Authorization: Bearer <token>
Content-Type: multipart/form-data

image: [文件]
description: "舌诊图片"
```

### 病历生成
```http
POST /api/record/ai-generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "symptoms": "发热、咳嗽",
  "diagnosis_info": "初步诊断为感冒"
}
```

## 错误处理

### 常见错误码
- **400**: 参数错误（如缺少必需参数）
- **401**: 未授权（需要登录）
- **413**: 文件过大
- **500**: AI服务不可用
- **504**: AI服务超时

### 错误响应格式
```json
{
  "success": false,
  "message": "错误描述",
  "error": "详细错误信息"
}
```

## 数据格式

### 成功响应格式
```json
{
  "success": true,
  "message": "操作成功",
  "data": {
    // 具体数据
  }
}
```

## 注意事项

1. **服务依赖**: AI功能需要AI Chatbot服务在端口8080运行
2. **认证要求**: 所有AI接口都需要用户认证
3. **文件限制**: 图片文件不超过10MB，支持JPEG、PNG等格式
4. **网络超时**: AI分析可能需要较长时间，设置了30秒超时
5. **错误重试**: 自动重试机制，最多重试3次

## 开发说明

### 添加新的AI功能
1. 在`aiService.js`中添加新的方法
2. 在对应的路由文件中添加新的端点
3. 在前端`aiService.ts`中添加对应的接口方法
4. 更新AI状态管理和组件

### 自定义AI配置
修改`backend/config/aiConfig.js`文件来调整AI服务的配置参数。

### 扩展AI数据存储
在`frontend/src/store/aiStore.ts`中添加新的状态字段和操作方法。

## 故障排查

1. **AI服务连接失败**: 检查AI Chatbot服务是否正常运行
2. **文件上传失败**: 检查文件大小和格式限制
3. **认证失败**: 确保用户已登录且token有效
4. **响应超时**: 检查网络连接和AI服务性能

## 更新日志

- v1.0.0: 完成AI服务集成，支持四大核心功能
- 后续版本将增加更多AI功能和优化
