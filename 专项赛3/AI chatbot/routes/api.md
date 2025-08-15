# 医疗诊断API接口说明与输入格式指南

## 🏥 系统概述

**服务地址**：`http://localhost:8080`  
**支持格式**：JSON, multipart/form-data  
**字符编码**：UTF-8  

---

## 📋 API接口列表

| 编号 | 接口名称 | 路径 | 方法 | 功能描述 |
|------|----------|------|------|----------|
| 3.1 | 疾病搜索 | `/api/search` | GET | 根据关键词搜索相关疾病信息 |
| 4.1 | 图片望诊 | `/api/watch` | POST | 上传医学图像进行望诊分析 |
| 4.2 | 望诊补充 | `/api/watch/complete` | POST | 补充望诊信息和分析 |
| 5.1 | 初步问诊 | `/api/inquiry` | POST | 基于症状进行中医问诊分析 |
| 5.2 | 补充问诊 | `/api/inquiry/complete` | POST | 补充问诊信息和深度分析 |
| 6.1 | 病历生成 | `/api/record` | POST | 从系统状态生成完整病历报告 |
| 6.2 | 文档导入 | `/api/import` | POST | OCR识别医学文档并提取信息 |

---

## 📥 详细输入格式说明

### 1. 疾病搜索API (3.1)

**接口**：`GET /api/search`

#### 输入格式
```
URL参数：
?search=关键词
```

#### 输入示例
```bash
GET /api/search?search=头痛
GET /api/search?search=高血压
GET /api/search?search=发热咳嗽
```

#### 参数说明
- `search` (必填): 搜索关键词，支持中文症状、疾病名称

---

### 2. 图片望诊API (4.1)

**接口**：`POST /api/watch`

#### 输入格式
```
Content-Type: multipart/form-data

表单字段：
image (必填) - 医学图像文件
```

#### 输入示例
```bash
curl -X POST http://localhost:8080/api/watch \
  -F "image=@tongue_image.jpg"
```

#### 文件要求
- **支持格式**：JPG, JPEG, PNG, BMP, GIF
- **文件大小**：建议 < 10MB
- **图像类型**：舌象、面部、眼部等医学图像

---

### 3. 望诊补充API (4.2)

**接口**：`POST /api/watch/complete`

#### 输入格式
```
Content-Type: multipart/form-data

表单字段：
image (可选) - 补充医学图像
additional_info (可选) - 补充文字信息
patient_info (可选) - 患者基本信息
```

#### 输入示例
```bash
curl -X POST http://localhost:8080/api/watch/complete \
  -F "image=@additional_image.jpg" \
  -F "additional_info=患者反映舌苔厚腻" \
  -F "patient_info=女，45岁"
```

---

### 4. 初步问诊API (5.1)

**接口**：`POST /api/inquiry`

#### 输入格式
```json
{
  "symptoms": "症状描述",
  "duration": "持续时间",
  "severity": "严重程度",
  "patientInfo": {
    "age": 年龄数字,
    "gender": "性别",
    "medicalHistory": "既往病史"
  },
  "additionalQuestions": [
    {
      "question": "问题内容",
      "answer": "回答内容"
    }
  ]
}
```

#### 输入示例
```bash
curl -X POST http://localhost:8080/api/inquiry \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": "头痛、头晕、心悸",
    "duration": "3天",
    "severity": "中等",
    "patientInfo": {
      "age": 35,
      "gender": "男",
      "medicalHistory": "高血压家族史"
    },
    "additionalQuestions": [
      {
        "question": "是否伴有恶心呕吐？",
        "answer": "偶有恶心"
      }
    ]
  }'
```

#### 字段说明
- `symptoms` (必填): 主要症状描述
- `duration` (可选): 症状持续时间
- `severity` (可选): 症状严重程度（轻度/中度/重度）
- `patientInfo` (可选): 患者基本信息对象
- `additionalQuestions` (可选): 问答数组

---

### 5. 补充问诊API (5.2)

**接口**：`POST /api/inquiry/complete`

#### 输入格式
```
Content-Type: multipart/form-data

表单字段：
inquiry_data (必填) - JSON格式的基础问诊数据
supplementary_info (可选) - 补充问诊信息
follow_up_questions (可选) - 后续问题回答
```

#### 输入示例
```bash
curl -X POST http://localhost:8080/api/inquiry/complete \
  -F 'inquiry_data={"symptoms":"头痛","duration":"3天"}' \
  -F 'supplementary_info=患者工作压力大，经常熬夜' \
  -F 'follow_up_questions=睡眠质量差，多梦易醒'
```

---

### 6. 病历生成API (6.1)

**接口**：`POST /api/record`

#### 输入格式
```json
{
  "patientInfo": "患者基本信息描述",
  "watchResults": "望诊分析结果",
  "inquiryResults": "问诊分析结果"
}
```

#### 输入示例
```bash
curl -X POST http://localhost:8080/api/record \
  -H "Content-Type: application/json" \
  -d '{
    "patientInfo": "男，35岁，有高血压家族病史",
    "watchResults": "遗传因素、饮食不当、缺乏运动。头痛、头晕、心悸。",
    "inquiryResults": "患者有家族史，症状持续一周。"
  }'
```

#### 字段说明
- `patientInfo` (必填): 患者基本信息字符串
- `watchResults` (必填): 望诊分析结果字符串
- `inquiryResults` (必填): 问诊分析结果字符串

#### 特殊说明
- 需要系统Graph实例支持
- 从Graph状态提取最终数据生成病历

---

### 7. 文档导入API (6.2)

**接口**：`POST /api/import`

#### 输入格式方式一：文件上传
```
Content-Type: multipart/form-data

表单字段：
document (必填) - 医学文档文件
```

#### 输入格式方式二：JSON文本
```json
{
  "content": "医学文档文本内容",
  "type": "文档类型标识"
}
```

#### 输入示例
```bash
# 方式1：上传PDF文件
curl -X POST http://localhost:8080/api/import \
  -F "document=@medical_report.pdf"

# 方式2：上传图片文件
curl -X POST http://localhost:8080/api/import \
  -F "document=@medical_image.jpg"

# 方式3：直接传文本
curl -X POST http://localhost:8080/api/import \
  -H "Content-Type: application/json" \
  -d '{
    "content": "患者主诉：头痛、发热3天。体查：血压150/90mmHg。诊断：高血压。处理：降压治疗，低盐饮食。",
    "type": "medical_report"
  }'
```

#### 文件要求
- **PDF文件**：支持多页，最多处理10页
- **图像文件**：JPG, JPEG, PNG, BMP, GIF
- **文件大小**：建议 < 50MB
- **OCR处理**：自动识别中文医学文档

#### 字段说明
- `document`: 文档文件（文件上传方式）
- `content`: 文档文本内容（JSON方式）
- `type`: 文档类型，如"medical_report", "prescription"

---

## 📤 统一输出格式

所有API都遵循统一的输出格式：

### 成功响应
```json
{
  "success": true,
  "message": "操作成功信息",
  "data": {
    // 具体业务数据
  }
}
```

### 错误响应
```json
{
  "success": false,
  "message": "错误描述信息",
  "data": {}
}
```

---

## 🔧 使用建议

### 1. Content-Type设置
```bash
# JSON请求
Content-Type: application/json

# 文件上传请求
Content-Type: multipart/form-data
```

### 2. 字符编码
所有文本数据使用UTF-8编码

### 3. 文件上传注意事项
- 确保文件格式正确
- 控制文件大小在合理范围
- PDF文件OCR处理耗时较长

### 4. JSON数据格式
- 使用标准JSON格式
- 字符串使用双引号
- 数字类型不加引号
- 布尔值使用true/false

### 5. 错误处理
- 检查HTTP状态码
- 解析response中的success字段
- 处理message中的错误信息

---

## 📋 快速测试模板

### 测试疾病搜索
```bash
curl "http://localhost:8080/api/search?search=头痛"
```

### 测试图像上传
```bash
curl -X POST http://localhost:8080/api/watch \
  -F "image=@test_image.jpg"
```

### 测试问诊分析
```bash
curl -X POST http://localhost:8080/api/inquiry \
  -H "Content-Type: application/json" \
  -d '{"symptoms":"头痛头晕","duration":"3天"}'
```

### 测试病历生成
```bash
curl -X POST http://localhost:8080/api/record \
  -H "Content-Type: application/json" \
  -d '{"patientInfo":"男，35岁","watchResults":"症状明显","inquiryResults":"需要治疗"}'
```

### 测试文档导入
```bash
# 上传文件
curl -X POST http://localhost:8080/api/import \
  -F "document=@medical_doc.pdf"

# 提交文本
curl -X POST http://localhost:8080/api/import \
  -H "Content-Type: application/json" \
  -d '{"content":"患者主诉头痛发热。诊断：感冒。建议：多休息。"}'
```

---

## 🚀 开发建议

### JavaScript示例
```javascript
// 疾病搜索
const searchResult = await fetch('/api/search?search=头痛').then(r => r.json());

// 文件上传
const formData = new FormData();
formData.append('image', fileInput.files[0]);
const imageResult = await fetch('/api/watch', {
  method: 'POST', 
  body: formData
}).then(r => r.json());

// JSON提交
const inquiryResult = await fetch('/api/inquiry', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    symptoms: '头痛头晕',
    duration: '3天'
  })
}).then(r => r.json());
```

### Python示例
```python
import requests

# 疾病搜索
response = requests.get('http://localhost:8080/api/search', 
                       params={'search': '头痛'})

# 文件上传
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8080/api/watch', 
                           files={'image': f})

# JSON提交
response = requests.post('http://localhost:8080/api/inquiry',
                        json={'symptoms': '头痛头晕', 'duration': '3天'})
```

---

## ❓ 常见问题

**Q: 文件上传失败怎么办？**
A: 检查文件格式、大小，确保使用multipart/form-data格式

**Q: JSON格式错误怎么解决？**
A: 验证JSON格式，确保字段名称正确，数据类型匹配

**Q: OCR识别效果不好？**
A: 确保图片清晰，文字不倾斜，PDF文件质量良好

**Q: Graph实例未设置？**
A: RecordAPI需要Graph支持，启动时传入Graph实例

**Q: 跨域请求被阻止？**
A: 服务器已启用CORS，检查请求头设置