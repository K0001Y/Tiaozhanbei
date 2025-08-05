# 眼部疾病诊断系统 API 文档

## 目录
- [图像诊断](#图像诊断)
- [对话交互](#对话交互)

---

## 图像诊断

### 基本信息

- 路径: `/api/diagnose`
- 方法: `POST`
- 描述: 上传眼部图像进行AI诊断分析

### 请求参数

| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| groups | Array | 是 | 图像组数组，每组包含普通图片和紫外线图片 |

#### groups 数组中每个对象的结构

| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| groupId | String | 是 | 图像组编号 |
| normalImage | String | 是 | 普通图片的 base64 编码数据 |
| uvImage | String | 是 | 紫外线图片的 base64 编码数据 |

### 请求体示例

```json
{
  "groups": [
    {
      "groupId": "1",
      "normalImage": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
      "uvImage": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
    },
    {
      "groupId": "2", 
      "normalImage": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
      "uvImage": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
    }
  ]
}
```

### 响应参数

| 状态码 | 内容类型         | 描述                        |
| ------ | ---------------- | --------------------------- |
| 200    | application/json | 诊断成功                    |
| 400    | application/json | 参数错误或图片格式不正确     |
| 413    | application/json | 图片文件过大                |
| 500    | application/json | 服务器错误                  |

### 响应示例

- 成功响应 (状态码：200)

  ```json
  {
    "success": true,
    "results": [
      {
        "groupId": "1",
        "result": "眼部健康状况良好，未发现明显异常"
      },
      {
        "groupId": "2",
        "result": "疑似轻微干眼症，建议使用人工泪液并注意用眼卫生"
      }
    ]
  }
  ```

- 参数错误 (状态码：400)

  ```json
  {
    "success": false,
    "message": "缺少必要参数或图片格式不正确"
  }
  ```

- 图片文件过大 (状态码：413)

  ```json
  {
    "success": false,
    "message": "上传的图片文件过大，请压缩后重试"
  }
  ```

- 服务器错误 (状态码：500)

  ```json
  {
    "success": false,
    "message": "服务器内部错误，请稍后重试"
  }
  ```

**备注**

- 图片大小限制：单张图片不超过 10MB
- 支持的图片格式：JPEG、PNG、BMP
- 该接口使用了请求频率限制 (rate limit)
- 图片数据需要使用 base64 编码传输

---

## 对话交互

### 基本信息

- 路径: `/api/dialog`
- 方法: `POST`
- 描述: 与AI助手进行眼部疾病相关的对话咨询

### 请求参数

| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| message | String | 是 | 用户输入的问题或描述 |
| context | String | 否 | 对话上下文，用于保持对话连贯性 |

### 请求体示例

```json
{
  "message": "我的眼睛最近经常感到干涩，这是什么原因？",
  "context": "eye_disease_consultation"
}
```

### 响应参数

| 状态码 | 内容类型         | 描述                        |
| ------ | ---------------- | --------------------------- |
| 200    | application/json | 对话成功                    |
| 400    | application/json | 参数错误或消息为空          |
| 429    | application/json | 请求过于频繁                |
| 500    | application/json | 服务器错误                  |

### 响应示例

- 成功响应 (状态码：200)

  ```json
  {
    "success": true,
    "reply": "眼干涩是现代人常见问题，多与长时间用眼、环境干燥有关。建议您使用人工泪液，保持室内湿度，定时远眺放松眼部肌肉。如果症状持续，建议就医检查。"
  }
  ```

- 参数错误 (状态码：400)

  ```json
  {
    "success": false,
    "message": "消息内容不能为空"
  }
  ```

- 请求过于频繁 (状态码：429)

  ```json
  {
    "success": false,
    "message": "请求过于频繁，请稍后再试"
  }
  ```

- 服务器错误 (状态码：500)

  ```json
  {
    "success": false,
    "message": "AI服务暂时不可用，请稍后重试"
  }
  ```

**备注**

- 单次消息长度限制：1000 字符
- 该接口使用了请求频率限制：每分钟最多 20 次请求
- AI 回复专注于眼部疾病相关咨询，不提供其他医疗建议
- 仅供参考，不能替代专业医疗诊断

---