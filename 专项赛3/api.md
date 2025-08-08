# 医疗辅助诊疗系统 API 文档

[toc]

## 1. 认证相关接口 (authRoutes)

### 1.1 用户注册

**基本信息**

- 路径: `/api/auth/register`
- 方法: `POST`
- 描述: 注册新用户（患者）

**请求参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| username | String | 是 | 用户名，必须唯一 |
| password | String | 是 | 用户密码 |
| name | String | 是 | 用户真实姓名 |
| age | Number | 是 | 用户年龄 |
| gender | String | 是 | 用户性别 |
| phone | String | 否 | 联系电话 |

**请求体示例**

```json
{
  "username": "patient001",
  "password": "password123",
  "name": "张三",
  "age": 35,
  "gender": "男",
  "phone": "13800138000"
}
```

**响应参数**

| 状态码 | 内容类型         | 描述                        |
| ------ | ---------------- | --------------------------- |
| 201    | application/json | 注册成功                    |
| 400    | application/json | 参数错误或用户已存在        |
| 500    | application/json | 服务器错误                  |

**响应示例**

- 成功响应 (状态码：201)

  ```json
  {
    "success": true,
    "message": "注册成功"
  }
  ```

- 错误响应 (状态码：400)

  ```json
  {
    "success": false,
    "message": "用户名已存在"
  }
  ```

---

### 1.2 用户登录

**基本信息**

- 路径: `/api/auth/login`
- 方法: `POST`
- 描述: 用户登录

**请求参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| username | String | 是 | 用户名 |
| password | String | 是 | 用户密码 |

**请求体示例**

```json
{
  "username": "patient001",
  "password": "password123"
}
```

**响应示例**

- 成功响应 (状态码：200)

  ```json
  {
    "success": true,
    "message": "登录成功",
    "data": {
      "user": {
        "userId": 1,
        "username": "patient001",
        "name": "张三",
        "age": 35,
        "gender": "男",
        "phone": "13800138000"
      },
      "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    }
  }
  ```

- 错误响应 (状态码：401)

  ```json
  {
    "success": false,
    "message": "用户名或密码错误"
  }
  ```

---

### 1.3 获取用户信息

**基本信息**

- 路径: `/api/auth/profile`
- 方法: `GET`
- 描述: 获取当前用户信息
- 认证: 需要Bearer Token

**响应示例**

```json
{
  "success": true,
  "message": "获取用户信息成功",
  "data": {
    "user": {
      "userId": 1,
      "username": "patient001",
      "name": "张三",
      "age": 35,
      "gender": "男",
      "phone": "13800138000"
    }
  }
}
```

---

## 2. 知识库信息接口 (libraryRoutes)

### 2.1 获取知识库列表

**基本信息**

- 路径: `/api/library`
- 方法: `GET`
- 描述: 分页获取知识库列表资料信息
- 认证: 需要Bearer Token

**查询参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| page | Number | 否 | 页码，默认为1 |
| limit | Number | 否 | 每页数量，默认为20 |

**响应示例**

```json
{
  "success": true,
  "message": "获取知识库列表成功",
  "data": {
    "page": 1,
    "limit": 20,
    "total": 50,
    "libraries": [
      {
        "libraryId": 1,
        "filePath": "/api/files/1.txt",
        "fileName": "高血压诊疗指南.txt",
        "uploadTime": "2024-01-01T10:00:00.000Z"
      },
      {
        "libraryId": 2,
        "filePath": "/api/files/2.txt",
        "fileName": "糖尿病治疗手册.txt",
        "uploadTime": "2024-01-02T10:00:00.000Z"
      }
    ]
  }
}
```

---

### 2.2 上传资料至知识库

**基本信息**

- 路径: `/api/library`
- 方法: `POST`
- 描述: 上传新的资料到知识库
- 认证: 需要Bearer Token
- 内容类型: multipart/form-data

**查询参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| file | File | 是 | 要上传的资料文件 |
| fileName | String | 是 | 资料文件的名称 |

**请求体示例**

```
file: [资料文件]
fileName: 高血压诊疗指南.txt
```

**响应示例**

```json
{
  "success": true,
  "message": "资料上传成功"
}
```

---

## 3. 病理检索接口 (diseaseRoutes)

### 3.1 获取检索结果

**基本信息**

- 路径: `/api/search`
- 方法: `GET`
- 描述: 获取疾病列表
- 认证: 需要Bearer Token

**查询参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| search | String | 是 | 搜索关键词 |

**响应示例**

```json
{
  "success": true,
  "message": "检索成功",
  "data": {
    "diseases": [
      {
        "diseaseId": 1,
        "diseaseName": "高血压",
        "description": "遗传因素、饮食不当、缺乏运动。头痛、头晕、心悸。药物治疗、生活方式改变。低盐饮食、适量运动、戒烟限酒。",
        "source": "内科学(第八版)",
        "relevance": "92%"
      }
    ]
  }
}
```

**说明**
返回的疾病最多5个，若为空则表示没有相关疾病。

---

## 4. 辅助望诊接口 (searchRoutes)

### 4.1 图片望诊

**基本信息**

- 路径: `/api/watch`
- 方法: `POST`
- 描述: 上传图片进行初步望诊
- 认证: 需要Bearer Token
- 内容类型: multipart/form-data

**请求参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| image | File | 是 | 症状图片 |
| description | String | 否 | 图片描述信息 |

**请求体示例**

```
description: CT图
image: [图片文件]
```

**响应示例**

```json
{
  "success": true,
  "message": "望诊分析成功",
  "data": {
    "results": "遗传因素、饮食不当、缺乏运动。头痛、头晕、心悸。药物治疗、生活方式改变。低盐饮食、适量运动、戒烟限酒。",
    "analysisId": "watch_123456789"
  }
}
```

---

### 4.2 望诊补充

**基本信息**

- 路径: `/api/watch/complete`
- 方法: `POST`
- 描述: 补充望诊信息
- 认证: 需要Bearer Token
- 内容类型: multipart/form-data

**请求参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| analysisId | String | 是 | 之前望诊的分析ID |
| additionalInfo | String | 是 | 补充的望诊信息 |
| additionalFile | File | 否 | 附加的图片文件 |

**请求体示例**

```
analysisId: watch_123456789
additionalInfo: 患者有家族史，症状持续一周，伴有轻微疼痛
additionalFile: [图片文件]
```

**响应示例**

```json
{
  "success": true,
  "message": "补充望诊信息成功",
  "data": {
    "results": "遗传因素、饮食不当、缺乏运动。头痛、头晕、心悸。药物治疗、生活方式改变。低盐饮食、适量运动、戒烟限酒。"
  }
}
```

---

## 5. 辅助问诊接口 (inquiryRoutes)

### 5.1 初步问诊

**基本信息**

- 路径: `/api/inquiry`
- 方法: `POST`
- 描述: 提交初步问诊信息
- 认证: 需要Bearer Token

**请求参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| patientInfo | String | 是 | 患者信息 |
| symptoms | String | 是 | 患者症状描述 |

**请求体示例**

```json
{
  "age": 35,
  "gender": "男",
  "symptoms": "头痛、头晕"
}
```

**响应示例**

```json
{
  "success": true,
  "message": "问诊分析成功",
  "data": {
    "results": "遗传因素、饮食不当、缺乏运动。头痛、头晕、心悸。药物治疗、生活方式改变。低盐饮食、适量运动、戒烟限酒。",
    "analysisId": "inquiry_123456789"
  }
}
```

---

### 5.2 问诊补充

**基本信息**

- 路径: `/api/inquiry/complete`
- 方法: `POST`
- 描述: 补充问诊信息
- 认证: 需要Bearer Token

**请求参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| additionalInfo | String | 是 | 补充的问诊信息 |
| additionalFile | File | 否 | 附加的图片文件 |

**请求体示例**

```
inquiryId: inquiry_123456789
additionalInfo: 患者有高血压家族史，平时工作压力大，经常熬夜
additionalFile: [检查报告图片]
```

**响应示例**

```json
{
  "success": true,
  "message": "补充问诊信息成功",
  "data": {
    "results": "遗传因素、饮食不当、缺乏运动。头痛、头晕、心悸。药物治疗、生活方式改变。低盐饮食、适量运动、戒烟限酒。"
  }
}
```

---

## 病历生成接口 (recordRoutes)

### 6.1 生成病历

**基本信息**

- 路径: `/api/record`
- 方法: `POST`
- 描述: 根据问诊和望诊结果生成病历
- 认证: 需要Bearer Token

**请求参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| watchResults | String | 否 | 望诊分析结果 |
| inquiryResults | String | 否 | 问诊分析结果 |

**请求体示例**

```json
{
  "patientInfo": "男，35岁，有XX病家族病史",
  "watchResults": "遗传因素、饮食不当、缺乏运动。头痛、头晕、心悸。",
  "inquiryResults": "患者有家族史，症状持续一周。"
}
```

**响应示例**

```json
{
  "success": true,
  "message": "病历生成成功",
  "data": {
    "symptoms": "主诉头痛、头晕。",
    "disease": "经望诊分析，可能与遗传因素、饮食不当、缺乏运动有关。问诊补充信息显示患者有家族史，症状持续一周。",
    "prescription": "建议低盐饮食、适量运动、戒烟限酒。"
  }
}
```

### 6.2 导入病历

**基本信息**

- 路径: `/api/record/import`
- 方法: `POST`
- 描述: 导入已有病历图片，生成新病历
- 认证: 需要Bearer Token
- 内容类型: multipart/form-data

**请求参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| recordImage | File | 是 | 要导入的病历图片 |

**请求体示例**

```
recordImage: [file]
```

**响应示例**

```json
{
  "success": true,
  "message": "病历导入成功",
  "data": {
    "symptoms": "主诉头痛、头晕。",
    "disease": "经望诊分析，可能与遗传因素、饮食不当、缺乏运动有关。问诊补充信息显示患者有家族史，症状持续一周。",
    "prescription": "建议低盐饮食、适量运动、戒烟限酒。"
  }
}
```

### 6.3 保存病历

**基本信息**

- 路径: `/api/record/save`
- 方法: `POST`
- 描述: 保存生成的病历至患者病历记录
- 认证: 需要Bearer Token

**请求参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| recordImage | File | 是 | 要保存的病历图片 |
| patientId | Number | 是 | 患者的唯一标识 |

**请求体示例**

```
recordImage: [file],
patientId: 1
```

**响应示例**

```json
{
  "success": true,
  "message": "病历保存成功"
}
```

---

需要接入AI配合实现的接口有 3.1, 4.1, 4.2, 5.1, 5.2, 6.1, 6.2。
