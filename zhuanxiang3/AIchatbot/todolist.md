PS F:\Files\bisai\Tiaozhanbei\zhuanxiang3\AIchatbot> python tests/server_test.py
🚀 API服务器集成测试 - 修复版（严格匹配API文档）
======================================================================
⏳ 等待服务器启动...
✅ 服务器已启动 (1s)

🔍 测试基础接口
----------------------------------------
✅ 服务器信息
   📈 服务器版本: 2.0.0
✅ 健康检查
   ⏱️ 运行时间: 26秒
✅ Graph统计
   📊 数据字段: ['graph_compiled', 'last_compiled_time', 'compilation_lock', 'uptime_hours']
✅ 404错误处理
   ⚠️ API返回失败: 接口不存在

🔍 测试搜索API
----------------------------------------
✅ 疾病搜索-头痛
   📊 数据字段: ['diseases']
✅ 疾病搜索-发热
   📊 数据字段: ['diseases']
❌ 空搜索词 - 期望200, 实际400
   错误: 搜索关键词不能为空
✅ 缺少搜索参数
   ⚠️ API返回失败: 搜索关键词不能为空

🔍 测试望诊API
----------------------------------------
✅ 4.1-舌诊图片分析
   📊 数据字段: ['results']
✅ 4.2-望诊补充分析
   📊 数据字段: ['results']
✅ 4.2-缺少additionalInfo
   ⚠️ API返回失败: 必须提供之前的分析结果或补充信息
✅ 4.2-缺少prevAnalysis
   ⚠️ API返回失败: 必须提供之前的分析结果或补充信息
✅ 4.1-缺少图片文件
   ⚠️ API返回失败: 未找到上传的图像文件

🔍 测试问诊API
----------------------------------------
✅ 5.1-初步问诊
   📊 数据字段: ['results']
❌ 5.2-问诊补充 - 期望200, 实际400
   错误: 必须提供之前的问诊结果或补充信息
✅ 5.1边界测试-缺少所有参数
   ⚠️ API返回失败: 请求数据格式错误，需要JSON格式
✅ 5.1边界测试-患者信息为空
   ⚠️ API返回失败: 年龄不能为空
✅ 5.1边界测试-症状为空
   ⚠️ API返回失败: 年龄不能为空
✅ 5.1边界测试-缺少症状
   ⚠️ API返回失败: 年龄不能为空
✅ 5.1边界测试-缺少患者信息
   ⚠️ API返回失败: 年龄不能为空
✅ 5.1边界测试-症状为空
   ⚠️ API返回失败: 年龄不能为空
✅ 5.1边界测试-患者信息为空
   ⚠️ API返回失败: 年龄不能为空
✅ 5.1边界测试-症状过长
   ⚠️ API返回失败: 年龄不能为空
✅ 5.2-缺少additionalInfo
   ⚠️ API返回失败: 必须提供之前的问诊结果或补充信息
✅ 5.2-缺少prevInquiry
   ⚠️ API返回失败: 必须提供之前的问诊结果或补充信息

🔍 测试病历生成API
----------------------------------------
✅ 6.1-病历生成
   📊 数据字段: ['symptoms', 'disease', 'prescription']
   📋 症状: 主诉：暂无明确症状。...
   🔍 诊断: 补充问诊：尊敬的患者：

您好！根据您描述的症状和中医辨证分析，现将您的情况整理如下，以便您更好地理...
   💊 处方: 暂无特殊建议。...
❌ 6.1-仅望诊结果 - 期望200, 实际500
   错误: 病历生成失败: 'dict' object has no attribute 'strip'
❌ 6.1-仅问诊结果 - 期望200, 实际500
   错误: 病历生成失败: 'dict' object has no attribute 'strip'
✅ 6.1-空病历数据
   ⚠️ API返回失败: 请求数据格式错误，需要JSON格式
✅ 6.1-空字符串参数
   ⚠️ API返回失败: watchResults 和 inquiryResults 至少需要提供一个

🔍 测试文档导入API
----------------------------------------
✅ 6.2-医学文档图片导入
   📊 数据字段: ['symptoms', 'disease', 'prescription']
✅ 6.2-空导入数据
   ⚠️ API返回失败: 未找到上传的病历图片文件，参数名应为'recordImage'
✅ 6.2-无效文件格式
   ⚠️ API返回失败: 不支持的文件格式，支持的格式: jpg, gif, pdf, bmp, jpeg, png
✅ 6.2-错误参数名
   ⚠️ API返回失败: 未找到上传的病历图片文件，参数名应为'recordImage'

🔍 测试AI智能分析API
----------------------------------------
✅ 7.1-AI文本分析
   📊 数据字段: ['solution']
✅ 7.1-AI图文分析
   📊 数据字段: ['solution']
✅ 7.1-AI纯图片分析
   📊 数据字段: ['solution']
✅ 7.1-contextMode=auto
   📊 数据字段: ['solution']
✅ 7.1-contextMode=simple
   📊 数据字段: ['solution']
✅ 7.1-contextMode=comprehensive
   📊 数据字段: ['solution']
✅ 7.1-AI分析无输入
   ⚠️ API返回失败: 请提供查询文本或上传文件
✅ 7.1-仅有contextMode
   ⚠️ API返回失败: 请提供查询文本或上传文件

🔍 测试Session集成功能
----------------------------------------
✅ Session测试-4.1舌诊
   📊 数据字段: ['results']
❌ Session测试-5.1问诊 - 期望200, 实际400
   错误: 年龄不能为空
✅ Session测试-7.1AI综合分析
   📊 数据字段: ['solution']
❌ Session测试-6.1病历生成 - 期望200, 实际500
   错误: 病历生成失败: 'dict' object has no attribute 'strip'
   ⚠️ Session集成流程部分失败

🔍 测试错误处理
----------------------------------------
❌ 错误Content-Type处理
   🔥 并发请求测试
   📊 并发成功率: 100.0% (10/10)
   ✅ 并发测试通过

======================================================================
📊 测试结果总结
======================================================================
📈 总测试数: 46
❌ 失败数: 6
✅ 成功数: 40
📊 成功率: 87.0%
⏱️ 总耗时: 320.46秒

🔍 API规范匹配验证:
   4.1 图片望诊: image参数 ✅
   4.2 望诊补充: prevAnalysis+additionalInfo都必选 ✅
   5.1 初步问诊: patientInfo+symptoms参数 ✅
   5.2 问诊补充: prevInquiry+additionalInfo都必选 ✅
   6.1 病历生成: watchResults+inquiryResults至少一个 ✅
   6.2 导入病历: recordImage参数 ✅
   7.1 AI分析: query+file至少一个，支持contextMode ✅

🔍 功能验证:
   ✅ search_headache: 正常
   ✅ watch_result: 正常
   ✅ inquiry_result: 正常
   ✅ import_result: 正常
   ✅ ai_analysis: 正常

💥 6 个测试失败，请检查API实现


核心修改：患者信息更改为年龄+性别

### 4. 辅助望诊接口 (searchRoutes)

#### 4.1 图片望诊

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


#### 4.2 望诊补充

**基本信息**

- 路径: `/api/watch/complete`
- 方法: `POST`
- 描述: 补充望诊信息
- 认证: 需要Bearer Token
- 内容类型: multipart/form-data

**请求参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| prevAnalysis | String | 是 | 之前望诊的分析 |
| additionalInfo | String | 是 | 补充的望诊信息 |
| additionalFile | File | 否 | 附加的图片文件 |


#### 5.1 初步问诊

**基本信息**

- 路径: `/api/inquiry`
- 方法: `POST`
- 描述: 提交初步问诊信息
- 认证: 需要Bearer Token

**请求参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| age | String | 是 | 患者信息 |
| gender| String | 是 | 患者信息 |
| symptoms | String | 是 | 患者症状描述 |

#### 5.2 问诊补充

**基本信息**

- 路径: `/api/inquiry/complete`
- 方法: `POST`
- 描述: 补充问诊信息
- 认证: 需要Bearer Token

**请求参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| prevInquiry | String | 是 | 之前问诊的分析 |
| additionalInfo | String | 是 | 补充的问诊信息 |
| additionalFile | File | 否 | 附加的图片文件 |

### 6. 病历生成接口 (recordRoutes)

#### 6.1 生成病历

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

### 7. AI智能助手接口 (aiRoutes)

#### 7.1 AI智能分析

**基本信息**

- 路径: `/api/ai/analyze`
- 方法: `POST`
- 描述: AI智能分析文本或图片内容，提供医疗建议
- 认证: 需要Bearer Token
- 内容类型: multipart/form-data

**请求参数**
| 参数名 | 类型 | 必选 | 描述 |
| ----------- | ------ | ---- | ------------------------ |
| query | String | 否 | 用户查询的文本内容 |
| file | File | 否 | 上传的图片或文档文件 |
| contextMode | String | 否 | 上下文分析模式，默认为auto |

*注：query和file至少需要提供一个

**contextMode取值说明**
- `auto`：自动模式，有历史诊断记录时进行综合分析，否则进行简单分析（推荐）
- `simple`：简单模式，仅基于当前输入进行分析
- `comprehensive`：综合模式，强制结合历史诊断记录进行分析
