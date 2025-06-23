# LangChain 聊天机器人

这是一个基于 LangChain 和 DeepSeek API 的智能聊天机器人，支持 RAG（检索增强生成）功能。

## 📁 文件结构

```
chatbot/
├── config.py              # 配置文件 - 包含默认设置和常量
├── chatbot.py              # 聊天机器人核心类
├── rag_manager.py          # RAG功能管理器
├── utils.py               # 工具函数和数据库管理器
├── main.py                # Web API服务器入口
├── frontend_example.html  # 前端界面示例
├── requirements.txt       # 依赖列表
└── README.md             # 使用说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

在 `config.py` 文件中修改你的 DeepSeek API 密钥：

```python
DEFAULT_API_KEY = "your-api-key-here"
```

或者通过命令行参数传入：

```bash
python main.py --api_key your-api-key-here
```

### 3. 启动Web API服务器

```bash
python main.py
```

服务器将在 `http://localhost:8000` 启动

### 4. 访问前端界面

打开 `frontend_example.html` 文件即可使用Web界面与聊天机器人交互。

或者直接访问API文档：
- Swagger文档: `http://localhost:8000/docs`
- ReDoc文档: `http://localhost:8000/redoc`

## 💡 使用方法

### Web界面 (推荐)

1. 启动API服务器：`python main.py`
2. 打开 `frontend_example.html` 文件
3. 通过友好的Web界面与AI对话

### API调用

#### 基本聊天
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "你好", "session_id": "test"}'
```

#### 上传文档
```bash
curl -X POST "http://localhost:8000/documents/upload" \
     -F "file=@document.txt" \
     -F "session_id=test"
```

#### 获取配置
```bash
curl "http://localhost:8000/config?session_id=test"
```

### 主要API端点

- `POST /chat` - 发送聊天消息
- `POST /documents/upload` - 上传文档文件
- `POST /documents/load` - 从路径加载文档
- `GET /config` - 获取配置信息
- `POST /config` - 更新配置
- `POST /vector-store/save` - 保存向量存储
- `POST /vector-store/load` - 加载向量存储
- `GET /sessions` - 获取会话信息
- `DELETE /sessions/{session_id}` - 删除会话
- `GET /stats` - 获取性能统计

## ⚙️ 命令行参数

```bash
python main.py [选项]

选项:
  --api_key API_KEY         DeepSeek API密钥
  --model MODEL             使用的模型名称 (默认: deepseek-chat)
  --memory {buffer,summary} 记忆类型 (默认: buffer)
  --temperature TEMP        温度参数 (默认: 0.7)
  --max_tokens TOKENS       最大token数 (默认: 1024)
  --embedding_path PATH     嵌入模型保存位置
  --auto_load PATH          启动时自动加载文档
  --auto_import PATH        启动时自动导入向量存储
  --db_url URL              数据库后端URL
  --db_token TOKEN          数据库认证令牌
```

## 🔧 模块说明

### config.py
包含所有配置常量和默认设置：
- API配置（密钥、URL、模型名称）
- RAG配置（分块大小、嵌入模型等）
- 提示模板
- 支持的文件编码

### chatbot.py
聊天机器人的核心类 `LangChainChatBot`：
- LLM初始化和配置
- 对话链管理
- 记忆管理
- 响应生成
- 配置更新方法

### rag_manager.py
RAG功能管理器 `RAGManager`：
- 文档加载和处理
- 向量存储创建和管理
- 检索器配置
- QA链创建
- 嵌入模型管理

### utils.py
工具函数和辅助类：
- `DatabaseManager` - 数据库交互
- `PerformanceMonitor` - 性能监控
- `direct_deepseek_call` - 直接API调用
- 各种辅助函数

### main.py
Web API服务器和用户界面：
- FastAPI应用和路由定义
- 会话管理和多用户支持
- 文件上传处理
- API端点实现
- 性能监控集成

## 📊 功能特性

### ✅ 已实现
- **Web API服务器** - 基于FastAPI的RESTful API
- **前端界面** - 现代化的Web聊天界面
- **多会话支持** - 支持同时多个用户会话
- **文件上传** - 支持通过Web界面上传文档
- **基本对话功能** - 智能问答和对话记忆
- **RAG文档检索** - 支持文档内容检索增强
- **多种文件格式** - 支持 .txt, .pdf, 目录批量处理
- **自动编码检测** - 智能识别文件编码格式
- **向量存储持久化** - 本地保存和加载向量数据
- **性能监控** - 实时响应时间和使用统计
- **数据库集成** - 可选的数据库后端支持
- **自动API文档** - Swagger/OpenAPI文档自动生成

### 🔄 Web功能优势
- **跨平台访问** - 任何设备的浏览器都能使用
- **实时交互** - 现代化的聊天界面体验
- **文件拖拽上传** - 便捷的文档处理方式
- **会话管理** - 支持多用户同时使用
- **状态监控** - 实时查看系统状态和性能
- **API调试** - 内置的API测试界面

## 🛠️ 开发和调试

### Web API开发

启动开发服务器（自动重载）：
```bash
python main.py --reload --host 127.0.0.1 --port 8000
```

API文档访问：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 单独测试模块

```python
# 测试RAG管理器
from rag_manager import RAGManager
rag = RAGManager()
rag.load_documents("test.txt")

# 测试聊天机器人
from chatbot import LangChainChatBot
bot = LangChainChatBot(api_key="your-key")
response = bot.generate_response("Hello")

# 测试数据库管理器
from utils import DatabaseManager
db = DatabaseManager("http://localhost:8000", "token")
```

### 前端开发

修改 `frontend_example.html` 中的 `API_BASE_URL` 来连接不同的API服务器：

```javascript
const API_BASE_URL = 'http://your-server:8000';
```

### 自定义配置

修改 `config.py` 中的常量来调整默认行为：

```python
DEFAULT_CHUNK_SIZE = 500  # 减小分块大小
DEFAULT_EMBEDDING_MODEL = "your-model"  # 使用不同的嵌入模型
```

### API测试示例

```bash
# 健康检查
curl http://localhost:8000/

# 发送聊天消息
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "session_id": "test"}'

# 获取统计信息
curl http://localhost:8000/stats
```

## 🔍 故障排除

### 常见问题

1. **Web服务器启动失败**
   - 检查端口是否被占用：`netstat -an | grep 8000`
   - 尝试使用其他端口：`python main.py --port 8001`

2. **前端无法连接API**
   - 确认API服务器正在运行
   - 检查CORS设置和防火墙
   - 验证API_BASE_URL配置

3. **模块导入错误**
   - 确保所有文件在同一目录下
   - 检查Python路径和依赖安装

4. **API调用失败**
   - 检查API密钥和网络连接
   - 查看服务器日志输出
   - 验证请求格式和参数

5. **文档上传失败**
   - 检查文件大小限制
   - 验证文件格式和编码
   - 确保有足够的磁盘空间

6. **向量存储错误**
   - 检查磁盘空间和权限
   - 验证嵌入模型下载
   - 清理损坏的向量存储文件

### 调试技巧

- 使用 `--reload` 参数启动开发服务器
- 查看浏览器开发者工具的网络选项卡
- 检查API服务器的控制台输出
- 使用API文档页面测试端点

## 📝 注意事项

### 安全性
- 确保API密钥的安全性，不要提交到版本控制
- 在生产环境中限制CORS允许的域名
- 考虑添加API访问认证和限流

### 性能
- 大文档处理可能需要较长时间
- 向量存储会占用磁盘空间
- 建议定期清理无用的向量存储文件
- 可考虑使用GPU版本的FAISS提升性能

### 部署
- 生产环境建议使用反向代理（如Nginx）
- 可使用Docker容器化部署
- 建议配置日志文件和监控

### 扩展性
- 支持水平扩展（多个API实例）
- 可集成Redis等缓存系统
- 支持外部数据库存储会话信息



# LangChainChatBot 函数速查表

## 🚀 核心对话功能

| 函数 | 用途 | 关键参数 | 返回值 |
|------|------|----------|--------|
| `generate_response()` | 生成AI回复（主要接口） | `user_input: str` | `str` |
| `process_file()` | 处理文件内容 | `file_path: str` | `str` |

## ⚙️ 配置管理

| 函数 | 用途 | 关键参数 | 返回值 |
|------|------|----------|--------|
| `update_pipeline_config()` | 更新AI参数 | `temperature, max_tokens, top_p` | `None` |
| `update_model_config()` | 更新模型配置 | `config_updates: dict` | `None` |
| `set_memory_type()` | 切换记忆类型 | `memory_type: str` | `None` |
| `customize_prompt()` | 自定义提示模板 | `new_template: str` | `None` |
| `clear_history()` | 清除对话历史 | 无 | `None` |

## 📚 RAG知识库管理

| 函数 | 用途 | 关键参数 | 返回值 |
|------|------|----------|--------|
| `load_documents()` | 加载文档到知识库 | `file_paths, chunk_size=1000` | `int` |
| `save_vector_store()` | 保存向量库到磁盘 | `path="RAG"` | `str` |
| `load_vector_store()` | 从磁盘加载向量库 | `path="RAG"` | `str` |

## 📊 状态监控

| 函数 | 用途 | 关键参数 | 返回值 |
|------|------|----------|--------|
| `get_status()` | 获取系统状态 | 无 | `dict` |
| `get_detailed_status()` | 获取详细状态报告 | 无 | `str` |

## 🔧 内部初始化函数（私有）

| 函数 | 用途 |
|------|------|
| `_init_llm()` | 初始化大语言模型 |
| `_init_memory()` | 初始化对话记忆 |
| `_init_conversation()` | 初始化对话链 |
| `_init_rag()` | 初始化RAG系统 |
| `_init_database()` | 初始化数据库连接 |
| `_generate_rag_response()` | RAG模式回复生成 |

---

## 🏷️ 使用标签说明

- **🔥 高频**：`generate_response()`, `get_status()`, `clear_history()`
- **⚙️ 配置**：`update_pipeline_config()`, `set_memory_type()` 
- **📚 知识库**：`load_documents()`, `save_vector_store()`, `load_vector_store()`
- **🔧 管理**：`process_file()`, `get_detailed_status()`

---

## 📝 快速使用示例

```python
# 基础对话
bot = LangChainChatBot(api_key="your_key")
response = bot.generate_response("你好")

# 知识库模式
bot.load_documents(["doc1.pdf", "doc2.txt"])
bot.save_vector_store("my_kb")

# 下次快速启动
bot.load_vector_store("my_kb")

# 调整参数
bot.update_pipeline_config(temperature=0.5)

# 状态检查
status = bot.get_status()
```


# main.py接口总结


## 📊 系统状态

| 方法 | 路径       | 作用                         |
|:------|:------------|:----------------------------|
| `GET` | `/`          | 返回前端页面（frontend.html）|
| `GET` | `/status`    | 获取系统与聊天机器人状态     |

## 💬 聊天功能

| 方法   | 路径            | 作用                              |
|:----------|:-----------------|:----------------------------------|
| `POST`    | `/chat`            | 发送消息，获取聊天回复            |
| `POST`    | `/chat/file`       | 上传文件，结合消息对文件内容对话分析 |
| `POST`    | `/chat/clear`      | 清空对话历史                       |

## 📄 RAG 文档管理

| 方法   | 路径            | 作用                        |
|:----------|:-----------------|:-----------------------------|
| `POST`    | `/rag/upload`      | 上传文档到 RAG 向量库并分块入库 |
| `POST`    | `/rag/save`        | 手动保存当前向量存储到磁盘       |
| `POST`    | `/rag/load`        | 从指定路径加载向量存储           |
| `POST`    | `/rag/reload`      | 重新加载当前向量存储             |

## ⚙️ 配置管理

| 方法   | 路径               | 作用                         |
|:----------|:--------------------|:------------------------------|
| `POST`    | `/config/pipeline`     | 更新聊天模型 pipeline 配置      |
| `POST`    | `/config/memory`       | 设置记忆类型（buffer / summary）|
| `POST`    | `/config/prompt`       | 更新聊天提示词模板               |

## 📌 总览  
共 **13 个 API**，涵盖：
- ✅ 系统状态
- ✅ 聊天功能
- ✅ RAG文档管理
- ✅ 配置管理
- ✅ 会话管理

