"""
增强版main.py - 完整的聊天机器人API接口
包含聊天、文档上传、RAG管理、状态查询等功能
保留原有的错误检测和日志系统
修复时间戳获取问题
"""

import logging
import traceback
import os
import tempfile
import shutil
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 设置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ChatBot API",
    description="完整的聊天机器人API，支持RAG、文档上传和对话管理",
    version="2.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
chatbot = None
initialization_error = None

# Pydantic模型
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str
    metadata: dict = {}
    error: str = None

class ConfigUpdateRequest(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None

class PromptUpdateRequest(BaseModel):
    template: str

class MemoryTypeRequest(BaseModel):
    memory_type: str  # "buffer" or "summary"

class StatusResponse(BaseModel):
    status: str
    chatbot_ready: bool
    chatbot_status: dict = {}
    detailed_status: str = ""
    error: str = None
    message: str = ""

def get_current_timestamp():
    """获取当前时间戳的安全方法"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def initialize_chatbot():
    """初始化聊天机器人，包含详细错误报告"""
    global chatbot, initialization_error
    
    try:
        logger.info("开始初始化聊天机器人...")
        
        # 1. 检查配置导入
        try:
            from config import (
                DEFAULT_MODEL_NAME, 
                DEFAULT_BASE_URL, 
                DEFAULT_MODEL_CONFIGS,
                DEFAULT_API_KEY
            )
            logger.info("✓ 配置文件导入成功")
            logger.info(f"模型名称: {DEFAULT_MODEL_NAME}")
            logger.info(f"API URL: {DEFAULT_BASE_URL}")
            logger.info(f"API Key设置: {'已设置' if DEFAULT_API_KEY else '未设置'}")
        except ImportError as e:
            error_msg = f"配置文件导入失败: {str(e)}"
            logger.error(error_msg)
            initialization_error = error_msg
            return False
        except Exception as e:
            error_msg = f"配置检查失败: {str(e)}"
            logger.error(error_msg)
            initialization_error = error_msg
            return False
        
        # 2. 检查必要配置项
        if not DEFAULT_API_KEY or DEFAULT_API_KEY.strip() == "":
            error_msg = "API_KEY未设置或为空"
            logger.error(error_msg)
            initialization_error = error_msg
            return False
        
        if not DEFAULT_MODEL_NAME or DEFAULT_MODEL_NAME.strip() == "":
            error_msg = "DEFAULT_MODEL_NAME未设置或为空"
            logger.error(error_msg)
            initialization_error = error_msg
            return False
        
        if not DEFAULT_BASE_URL or DEFAULT_BASE_URL.strip() == "":
            error_msg = "DEFAULT_BASE_URL未设置或为空"
            logger.error(error_msg)
            initialization_error = error_msg
            return False
        
        # 3. 导入chatbot类
        try:
            from chatbot import LangChainChatBot
            logger.info("✓ ChatBot类导入成功")
        except ImportError as e:
            error_msg = f"ChatBot类导入失败: {str(e)}"
            logger.error(error_msg)
            initialization_error = error_msg
            return False
        
        # 4. 创建chatbot实例
        try:
            logger.info("创建ChatBot实例...")
            chatbot = LangChainChatBot(
                api_key=DEFAULT_API_KEY,
                model_name=DEFAULT_MODEL_NAME,
                model_configs=DEFAULT_MODEL_CONFIGS,
                base_url=DEFAULT_BASE_URL,
                auto_load_vector_store=True,  # 启用自动加载向量库
                vector_store_path="RAG"  # 默认向量库路径
            )
            logger.info("✓ ChatBot实例创建成功")
            
            # 5. 检查chatbot状态
            status = chatbot.get_status()
            logger.info(f"ChatBot状态: {status}")
            
            # 6. 测试chatbot响应
            try:
                test_response = chatbot.generate_response("测试消息")
                logger.info(f"✓ ChatBot测试成功，响应长度: {len(test_response)}")
            except Exception as test_e:
                logger.warning(f"ChatBot测试失败，但实例已创建: {str(test_e)}")
            
            return True
            
        except Exception as e:
            error_msg = f"ChatBot实例创建失败: {str(e)}"
            logger.error(error_msg)
            logger.error(f"详细错误: {traceback.format_exc()}")
            initialization_error = error_msg
            return False
    
    except Exception as e:
        error_msg = f"初始化过程中发生未预期的错误: {str(e)}"
        logger.error(error_msg)
        logger.error(f"详细错误: {traceback.format_exc()}")
        initialization_error = error_msg
        return False

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化chatbot"""
    logger.info("FastAPI应用启动中...")
    
    # 创建必要的目录
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("RAG", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    success = initialize_chatbot()
    if success:
        logger.info("✓ 聊天机器人初始化成功")
    else:
        logger.error(f"✗ 聊天机器人初始化失败: {initialization_error}")

# ===== 基础路由 =====
@app.get("/")
async def read_root():
    """根路径，返回前端页面"""
    return FileResponse("frontend.html")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """获取系统状态"""
    global chatbot, initialization_error
    
    if chatbot is None:
        return StatusResponse(
            status="未初始化",
            chatbot_ready=False,
            error=initialization_error,
            message="聊天机器人未成功初始化"
        )
    
    try:
        chatbot_status = chatbot.get_status()
        return StatusResponse(
            status="已初始化",
            chatbot_ready=True,
            chatbot_status=chatbot_status,
            detailed_status=chatbot.get_detailed_status()
        )
    except Exception as e:
        return StatusResponse(
            status="状态检查失败",
            chatbot_ready=False,
            error=str(e)
        )

# ===== 聊天功能 =====
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """聊天端点"""
    global chatbot, initialization_error
    
    try:
        # 详细的状态检查
        if chatbot is None:
            error_msg = f"机器人未初始化。初始化错误: {initialization_error}"
            logger.error(error_msg)
            logger.error("尝试重新初始化...")
            
            # 尝试重新初始化
            if initialize_chatbot():
                logger.info("重新初始化成功")
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"机器人初始化失败: {initialization_error}"
                )
        
        # 验证输入
        if not request.message or request.message.strip() == "":
            raise HTTPException(status_code=400, detail="消息不能为空")
        
        # 生成回复
        logger.info(f"处理聊天请求: {request.message[:50]}...")
        response = chatbot.generate_response(request.message)
        
        # 获取状态信息
        status = chatbot.get_status()
        metadata = {
            "rag_enabled": status.get("rag_ready", False),
            "vector_store_loaded": status.get("vector_store_loaded", False),
            "timestamp": get_current_timestamp()
        }
        
        logger.info(f"✓ 聊天处理成功，响应长度: {len(response)}")
        
        return ChatResponse(
            response=response,
            status="success",
            metadata=metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"聊天处理失败: {str(e)}"
        logger.error(error_msg)
        logger.error(f"详细错误: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@app.post("/chat/file")
async def chat_with_file(
    file: UploadFile = File(...),
    message: str = Form(default="请分析这个文件的内容")
):
    """上传文件并与文件内容对话"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=500, detail="聊天机器人未初始化")
    
    temp_file_path = None
    try:
        # 检查文件类型
        allowed_extensions = {'.txt', '.md', '.pdf', '.docx', '.doc'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型: {file_ext}. 支持的类型: {', '.join(allowed_extensions)}"
            )
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
        
        logger.info(f"处理文件: {file.filename}")
        
        # 读取文件内容
        if file_ext == '.txt' or file_ext == '.md':
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        elif file_ext == '.pdf':
            # 这里需要实现PDF读取逻辑
            file_content = f"PDF文件: {file.filename} (需要实现PDF解析)"
        else:
            file_content = f"文档文件: {file.filename} (需要实现文档解析)"
        
        # 构建包含文件内容的消息
        full_message = f"{message}\n\n文件名: {file.filename}\n文件内容:\n{file_content[:5000]}..."  # 限制长度
        
        # 生成回复
        response = chatbot.generate_response(full_message)
        
        return ChatResponse(
            response=response,
            status="success",
            metadata={
                "filename": file.filename,
                "file_type": file_ext,
                "file_size": len(file_content),
                "timestamp": get_current_timestamp()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"文件处理失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# ===== RAG功能 =====
@app.post("/rag/upload")
async def upload_documents_to_rag(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200)
):
    """上传文档到RAG向量库"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=500, detail="聊天机器人未初始化")
    
    if not chatbot.rag_manager:
        raise HTTPException(status_code=500, detail="RAG管理器未初始化")
    
    temp_files = []
    try:
        # 保存上传的文件
        file_paths = []
        for file in files:
            # 检查文件类型
            allowed_extensions = {'.txt', '.md', '.pdf', '.docx', '.doc'}
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件 {file.filename} 类型不支持: {file_ext}"
                )
            
            # 保存到临时目录
            temp_path = os.path.join("temp", file.filename)
            with open(temp_path, "wb") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
            
            file_paths.append(temp_path)
            temp_files.append(temp_path)
        
        logger.info(f"开始加载 {len(file_paths)} 个文档到RAG系统")
        
        # 加载文档到RAG
        num_chunks = chatbot.load_documents(file_paths, chunk_size, chunk_overlap)
        
        if num_chunks > 0:
            # 自动保存向量存储
            save_result = chatbot.save_vector_store()
            
            return {
                "status": "success",
                "message": f"成功加载 {len(file_paths)} 个文档，生成 {num_chunks} 个文本块",
                "files_processed": [f.filename for f in files],
                "chunks_generated": num_chunks,
                "vector_store_saved": "成功" in save_result,
                "save_result": save_result,
                "timestamp": get_current_timestamp()
            }
        else:
            return {
                "status": "failed",
                "message": "文档加载失败，未生成文本块",
                "files_processed": [f.filename for f in files],
                "timestamp": get_current_timestamp()
            }
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"RAG文档上传失败: {str(e)}"
        logger.error(error_msg)
        logger.error(f"详细错误: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

@app.post("/rag/save")
async def save_vector_store(path: str = Form(default="RAG")):
    """保存向量存储"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=500, detail="聊天机器人未初始化")
    
    try:
        result = chatbot.save_vector_store(path)
        
        return {
            "status": "success" if "成功" in result else "failed",
            "message": result,
            "path": path,
            "timestamp": get_current_timestamp()
        }
    except Exception as e:
        error_msg = f"保存向量存储失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/rag/load")
async def load_vector_store(path: str = Form(default="RAG")):
    """加载向量存储"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=500, detail="聊天机器人未初始化")
    
    try:
        result = chatbot.load_vector_store(path)
        
        return {
            "status": "success" if "成功" in result else "failed",
            "message": result,
            "path": path,
            "chatbot_status": chatbot.get_status(),
            "timestamp": get_current_timestamp()
        }
    except Exception as e:
        error_msg = f"加载向量存储失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/rag/reload")
async def reload_vector_store():
    """重新加载向量存储"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=500, detail="聊天机器人未初始化")
    
    try:
        result = chatbot.reload_vector_store()
        
        return {
            "status": "success",
            "message": result,
            "chatbot_status": chatbot.get_status(),
            "timestamp": get_current_timestamp()
        }
    except Exception as e:
        error_msg = f"重新加载向量存储失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# ===== 配置管理 =====
@app.post("/config/pipeline")
async def update_pipeline_config(request: ConfigUpdateRequest):
    """更新pipeline配置"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=500, detail="聊天机器人未初始化")
    
    try:
        chatbot.update_pipeline_config(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            presence_penalty=request.presence_penalty
        )
        
        return {
            "status": "success",
            "message": "Pipeline配置更新成功",
            "config": request.dict(exclude_none=True),
            "timestamp": get_current_timestamp()
        }
    except Exception as e:
        error_msg = f"Pipeline配置更新失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/config/memory")
async def set_memory_type(request: MemoryTypeRequest):
    """设置记忆类型"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=500, detail="聊天机器人未初始化")
    
    if request.memory_type not in ["buffer", "summary"]:
        raise HTTPException(status_code=400, detail="记忆类型必须是 'buffer' 或 'summary'")
    
    try:
        chatbot.set_memory_type(request.memory_type)
        
        return {
            "status": "success",
            "message": f"记忆类型设置为: {request.memory_type}",
            "timestamp": get_current_timestamp()
        }
    except Exception as e:
        error_msg = f"设置记忆类型失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/config/prompt")
async def update_prompt_template(request: PromptUpdateRequest):
    """更新提示模板"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=500, detail="聊天机器人未初始化")
    
    try:
        chatbot.customize_prompt(request.template)
        
        return {
            "status": "success",
            "message": "提示模板更新成功",
            "timestamp": get_current_timestamp()
        }
    except Exception as e:
        error_msg = f"提示模板更新失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# ===== 会话管理 =====
@app.post("/chat/clear")
async def clear_chat_history():
    """清除对话历史"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=500, detail="聊天机器人未初始化")
    
    try:
        chatbot.clear_history()
        
        return {
            "status": "success",
            "message": "对话历史已清除",
            "timestamp": get_current_timestamp()
        }
    except Exception as e:
        error_msg = f"清除历史失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# ===== 系统管理 =====
@app.post("/system/reinitialize")
async def reinitialize_chatbot():
    """手动重新初始化chatbot"""
    global chatbot, initialization_error
    
    logger.info("手动重新初始化chatbot...")
    chatbot = None
    initialization_error = None
    
    success = initialize_chatbot()
    
    if success:
        return {
            "status": "success",
            "message": "ChatBot重新初始化成功",
            "chatbot_status": chatbot.get_status(),
            "timestamp": get_current_timestamp()
        }
    else:
        return {
            "status": "failed",
            "message": f"ChatBot重新初始化失败: {initialization_error}",
            "error": initialization_error,
            "timestamp": get_current_timestamp()
        }

@app.get("/system/health")
async def health_check():
    """健康检查"""
    global chatbot
    
    try:
        if chatbot is None:
            return {
                "status": "unhealthy", 
                "message": "ChatBot未初始化",
                "timestamp": get_current_timestamp()
            }
        
        # 简单的健康检查
        status = chatbot.get_status()
        is_healthy = status.get("initialized", False) and status.get("llm_ready", False)
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": get_current_timestamp(),
            "chatbot_status": status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": get_current_timestamp()
        }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("启动增强版ChatBot API服务器...")
    logger.info("访问 http://localhost:8000/docs 查看API文档")
    logger.info("访问 http://localhost:8000/status 查看系统状态")
    logger.info("访问 http://localhost:8000 使用聊天界面")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)