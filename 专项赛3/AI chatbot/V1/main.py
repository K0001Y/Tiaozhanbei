"""
增强版main.py - 完整的聊天机器人API接口
包含聊天、文档上传、RAG管理、状态查询、病历上传等功能
保留原有的错误检测和日志系统，增加完整的异常处理
"""

import logging
import traceback
import os
import tempfile
import shutil
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
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
    title="Enhanced ChatBot API",
    description="完整的中医AI聊天机器人API，支持RAG、文档上传、病历管理和对话管理",
    version="3.0.0"
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
    memory_type: str  # "buffer", "summary", "window"

class StatusResponse(BaseModel):
    status: str
    chatbot_ready: bool
    chatbot_status: dict = {}
    detailed_status: str = ""
    error: str = None
    message: str = ""

class MedicalUploadResponse(BaseModel):
    status: str
    message: str
    document_path: str = ""
    content_length: int = 0
    processed: bool = False
    error: str = None
    timestamp: str = ""

class RAGUploadResponse(BaseModel):
    status: str
    message: str
    files_processed: List[str] = []
    documents_added: int = 0
    error: str = None
    timestamp: str = ""

def get_current_timestamp() -> str:
    """获取当前时间戳的安全方法"""
    try:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"获取时间戳失败: {str(e)}")
        return "时间戳获取失败"

def safe_json_response(data: Dict[str, Any], status_code: int = 200) -> JSONResponse:
    """安全的JSON响应包装器"""
    try:
        # 确保时间戳字段存在
        if 'timestamp' not in data:
            data['timestamp'] = get_current_timestamp()
        
        return JSONResponse(content=data, status_code=status_code)
    except Exception as e:
        logger.error(f"创建JSON响应失败: {str(e)}")
        return JSONResponse(
            content={
                "status": "error",
                "error": f"响应创建失败: {str(e)}",
                "timestamp": get_current_timestamp()
            },
            status_code=500
        )

def initialize_chatbot() -> bool:
    """初始化聊天机器人，包含详细错误报告"""
    global chatbot, initialization_error
    
    try:
        logger.info("开始初始化聊天机器人...")
        
        # 1. 检查配置导入
        try:
            from V2.config import (
                DEFAULT_MODEL_NAME, 
                DEFAULT_BASE_URL, 
                DEFAULT_MODEL_CONFIGS,
                DEFAULT_API_KEY,
                DEFAULT_PROMPT_TEMPLATE
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
            logger.error(traceback.format_exc())
            initialization_error = error_msg
            return False
        
        # 2. 检查必要配置项
        try:
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
                
        except Exception as e:
            error_msg = f"配置验证失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            initialization_error = error_msg
            return False
        
        # 3. 导入ChatBot类
        try:
            from Chat_bot import ChatBot  # 导入我们完善的ChatBot类
            logger.info("✓ ChatBot类导入成功")
        except ImportError as e:
            error_msg = f"ChatBot类导入失败: {str(e)}"
            logger.error(error_msg)
            logger.error("请确保chatbot.py文件在正确的位置")
            initialization_error = error_msg
            return False
        except Exception as e:
            error_msg = f"ChatBot导入过程异常: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            initialization_error = error_msg
            return False
        
        # 4. 创建ChatBot实例
        try:
            logger.info("创建ChatBot实例...")
            
            # 设置API KEY到环境变量
            os.environ['OPENAI_API_KEY'] = DEFAULT_API_KEY
            
            chatbot = ChatBot(
                model_name=DEFAULT_MODEL_NAME,
                base_url=DEFAULT_BASE_URL,
                model_configs=DEFAULT_MODEL_CONFIGS,
                prompt_template=DEFAULT_PROMPT_TEMPLATE
            )
            logger.info("✓ ChatBot实例创建成功")
            
        except Exception as e:
            error_msg = f"ChatBot实例创建失败: {str(e)}"
            logger.error(error_msg)
            logger.error(f"详细错误: {traceback.format_exc()}")
            initialization_error = error_msg
            return False
        
        # 5. 检查ChatBot状态
        try:
            status = chatbot.get_status()
            logger.info(f"ChatBot状态: {status}")
            
            if not status.get('initialized', False):
                error_msg = f"ChatBot初始化状态异常: {status}"
                logger.error(error_msg)
                initialization_error = error_msg
                return False
                
        except Exception as e:
            error_msg = f"ChatBot状态检查失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            initialization_error = error_msg
            return False
        
        # 6. 测试ChatBot响应
        try:
            test_response = chatbot.generate_response("测试消息")
            if test_response and hasattr(test_response, 'content'):
                logger.info(f"✓ ChatBot测试成功，响应长度: {len(test_response.content)}")
            else:
                logger.warning("ChatBot测试响应格式异常，但实例已创建")
                
        except Exception as test_e:
            logger.warning(f"ChatBot测试失败，但实例已创建: {str(test_e)}")
            # 不阻止初始化完成
        
        logger.info("ChatBot初始化完全成功")
        return True
        
    except Exception as e:
        error_msg = f"初始化过程中发生未预期的错误: {str(e)}"
        logger.error(error_msg)
        logger.error(f"详细错误: {traceback.format_exc()}")
        initialization_error = error_msg
        return False

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化chatbot"""
    try:
        logger.info("FastAPI应用启动中...")
        
        # 创建必要的目录
        directories = ["uploads", "RAG", "temp", "medical_records"]
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"✓ 目录创建成功: {directory}")
            except Exception as e:
                logger.error(f"创建目录失败 {directory}: {str(e)}")
        
        success = initialize_chatbot()
        if success:
            logger.info("✓ 聊天机器人初始化成功")
        else:
            logger.error(f"✗ 聊天机器人初始化失败: {initialization_error}")
            
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        logger.error(traceback.format_exc())

# ===== 基础路由 =====
@app.get("/")
async def read_root():
    """根路径，返回前端页面"""
    try:
        if os.path.exists("frontend.html"):
            return FileResponse("frontend.html")
        else:
            return JSONResponse(
                content={
                    "message": "Enhanced ChatBot API is running",
                    "docs": "/docs",
                    "status": "/status",
                    "timestamp": get_current_timestamp()
                }
            )
    except Exception as e:
        logger.error(f"根路径访问失败: {str(e)}")
        return safe_json_response({
            "status": "error",
            "error": f"根路径访问失败: {str(e)}"
        }, 500)

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """获取系统状态"""
    global chatbot, initialization_error
    
    try:
        if chatbot is None:
            return StatusResponse(
                status="未初始化",
                chatbot_ready=False,
                error=initialization_error or "ChatBot未初始化",
                message="聊天机器人未成功初始化"
            )
        
        try:
            chatbot_status = chatbot.get_status()
            detailed_status = chatbot.get_detailed_status()
            
            return StatusResponse(
                status="已初始化",
                chatbot_ready=chatbot_status.get('initialized', False),
                chatbot_status=chatbot_status,
                detailed_status=detailed_status,
                message="ChatBot状态正常"
            )
        except Exception as e:
            error_msg = f"获取ChatBot状态失败: {str(e)}"
            logger.error(error_msg)
            return StatusResponse(
                status="状态检查失败",
                chatbot_ready=False,
                error=error_msg
            )
            
    except Exception as e:
        error_msg = f"状态查询失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return StatusResponse(
            status="系统错误",
            chatbot_ready=False,
            error=error_msg
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
            logger.info("尝试重新初始化...")
            
            # 尝试重新初始化
            if initialize_chatbot():
                logger.info("重新初始化成功")
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"机器人初始化失败: {initialization_error}"
                )
        
        # 验证输入
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")
        
        # 生成回复
        logger.info(f"处理聊天请求: {request.message[:50]}...")
        
        try:
            response_message = chatbot.generate_response(request.message)
            
            # 提取响应内容
            if hasattr(response_message, 'content'):
                response_content = response_message.content
            else:
                response_content = str(response_message)
                
        except Exception as e:
            error_msg = f"生成回复失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
        
        # 获取状态信息
        try:
            status = chatbot.get_status()
            metadata = {
                "rag_enabled": status.get("rag_ready", False),
                "vector_store_loaded": status.get("vector_store_loaded", False),
                "memory_ready": status.get("memory_ready", False),
                "timestamp": get_current_timestamp()
            }
        except Exception as e:
            logger.warning(f"获取状态信息失败: {str(e)}")
            metadata = {"timestamp": get_current_timestamp()}
        
        logger.info(f"✓ 聊天处理成功，响应长度: {len(response_content)}")
        
        return ChatResponse(
            response=response_content,
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
    
    temp_file_path = None
    try:
        if chatbot is None:
            raise HTTPException(status_code=500, detail="聊天机器人未初始化")
        
        # 检查文件类型
        allowed_extensions = {'.txt', '.md', '.pdf', '.docx', '.doc'}
        file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ''
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型: {file_ext}. 支持的类型: {', '.join(allowed_extensions)}"
            )
        
        # 保存临时文件
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file_path = temp_file.name
                shutil.copyfileobj(file.file, temp_file)
                
            logger.info(f"处理文件: {file.filename}")
        except Exception as e:
            error_msg = f"保存临时文件失败: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # 读取文件内容
        try:
            if file_ext in ['.txt', '.md']:
                # 使用ChatBot的文档处理功能
                if hasattr(chatbot, 'rag_manager') and chatbot.rag_manager:
                    docs = chatbot.rag_manager.document_loader.load_single_document(temp_file_path)
                    if docs:
                        content_parts = []
                        for doc in docs:
                            if isinstance(doc, dict):
                                content_parts.append(doc.get('page_content', ''))
                            else:
                                content_parts.append(doc.page_content)
                        file_content = '\n'.join(content_parts)
                    else:
                        raise Exception("无法解析文件内容")
                else:
                    # 备用方案
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
            else:
                file_content = f"文档文件: {file.filename} (需要实现{file_ext}解析)"
                
        except Exception as e:
            error_msg = f"读取文件内容失败: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # 构建包含文件内容的消息
        try:
            full_message = f"{message}\n\n文件名: {file.filename}\n文件内容:\n{file_content[:5000]}..."  # 限制长度
            
            # 生成回复
            response_message = chatbot.generate_response(full_message)
            
            # 提取响应内容
            if hasattr(response_message, 'content'):
                response_content = response_message.content
            else:
                response_content = str(response_message)
                
        except Exception as e:
            error_msg = f"处理文件对话失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
        
        return ChatResponse(
            response=response_content,
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
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"清理临时文件失败: {str(e)}")

# ===== 病历上传功能 (新增) =====
@app.post("/medical/upload", response_model=MedicalUploadResponse)
async def upload_medical_record(
    file: UploadFile = File(...),
    patient_name: str = Form(default=""),
    description: str = Form(default="")
):
    """上传病历文件专用接口"""
    global chatbot
    
    saved_file_path = None
    try:
        if chatbot is None:
            raise HTTPException(status_code=500, detail="聊天机器人未初始化")
        
        # 检查文件类型
        allowed_extensions = {'.txt', '.md', '.pdf', '.docx', '.doc'}
        file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ''
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的病历文件类型: {file_ext}. 支持的类型: {', '.join(allowed_extensions)}"
            )
        
        # 生成安全的文件名
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"medical_{timestamp}_{file.filename}"
            saved_file_path = os.path.join("medical_records", safe_filename)
            
            # 确保目录存在
            os.makedirs("medical_records", exist_ok=True)
            
            # 保存病历文件
            with open(saved_file_path, "wb") as saved_file:
                shutil.copyfileobj(file.file, saved_file)
                
            logger.info(f"病历文件保存成功: {saved_file_path}")
            
        except Exception as e:
            error_msg = f"保存病历文件失败: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # 使用ChatBot的病历处理功能
        try:
            upload_result = chatbot.upload_document(saved_file_path)
            
            if not upload_result.get('success', False):
                raise Exception(upload_result.get('error', '病历处理失败'))
                
            logger.info("病历文件处理成功")
            
        except Exception as e:
            error_msg = f"ChatBot处理病历失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
        
        # 保存病历元数据
        try:
            metadata = {
                "original_filename": file.filename,
                "saved_path": saved_file_path,
                "patient_name": patient_name,
                "description": description,
                "upload_time": get_current_timestamp(),
                "file_size": upload_result.get('content_length', 0),
                "processed": True
            }
            
            metadata_file = os.path.join("medical_records", f"metadata_{timestamp}.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            logger.info(f"病历元数据保存成功: {metadata_file}")
            
        except Exception as e:
            logger.warning(f"保存病历元数据失败: {str(e)}")
            # 不阻止主流程
        
        return MedicalUploadResponse(
            status="success",
            message=f"病历文件 '{file.filename}' 上传并处理成功",
            document_path=saved_file_path,
            content_length=upload_result.get('content_length', 0),
            processed=True,
            timestamp=get_current_timestamp()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"病历上传失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # 清理失败的文件
        if saved_file_path and os.path.exists(saved_file_path):
            try:
                os.unlink(saved_file_path)
            except:
                pass
                
        return MedicalUploadResponse(
            status="error",
            message="病历上传失败",
            error=error_msg,
            processed=False,
            timestamp=get_current_timestamp()
        )

@app.get("/medical/list")
async def list_medical_records():
    """获取已上传的病历列表"""
    try:
        medical_dir = "medical_records"
        if not os.path.exists(medical_dir):
            return safe_json_response({
                "status": "success",
                "records": [],
                "message": "暂无病历记录"
            })
        
        records = []
        try:
            for file in os.listdir(medical_dir):
                if file.startswith("metadata_") and file.endswith(".json"):
                    metadata_path = os.path.join(medical_dir, file)
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        records.append(metadata)
                    except Exception as e:
                        logger.warning(f"读取病历元数据失败 {file}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"遍历病历目录失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"获取病历列表失败: {str(e)}")
        
        # 按时间排序
        records.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
        
        return safe_json_response({
            "status": "success",
            "records": records,
            "count": len(records),
            "message": f"找到 {len(records)} 条病历记录"
        })
        
    except Exception as e:
        error_msg = f"获取病历列表失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# ===== RAG功能 =====
@app.post("/rag/upload", response_model=RAGUploadResponse)
async def upload_documents_to_rag(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200)
):
    """上传文档到RAG向量库"""
    global chatbot
    
    temp_files = []
    try:
        if chatbot is None:
            raise HTTPException(status_code=500, detail="聊天机器人未初始化")
        
        if not hasattr(chatbot, 'rag_manager') or not chatbot.rag_manager:
            raise HTTPException(status_code=500, detail="RAG管理器未初始化")
        
        # 验证文件
        file_paths = []
        try:
            for file in files:
                # 检查文件类型
                allowed_extensions = {'.txt', '.md', '.pdf', '.docx', '.doc'}
                file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ''
                
                if file_ext not in allowed_extensions:
                    raise HTTPException(
                        status_code=400,
                        detail=f"文件 {file.filename} 类型不支持: {file_ext}"
                    )
                
                # 保存到临时目录
                temp_path = os.path.join("temp", file.filename)
                try:
                    with open(temp_path, "wb") as temp_file:
                        shutil.copyfileobj(file.file, temp_file)
                    
                    file_paths.append(temp_path)
                    temp_files.append(temp_path)
                    
                except Exception as e:
                    raise Exception(f"保存临时文件 {file.filename} 失败: {str(e)}")
                    
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"文件预处理失败: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info(f"开始加载 {len(file_paths)} 个文档到RAG系统")
        
        # 加载文档到RAG
        try:
            result = chatbot.add_documents_to_rag(file_paths)
            
            if result.get('success', False):
                return RAGUploadResponse(
                    status="success",
                    message=result.get('message', f"成功处理 {len(file_paths)} 个文档"),
                    files_processed=[f.filename for f in files],
                    documents_added=result.get('document_count', len(file_paths)),
                    timestamp=get_current_timestamp()
                )
            else:
                error_msg = result.get('error', 'RAG文档处理失败')
                raise Exception(error_msg)
                
        except Exception as e:
            error_msg = f"RAG文档处理失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"RAG文档上传失败: {str(e)}"
        logger.error(error_msg)
        logger.error(f"详细错误: {traceback.format_exc()}")
        
        return RAGUploadResponse(
            status="error",
            message="RAG文档上传失败",
            error=error_msg,
            files_processed=[],
            documents_added=0,
            timestamp=get_current_timestamp()
        )
    
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"清理临时文件失败 {temp_file}: {str(e)}")

@app.get("/rag/status")
async def get_rag_status():
    """获取RAG系统状态"""
    global chatbot
    
    try:
        if chatbot is None:
            raise HTTPException(status_code=500, detail="聊天机器人未初始化")
        
        if not hasattr(chatbot, 'rag_manager') or not chatbot.rag_manager:
            return safe_json_response({
                "status": "未初始化",
                "message": "RAG管理器未初始化",
                "vector_store_loaded": False
            })
        
        try:
            # 获取向量库信息
            vector_info = chatbot.rag_manager.get_vector_store_info()
            
            # 获取ChatBot状态
            chatbot_status = chatbot.get_status()
            
            return safe_json_response({
                "status": "正常" if chatbot_status.get('rag_ready', False) else "异常",
                "vector_store_loaded": chatbot_status.get('vector_store_loaded', False),
                "vector_info": vector_info,
                "rag_ready": chatbot_status.get('rag_ready', False),
                "message": "RAG系统状态获取成功"
            })
            
        except Exception as e:
            error_msg = f"获取RAG状态失败: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"RAG状态查询失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# ===== 配置管理 =====
@app.post("/config/memory")
async def update_memory_config(request: MemoryTypeRequest):
    """更新内存配置"""
    global chatbot
    
    try:
        if chatbot is None:
            raise HTTPException(status_code=500, detail="聊天机器人未初始化")
        
        # 验证内存类型
        allowed_types = ["buffer", "summary", "window"]
        if request.memory_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的内存类型: {request.memory_type}. 支持的类型: {', '.join(allowed_types)}"
            )
        
        try:
            chatbot._configure_memory(request.memory_type)
            
            return safe_json_response({
                "status": "success",
                "message": f"内存配置更新为: {request.memory_type}",
                "memory_type": request.memory_type
            })
            
        except Exception as e:
            error_msg = f"内存配置更新失败: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"内存配置更新过程失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# ===== 会话管理 =====
@app.post("/chat/clear")
async def clear_chat_history():
    """清除对话历史"""
    global chatbot
    
    try:
        if chatbot is None:
            raise HTTPException(status_code=500, detail="聊天机器人未初始化")
        
        try:
            chatbot.clear_memory()
            
            return safe_json_response({
                "status": "success",
                "message": "对话历史已清除"
            })
            
        except Exception as e:
            error_msg = f"清除历史失败: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"清除历史过程失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# ===== 系统管理 =====
@app.post("/system/reinitialize")
async def reinitialize_chatbot():
    """手动重新初始化chatbot"""
    global chatbot, initialization_error
    
    try:
        logger.info("手动重新初始化chatbot...")
        chatbot = None
        initialization_error = None
        
        success = initialize_chatbot()
        
        if success:
            try:
                chatbot_status = chatbot.get_status()
                return safe_json_response({
                    "status": "success",
                    "message": "ChatBot重新初始化成功",
                    "chatbot_status": chatbot_status
                })
            except Exception as e:
                logger.warning(f"获取重新初始化后状态失败: {str(e)}")
                return safe_json_response({
                    "status": "success",
                    "message": "ChatBot重新初始化成功，但状态获取失败",
                    "warning": str(e)
                })
        else:
            return safe_json_response({
                "status": "failed",
                "message": f"ChatBot重新初始化失败: {initialization_error}",
                "error": initialization_error
            }, 500)
            
    except Exception as e:
        error_msg = f"重新初始化过程失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return safe_json_response({
            "status": "error",
            "message": "重新初始化过程异常",
            "error": error_msg
        }, 500)

@app.get("/system/health")
async def health_check():
    """健康检查"""
    global chatbot
    
    try:
        if chatbot is None:
            return safe_json_response({
                "status": "unhealthy", 
                "message": "ChatBot未初始化",
                "components": {
                    "chatbot": False,
                    "rag": False,
                    "memory": False,
                    "llm": False
                }
            })
        
        # 详细的健康检查
        try:
            status = chatbot.get_status()
            components = {
                "chatbot": status.get("initialized", False),
                "rag": status.get("rag_ready", False),
                "memory": status.get("memory_ready", False),
                "llm": status.get("llm_ready", False),
                "database": status.get("db_ready", False),
                "vector_store": status.get("vector_store_loaded", False)
            }
            
            is_healthy = status.get("initialized", False) and status.get("llm_ready", False)
            
            return safe_json_response({
                "status": "healthy" if is_healthy else "unhealthy",
                "message": "系统状态正常" if is_healthy else "系统存在问题",
                "components": components,
                "chatbot_status": status
            })
            
        except Exception as e:
            logger.error(f"健康检查状态获取失败: {str(e)}")
            return safe_json_response({
                "status": "unhealthy",
                "message": "状态检查失败",
                "error": str(e)
            })
            
    except Exception as e:
        error_msg = f"健康检查失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return safe_json_response({
            "status": "unhealthy",
            "message": "健康检查异常",
            "error": error_msg
        })

@app.get("/api/info")
async def get_api_info():
    """获取API信息"""
    try:
        return safe_json_response({
            "title": app.title,
            "version": app.version,
            "description": app.description,
            "endpoints": {
                "chat": "/chat",
                "medical_upload": "/medical/upload",
                "rag_upload": "/rag/upload",
                "status": "/status",
                "health": "/system/health",
                "docs": "/docs"
            },
            "features": [
                "中医AI对话",
                "病历文件上传",
                "RAG知识库管理",
                "对话历史管理",
                "系统状态监控"
            ]
        })
    except Exception as e:
        error_msg = f"获取API信息失败: {str(e)}"
        logger.error(error_msg)
        return safe_json_response({
            "status": "error",
            "error": error_msg
        }, 500)

if __name__ == "__main__":
    try:
        import uvicorn
        
        logger.info("启动增强版ChatBot API服务器...")
        logger.info("=" * 60)
        logger.info("🚀 Enhanced ChatBot API v3.0.0")
        logger.info("=" * 60)
        logger.info("📖 API文档: http://localhost:8000/docs")
        logger.info("🏥 系统状态: http://localhost:8000/status")
        logger.info("💬 聊天界面: http://localhost:8000")
        logger.info("🔍 健康检查: http://localhost:8000/system/health")
        logger.info("📋 病历上传: http://localhost:8000/medical/upload")
        logger.info("📚 RAG管理: http://localhost:8000/rag/upload")
        logger.info("=" * 60)
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level="info",
            reload=False  # 生产环境建议设为False
        )
        
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        logger.error(traceback.format_exc())