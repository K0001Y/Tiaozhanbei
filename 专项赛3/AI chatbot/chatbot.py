"""
聊天机器人核心类 - 精简增强版（自动导入向量库）
保留核心错误报告功能，简化冗余部分，增加自动向量库导入功能
"""

import datetime
import logging
import traceback
import os
from typing import Optional, Dict, Any
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_CONFIGS,
    DEFAULT_PROMPT_TEMPLATE
)
from rag_manager import RAGManager
from utils import DatabaseManager


class LangChainChatBot:
    """LangChain聊天机器人主类 - 精简增强版（自动导入向量库）"""
    
    def __init__(self, 
                 api_key: str,
                 model_name: str = DEFAULT_MODEL_NAME,
                 model_configs: Optional[Dict[str, Any]] = None,
                 memory_type: str = "buffer",
                 base_url: str = DEFAULT_BASE_URL,
                 db_url: Optional[str] = None,
                 db_token: Optional[str] = None,
                 use_async_db: bool = False,
                 embedding_model_path: str = "model",
                 auto_load_vector_store: bool = True,
                 vector_store_path: str = "RAG"):
        
        # 设置简单日志
        self.logger = logging.getLogger(f"ChatBot")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # 状态跟踪
        self.status = {
            "initialized": False,
            "llm_ready": False,
            "memory_ready": False,
            "rag_ready": False,
            "db_ready": False,
            "vector_store_loaded": False
        }
        self.last_error = None
        
        # 存储向量库路径用于自动加载
        self.vector_store_path = vector_store_path
        self.auto_load_vector_store = auto_load_vector_store
        
        try:
            # 验证基本参数
            if not api_key or not model_name or not base_url:
                raise ValueError("API密钥、模型名称和基础URL都不能为空")
            
            # 基本配置
            self.api_key = api_key
            self.model_name = model_name
            self.model_configs = model_configs or DEFAULT_MODEL_CONFIGS.copy()
            self.base_url = base_url
            
            # 初始化组件
            self._init_llm()
            self._init_memory(memory_type)
            self._init_conversation()
            self._init_rag(embedding_model_path)
            self._init_database(db_url, db_token, use_async_db)
            
            self.status["initialized"] = True
            self.logger.info(f"✓ 聊天机器人初始化成功 (模型: {self.model_name})")
            
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"✗ 初始化失败: {str(e)}")
            raise
    
    def _init_llm(self):
        """初始化LLM"""
        try:
            llm_kwargs = {
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.95,
                "presence_penalty": 0.1,
                **self.model_configs.get("llm_kwargs", {})
            }
            
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url,
                **llm_kwargs
            )
            
            self.status["llm_ready"] = True
            self.logger.info("✓ LLM初始化成功")
            
        except Exception as e:
            self.logger.error(f"✗ LLM初始化失败: {str(e)}")
            raise
    
    def _init_memory(self, memory_type: str):
        """初始化对话记忆"""
        try:
            template = DEFAULT_PROMPT_TEMPLATE + """

        Human: {input}
        Assistant:"""
            
            self.prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=template
            )
            
            if memory_type == "summary":
                self.memory = ConversationSummaryMemory(
                    llm=self.llm,
                    return_messages=True,
                    memory_key="history"
                )
            else:
                self.memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="history"
                )
            
            self.status["memory_ready"] = True
            self.logger.info(f"✓ 记忆初始化成功 (类型: {memory_type})")
            
        except Exception as e:
            self.logger.error(f"✗ 记忆初始化失败: {str(e)}")
            raise
    
    def _init_conversation(self):
        """初始化对话链"""
        try:
            self.conversation = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                prompt=self.prompt,
                verbose=False
            )
            self.logger.info("✓ 对话链初始化成功")
            
        except Exception as e:
            self.logger.error(f"✗ 对话链初始化失败: {str(e)}")
            raise
    
    def _init_rag(self, embedding_model_path: str):
        """初始化RAG管理器并自动加载向量库"""
        try:
            # 初始化RAG管理器
            self.rag_manager = RAGManager(embedding_model_path)
            self.status["rag_ready"] = True
            self.logger.info("✓ RAG管理器初始化成功")
            
            # 自动加载向量库（如果启用且存在）
            #if self.auto_load_vector_store:
             #   self._auto_load_vector_store()
                
        except Exception as e:
            self.rag_manager = None
            self.logger.warning(f"RAG管理器初始化失败: {str(e)}")
    
    def _auto_load_vector_store(self):
        """自动加载向量库"""
        try:
            # 检查向量库路径是否存在
            if os.path.exists(self.vector_store_path):
                self.logger.info(f"发现向量库路径: {self.vector_store_path}，开始自动加载...")
                
                # 尝试加载向量存储
                result = self.rag_manager.load_vector_store(self.vector_store_path)
                
                if "成功" in result:
                    # 创建QA链
                    self.rag_manager.create_qa_chain(self.llm)
                    self.status["vector_store_loaded"] = True
                    self.logger.info("✓ 向量库自动加载成功")
                else:
                    self.logger.warning(f"向量库加载失败: {result}")
            else:
                self.logger.info(f"未找到向量库路径: {self.vector_store_path}，跳过自动加载")
                
        except Exception as e:
            self.logger.warning(f"自动加载向量库失败: {str(e)}")
    
    def _init_database(self, db_url: Optional[str], db_token: Optional[str], use_async_db: bool):
        """初始化数据库管理器"""
        try:
            if db_url:
                self.db_manager = DatabaseManager(db_url, db_token, use_async_db)
                self.status["db_ready"] = True
                self.logger.info("✓ 数据库连接成功")
            else:
                self.db_manager = None
                self.logger.info("未配置数据库")
        except Exception as e:
            self.db_manager = None
            self.logger.warning(f"数据库连接失败: {str(e)}")
    
    def generate_response(self, user_input: str) -> str:
        """生成回复"""
        try:
            if not user_input or not user_input.strip():
                raise ValueError("用户输入不能为空")
            
            # 使用RAG或普通模式
            if self.rag_manager and self.rag_manager.is_ready():
                response = self._generate_rag_response(user_input)
                metadata = {"rag_enabled": True, "vector_store_loaded": self.status["vector_store_loaded"]}
            else:
                response = self.conversation.predict(input=user_input)
                metadata = {"rag_enabled": False, "vector_store_loaded": False}
            
            # 更新记忆
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response)
            
            # 发送到数据库
            if self.db_manager:
                try:
                    self.db_manager.send_response(response, user_input, metadata)
                except Exception as e:
                    self.logger.warning(f"数据库发送失败: {str(e)}")
            
            return response
            
        except Exception as e:
            error_msg = f"生成回复失败: {str(e)}"
            self.last_error = error_msg
            self.logger.error(error_msg)
            return error_msg
    
    def _generate_rag_response(self, user_input: str) -> str:
        """使用RAG生成回复"""
        try:
            docs = self.rag_manager.get_relevant_documents(user_input)
            result = self.rag_manager.qa_chain({"query": user_input})
            return result["result"].strip()
        except Exception as e:
            self.logger.warning(f"RAG生成失败，使用普通模式: {str(e)}")
            return self.conversation.predict(input=user_input)
    
    def process_file(self, file_path: str) -> str:
        """处理文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
            
            user_input = f"请处理以下文本内容：\n\n{file_content}"
            return self.generate_response(user_input)
        except Exception as e:
            error_msg = f"处理文件失败: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    # 配置更新方法
    def update_model_config(self, config_updates: Dict[str, Any]):
        """更新模型配置"""
        try:
            self.model_configs.update(config_updates)
            self._init_llm()
            self._init_conversation()
            # 如果RAG已加载向量库，需要重新创建QA链
            if self.rag_manager and self.status["vector_store_loaded"]:
                self.rag_manager.create_qa_chain(self.llm)
            self.logger.info("模型配置更新成功")
        except Exception as e:
            error_msg = f"模型配置更新失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def update_pipeline_config(self, 
                             temperature: Optional[float] = None,
                             max_tokens: Optional[int] = None,
                             top_p: Optional[float] = None,
                             presence_penalty: Optional[float] = None):
        """更新pipeline配置"""
        try:
            llm_kwargs = self.model_configs.get("llm_kwargs", {})
            
            if temperature is not None:
                llm_kwargs["temperature"] = temperature
            if max_tokens is not None:
                llm_kwargs["max_tokens"] = max_tokens
            if top_p is not None:
                llm_kwargs["top_p"] = top_p
            if presence_penalty is not None:
                llm_kwargs["presence_penalty"] = presence_penalty
            
            self.model_configs["llm_kwargs"] = llm_kwargs
            self._init_llm()
            self._init_conversation()
            # 如果RAG已加载向量库，需要重新创建QA链
            if self.rag_manager and self.status["vector_store_loaded"]:
                self.rag_manager.create_qa_chain(self.llm)
            self.logger.info("Pipeline配置更新成功")
        except Exception as e:
            error_msg = f"Pipeline配置更新失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def set_memory_type(self, memory_type: str):
        """设置记忆类型"""
        try:
            self._init_memory(memory_type)
            self._init_conversation()
            self.logger.info(f"记忆类型设置为: {memory_type}")
        except Exception as e:
            error_msg = f"设置记忆类型失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def customize_prompt(self, new_template: str):
        """自定义提示模板"""
        try:
            self.prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=new_template
            )
            self._init_conversation()
            self.logger.info("提示模板更新成功")
        except Exception as e:
            error_msg = f"提示模板更新失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def clear_history(self):
        """清除对话历史"""
        try:
            if self.memory:
                self.memory.clear()
                self.logger.info("对话历史已清除")
        except Exception as e:
            self.logger.error(f"清除历史失败: {str(e)}")
    
    # RAG相关方法
    def load_documents(self, file_paths, chunk_size=1000, chunk_overlap=200):
        """加载文档到RAG系统"""
        try:
            if not self.rag_manager:
                raise Exception("RAG管理器未初始化")
            
            num_chunks = self.rag_manager.load_documents(file_paths, chunk_size, chunk_overlap)
            if num_chunks > 0:
                self.rag_manager.create_qa_chain(self.llm)
                self.status["vector_store_loaded"] = True
                self.logger.info(f"文档加载成功，生成 {num_chunks} 个块")
            return num_chunks
        except Exception as e:
            self.logger.error(f"文档加载失败: {str(e)}")
            return 0
    
    def save_vector_store(self, path=None, save_embedding_model=True):
        """保存向量存储"""
        try:
            if not self.rag_manager:
                raise Exception("RAG管理器未初始化")
            
            # 如果没有指定路径，使用默认路径
            if path is None:
                path = self.vector_store_path
                
            result = self.rag_manager.save_vector_store(path)
            if "成功" in result:
                self.logger.info(f"向量存储已保存到: {path}")
            return result
        except Exception as e:
            error_msg = f"保存向量存储失败: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def load_vector_store(self, path=None, custom_embedding_model=None):
        """手动加载向量存储"""
        try:
            if not self.rag_manager:
                raise Exception("RAG管理器未初始化")
            
            # 如果没有指定路径，使用默认路径
            if path is None:
                path = self.vector_store_path
            
            result = self.rag_manager.load_vector_store(path, custom_embedding_model)
            if "成功" in result:
                self.rag_manager.create_qa_chain(self.llm)
                self.status["vector_store_loaded"] = True
                self.logger.info("向量存储手动加载成功")
            return result
        except Exception as e:
            error_msg = f"加载向量存储失败: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def reload_vector_store(self):
        """重新加载向量存储"""
        try:
            if not self.rag_manager:
                raise Exception("RAG管理器未初始化")
            
            self.status["vector_store_loaded"] = False
            self._auto_load_vector_store()
            return "向量存储重新加载完成"
        except Exception as e:
            error_msg = f"重新加载向量存储失败: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    # 状态查询方法
    def get_status(self) -> dict:
        """获取系统状态"""
        rag_ready = False
        if self.rag_manager:
            try:
                rag_ready = self.rag_manager.is_ready()
            except:
                pass
        
        return {
            "initialized": self.status["initialized"],
            "llm_ready": self.status["llm_ready"],
            "memory_ready": self.status["memory_ready"],
            "rag_ready": rag_ready,
            "db_ready": self.status["db_ready"],
            "vector_store_loaded": self.status["vector_store_loaded"],
            "vector_store_path": self.vector_store_path,
            "auto_load_enabled": self.auto_load_vector_store,
            "last_error": self.last_error
        }
    
    def get_detailed_status(self) -> str:
        """获取详细状态报告"""
        status = self.get_status()
        
        lines = [
            "=== 聊天机器人状态 ===",
            f"总体状态: {'正常' if status['initialized'] else '异常'}",
            f"LLM: {'✓' if status['llm_ready'] else '✗'}",
            f"记忆: {'✓' if status['memory_ready'] else '✗'}",
            f"RAG: {'✓' if status['rag_ready'] else '✗'}",
            f"数据库: {'✓' if status['db_ready'] else '✗'}",
            f"向量库: {'✓' if status['vector_store_loaded'] else '✗'}",
            f"自动加载: {'启用' if status['auto_load_enabled'] else '禁用'}",
            f"向量库路径: {status['vector_store_path']}",
        ]
        
        if status['last_error']:
            lines.append(f"最后错误: {status['last_error']}")
        
        lines.append("=" * 25)
        return "\n".join(lines)