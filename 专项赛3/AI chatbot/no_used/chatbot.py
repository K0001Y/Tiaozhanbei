"""
聊天机器人核心类 - 精简增强版（自动导入向量库）
保留核心错误报告功能，简化冗余部分，增加自动向量库导入功能
"""

import datetime
import logging
import traceback
import os
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from typing import Optional, Dict, Any
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_CONFIGS,
    DEFAULT_PROMPT_TEMPLATE
)
from no_used.rag_manager import RAGManager
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
            # 创建中医诊疗专用prompt模板
            self.prompt = PromptTemplate(
                input_variables=["history", "context", "input"],
                template=DEFAULT_PROMPT_TEMPLATE
            )
            
            if memory_type == "summary":
                # 摘要记忆：适合长期诊疗跟踪
                self.memory = ConversationSummaryMemory(
                    llm=self.llm,
                    return_messages=False,  # 中医场景使用文本格式更好
                    memory_key="history",
                    max_token_limit=500  # 限制摘要长度
                )
                self.logger.info("使用摘要记忆模式（适合长期跟踪）")
            
            elif memory_type == "window":
            
            # 窗口记忆：适合单次诊疗会话
                self.memory = ConversationBufferWindowMemory(
                    k=8,  # 保留最近4轮对话（每轮包含患者+医师）
                    return_messages=False,
                    memory_key="history"
                )
                self.logger.info("使用窗口记忆模式（保留最近4轮对话）")
            
            else:  # buffer 或其他
                # 缓冲记忆：适合短期诊疗会话
                self.memory = ConversationBufferMemory(
                    return_messages=False,
                    memory_key="history",
                    max_token_limit=1000  # 防止上下文过长
                )
                self.logger.info("使用缓冲记忆模式（完整对话记录）")
        
            # 中医诊疗记忆增强配置
            self._configure_tcm_memory()
        
            self.status["memory_ready"] = True
            self.logger.info(f"✓ 中医诊疗记忆初始化成功 (类型: {memory_type})")
        
        except Exception as e:
            self.logger.error(f"✗ 记忆初始化失败: {str(e)}")
            raise
    def _configure_tcm_memory(self):
        """配置中医诊疗记忆增强功能"""
        try:
            # 中医关键信息标记
            self.tcm_key_terms = {
                '症状': ['头痛', '发热', '咳嗽', '胸闷', '腹痛', '便秘', '腹泻', '失眠', '心悸'],
                '体征': ['舌质', '舌苔', '脉象', '面色', '精神', '二便', '汗出', '寒热'],
                '证型': ['气虚', '血瘀', '阴虚', '阳虚', '痰湿', '湿热', '肝郁', '脾虚', '肾虚'],
                '治法': ['补气', '活血', '滋阴', '温阳', '化痰', '清热', '疏肝', '健脾', '补肾'],
                '方剂': ['四君子汤', '四物汤', '六味地黄丸', '逍遥散', '小柴胡汤']
            }
        
            # 记忆优化配置
            self.memory_config = {
                'prioritize_symptoms': True,    # 优先保留症状信息
                'track_progress': True,         # 跟踪病情变化
                'preserve_diagnosis': True      # 保留诊断结论
            }
        
            self.logger.info("✓ 中医记忆增强配置完成")
        
        except Exception as e:
            self.logger.warning(f"中医记忆配置失败: {str(e)}")    
    
    def _init_conversation(self):
        """初始化中医诊疗对话链"""
        try:
            # 检查prompt是否已由记忆模块初始化
            if not hasattr(self, 'prompt') or self.prompt is None:
                self.logger.warning("Prompt未初始化，请先调用_init_memory")
                raise ValueError("需要先初始化记忆模块")
        
            # 创建对话链（与记忆系统集成）
            self.conversation = LLMChain(
                llm=self.llm,
                prompt=self.prompt,
                memory=self.memory,  # 集成记忆系统
                verbose=False
            )
        
            # 初始化中医诊疗安全检查
            self._init_tcm_safety_check()
        
            # 设置对话链状态
            self.status["conversation_ready"] = True
            self.logger.info("✓ 中医诊疗对话链初始化成功")
        
        except Exception as e:
            self.logger.error(f"✗ 对话链初始化失败: {str(e)}")
            raise

    def _init_tcm_safety_check(self):
        """初始化中医诊疗安全检查机制"""
            # 急诊关键词（需要立即就医）
        self.emergency_keywords = [
                '昏迷', '意识不清', '大出血', '心脏骤停', '呼吸衰竭', 
                '休克', '窒息', '抽搐不止', '高热不退', '严重胸痛',
                '剧烈腹痛', '吐血', '便血', '呼吸困难'
            ]
        
            # 用药安全剂量警戒线（克）
        self.drug_safety_limits = {
                '附子': 15, '大黄': 12, '甘遂': 3, '芫花': 3, 
                '大戟': 3, '商陆': 3, '巴豆': 1, '蟾酥': 0.3
            }
        
            # 配伍禁忌检查（十八反十九畏）
        self.incompatible_drugs = {
                '甘草': ['甘遂', '大戟', '海藻', '芫花'],
                '乌头': ['贝母', '瓜蒌', '半夏', '白蔹', '白及'],
                '藜芦': ['人参', '沙参', '丹参', '玄参', '细辛', '芍药']
            }            
    
    def _init_rag(self, embedding_model_path: str):
        """初始化RAG管理器并自动加载向量库"""
        try:
            # 初始化RAG管理器
            self.rag_manager = RAGManager(embedding_model_path)
            self.status["rag_ready"] = True
            self.logger.info("✓ RAG管理器初始化成功")
            
            # 自动加载向量库（如果启用且存在）
            if self.auto_load_vector_store:
                self._auto_load_vector_store()
                
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


        
    #def generate_response(self, user_input: str) -> str:
       
    
    

    
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
            