"""
聊天机器人核心类 - 精简增强版（自动导入向量库）
保留核心错误报告功能，简化冗余部分，增加自动向量库导入功能
"""

import datetime
import logging
import traceback
import os
import json
import re
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from typing import Optional, Dict, Any, List, Union
from langchain.chains import LLMChain, ConversationChain
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
from Rag_manager import RAGManager
from utils import DatabaseManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    """
    聊天机器人核心类 - 处理用户输入、生成响应、管理对话状态和RAG功能
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        base_url: str = DEFAULT_BASE_URL,
        model_configs: Optional[Dict[str, Any]] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    ):
        """
        初始化聊天机器人
        :param model_name: 模型名称
        :param base_url: API基础URL
        :param model_configs: 模型配置参数
        :param prompt_template: 提示模板
        """
        try:
            logger.info("开始初始化ChatBot...")
            
            self.model_name = model_name
            self.base_url = base_url
            self.model_configs = model_configs or DEFAULT_MODEL_CONFIGS
            self.prompt_template = prompt_template
            
            # 初始化状态标记
            self._initialized = False
            self._llm_ready = False
            self._memory_ready = False
            self._rag_ready = False
            self._db_ready = False
            self._vector_store_loaded = False
            self._auto_load_enabled = True
            self._last_error = None
            
            # 初始化核心组件
            self.llm = None
            self.first_round_memory = None  # 第一轮对话内存
            self.document_content = None    # 上传的病历文件内容
            self.current_chain = None       # 当前使用的对话链
            
            # 初始化RAG管理器和数据库管理器
            self._init_rag_manager()
            self._init_database_manager()
            
            # 初始化对话内存
            self._init_memory()
            
            # 初始化LLM模型
            self._init_llm()
            
            # 尝试自动加载向量库
            self._auto_load_vector_store()
            
            self._initialized = True
            logger.info("ChatBot初始化成功")

            INITIAL = "initial"              # 初始状态
            SYMPTOM_DESCRIPTION = "symptom_description"  # 症状描述
            DIAGNOSIS = "diagnosis"          # 诊断阶段
            TREATMENT_PLAN = "treatment_plan"  # 治疗方案
            QUESTION_ANSWERING = "question_answering"  # 问答阶段
            
        except Exception as e:
            error_msg = f"ChatBot初始化失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise Exception(error_msg)

    def _init_rag_manager(self):
        """
        初始化RAG管理器
        """
        try:
            logger.info("初始化RAG管理器...")
            self.rag_manager = RAGManager()
            
            # 启动RAG管理器
            self.rag_manager._start_rag_manager()
            self._rag_ready = True
            
            logger.info("RAG管理器初始化成功")
            
        except Exception as e:
            error_msg = f"RAG管理器初始化失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            self._rag_ready = False
            # 不抛出异常，允许继续初始化其他组件

    def _init_database_manager(self):
        """
        初始化数据库管理器
        """
        try:
            logger.info("初始化数据库管理器...")
            
            # 从环境变量或配置文件读取数据库配置
            db_url = os.getenv('DB_URL', 'http://localhost:8000/api')
            db_token = os.getenv('DB_TOKEN', None)
            
            self.database_manager = DatabaseManager(
                db_url=db_url,
                db_token=db_token,
                use_async=False,
                timeout=30,
                max_retries=3
            )
            
            # 测试数据库连接
            connection_test = self.database_manager.test_connection()
            if connection_test.get('success', False):
                self._db_ready = True
                logger.info("数据库管理器初始化成功")
            else:
                logger.warning(f"数据库连接测试失败: {connection_test.get('error', '未知错误')}")
                self._db_ready = False
                
        except Exception as e:
            error_msg = f"数据库管理器初始化失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            self._db_ready = False
            # 不抛出异常，允许继续初始化其他组件

    def _init_llm(self):
        """
        初始化LLM模型
        """
        try:
            logger.info("初始化LLM模型...")
            
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                base_url=self.base_url,
                **self.model_configs
            )
            
            self._llm_ready = True
            logger.info("LLM模型初始化成功")
            
        except Exception as e:
            error_msg = f"LLM模型初始化失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            self._llm_ready = False
            raise Exception(error_msg)

    def _init_memory(self):
        """
        初始化对话内存
        """
        try:
            logger.info("初始化对话内存...")
            
            # 初始化主对话内存
            self.memory = ConversationBufferMemory(
                memory_key="history", 
                return_messages=True,
                input_key="input",
                output_key="response"
            )
            
            # 初始化第一轮对话专用内存
            self.first_round_memory = ConversationBufferMemory(
                memory_key="first_round_history",
                return_messages=True
            )
            
            self._memory_ready = True
            logger.info("对话内存初始化成功")
            
        except Exception as e:
            error_msg = f"对话内存初始化失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            self._memory_ready = False
            raise Exception(error_msg)

    def _auto_load_vector_store(self):
        """
        自动加载向量库
        """
        try:
            if not self._auto_load_enabled:
                logger.info("自动加载向量库已禁用")
                return
                
            logger.info("尝试自动加载向量库...")
            
            if not self.rag_manager or not self._rag_ready:
                logger.warning("RAG管理器未就绪，跳过向量库加载")
                return
            
            # 检查向量库信息
            vector_info = self.rag_manager.get_vector_store_info()
            
            if vector_info.get('status') == '已加载' and vector_info.get('count', 0) > 0:
                self._vector_store_loaded = True
                logger.info(f"向量库自动加载成功，包含 {vector_info.get('count')} 个文档")
            else:
                logger.info("未发现可用的向量库或向量库为空")
                self._vector_store_loaded = False
                
        except Exception as e:
            error_msg = f"自动加载向量库失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            self._vector_store_loaded = False

    def _save_first_round(self, user_input: str, ai_response: str):
        """
        保存第一轮对话状态
        :param user_input: 用户输入
        :param ai_response: AI回复
        """
        try:
            logger.info("保存第一轮对话状态...")
            
            if not self.first_round_memory:
                raise Exception("第一轮对话内存未初始化")
            
            if not user_input or not user_input.strip():
                raise ValueError("用户输入不能为空")
            
            # 将第一轮对话存入专用内存
            self.first_round_memory.save_context(
                {"input": user_input}, 
                {"output": ai_response}
            )
            
            # 同时保存到主内存
            self.memory.save_context(
                {"input": user_input}, 
                {"response": ai_response}
            )
            
            logger.info("第一轮对话状态保存成功")
            
        except Exception as e:
            error_msg = f"保存第一轮对话状态失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise Exception(error_msg)

    def _save_document(self, document_path: str):
        """
        保存上传的病历文件内容
        :param document_path: 上传的病历文件路径
        """
        try:
            logger.info(f"保存病历文件内容: {document_path}")
            
            if not document_path or not document_path.strip():
                raise ValueError("文档路径不能为空")
            
            if not os.path.exists(document_path):
                raise FileNotFoundError(f"文档文件不存在: {document_path}")
            
            # 使用utils中的文档加载器读取文件内容
            if hasattr(self.rag_manager, 'document_loader'):
                doc_loader = self.rag_manager.document_loader
                
                # 验证文件路径
                validation = doc_loader.validate_file_path(document_path)
                if not validation['valid']:
                    raise ValueError(f"文件验证失败: {validation['reason']}")
                
                # 加载文档
                documents = doc_loader.load_single_document(document_path)
                
                if not documents:
                    raise Exception("未能加载任何文档内容")
                
                # 提取文本内容
                content_parts = []
                for doc in documents:
                    if isinstance(doc, dict):
                        content_parts.append(doc.get('page_content', ''))
                    else:
                        content_parts.append(doc.page_content)
                
                self.document_content = '\n'.join(content_parts)
                
                # 同时将文档添加到向量库（可选）
                if self._rag_ready and self.rag_manager:
                    try:
                        self.rag_manager.add_documents_to_store([document_path])
                        logger.info("文档已添加到向量库")
                    except Exception as e:
                        logger.warning(f"添加文档到向量库失败: {str(e)}")
                
            else:
                # 备用方案：直接读取文件
                from utils import FileHandler
                self.document_content = FileHandler.read_file(document_path)
            
            logger.info("病历文件内容保存成功")
            
        except Exception as e:
            error_msg = f"保存病历文件内容失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise Exception(error_msg)
    def recognize_intent(self,text):
        """
        根据用户输入识别意图
        :param text: 用户输入文本
        :return: 意图 ("question", "diagnosis", "unclear")
        """
        question_keywords = ["什么是", "我想知道", "治疗方案", "诊断结果"]
        diagnosis_keywords = ["我的症状是", "我感觉", "我得了", "头痛", "脉象","我现在"]
        document_keywords = ["病历", "检查报告"]

        text_lower = text.lower()
    
    # 优先检查文档相关意图
        if any(keyword in text_lower for keyword in document_keywords):
            return "diagnosis"
    
    # 检查提问意图
        if any(keyword in text_lower for keyword in question_keywords):
            return "question"
    
    # 检查诊断意图
        if any(keyword in text_lower for keyword in diagnosis_keywords):
            return "diagnosis"
    
        return "unclear"
    def _choose_chain(self,user_input):
        """
        每次生成回复之前判断用户提问状态并选择对应的对话链
        LLMChain或者ConversationChain
        """
        try:
            logger.info("选择对话链...")

            intent = self.recognize_intent(user_input)
            
            if not self._llm_ready or not self.llm:
                raise Exception("LLM模型未就绪")
            
            if intent == "question":
                return self._create_conversation_chain()
            elif intent == "diagnosis":
                return self._create_llm_chain()
            else:
                logger.info("选择ConversationChain ")
                return self._create_conversation_chain()
                
        except Exception as e:
            error_msg = f"选择对话链失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise Exception(error_msg)

    def _create_llm_chain(self):
        """
        创建LLMChain
        """
        try:
            logger.info("创建LLMChain...")
            
            # 症状采集阶段模板
            symptom_prompt = PromptTemplate(
                input_variables=["input"],
                template="""
                作为中医AI，请从用户描述中提取症状（输出JSON格式）：
                {{
                    "symptoms": ["症状1", "症状2"], 
                    "missing_info": ["需追问的症状"]  # 如未提及舌象/脉象
                }}
                用户输入：{input}


                # 回答要求：

                    ## 始终保持礼貌和专业的态度
                    ## 给出清晰、准确的回答
                    ## 在不确定的情况下诚实承认
                    ## 避免有害或不当的内容
                    ## 使用用户的语言进行回复
                """

                
            )
            
            chain = LLMChain(llm=self.llm, prompt=symptom_prompt)
            logger.info("LLMChain创建成功")
            print(hasattr(chain, 'predict'))
            return chain
            
        except Exception as e:
            error_msg = f"创建LLMChain失败: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _create_conversation_chain(self):
        """
        创建ConversationChain
        """
        try:
            logger.info("创建ConversationChain...")
            
            # 增强的中医对话模板
            enhanced_prompt = PromptTemplate(
                input_variables=["history", "input"],
                template="""
                作为专业中医AI助手，基于以下信息进行诊疗分析：
                
                历史对话: {history}
                
                当前输入: {input}
                
                请结合历史信息和当前问题，提供专业的中医诊疗建议。
                注意：
                1. 症状采集要全面
                2. 辨证分析要准确
                3. 用药建议要安全
                4. 避免绝对化表述
                 
                # 回答要求：

                    ## 始终保持礼貌和专业的态度
                    ## 给出清晰、准确的回答
                    ## 在不确定的情况下诚实承认
                    ## 避免有害或不当的内容
                    ## 使用用户的语言进行回复

                    ## 所有知识输出必须标注三维来源：
                    ### [典籍]《金匮要略·痰饮篇》§3.2 | [医案]国医大师邓铁涛案1987-021 | [指南]2020版消渴病诊疗规范
                """
                
            )
            
            chain = ConversationChain(
                llm=self.llm,
                prompt=enhanced_prompt,
                memory=self.memory,
                verbose=True
            )
            
            logger.info("ConversationChain创建成功")
            return chain
            
        except Exception as e:
            error_msg = f"创建ConversationChain失败: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _llm_chain_response(self, user_input: str):
        """
        使用LLMChain生成回复
        :param user_input: 用户输入
        :return: AI回复消息
        """
        try:
            logger.info("使用LLMChain生成回复...")
            
            if not user_input or not user_input.strip():
                raise ValueError("用户输入不能为空")
            
            # 使用RAG增强回答
            relevant_context = ""
            if self._rag_ready and self.rag_manager and self._vector_store_loaded:
                try:
                    retrieved_docs = self.rag_manager._retrieve(user_input, k=3)
                    if retrieved_docs:
                        context_parts = [doc.get('content', '') for doc in retrieved_docs]
                        relevant_context = '\n'.join(context_parts)
                        logger.info(f"获取到RAG上下文，长度: {len(relevant_context)}")
                except Exception as e:
                    logger.warning(f"RAG检索失败: {str(e)}")
            
            # 创建LLM链进行阶段化处理
            if not self.llm:
                raise Exception("LLM模型未初始化")
            
            # 症状采集阶段
            symptom_prompt = PromptTemplate(
                input_variables=["input", "context"],
                template="""
                作为中医AI，请从用户描述中提取症状（输出JSON格式）,忽略礼貌用语（如“你好”）和修饰词（如“有点”），仅提取核心症状：
                
                示例：
                    输入：“你好，我感觉头有点痛，嗓子干”
                    输出：{{"symptoms": ["头痛", "嗓子干"], "missing_info": ["头痛位置", "持续时间"], "stage": "symptom_collection"}}

                相关知识: {context}
                
                {{
                    "symptoms": ["症状1", "症状2"], 
                    "missing_info": ["需追问的症状"],
                    "stage": "symptom_collection"
                }}
                用户输入：{input}

                # 回答要求：

                    ## 始终保持礼貌和专业的态度
                    ## 给出清晰、准确的回答
                    ## 在不确定的情况下诚实承认
                    ## 避免有害或不当的内容
                    ## 使用用户的语言进行回复
                
                """
            )
            
            # 辩证推理阶段  
            diagnosis_prompt = PromptTemplate(
                input_variables=["symptoms", "context"],
                template="""
                根据以下症状进行中医辨证,（输出JSON格式）。如果症状不足，提供初步推测并说明依据
                
                相关知识: {context}
                症状列表：{symptoms}
                
                必须输出：{{
                    "证型": "标准证型名称", 
                    "病机": "不超过50字分析",
                    "依据": "《黄帝内经》等古籍引文",
                    "stage": "diagnosis"
                }}
                """
            )
            
            # 治疗方案生成阶段
            prescription_prompt = PromptTemplate(
                input_variables=["diagnosis", "context"],
                template="""
                推荐适合{diagnosis}的方剂,（输出JSON格式）。如果诊断为空，建议线下就诊或上传更多资料：
                
                相关知识: {context}
                
                输出格式：{{
                    "方剂名": "",
                    "组成": ["药材1 用量", "药材2 用量"],
                    "禁忌检查": true/false,
                    "stage": "prescription"
                }}
                """
            )
            
            # 执行阶段化处理
            symptom_chain = LLMChain(llm=self.llm, prompt=symptom_prompt)
            diagnosis_chain = LLMChain(llm=self.llm, prompt=diagnosis_prompt)
            prescription_chain = LLMChain(llm=self.llm, prompt=prescription_prompt)
            
            # 第一阶段：症状采集
            symptom_result = symptom_chain.run(input=user_input, context=relevant_context)
            
            # 解析症状结果
            try:
                symptom_data = json.loads(symptom_result)
            except:
                symptom_data = {"symptoms": [user_input], "missing_info": [],"stage": "symptom_collection"}
            response_lines = []
            if symptom_data["symptoms"]:
                response_lines.append(f"您的主要症状为：{', '.join(symptom_data['symptoms'])}。")
            if symptom_data["missing_info"]:
                response_lines.append("为了更准确地进行中医辨证，请补充以下信息：")
                response_lines.extend(f"{i+1}. {info}" for i, info in enumerate(symptom_data["missing_info"]))
            # 第二阶段：辨证分析
            diagnosis_result = diagnosis_chain.run(
                symptoms=symptom_data.get("symptoms", []), 
                context=relevant_context
            )
            try:
                diagnosis_data = json.loads(diagnosis_result)
                response_lines.append(f"\n初步诊断：{diagnosis_data['证型']}")
                response_lines.append(f"病机分析：{diagnosis_data['病机']}")
                response_lines.append(f"依据：{diagnosis_data['依据']}")
            except:
                response_lines.append("\n暂无诊断结果。")
            # 第三阶段：方剂推荐
            try:
                prescription_result = prescription_chain.run(
                    diagnosis=diagnosis_data.get("证型", ""), 
                    context=relevant_context
                )
                prescription_data = json.loads(prescription_result)
                if prescription_data["方剂名"]:
                    response_lines.append(f"\n推荐方剂：{prescription_data['方剂名']}")
                    response_lines.append("组成：" + ", ".join(prescription_data["组成"]))
                    response_lines.append(f"禁忌检查：{'通过' if prescription_data['禁忌检查'] else '未通过'}")
                else:
                    response_lines.append("\n暂无方剂推荐。")
            except:
                response_lines.append("\n暂无方剂推荐。")
            
            # 整合结果
            response = "\n".join(response_lines)
            if relevant_context:
                response += "\n\n（分析参考了相关中医知识库）"
            
            logger.info("LLMChain回复生成成功")
            return response
            
        except Exception as e:
            error_msg = f"LLMChain生成回复失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise Exception(error_msg)

    def _conversation_chain_response(self, user_input: str):
        """
        使用ConversationChain生成回复
        :param user_input: 用户输入
        :return: AI回复消息
        """
        try:
            logger.info("使用ConversationChain生成回复...")
            
            if not user_input or not user_input.strip():
                raise ValueError("用户输入不能为空")
            
            # 使用RAG增强回答
            relevant_context = ""
            if self._rag_ready and self.rag_manager and self._vector_store_loaded:
                try:
                    retrieved_docs = self.rag_manager._retrieve(user_input, k=4)
                    if retrieved_docs:
                        context_parts = [doc.get('content', '') for doc in retrieved_docs]
                        relevant_context = '\n'.join(context_parts)
                        logger.info(f"获取到RAG上下文，长度: {len(relevant_context)}")
                except Exception as e:
                    logger.warning(f"RAG检索失败: {str(e)}")
            
            # 从第一轮对话或病历文件中提取关键中医实体
            historical_context = ""
            if self.first_round_memory and len(self.first_round_memory.chat_memory.messages) > 0:
                historical_context += "第一轮对话内容: " + str(self.first_round_memory.chat_memory.messages)
            
            if self.document_content:
                historical_context += f"\n病历文件内容: {self.document_content[:500]}..."  # 限制长度
            
            # 创建增强的对话链
            chain = self._create_conversation_chain()
            
            # 构造增强的输入
            enhanced_input = f"""
            用户问题: {user_input}
            
            相关知识: {relevant_context}
            
            历史信息: {historical_context}
            """
            
            # 生成回复
            response = chain.predict(input=enhanced_input)
            
            logger.info("ConversationChain回复生成成功")
            return response
            
        except Exception as e:
            error_msg = f"ConversationChain生成回复失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise Exception(error_msg)

    def _configure_memory(self, memory_type: str = "buffer"):
        """
        配置对话内存
        :param memory_type: 内存类型 (buffer, summary, window)
        """
        try:
            logger.info(f"配置对话内存类型: {memory_type}")
            
            if memory_type == "buffer":
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    input_key="input",
                    output_key="response"
                )
            elif memory_type == "summary":
                if not self.llm:
                    raise Exception("LLM模型未初始化，无法使用summary内存")
                    
                self.memory = ConversationSummaryMemory(
                    llm=self.llm,
                    memory_key="chat_history",
                    return_messages=True
                )
            elif memory_type == "window":
                self.memory = ConversationBufferWindowMemory(
                    k=5,  # 保留最近5轮对话
                    memory_key="chat_history",
                    return_messages=True,
                    input_key="input",
                    output_key="response"
                )
            else:
                raise ValueError(f"不支持的内存类型: {memory_type}")
            
            logger.info("对话内存配置成功")
            
        except Exception as e:
            error_msg = f"配置对话内存失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise Exception(error_msg)

    def _remind_user(self):
        """
        提醒用户优先导入简历或进行病情描述
        """
        try:
            logger.info("生成用户提醒信息...")
            
            # 检查是否已有第一轮对话或病历文件
            has_first_round = (self.first_round_memory and 
                             len(self.first_round_memory.chat_memory.messages) > 0)
            has_document = self.document_content is not None
            
            if not has_first_round and not has_document:
                reminder_message = (
                    "🏥 中医AI诊疗助手提醒您：\n\n"
                    "为了提供更准确的中医诊疗建议，建议您：\n"
                    "1. 📋 上传病历文件（支持txt、pdf格式）\n"
                    "2. 💬 详细描述当前症状、舌象、脉象等信息\n"
                    "3. 🕐 提供症状持续时间和诱发因素\n\n"
                    "这样我能更好地为您进行中医辨证分析和治疗建议。\n"
                    "您可以直接开始描述症状，我会引导您提供必要信息。"
                )
                logger.info("用户提醒信息生成成功")
                return reminder_message
            else:
                logger.info("用户已有历史信息，无需提醒")
                return None
                
        except Exception as e:
            error_msg = f"生成用户提醒信息失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return "系统提醒功能暂时不可用，请直接开始对话。"

    def safe_guard(self, user_input: str, ai_response: str = None) -> Dict[str, Any]:
        """
        判断用户输入和AI回复是否符合安全协议
        :param user_input: 用户输入
        :param ai_response: AI回复（可选）
        :return: 安全检查结果
        """
        try:
            logger.info("执行安全协议检查...")
            
            if not user_input:
                return {"safe": True, "warnings": [], "block": False}
            
            warnings = []
            should_block = False
            emergency_alert = False
            
            # 1. 急诊触发关键词检查
            emergency_keywords = [
                "胸痛", "胸闷", "呼吸困难", "昏迷", "休克", "大出血", 
                "严重外伤", "中毒", "窒息", "心梗", "脑梗", "急性腹痛",
                "高热不退", "抽搐", "意识不清", "急救", "120", "999"
            ]
            
            for keyword in emergency_keywords:
                if keyword in user_input:
                    emergency_alert = True
                    warnings.append(f"检测到急诊关键词：{keyword}")
                    break
            
            # 2. 剂量红线检查（如果AI回复中包含用药建议）
            if ai_response:
                # 检查是否包含过量用药建议
                dangerous_doses = [
                    r"(\d+)g.*附子.*(\d+)", r"(\d+)g.*川乌", r"(\d+)g.*草乌",
                    r"(\d+)g.*雄黄", r"(\d+)g.*朱砂", r"(\d+)g.*轻粉"
                ]
                
                for pattern in dangerous_doses:
                    matches = re.findall(pattern, ai_response)
                    if matches:
                        warnings.append("检测到可能的危险用药剂量")
                        break
            
            # 3. 地域禁忌检查
            regional_herbs = {
                "南方": ["麻黄", "桂枝"],
                "北方": ["生地", "玄参"],
                "沿海": ["海藻", "昆布"],
                "内陆": ["紫菜", "海带"]
            }
            
            # 这里可以根据用户地域信息进行检查（需要用户提供地域信息）
            
            # 4. 十八反十九畏检查
            if ai_response:
                eighteen_contradictions = [
                    ["甘草", "甘遂"], ["甘草", "大戟"], ["甘草", "海藻"], ["甘草", "芫花"],
                    ["乌头", "贝母"], ["乌头", "瓜蒌"], ["乌头", "半夏"], ["乌头", "白蔹"], ["乌头", "白及"],
                    ["藜芦", "人参"], ["藜芦", "沙参"], ["藜芦", "丹参"], ["藜芦", "玄参"], 
                    ["藜芦", "细辛"], ["藜芦", "芍药"]
                ]
                
                for contradiction in eighteen_contradictions:
                    if all(herb in ai_response for herb in contradiction):
                        warnings.append(f"检测到十八反配伍：{' + '.join(contradiction)}")
                        should_block = True
            
            # 构造返回结果
            result = {
                "safe": not should_block and not emergency_alert,
                "warnings": warnings,
                "block": should_block,
                "emergency": emergency_alert,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            if emergency_alert:
                result["emergency_message"] = (
                    "⚠️ 紧急提醒：检测到可能的急诊症状！\n"
                    "建议立即前往最近的医院急诊科就诊，或拨打急救电话：\n"
                    "• 大陆地区：120\n"
                    "• 香港地区：999\n"
                    "• 台湾地区：119\n"
                    "中医诊疗不能替代急诊医疗！"
                )
            
            logger.info(f"安全协议检查完成，结果: {result}")
            return result
            
        except Exception as e:
            error_msg = f"安全协议检查失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {
                "safe": False,
                "warnings": [f"安全检查系统错误: {str(e)}"],
                "block": True,
                "emergency": False,
                "error": error_msg
            }

    def generate_response(self, user_input: str) -> BaseMessage:
        """
        生成AI回复
        :param user_input: 用户输入
        :return: AI回复消息
        """
        try:
            logger.info(f"开始生成回复，用户输入长度: {len(user_input)}")
            
            if not user_input or not user_input.strip():
                raise ValueError("用户输入不能为空")
            
            # 检查用户输入是否符合安全协议
            safety_check = self.safe_guard(user_input)
            
            # 若触发急诊关键词，优先返回急诊提醒
            if safety_check.get('emergency', False):
                emergency_msg = safety_check.get('emergency_message', '检测到紧急情况，建议立即就医')
                logger.warning("触发急诊提醒")
                return AIMessage(content=emergency_msg)
            
            # 检查是否需要用户提醒
            #reminder = self._remind_user()
            #if reminder:
            #    logger.info("返回用户提醒信息")
            #    return AIMessage(content=reminder)
            
            # 保存用户输入到对话内存
            try:
                self.memory.save_context({"input": user_input}, {})
            except Exception as e:
                logger.warning(f"保存用户输入到内存失败: {str(e)}")
            
            # 选择合适的对话链
            chain = self._choose_chain(user_input)
            logger.info(f"选择的链类型: {type(chain).__name__}")
            # 生成回复
            if isinstance(chain, ConversationChain):
                logger.info("执行ConversationChain")
                response = self._conversation_chain_response(user_input)
            elif isinstance(chain, LLMChain):
                logger.info("执行LLMChain")
                response = self._llm_chain_response(user_input)
            else:
                logger.info(f"未知的链类型: {type(chain)}")
                response = self._conversation_chain_response(user_input)
            
            # 对AI回复进行安全检查
            response_safety = self.safe_guard(user_input, response)
            
            if response_safety.get('block', False):
                blocked_msg = (
                    "⚠️ 检测到用药安全风险，建议咨询专业中医师。\n"
                    f"风险提示：{'; '.join(response_safety.get('warnings', []))}"
                )
                logger.warning("AI回复被安全协议阻止")
                response = blocked_msg
            
            # 保存回复到对话内存
            try:
                self.memory.save_context({}, {"response": response})
            except Exception as e:
                logger.warning(f"保存AI回复到内存失败: {str(e)}")
            
            # 保存到数据库（如果可用）
            if self._db_ready and self.database_manager:
                try:
                    metadata = {
                        "safety_check": safety_check,
                        "response_safety": response_safety,
                        "chain_type": type(chain).__name__,
                        "rag_used": self._vector_store_loaded
                    }
                    self.database_manager.send_response(response, user_input, metadata)
                except Exception as e:
                    logger.warning(f"保存到数据库失败: {str(e)}")
            
            logger.info("AI回复生成成功")
            return AIMessage(content=response)
        
        except Exception as e:
            error_msg = f"生成回复失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return AIMessage(content="抱歉，生成回复时发生错误。请稍后再试，或联系技术支持。")

    def clear_memory(self):
        """
        清除对话内存
        """
        try:
            logger.info("清除对话内存...")
            
            if self.memory:
                self.memory.clear()
            
            if self.first_round_memory:
                self.first_round_memory.clear()
            
            # 清除文档内容
            self.document_content = None
            
            logger.info("对话内存已清除")
            
        except Exception as e:
            error_msg = f"清除对话内存失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise Exception(error_msg)

    def get_status(self) -> Dict[str, Any]:
        """
        获取ChatBot状态
        """
        try:
            vector_store_path = "vector_store"
            if self.rag_manager:
                try:
                    vector_info = self.rag_manager.get_vector_store_info()
                    vector_store_path = vector_info.get('metadata', {}).get('vector_store_path', 'vector_store')
                except:
                    pass
            
            status = {
                'initialized': self._initialized,
                'llm_ready': self._llm_ready,
                'memory_ready': self._memory_ready,
                'rag_ready': self._rag_ready,
                'db_ready': self._db_ready,
                'vector_store_loaded': self._vector_store_loaded,
                'auto_load_enabled': self._auto_load_enabled,
                'vector_store_path': vector_store_path,
                'last_error': self._last_error
            }
            
            return status
            
        except Exception as e:
            error_msg = f"获取状态失败: {str(e)}"
            logger.error(error_msg)
            return {
                'initialized': False,
                'error': error_msg
            }
        
    def get_detailed_status(self) -> str:
        """获取详细状态报告"""
        try:
            logger.info("生成详细状态报告...")
            
            status = self.get_status()
            
            lines = [
                "=== 聊天机器人状态 ===",
                f"总体状态: {'✅ 正常' if status['initialized'] else '❌ 异常'}",
                f"LLM: {'✅' if status['llm_ready'] else '❌'}",
                f"记忆: {'✅' if status['memory_ready'] else '❌'}",
                f"RAG: {'✅' if status['rag_ready'] else '❌'}",
                f"数据库: {'✅' if status['db_ready'] else '❌'}",
                f"向量库: {'✅' if status['vector_store_loaded'] else '❌'}",
                f"自动加载: {'启用' if status['auto_load_enabled'] else '禁用'}",
                f"向量库路径: {status['vector_store_path']}",
            ]
            
            # 添加内存状态
            if self.memory:
                try:
                    msg_count = len(self.memory.chat_memory.messages)
                    lines.append(f"对话轮数: {msg_count // 2}")
                except:
                    lines.append("对话轮数: 无法获取")
            
            # 添加向量库详细信息
            if self._rag_ready and self.rag_manager:
                try:
                    vector_info = self.rag_manager.get_vector_store_info()
                    if vector_info.get('status') == '已加载':
                        lines.append(f"向量库文档数: {vector_info.get('count', 0)}")
                        metadata = vector_info.get('metadata', {})
                        if metadata:
                            lines.append(f"文档总数: {metadata.get('documents_count', 0)}")
                            lines.append(f"文档块数: {metadata.get('chunks_count', 0)}")
                except Exception as e:
                    lines.append(f"向量库信息获取失败: {str(e)}")
            
            if status.get('last_error'):
                lines.append(f"最后错误: {status['last_error']}")
            
            lines.append("=" * 30)
            
            result = "\n".join(lines)
            logger.info("详细状态报告生成成功")
            return result
            
        except Exception as e:
            error_msg = f"生成详细状态报告失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return f"状态报告生成失败: {error_msg}"

    def upload_document(self, document_path: str) -> Dict[str, Any]:
        """
        上传并处理病历文档
        :param document_path: 文档路径
        :return: 处理结果
        """
        try:
            logger.info(f"上传文档: {document_path}")
            
            # 保存文档内容
            self._save_document(document_path)
            
            result = {
                "success": True,
                "message": "文档上传并处理成功",
                "document_path": document_path,
                "content_length": len(self.document_content) if self.document_content else 0,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            logger.info("文档上传处理成功")
            return result
            
        except Exception as e:
            error_msg = f"上传文档失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": error_msg,
                "timestamp": datetime.datetime.now().isoformat()
            }

    def add_documents_to_rag(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        添加文档到RAG向量库
        :param document_paths: 文档路径列表
        :return: 处理结果
        """
        try:
            logger.info(f"添加 {len(document_paths)} 个文档到RAG向量库")
            
            if not self._rag_ready or not self.rag_manager:
                raise Exception("RAG管理器未就绪")
            
            # 添加文档到向量库
            self.rag_manager.add_documents_to_store(document_paths)
            
            # 更新状态
            self._vector_store_loaded = True
            
            # 获取更新后的向量库信息
            vector_info = self.rag_manager.get_vector_store_info()
            
            result = {
                "success": True,
                "message": f"成功添加 {len(document_paths)} 个文档到向量库",
                "document_count": len(document_paths),
                "vector_info": vector_info,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            logger.info("文档添加到RAG向量库成功")
            return result
            
        except Exception as e:
            error_msg = f"添加文档到RAG向量库失败: {str(e)}"
            self._last_error = error_msg
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": error_msg,
                "timestamp": datetime.datetime.now().isoformat()
            }