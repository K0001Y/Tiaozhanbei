import logging
import json
import os
from typing import Dict, List, Any, TypedDict, Optional, Literal, Union

# 根据需要导入不同的LLM客户端
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_BASE_URL,
    DEFAULT_API_KEY,
    DEFAULT_MODEL_CONFIGS,
    ALI_API_KEY,
    ALI_BASE_URL
)

ali_url = ALI_BASE_URL
ali_api = ALI_API_KEY

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 状态类型定义
class State(TypedDict):
    """LangGraph状态类型"""
    user_input: str  # 用户输入
    query: Optional[str]  # 查询（处理后的用户输入）
    messages: List[Any]  # 消息历史
    memory: Optional[Any]  # 对话内存
    documents: Optional[List[Dict[str, Any]]]  # RAG检索结果
    response: Optional[str]  # 最终响应
    error: Optional[str]  # 错误信息
    config: Dict[str, Any]  # 配置信息
    safety_check: Optional[Dict[str, Any]]  # 安全检查结果
    intent: Optional[str]  # 用户意图
    intent_details: Optional[Dict[str, Any]]  # 意图详细信息

# 定义医疗相关关键词
MEDICAL_KEYWORDS = [
    "病历", "症状", "诊断", "治疗", "病情", "检查", "病因",
    "用药", "副作用", "病史", "化验", "CT", "核磁", "X光", "B超"
]

# =============== LLM配置 ===============

# 模型选择 - 直接在代码中配置
# 可选项: "openai", "azure", "local"
LLM_PROVIDER = "openai"

# OpenAI配置
OPENAI_CONFIG = {
    "model": "tongyi-intent-detect-v3",      # 模型名称
    "temperature": 0,              # 温度参数（0表示最确定性）
    "api_key": ali_api,  # 你的API密钥
    "api_base": ali_url, # API基础URL
    "timeout": 30,                 # 请求超时时间（秒）
    "max_tokens": 1000             # 最大输出标记数
}

# Azure OpenAI配置
AZURE_CONFIG = {
    "deployment_name": "gpt-4",    # Azure部署名称
    "api_version": "2023-05-15",   # API版本
    "api_key": "your-azure-key",   # Azure API密钥
    "api_base": "https://your-resource.openai.azure.com/" # Azure端点
}

# 本地模型配置
LOCAL_MODEL_CONFIG = {
    "model_path": "/path/to/model.gguf", # 本地模型路径
    "n_ctx": 2048,                       # 上下文窗口大小
    "n_threads": 4                       # 线程数
}

# =============== 节点实现 ===============

class RecognizeIntentNode:
    """意图识别节点类"""
    
    def __init__(self):
        """初始化意图识别节点"""
        # 初始化LLM
        self.llm = self._create_llm()
        logger.info(f"已初始化意图识别LLM: {LLM_PROVIDER}")
    
    def _create_llm(self):
        """根据配置创建LLM实例"""
        if LLM_PROVIDER == "openai":
            return ChatOpenAI(
                model=OPENAI_CONFIG["model"],
                temperature=OPENAI_CONFIG["temperature"],
                openai_api_key=OPENAI_CONFIG["api_key"],
                openai_api_base=OPENAI_CONFIG["api_base"],
                request_timeout=OPENAI_CONFIG["timeout"],
                max_tokens=OPENAI_CONFIG["max_tokens"]
            )
        
        elif LLM_PROVIDER == "azure":
            return AzureChatOpenAI(
                deployment_name=AZURE_CONFIG["deployment_name"],
                openai_api_version=AZURE_CONFIG["api_version"],
                openai_api_key=AZURE_CONFIG["api_key"],
                openai_api_base=AZURE_CONFIG["api_base"],
                temperature=OPENAI_CONFIG["temperature"]
            )
        
        elif LLM_PROVIDER == "local":
            return LlamaCpp(
                model_path=LOCAL_MODEL_CONFIG["model_path"],
                temperature=OPENAI_CONFIG["temperature"],
                max_tokens=OPENAI_CONFIG["max_tokens"],
                n_ctx=LOCAL_MODEL_CONFIG["n_ctx"],
                n_threads=LOCAL_MODEL_CONFIG["n_threads"]
            )
        
        else:
            raise ValueError(f"不支持的LLM提供商: {LLM_PROVIDER}")
    
    def __call__(self, state: State) -> State:
        """节点主函数"""
        try:
            # 获取用户输入
            user_input = state.get("user_input", "")
            
            # 设置输出解析器
            response_schemas = [
                ResponseSchema(name="intent", 
                              description="用户的主要意图，必须是以下之一：'diagnosis'（寻求诊断或医疗建议）, 'question'（一般问题）, 'unclear'（意图不明确）", 
                              type="string"),
                ResponseSchema(name="confidence", 
                              description="意图识别的置信度，0到1之间的浮点数", 
                              type="number"),
                ResponseSchema(name="keywords", 
                              description="从用户输入中提取的关键词列表", 
                              type="array"),
                ResponseSchema(name="reasoning", 
                              description="识别该意图的推理过程", 
                              type="string")
            ]
            
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()
            
            # 创建提示模板
            template = """
            你是一个医疗对话系统的意图识别组件。分析用户输入并确定他们的主要意图。
            
            可能的意图类别：
            - diagnosis: 用户正在寻求医疗诊断、治疗建议或描述症状
            - question: 用户在询问一般问题，不是寻求具体的医疗诊断
            - unclear: 用户意图不清晰或无法分类
            
            特别注意：如果用户提到"病历"、"症状"、"诊断"等医疗相关词汇，这通常表明是diagnosis意图。

            用户输入: {user_input}
            
            {format_instructions}
            """
            
            # 创建提示
            prompt = ChatPromptTemplate.from_template(template)
            
            # 准备提示输入
            prompt_input = {
                "user_input": user_input,
                "format_instructions": format_instructions
            }
            
            # 获取LLM响应
            messages = prompt.format_messages(**prompt_input)
            response = self.llm.predict_messages(messages)
            
            # 解析LLM响应
            parsed_response = output_parser.parse(response.content)
            
            # 获取识别的意图
            intent = parsed_response.get("intent", "unclear")
            confidence = parsed_response.get("confidence", 0.0)
            keywords = parsed_response.get("keywords", [])
            reasoning = parsed_response.get("reasoning", "")
            
            # 检查是否包含医疗关键词，增强诊断意图识别
            medical_keywords_found = [kw for kw in MEDICAL_KEYWORDS if kw in user_input]
            if medical_keywords_found and intent != "diagnosis" and confidence < 0.8:
                intent = "diagnosis"
                reasoning += f"\n[调整] 检测到医疗关键词: {', '.join(medical_keywords_found)}，调整为诊断意图。"
            
            # 记录识别结果
            intent_details = {
                "intent": intent,
                "confidence": confidence,
                "keywords": keywords + medical_keywords_found,
                "reasoning": reasoning
            }
            
            logger.info(f"意图识别结果: {intent} (置信度: {confidence})")
            
            # 更新状态
            updated_state = {
                **state,
                "intent": intent,
                "intent_details": intent_details
            }
            
            # 根据意图决定路由
            if intent == "diagnosis":
                return updated_state
            else:
                return updated_state
        
        except Exception as e:
            error_msg = f"意图识别过程中出错: {str(e)}"
            logger.error(error_msg)
            
            # 发生错误时默认为conversation路由
            return {
                **state,
                "error": error_msg,
                "intent": "unclear",
                "intent_details": {
                    "intent": "unclear",
                    "confidence": 0.0,
                    "keywords": [],
                    "reasoning": f"处理过程中出错: {str(e)}"
                }
            }

# 导出节点实例以便在图中使用
recognize_intent_node = RecognizeIntentNode()