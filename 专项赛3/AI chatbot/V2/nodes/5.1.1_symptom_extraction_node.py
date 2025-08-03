import logging
import json
from typing import Dict, List, Any, TypedDict, Optional, Union, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain

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
    relevant_context: Optional[str]  # RAG检索相关上下文
    symptoms_list: Optional[List[Dict[str, Any]]]  # 提取的症状列表
    missing_info_list: Optional[List[str]]  # 缺失的信息列表

# =============== LLM配置 ===============

# 模型选择 - 直接在代码中配置
# 可选项: "openai", "azure", "local"
LLM_PROVIDER = "openai"

# OpenAI配置
OPENAI_CONFIG = {
    "model": "gpt-4",            # 使用更高能力的模型提取症状
    "temperature": 0.1,          # 低温度确保更稳定的结果
    "api_key": "sk-your-api-key", # 你的API密钥
    "api_base": "https://api.openai.com/v1", # API基础URL
    "timeout": 60,               # 较长的超时时间
    "max_tokens": 2000           # 充足的输出长度
}

# Azure OpenAI配置
AZURE_CONFIG = {
    "deployment_name": "gpt-4",  # Azure部署名称
    "api_version": "2023-05-15", # API版本
    "api_key": "your-azure-key", # Azure API密钥
    "api_base": "https://your-resource.openai.azure.com/" # Azure端点
}

# 本地模型配置
LOCAL_MODEL_CONFIG = {
    "model_path": "/path/to/medical-model.gguf", # 本地医疗专用模型路径
    "n_ctx": 4096,               # 更大的上下文窗口
    "n_threads": 8               # 更多线程以提高性能
}

# 症状提取阈值配置
SYMPTOM_THRESHOLD = 2  # 提取的症状数量阈值，少于这个值会要求更多信息

# =============== 节点实现 ===============

class SymptomExtractionNode:
    """症状提取节点类"""
    
    def __init__(self):
        """初始化症状提取节点"""
        # 初始化LLM
        self.llm = self._create_llm()
        
        # 创建输出解析器
        self.output_parser = self._create_output_parser()
        
        # 创建症状提取提示
        self.symptom_prompt = self._create_symptom_prompt()
        
        # 创建LLMChain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.symptom_prompt,
            output_parser=self.output_parser,
            verbose=True
        )
        
        logger.info(f"症状提取节点初始化完成，使用LLM提供商: {LLM_PROVIDER}")
    
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
                temperature=OPENAI_CONFIG["temperature"],
                max_tokens=OPENAI_CONFIG["max_tokens"]
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
    
    def _create_output_parser(self):
        """创建结构化输出解析器"""
        response_schemas = [
            ResponseSchema(
                name="symptoms",
                description="从用户描述中提取的症状列表，每个症状应包含症状名称、描述、严重程度和持续时间",
                type="array"
            ),
            ResponseSchema(
                name="missing_info",
                description="需要从用户那里获取的缺失信息列表，这些信息对诊断很重要",
                type="array"
            ),
            ResponseSchema(
                name="stage",
                description="当前会话阶段，应为'symptom_collection'",
                type="string"
            )
        ]
        
        return StructuredOutputParser.from_response_schemas(response_schemas)
    
    def _create_symptom_prompt(self):
        """创建症状提取提示模板"""
        template = """
        你是一个专业的医疗症状提取助手。你的任务是从用户的描述中提取症状信息，并识别出可能缺失的重要信息。
        
        # 用户历史消息
        {message_history}
        
        # 当前用户输入
        {user_input}
        
        # 相关医疗参考信息
        {relevant_context}
        
        请从用户描述中提取症状信息，并识别出哪些信息是缺失的。对于每个症状，请提供：
        1. 症状名称
        2. 症状描述（根据用户原文）
        3. 严重程度（如果用户提到）
        4. 持续时间（如果用户提到）
        
        对于缺失信息，请列出为进行更准确的诊断而需要询问用户的问题。

        {format_instructions}
        """
        
        return ChatPromptTemplate.from_template(template)
    
    def _format_message_history(self, messages):
        """格式化消息历史"""
        if not messages:
            return "无历史消息"
        
        formatted_history = []
        for msg in messages:
            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                role = msg.type
                content = msg.content
                formatted_history.append(f"{role}: {content}")
        
        return "\n".join(formatted_history)
    
    def __call__(self, state: State) -> Tuple[State, str]:
        """节点主函数"""
        try:
            # 获取输入
            user_input = state.get("user_input", "")
            messages = state.get("messages", [])
            relevant_context = state.get("relevant_context", "无相关上下文")
            
            # 格式化消息历史
            message_history = self._format_message_history(messages)
            
            # 准备链的输入
            chain_input = {
                "user_input": user_input,
                "message_history": message_history,
                "relevant_context": relevant_context,
                "format_instructions": self.output_parser.get_format_instructions()
            }
            
            # 运行链
            logger.info("开始提取症状...")
            result = self.chain.invoke(chain_input)
            
            # 解析结果
            parsed_output = result
            symptoms = parsed_output.get("symptoms", [])
            missing_info = parsed_output.get("missing_info", [])
            
            logger.info(f"提取到 {len(symptoms)} 个症状和 {len(missing_info)} 个缺失信息")
            
            # 更新状态
            updated_state = {
                **state,
                "symptoms_list": symptoms,
                "missing_info_list": missing_info
            }
            
            # 确定路由
            if missing_info or len(symptoms) < SYMPTOM_THRESHOLD:
                logger.info("需要更多信息，路由到后续问题节点")
                return updated_state, "follow_up"
            else:
                logger.info("症状信息充足，路由到诊断节点")
                return updated_state, "diagnosis"
        
        except Exception as e:
            error_msg = f"症状提取过程中出错: {str(e)}"
            logger.error(error_msg)
            
            # 发生错误时路由到follow_up以获取更多信息
            return {
                **state,
                "error": error_msg,
                "symptoms_list": [],
                "missing_info_list": ["由于处理错误，需要您重新描述症状"]
            }, "follow_up"

# 导出节点实例以便在图中使用
symptom_extraction_node = SymptomExtractionNode()