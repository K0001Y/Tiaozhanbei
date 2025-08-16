import logging
from typing import Dict, List, Any, TypedDict, Optional, Union, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from config import ALI_API_KEY, ALI_BASE_URL

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

# =============== LLM配置 ===============

# 模型选择 - 直接在代码中配置
# 可选项: "openai", "azure", "local"
LLM_PROVIDER = "openai"

# OpenAI配置
OPENAI_CONFIG = {
    "model": "qwen-plus-2025-07-28",    # 对话问答可以使用3.5模型
    "temperature": 0.4,          # 适当的创造性
    "api_key": ALI_API_KEY, # 你的API密钥
    "api_base": ALI_BASE_URL, # API基础URL
    "timeout": 30,               # 请求超时时间
    "max_tokens": 2000           # 较长的回复空间
}

# Azure OpenAI配置
AZURE_CONFIG = {
    "deployment_name": "gpt-35-turbo", # Azure部署名称
    "api_version": "2023-05-15",       # API版本
    "api_key": "your-azure-key",       # Azure API密钥
    "api_base": "https://your-resource.openai.azure.com/" # Azure端点
}

# 本地模型配置
LOCAL_MODEL_CONFIG = {
    "model_path": "/path/to/chat-model.gguf", # 本地模型路径
    "n_ctx": 4096,               # 较大的上下文窗口
    "n_threads": 4               # 线程数
}

# =============== 节点实现 ===============

class ConversationChainNode:
    """对话链节点类"""
    
    def __init__(self):
        """初始化对话链节点"""
        # 初始化LLM
        self.llm = self._create_llm()
        
        # 创建对话提示模板
        self.conversation_prompt = self._create_conversation_prompt()
        
        # 创建LLM链（不再使用ConversationChain）
        self.conversation_chain = LLMChain(
            llm=self.llm,
            prompt=self.conversation_prompt,
            verbose=True
        )
        
        logger.info(f"对话链节点初始化完成，使用LLM提供商: {LLM_PROVIDER}")
    
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
    
    def _create_conversation_prompt(self):
        """创建对话提示模板"""
        template = """
        你是一位博学多识的中医助手，能够回答各种中医相关问题，并提供专业、准确的知识和建议。

        # 对话历史
        {history}
        
        # 参考知识
        {relevant_context}
        
        # 回答指南
        1. 回答应基于中医理论体系，确保专业准确
        2. 解释专业术语时要通俗易懂
        3. 说明中医理论与现代医学的关系时保持客观
        4. 对于缺乏明确科学证据的内容，清晰表明这是传统观点
        5. 不做具体诊断或开具处方，而是提供一般性知识
        6. 如果问题超出中医范畴，可以适当引入现代医学知识，但要明确区分
        7. 对于无法确定的信息，坦诚表明自己的局限性
        
        用户问题: {user_input}
        
        助手回答:
        """
        
        return ChatPromptTemplate.from_template(template)
    
    def _format_history(self, messages):
        """格式化消息历史"""
        if not messages:
            return "这是一次新的对话。"
        
        formatted_history = []
        for msg in messages:
            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                role = "用户" if msg.type in ['user', 'human'] else "助手"
                content = msg.content
                formatted_history.append(f"{role}: {content}")
        
        return "\n".join(formatted_history)
    
    def _prepare_chain_input(self, user_input, messages, relevant_context):
        """准备链的输入参数"""
        # 格式化历史消息
        history = self._format_history(messages)
        
        # 处理相关上下文
        if not relevant_context:
            relevant_context = (
                "中医是中国传统医学，有数千年历史。核心理论包括阴阳五行、脏腑经络、气血津液等。"
                "诊断方法有望闻问切，治疗手段包括中药、针灸、推拿、气功等。强调整体观念和辨证论治。"
            )
        
        return {
            "user_input": user_input,
            "history": history,
            "relevant_context": relevant_context
        }
    
    def _validate_response(self, response):
        """验证响应质量"""
        if not response or not response.strip():
            return False
        
        # 检查响应长度
        if len(response.strip()) < 10:
            return False
        
        # 检查是否包含基本的中医元素（可选）
        tcm_keywords = ["中医", "阴阳", "五行", "气血", "脏腑", "经络", "辨证", "治疗", "养生"]
        user_friendly_words = ["您", "建议", "可以", "帮助", "了解"]
        
        # 响应应该专业但友好
        has_tcm_content = any(keyword in response for keyword in tcm_keywords)
        has_friendly_tone = any(word in response for word in user_friendly_words)
        
        return has_tcm_content or has_friendly_tone
    
    def __call__(self, state: State) -> State:
        """节点主函数"""
        try:
            # 获取输入
            user_input = state.get("user_input", "")
            messages = state.get("messages", [])
            relevant_context = state.get("relevant_context", "")
            
            # 准备链的输入
            chain_input = self._prepare_chain_input(user_input, messages, relevant_context)
            
            # 记录调用信息
            logger.info(f"处理用户输入: {user_input[:50]}...")
            logger.info("开始生成对话回复...")
            
            # 运行对话链
            result = self.conversation_chain.invoke(chain_input)
            response = result.get('text', '').strip()
            
           
            
            logger.info(f"生成的回复: {response[:100]}...")
            
            # 更新状态
            updated_state = {
                **state,
                "response": response
            }
            
            return updated_state
        
        except Exception as e:
            error_msg = f"生成对话回复过程中出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            fallback_response += (
                "\n\n很抱歉，我在处理您的问题时遇到了技术困难。"
                "请重新表述您的问题，或稍后再试。"
            )
            
            
            
            return {
                **state,
                "error": error_msg,
                "response": fallback_response
            }

# 导出节点实例以便在图中使用
conversation_chain_node = ConversationChainNode()