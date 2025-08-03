import logging
from typing import Dict, List, Any, TypedDict, Optional, Union, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

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
    "model": "gpt-3.5-turbo",    # 对话问答可以使用3.5模型
    "temperature": 0.4,          # 适当的创造性
    "api_key": "sk-your-api-key", # 你的API密钥
    "api_base": "https://api.openai.com/v1", # API基础URL
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
        
        # 创建内存对象
        self.memory = ConversationBufferMemory()
        
        # 创建对话链
        self.chain = ConversationChain(
            llm=self.llm,
            prompt=self.conversation_prompt,
            memory=self.memory,
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
        
        用户问题: {input}
        
        助手回答:
        """
        
        return ChatPromptTemplate.from_template(template)
    
    def _format_history(self, messages):
        """格式化消息历史"""
        if not messages:
            return ""
        
        formatted_history = []
        for msg in messages:
            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                role = "用户" if msg.type in ['user', 'human'] else "助手"
                content = msg.content
                formatted_history.append(f"{role}: {content}")
        
        return "\n".join(formatted_history)
    
    def __call__(self, state: State) -> Tuple[State, str]:
        """节点主函数"""
        try:
            # 获取输入
            user_input = state.get("user_input", "")
            messages = state.get("messages", [])
            relevant_context = state.get("relevant_context", "")
            
            # 如果没有相关上下文，使用通用回复
            if not relevant_context:
                relevant_context = (
                    "中医是中国传统医学，有数千年历史。核心理论包括阴阳五行、脏腑经络、气血津液等。"
                    "诊断方法有望闻问切，治疗手段包括中药、针灸、推拿、气功等。强调整体观念和辨证论治。"
                )
            
            # 格式化历史消息
            history = self._format_history(messages)
            
            # 同步内存
            self.memory.clear()  # 清除旧内存
            if history:
                self.memory.chat_memory.add_user_message("历史对话：" + history)
            
            # 准备链的输入
            chain_input = {
                "input": user_input,
                "relevant_context": relevant_context,
                "history": history
            }
            
            # 运行链
            logger.info("生成对话回复...")
            result = self.chain.invoke(chain_input)
            response = result.get('response', '')
            
            # 如果响应为空，使用默认回复
            if not response:
                response = (
                    "很抱歉，我无法提供关于这个问题的具体回答。"
                    "您是想了解中医相关的内容吗？如果是，请提供更多细节，我会尽力协助您。"
                )
            
            logger.info(f"生成的回复: {response[:100]}...")
            
            # 更新状态
            updated_state = {
                **state,
                "response": response
            }
            
            return updated_state, "to_safety_check"
        
        except Exception as e:
            error_msg = f"生成对话回复过程中出错: {str(e)}"
            logger.error(error_msg)
            
            # 发生错误时的默认回复
            default_response = (
                "很抱歉，我在处理您的问题时遇到了技术困难。"
                "请重新表述您的问题，或稍后再试。如果您有关于中医的具体问题，我很乐意为您解答。"
            )
            
            return {
                **state,
                "error": error_msg,
                "response": default_response
            }, "to_safety_check"

# 导出节点实例以便在图中使用
conversation_chain_node = ConversationChainNode()