import logging
from typing import Dict, List, Any, TypedDict, Optional, Union
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义状态类型
class State(TypedDict):
    """LangGraph状态类型"""
    user_input: str  # 用户输入
    query: Optional[str]  # 查询（处理后的用户输入）
    messages: List[Any]  # 消息历史
    memory: Optional[ConversationBufferMemory]  # 对话内存
    documents: Optional[List[Dict[str, Any]]]  # RAG检索结果
    response: Optional[str]  # 最终响应
    error: Optional[str]  # 错误信息
    config: Dict[str, Any]  # 配置信息

def input_node(state: State) -> tuple[State, str]:
    """
    输入处理节点：校验用户输入，转换为HumanMessage，并保存到内存
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态和路由指示
    """
    try:
        # 获取用户输入
        user_input = state.get("user_input", "")
        
        # 校验用户输入（非空、非空白）
        if not user_input or user_input.isspace():
            # 输入无效，设置错误响应
            logger.warning("输入无效：空或仅包含空白字符")
            return {
                **state,
                "error": "输入不能为空或仅包含空白字符",
                "response": "请提供有效的输入。"
            }, "end"
        
        # 创建HumanMessage
        human_message = HumanMessage(content=user_input)
        
        # 更新messages列表
        messages = state.get("messages", [])
        messages.append(human_message)
        
        # 保存到内存
        memory = state.get("memory")
        if not memory:
            memory = ConversationBufferMemory()
        memory.chat_memory.add_user_message(user_input)
        
        # 设置查询（初始查询与用户输入相同）
        query = user_input
        
        logger.info(f"处理用户输入成功: '{user_input[:30]}...'（如果更长）")
        
        # 更新状态并路由到下一个节点
        return {
            **state,
            "user_input": user_input,
            "query": query,
            "messages": messages,
            "memory": memory,
            "error": None
        }, "next"
    
    except Exception as e:
        error_msg = f"处理用户输入时出错: {str(e)}"
        logger.error(error_msg)
        return {
            **state,
            "error": error_msg,
            "response": "处理您的输入时发生错误，请重试。"
        }, "end"