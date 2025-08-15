"""
输入处理节点
校验用户输入，转换为HumanMessage，并保存到内存
"""
import logging
from typing import Dict, List, Any, TypedDict, Optional, Tuple
from langchain.schema import HumanMessage
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
    symptoms_list: Optional[List[Dict[str, Any]]]  # 提取的症状列表
    missing_info_list: Optional[List[str]]  # 缺失的信息列表
    conversation_state: Optional[str]  # 对话状态标记
    diagnosis_data: Optional[Dict[str, Any]]  # 辨证分析结果

class InputProcessNode:
    """输入处理节点类"""
    
    def __init__(self):
        """初始化输入处理节点"""
        try:
            logger.info("初始化输入处理节点")
            
            # 节点配置
            self.max_input_length = 1000  # 最大输入长度限制
            
            logger.info("输入处理节点初始化完成")
            
        except Exception as e:
            logger.error(f"输入处理节点初始化失败: {str(e)}")
            raise Exception(f"输入处理节点初始化失败: {str(e)}")
    
    def _validate_input(self, user_input: str) -> bool:
        """
        校验用户输入
        
        :param user_input: 用户输入
        :return: 校验结果
        """
        if not user_input or user_input.isspace():
            return False
        
        if len(user_input) > self.max_input_length:
            return False
            
        return True
    
    def _create_human_message(self, user_input: str) -> HumanMessage:
        """
        创建HumanMessage对象
        
        :param user_input: 用户输入
        :return: HumanMessage对象
        """
        return HumanMessage(content=user_input)
    
    def _update_memory(self, memory: Optional[Any], user_input: str) -> Any:
        """
        更新对话内存
        
        :param memory: 当前内存对象
        :param user_input: 用户输入
        :return: 更新后的内存对象
        """
        if not memory:
            memory = ConversationBufferMemory()
        
        memory.chat_memory.add_user_message(user_input)
        return memory
    
    def __call__(self, state: State) -> State:
        """
        执行输入处理操作
        
        :param state: LangGraph状态
        :return: 更新后的状态和路由标识
        """
        try:
            logger.info("输入处理节点启动")
            
            # 获取用户输入
            user_input = state.get("user_input", "")
            
            # 校验用户输入
            if not self._validate_input(user_input):
                logger.warning("输入无效：空或仅包含空白字符")
                state['error'] = "输入不能为空或仅包含空白字符"
                state['response'] = "请提供有效的输入。"
                return state
            
            # 创建HumanMessage
            try:
                human_message = self._create_human_message(user_input)
                
                # 更新messages列表
                messages = state.get("messages", [])
                messages.append(human_message)
                state['messages'] = messages
                
                # 更新内存
                memory = state.get("memory")
                memory = self._update_memory(memory, user_input)
                state['memory'] = memory
                
                # 设置查询
                state['query'] = user_input
                
                # 清除之前的错误
                state['error'] = None
                
                logger.info(f"处理用户输入成功: '{user_input[:30]}{'...' if len(user_input) > 30 else ''}'")
                
            except Exception as e:
                logger.error(f"处理用户输入失败: {str(e)}")
                state['error'] = f"处理用户输入失败: {str(e)}"
                state['response'] = "处理您的输入时发生错误，请重试。"
                return state
            
            # 设置下一步路由
            return state
            
        except Exception as e:
            error_msg = f"输入处理节点执行失败: {str(e)}"
            logger.error(error_msg)
            
            # 更新状态
            state['error'] = error_msg
            state['response'] = "处理您的输入时发生错误，请重试。"
            
            # 出错时路由到结束
            return state, "end"

# 导出节点实例以便在图中使用
input_process_node = InputProcessNode()