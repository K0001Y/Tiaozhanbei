import logging
import time
from typing import Dict, List, Any, TypedDict, Optional, Union, Tuple
from langchain.schema import AIMessage

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
    prescription_data: Optional[Dict[str, Any]]  # 处方推荐数据
    safety_violations: Optional[List[Dict[str, Any]]]  # 安全违规记录

# 可选的数据库管理器（如果不需要，可以移除）
class DatabaseManager:
    """数据库管理器类，用于保存对话和元数据"""
    
    def __init__(self, connection_string=None):
        """初始化数据库管理器"""
        self.is_ready = connection_string is not None
        self.connection_string = connection_string
        
        if self.is_ready:
            try:
                # 这里应该有真正的数据库连接代码
                # 例如：self.client = MongoClient(connection_string)
                # 或者：self.engine = create_engine(connection_string)
                logger.info("数据库连接初始化成功")
            except Exception as e:
                logger.error(f"数据库连接初始化失败: {str(e)}")
                self.is_ready = False
    
    def save_conversation(self, user_input: str, response: str, metadata: Dict[str, Any]) -> bool:
        """保存对话及元数据到数据库"""
        if not self.is_ready:
            logger.warning("数据库未就绪，无法保存对话")
            return False
        
        try:
            # 生成会话记录
            conversation_record = {
                "timestamp": time.time(),
                "user_input": user_input,
                "response": response,
                "metadata": metadata
            }
            
            # 这里应该有真正的数据库保存代码
            # 例如：self.client.db.conversations.insert_one(conversation_record)
            logger.info("对话已保存到数据库")
            return True
        
        except Exception as e:
            logger.error(f"保存对话到数据库失败: {str(e)}")
            return False

# =============== 节点实现 ===============

class OutputNode:
    """输出处理节点类"""
    
    def __init__(self, db_connection_string=None):
        """初始化输出处理节点"""
        # 初始化数据库管理器（如果提供了连接字符串）
        self.db_manager = DatabaseManager(db_connection_string) if db_connection_string else None
        
        logger.info(f"输出处理节点初始化完成，数据库{'已连接' if db_connection_string else '未连接'}")
    
    def _create_default_error_message(self) -> str:
        """创建默认错误响应"""
        return (
            "很抱歉，在处理您的请求时遇到了问题。请尝试重新表述您的问题，"
            "或稍后再试。如果您有关于中医的问题，我们很乐意为您提供帮助。"
        )
    
    def _collect_metadata(self, state: State) -> Dict[str, Any]:
        """从状态中收集元数据"""
        metadata = {}
        
        # 收集关键元数据字段
        for key in [
            "intent", "intent_details", "safety_check", "conversation_state",
            "safety_violations"
        ]:
            if key in state and state[key] is not None:
                metadata[key] = state[key]
        
        # 添加处理时间戳
        metadata["timestamp"] = time.time()
        
        # 添加诊断和处方摘要（如果有）
        if "diagnosis_data" in state and state["diagnosis_data"]:
            diagnosis = state["diagnosis_data"]
            metadata["diagnosis_summary"] = {
                "pattern_type": diagnosis.get("pattern_type"),
                "confidence": diagnosis.get("confidence")
            }
        
        if "prescription_data" in state and state["prescription_data"]:
            prescription = state["prescription_data"]
            metadata["prescription_summary"] = {
                "formula_name": prescription.get("formula_name")
            }
        
        # 添加症状计数
        if "symptoms_list" in state and state["symptoms_list"]:
            metadata["symptoms_count"] = len(state["symptoms_list"])
        
        return metadata
    
    def __call__(self, state: State) -> State:
        """节点主函数"""
        try:
            # 获取输入
            response = state.get("response", "")
            messages = state.get("messages", [])
            memory = state.get("memory")
            user_input = state.get("user_input", "")
            
            # 如果响应为空且有错误，使用默认错误消息
            if (not response) and state.get("error"):
                response = self._create_default_error_message()
                logger.warning(f"使用默认错误消息代替空响应，错误: {state.get('error')}")
            
            # 确保有响应
            if not response:
                response = "很抱歉，我无法理解您的请求。请尝试用不同的方式表述您的问题。"
                logger.warning("响应为空，使用默认回复")
            
            # 创建AIMessage
            ai_message = AIMessage(content=response)
            
            # 添加到消息历史
            messages.append(ai_message)
            
            # 保存到内存
            if memory:
                memory.chat_memory.add_ai_message(response)
            
            # 收集元数据
            metadata = self._collect_metadata(state)
            
            # 保存到数据库（如果可用）
            if self.db_manager and self.db_manager.is_ready:
                saved = self.db_manager.save_conversation(user_input, response, metadata)
                if saved:
                    logger.info("对话已保存到数据库")
                else:
                    logger.warning("保存对话到数据库失败")
            
            # 清理不必要的大数据结构以减轻最终状态负担
            clean_state = {**state}
            for key in ["documents", "relevant_context"]:
                if key in clean_state:
                    clean_state[key] = None
            
            # 更新最终状态
            final_state = {
                **clean_state,
                "messages": messages,
                "response": response,
            }
            
            logger.info("输出处理完成，返回最终状态")
            return final_state
        
        except Exception as e:
            error_msg = f"输出处理过程中出错: {str(e)}"
            logger.error(error_msg)
            
            # 异常处理：确保即使在错误情况下也返回有效响应
            default_response = self._create_default_error_message()
            ai_message = AIMessage(content=default_response)
            
            # 确保消息列表存在
            if "messages" not in state or state["messages"] is None:
                state["messages"] = []
            
            state["messages"].append(ai_message)
            
            return {
                **state,
                "error": error_msg,
                "response": default_response
            }

# 导出节点实例以便在图中使用（默认不连接数据库）
output_node = OutputNode()

# 如果需要连接数据库，可以使用：
# output_node = OutputNode(db_connection_string="your_connection_string")