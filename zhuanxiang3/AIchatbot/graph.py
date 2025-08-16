from langgraph.graph import StateGraph, END
from typing import Dict, List, Any, TypedDict, Optional, Tuple
import logging
import logging.handlers
import os
import time

# 导入所有节点（保持不变）
from nodes.a_input_node import input_process_node
from nodes.b_safety_guard_node import safety_guard_node
from nodes.c_rag_retrieval_node import RAGRetrieveNode
from nodes.d_recognize_intent_node import recognize_intent_node
from nodes.e_symptom_extraction_node import symptom_extraction_node
from nodes.f_follow_up_question_node import follow_up_question_node
from nodes.g_diagnosis_node import diagnosis_node
from nodes.h_prescription_node import prescription_node
from nodes.i_conversation_chain_node import conversation_chain_node
from nodes.j_response_safety_node import response_safety_node
from nodes.k_output_node import output_node

# 日志配置（保持不变）
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "tcm_system.log")
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=10*1024*1024, backupCount=5
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
logger = logging.getLogger(__name__)

# 状态类型定义（保持不变）
class State(TypedDict):
    """LangGraph状态类型"""
    user_input: str
    query: Optional[str]
    messages: List[Any]
    memory: Optional[Any]
    documents: Optional[List[Dict[str, Any]]]
    response: Optional[str]
    error: Optional[str]
    config: Dict[str, Any]
    safety_check: Optional[Dict[str, Any]]
    intent: Optional[str]
    intent_details: Optional[Dict[str, Any]]
    relevant_context: Optional[str]
    symptoms_list: Optional[List[Dict[str, Any]]]
    missing_info_list: Optional[List[str]]
    conversation_state: Optional[str]
    diagnosis_data: Optional[Dict[str, Any]]
    prescription_data: Optional[Dict[str, Any]]
    safety_violations: Optional[List[Dict[str, Any]]]

class GraphManager:
    """
    Linus式Graph管理器 - 单例模式，避免重复编译
    """
    _instance = None
    _compiled_graph = None
    _last_compiled_time = None
    _compilation_lock = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_compiled_graph(self):
        """
        获取编译好的graph，避免每次重复编译这个蠢操作
        """
        current_time = time.time()
        
        # 防止并发编译
        if self._compilation_lock:
            # 如果正在编译，等待一下
            retry_count = 0
            while self._compilation_lock and retry_count < 50:  # 最多等5秒
                time.sleep(0.1)
                retry_count += 1
        
        # 如果没有编译过，或者超过1小时重新编译（防止内存泄漏）
        if (self._compiled_graph is None or 
            self._last_compiled_time is None or 
            current_time - self._last_compiled_time > 3600):
            
            logger.info("需要重新编译graph - 避免重复编译的关键优化")
            
            self._compilation_lock = True
            try:
                self._compiled_graph = self._create_and_compile_graph()
                self._last_compiled_time = current_time
                logger.info("Graph编译完成，将复用此实例")
            finally:
                self._compilation_lock = False
        else:
            logger.info("复用已编译的graph实例 - 性能优化生效")
        
        return self._compiled_graph
    
    def _create_and_compile_graph(self):
        """
        创建并编译图 - 从原来的create_tcm_graph提取出来
        """
        logger.info("开始创建中医智能对话系统图...")
        
        # 创建RAG检索节点实例
        rag_node = RAGRetrieveNode()
        
        # 创建图实例
        graph = StateGraph(State)
        
        # 添加所有节点
        logger.info("添加所有节点...")
        graph.add_node("input", input_process_node)
        graph.add_node("safety_guard", safety_guard_node)
        graph.add_node("rag", rag_node)
        graph.add_node("recognize_intent", recognize_intent_node)
        graph.add_node("symptom_extraction", symptom_extraction_node)
        graph.add_node("follow_up_question", follow_up_question_node)
        graph.add_node("diagnosis", diagnosis_node)
        graph.add_node("prescription", prescription_node)
        graph.add_node("conversation_chain", conversation_chain_node)
        graph.add_node("response_safety", response_safety_node)
        graph.add_node("output", output_node)
        
        # 设置入口点
        graph.set_entry_point("input")
        
        # 添加边和条件路由
        logger.info("配置节点间的路由逻辑...")
        
        # 1. 输入节点到安全检查节点或输出节点
        graph.add_conditional_edges(
            source="input",
            path=lambda x: "safety_guard" if not x.get("error") else "output",
            path_map={
                "safety_guard": "safety_guard",
                "output": "output"
            }
        )
        
        # 2. 安全检查节点到RAG节点或输出节点
        graph.add_conditional_edges(
            source="safety_guard",
            path=lambda x: "rag" if x.get("safety_check", {}).get("is_emergency") == False else "output",
            path_map={
                "rag": "rag",
                "output": "output"
            }
        )
        
        # 3. RAG节点到意图识别节点
        graph.add_edge("rag", "recognize_intent")
        
        # 4. 意图识别节点到症状提取或对话链
        graph.add_conditional_edges(
            source="recognize_intent",
            path=lambda x: "symptom_extraction" if x.get("intent") == "diagnosis" else "conversation_chain",
            path_map={
                "symptom_extraction": "symptom_extraction",
                "conversation_chain": "conversation_chain"
            }
        )
        
        # 5. 症状提取节点到追问或诊断
        graph.add_conditional_edges(
            source="symptom_extraction",
            path=lambda x: "follow_up_question" if len(x.get("symptoms_list", [])) < 2 else "diagnosis",
            path_map={
                "follow_up_question": "follow_up_question",
                "diagnosis": "diagnosis"
            }
        )
        
        # 6. 追问节点到输出或诊断
        graph.add_conditional_edges(
            source="follow_up_question",
            path=lambda x: "output" if x.get("conversation_state") == "awaiting_follow_up" else "diagnosis",
            path_map={
                "output": "output",
                "diagnosis": "diagnosis"
            }
        )
        
        # 7. 诊断节点到处方节点
        graph.add_edge("diagnosis", "prescription")
        
        # 8. 处方节点到安全检查节点
        graph.add_edge("prescription", "response_safety")
        
        # 9. 对话链节点到安全检查节点
        graph.add_edge("conversation_chain", "response_safety")
        
        # 10. 响应安全检查节点到输出节点
        graph.add_edge("response_safety", "output")
        
        # 编译图
        logger.info("编译图...")
        app = graph.compile()
        
        logger.info("中医智能对话系统图创建完成")
        return app

# 全局Graph管理器实例
_graph_manager = GraphManager()

def create_tcm_graph():
    """
    【已废弃】保持向后兼容的接口
    实际调用GraphManager，避免重复编译
    """
    logger.warning("create_tcm_graph() 已废弃，请使用 get_compiled_graph()")
    return _graph_manager.get_compiled_graph()

def get_compiled_graph():
    """
    获取编译好的graph实例 - 推荐使用的接口
    """
    return _graph_manager.get_compiled_graph()

def run_tcm_graph(user_input: str, messages: List[Any] = None, memory: Any = None, config: Dict[str, Any] = None):
    """
    运行中医图分析 - Linus修复版
    
    关键修复：
    1. 不再每次调用都重新编译graph
    2. 支持状态复用和增量计算
    3. 保持向后兼容性
    """
    logger.info(f"开始处理用户输入: {user_input}")
    
    # 构建初始状态
    initial_state = {
        "user_input": user_input,
        "messages": messages or [],
        "memory": memory,
        "config": config or {"retriever_k": 4},
        "query": None,
        "documents": None,
        "response": None,
        "error": None,
        "safety_check": None,
        "intent": None,
        "intent_details": None,
        "relevant_context": None,
        "symptoms_list": None,
        "missing_info_list": None,
        "conversation_state": None,
        "diagnosis_data": None,
        "prescription_data": None,
        "safety_violations": None
    }
    
    # 【核心修复】获取已编译的graph，而不是每次重新编译
    app = get_compiled_graph()
    
    try:
        logger.info("开始执行图...")
        result = app.invoke(initial_state)
        logger.info("图执行完成")
        return result
    except Exception as e:
        logger.error(f"图执行过程中发生错误: {str(e)}")
        initial_state["error"] = f"图执行失败: {str(e)}"
        initial_state["response"] = f"很抱歉，系统处理您的请求时发生错误。请稍后再试或联系管理员。错误信息: {str(e)}"
        return initial_state

def run_tcm_graph_with_state(user_input: str, existing_state: Dict[str, Any] = None, 
                            messages: List[Any] = None, memory: Any = None, 
                            config: Dict[str, Any] = None):
    """
    基于已有状态运行图 - 新增接口，支持增量计算
    
    :param user_input: 用户输入
    :param existing_state: 已有的图状态（用于增量计算）
    :param messages: 消息历史
    :param memory: 记忆数据
    :param config: 配置信息
    :return: 图执行结果
    """
    logger.info(f"基于已有状态处理用户输入: {user_input}")
    
    if existing_state:
        # 基于已有状态进行增量计算
        incremental_state = existing_state.copy()
        incremental_state["user_input"] = user_input
        
        # 更新messages
        if messages is not None:
            incremental_state["messages"] = messages
        elif "messages" not in incremental_state:
            incremental_state["messages"] = []
        
        # 添加新的用户输入到消息历史
        incremental_state["messages"].append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        
        # 更新其他参数
        if memory is not None:
            incremental_state["memory"] = memory
        if config is not None:
            incremental_state["config"] = config
    else:
        # 没有已有状态，创建新状态
        incremental_state = {
            "user_input": user_input,
            "messages": messages or [],
            "memory": memory,
            "config": config or {"retriever_k": 4},
            "query": None,
            "documents": None,
            "response": None,
            "error": None,
            "safety_check": None,
            "intent": None,
            "intent_details": None,
            "relevant_context": None,
            "symptoms_list": None,
            "missing_info_list": None,
            "conversation_state": None,
            "diagnosis_data": None,
            "prescription_data": None,
            "safety_violations": None
        }
    
    # 获取已编译的graph
    app = get_compiled_graph()
    
    try:
        logger.info("开始执行增量图计算...")
        result = app.invoke(incremental_state)
        logger.info("增量图执行完成")
        return result
    except Exception as e:
        logger.error(f"增量图执行过程中发生错误: {str(e)}")
        incremental_state["error"] = f"图执行失败: {str(e)}"
        incremental_state["response"] = f"很抱歉，系统处理您的请求时发生错误。请稍后再试或联系管理员。错误信息: {str(e)}"
        return incremental_state

def get_graph_stats():
    """
    获取Graph统计信息 - 用于监控和调试
    """
    return {
        "graph_compiled": _graph_manager._compiled_graph is not None,
        "last_compiled_time": _graph_manager._last_compiled_time,
        "compilation_lock": _graph_manager._compilation_lock,
        "uptime_hours": (time.time() - _graph_manager._last_compiled_time) / 3600 if _graph_manager._last_compiled_time else 0
    }

# 测试代码保持不变，但使用优化后的接口
if __name__ == "__main__":
    test_input = "我最近总是感到头晕 Encore encore，还伴有口干舌燥，请问是什么原因？"
    try:
        logger.info(f"测试用例 - 处理用户输入: {test_input}")
        
        # 测试原有接口
        result1 = run_tcm_graph(test_input)
        response1 = result1.get("response", "")
        logger.info(f"原接口响应: {response1}")
        
        # 测试新的增量接口
        additional_input = "补充信息：症状持续了一周"
        result2 = run_tcm_graph_with_state(additional_input, result1)
        response2 = result2.get("response", "")
        logger.info(f"增量接口响应: {response2}")
        
        # 输出统计信息
        stats = get_graph_stats()
        logger.info(f"Graph统计信息: {stats}")
        
    except Exception as e:
        logger.error(f"运行测试时出错: {str(e)}")