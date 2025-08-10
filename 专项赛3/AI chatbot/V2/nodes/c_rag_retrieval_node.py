"""
RAG检索节点
基于用户输入执行检索并提取相关上下文
"""
import logging
from typing import Dict, List, Any, TypedDict, Optional, Tuple
from config import DEFAULT_VECTOR_STORE_PATH
from RAG_system.rag_manager import RAGManager

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

class RAGRetrieveNode:
    """RAG检索节点类"""
    
    def __init__(self):
        """初始化RAG检索节点"""
        try:
            logger.info("初始化RAG检索节点")
            
            # 初始化RAG管理器
            self.rag_manager = RAGManager()
            
            # 启动RAG管理器
            self.rag_manager._start_rag_manager(DEFAULT_VECTOR_STORE_PATH)
            
            logger.info("RAG检索节点初始化完成")
            
        except Exception as e:
            logger.error(f"RAG检索节点初始化失败: {str(e)}")
            raise Exception(f"RAG检索节点初始化失败: {str(e)}")
    
    def _extract_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        从检索结果中提取并格式化上下文
        
        :param retrieved_docs: 检索结果文档列表
        :return: 格式化的上下文字符串
        """
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            # 提取内容和元数据
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            source = metadata.get('source', '未知来源')
            
            # 格式化为上下文片段
            context_part = f"[文档{i+1}] 来源: {source}\n{content}\n"
            context_parts.append(context_part)
        
        # 合并所有上下文
        return "\n".join(context_parts)
    
    def __call__(self, state: State) -> State:
        """
        执行RAG检索操作
        
        :param state: LangGraph状态
        :return: 更新后的状态和路由标识
        """
        try:
            logger.info("RAG检索节点启动")
            
            # 检查必要的输入
            user_input = state.get('user_input')
            if not user_input or not user_input.strip():
                logger.warning("用户输入为空，跳过检索")
                state['relevant_context'] = ""
                return state
            
            # 获取配置中的检索参数
            config = state.get('config', {})
            k = config.get('retriever_k', 4)  # 默认为4，可以根据链类型在配置中设置
            
            # 执行检索
            logger.info(f"执行检索，查询：{user_input}，k值：{k}")
            try:
                retrieved_docs = self.rag_manager._retrieve(user_input, k)
                
                # 保存原始检索结果到state中
                state['documents'] = retrieved_docs
                
                # 提取并组织相关上下文
                if retrieved_docs:
                    # 将检索结果格式化为上下文字符串
                    relevant_context = self._extract_context(retrieved_docs)
                    state['relevant_context'] = relevant_context
                    logger.info(f"成功提取上下文，共{len(retrieved_docs)}个文档")
                else:
                    # 没有检索到结果
                    state['relevant_context'] = ""
                    logger.info("检索未返回结果")
                    
            except Exception as e:
                # 检索失败
                logger.error(f"检索操作失败: {str(e)}")
                state['relevant_context'] = ""
                state['error'] = f"检索失败: {str(e)}"
            
            # 设置下一步路由
            return state
            
        except Exception as e:
            error_msg = f"RAG检索节点执行失败: {str(e)}"
            logger.error(error_msg)
            
            # 更新状态
            state['relevant_context'] = ""
            state['error'] = error_msg
            
            # 即使出错也继续流程
            return state

# 导出节点实例以便在图中使用
rag_retrieve_node = RAGRetrieveNode()