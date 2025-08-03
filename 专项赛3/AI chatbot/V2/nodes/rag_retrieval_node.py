"""
RAG检索节点 - 为LangGraph框架设计
专注于向量存储检索功能
"""
import os
import json
import logging
from typing import Dict, List, Any, TypedDict, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

# 导入之前拆分的各个组件
from RAG_system.embedding_model import EmbeddingModelManager
from RAG_system.vector_store import VectorStoreManager
from RAG_system.retriever import RetrieverManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义类型
class RAGState(TypedDict):
    """RAG节点的状态类型"""
    query: str
    documents: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    config: Dict[str, Any]


class RAGRetrievalNode:
    """
    RAG检索节点类
    作为LangGraph中的一个节点使用，仅实现检索功能
    """
    
    def __init__(self, 
                 model_path: str = "model",
                 vector_store_path: str = "RAG/vector_store",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化RAG检索节点
        
        Args:
            model_path: 嵌入模型存储路径
            vector_store_path: 向量存储路径
            embedding_model: 嵌入模型名称
        """
        try:
            # 初始化嵌入模型
            self.embedding_manager = EmbeddingModelManager(model_path)
            self.device = self.embedding_manager._get_device()
            self.embedding = self.embedding_manager.load_embedding_model(embedding_model)
            
            # 初始化向量存储管理器并加载向量存储
            self.vector_store_manager = VectorStoreManager(self.embedding)
            
            # 尝试加载现有向量存储
            try:
                self.vector_store = self.vector_store_manager.load_vector_store(vector_store_path)
                logger.info(f"成功加载向量存储: {vector_store_path}")
            except FileNotFoundError:
                logger.error(f"向量存储不存在: {vector_store_path}")
                raise
            
            # 初始化检索器
            self.retriever_manager = RetrieverManager()
            self.retriever = self.retriever_manager.create_retriever(self.vector_store)
            
            logger.info("RAG检索节点初始化成功")
        
        except Exception as e:
            logger.error(f"RAG检索节点初始化失败: {str(e)}")
            raise
    
    def __call__(self, state: RAGState) -> RAGState:
        """
        执行检索操作，作为LangGraph节点的主函数
        
        Args:
            state: 当前状态，包含查询和配置
            
        Returns:
            更新后的状态，包含检索结果
        """
        try:
            # 从状态中获取查询
            query = state.get("query")
            if not query:
                return {"error": "查询为空", "documents": [], **state}
            
            # 从状态中获取配置
            config = state.get("config", {})
            k = config.get("k", 4)  # 默认检索4个结果
            
            # 执行检索
            logger.info(f"正在检索: '{query}' (k={k})")
            results = self.retriever_manager.retrieve(query, k)
            
            # 更新状态
            return {
                "query": query,
                "documents": results,
                "error": None,
                "config": config
            }
        
        except Exception as e:
            error_msg = f"检索失败: {str(e)}"
            logger.error(error_msg)
            return {
                "query": state.get("query", ""),
                "documents": [],
                "error": error_msg,
                "config": state.get("config", {})
            }
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取RAG检索节点信息
        
        Returns:
            节点信息字典
        """
        try:
            vector_store_info = self.vector_store_manager.get_vector_store_info(self.device)
            
            return {
                "status": "正常",
                "device": self.device,
                "vector_store": vector_store_info,
                "embedding_model": self.embedding_manager.embedding_model_path
            }
        except Exception as e:
            return {
                "status": "错误",
                "error": str(e),
                "device": self.device
            }