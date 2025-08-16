"""
RAG(检索增强生成)功能管理器
负责向量存储、检索器等RAG相关功能
"""
import os
import json
import logging
from typing import List, Optional, Dict, Any
from langchain.schema import Document
from utils import FileHandler, DocumentLoader
from config import (
    DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, 
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_STORE_PATH,
    SUPPORTED_ENCODINGS,
    DEFAULT_EMBEDDING_PATH 
)

from RAG_system.embedding_model import EmbeddingModelManager
from RAG_system.vector_store import VectorStoreManager
from RAG_system.retriever import RetrieverManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGManager:
    """RAG功能管理器类"""
    
    def __init__(self, embedding_model_path: str = DEFAULT_EMBEDDING_PATH):
        """
        初始化RAG管理器
        :param embedding_model_path: 嵌入模型路径
        """
        try:
            self.embedding_model_path = embedding_model_path
            
            # 初始化组件管理器
            self.embedding_manager = EmbeddingModelManager(embedding_model_path)
            self.vector_store_manager = None  # 稍后初始化，需要embedding模型
            self.retriever_manager = RetrieverManager()
            
            # 辅助组件
            self.document_loader = DocumentLoader()
            self.file_handler = FileHandler()
            
            # 确保必要的目录存在
            os.makedirs(DEFAULT_EMBEDDING_PATH, exist_ok=True)
            os.makedirs(DEFAULT_VECTOR_STORE_PATH, exist_ok=True)
            
            # 获取设备
            self.device = self.embedding_manager._get_device()
            
            logger.info(f"RAG管理器初始化成功，使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"RAG管理器初始化失败: {str(e)}")
            raise Exception(f"RAG管理器初始化失败: {str(e)}")
    
    def load_embedding_model(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        加载嵌入模型
        :param model_name: 嵌入模型名称
        :return: 嵌入模型对象
        """
        return self.embedding_manager.load_embedding_model(model_name)
    
    def _create_empty_vector_store(self):
        """
        创建空的向量存储
        :return: 空向量存储对象
        """
        # 确保向量存储管理器已初始化
        if not self.vector_store_manager:
            self.vector_store_manager = VectorStoreManager(self.embedding_manager.embedding_model)
        
        return self.vector_store_manager.create_empty_vector_store(self.device)
    
    def add_documents_to_store(self, documents: List[str], chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
        """
        向现有向量存储添加文档
        :param documents: 文档路径列表
        :param chunk_size: 分块大小
        :param chunk_overlap: 分块重叠大小
        :return: 更新后的向量存储对象
        """
        try:
            logger.info(f"正在向向量存储添加 {len(documents)} 个文档")
            
            # 确保嵌入模型已加载
            if not self.embedding_manager.embedding_model:
                logger.info("嵌入模型未加载，正在加载默认模型")
                self.load_embedding_model()
            
            # 确保向量存储管理器已初始化
            if not self.vector_store_manager:
                self.vector_store_manager = VectorStoreManager(self.embedding_manager.embedding_model)
            
            # 确保向量存储已创建
            if not self.vector_store_manager.vector_store:
                self._create_empty_vector_store()
            
            # 加载新文档
            loaded_documents = []
            for doc_path in documents:
                try:
                    # 验证文件路径
                    validation = self.document_loader.validate_file_path(doc_path)
                    if not validation['valid']:
                        logger.warning(f"文件验证失败 {doc_path}: {validation['reason']}")
                        continue
                    
                    # 加载单个文档
                    docs = self.document_loader.load_single_document(doc_path)
                    if docs:
                        # 转换为LangChain Document格式
                        for doc in docs:
                            if isinstance(doc, dict):
                                doc_obj = Document(
                                    page_content=doc.get('page_content', ''),
                                    metadata=doc.get('metadata', {})
                                )
                                loaded_documents.append(doc_obj)
                            else:
                                loaded_documents.append(doc)
                except Exception as e:
                    logger.warning(f"加载文档失败 {doc_path}: {str(e)}")
                    continue
            
            if not loaded_documents:
                logger.warning("没有成功加载任何新文档")
                return self.vector_store_manager.vector_store
            
            # 添加文档到向量存储
            return self.vector_store_manager.add_documents_to_store(loaded_documents, chunk_size, chunk_overlap, self.device)
            
        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {str(e)}")
            raise Exception(f"添加文档到向量存储失败: {str(e)}")
    
    def _load_vector_store(self, vector_store_path: str):
        """
        从指定路径加载向量存储
        :param vector_store_path: 向量存储路径
        :return: 向量存储对象
        """
        # 确保向量存储管理器已初始化
        if not self.vector_store_manager:
            self.vector_store_manager = VectorStoreManager(self.embedding_manager.embedding_model)
        
        return self.vector_store_manager.load_vector_store(vector_store_path)
    
    def _create_retriever(self, vector_store):
        """
        创建检索器
        :param vector_store: 向量存储对象
        :return: 检索器对象
        """
        return self.retriever_manager.create_retriever(vector_store)
    
    def _start_rag_manager(self, vector_store_path: str = "vector_store"):
        """
        启动RAG管理器
        :param vector_store_path: 向量存储路径
        :return: RAG管理器对象
        """
        try:
            logger.info("正在启动RAG管理器")
            
            # 检查本地是否存在嵌入模型
            model_info_path = os.path.join(self.embedding_model_path, "model_info.json")
            
            if os.path.exists(model_info_path):
                logger.info("发现本地嵌入模型信息，正在加载")
                try:
                    with open(model_info_path, 'r', encoding='utf-8') as f:
                        model_info = json.load(f)
                    
                    # 加载第一个可用的模型
                    if model_info:
                        first_model = list(model_info.keys())[0]
                        self.load_embedding_model(first_model)
                    else:
                        self.load_embedding_model()
                except Exception as e:
                    logger.warning(f"加载本地模型信息失败，使用默认模型: {str(e)}")
                    self.load_embedding_model()
            else:
                logger.info("未发现本地嵌入模型，下载默认模型")
                self.load_embedding_model()
            
            # 初始化向量存储管理器
            self.vector_store_manager = VectorStoreManager(self.embedding_manager.embedding_model)
            
            
            vector_store_path = "vector_store" 
            # 检查本地向量存储并加载
            full_vector_path = os.path.join(DEFAULT_VECTOR_STORE_PATH, vector_store_path) 
            index_file = os.path.join(full_vector_path, "index.faiss")
            logger.info(f"检查向量存储路径: {index_file}")
            if os.path.exists(index_file):
                logger.info("发现本地向量存储，正在加载")
                self._load_vector_store(vector_store_path)
            else:
                logger.warning("未发现本地向量存储，已创建空的存储")
                self._create_empty_vector_store()
            
            # 创建检索器
            if self.vector_store_manager.vector_store:
                self._create_retriever(self.vector_store_manager.vector_store)
            
            logger.info("RAG管理器启动成功")
            return self
            
        except Exception as e:
            logger.error(f"启动RAG管理器失败: {str(e)}")
            raise Exception(f"启动RAG管理器失败: {str(e)}")
    
    def _retrieve(self, query: str, k: int = 4) -> List[dict]:
        """
        执行检索操作
        :param query: 查询字符串
        :param k: 返回结果数量
        :return: 检索结果列表
        """
        return self.retriever_manager.retrieve(query, k)
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """
        获取向量存储信息
        :return: 向量存储信息字典
        """
        if not self.vector_store_manager:
            return {"status": "未初始化", "count": 0, "device": self.device}
        
        return self.vector_store_manager.get_vector_store_info(self.device)
    
    def batch_load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        批量加载多个文档
        :param file_paths: 文件路径列表
        :return: 文档对象列表
        """
        try:
            logger.info(f"正在批量加载 {len(file_paths)} 个文档")
            
            # 使用DocumentLoader的批量加载功能
            all_docs = self.document_loader.load_multiple_documents(file_paths)
            
            # 转换为统一的Document格式
            documents = []
            for doc in all_docs:
                if isinstance(doc, dict):
                    doc_obj = Document(
                        page_content=doc.get('page_content', ''),
                        metadata=doc.get('metadata', {})
                    )
                    documents.append(doc_obj)
                else:
                    documents.append(doc)
            
            logger.info(f"批量加载完成，共获得 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"批量加载文档失败: {str(e)}")
            raise Exception(f"批量加载文档失败: {str(e)}")
    
    def clear_vector_store(self):
        """
        清空向量存储
        """
        if not self.vector_store_manager:
            self.vector_store_manager = VectorStoreManager(self.embedding_manager.embedding_model)
        
        self.vector_store_manager.clear_vector_store(self.device)
    
    def get_supported_formats(self) -> List[str]:
        """
        获取支持的文档格式
        :return: 支持的格式列表
        """
        try:
            return self.document_loader.get_supported_formats()
        except Exception as e:
            logger.error(f"获取支持格式失败: {str(e)}")
            return ['.txt']  # 默认支持txt格式