"""
向量存储管理组件
负责创建、加载和管理向量存储
"""
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import (
    DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, 
    DEFAULT_VECTOR_STORE_PATH,
    
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """向量存储管理类"""
    
    def __init__(self, embedding_model=None):
        """
        初始化向量存储管理器
        :param embedding_model: 嵌入模型对象
        """
        try:
            self.embedding_model = embedding_model
            self.vector_store = None
            
            # 确保必要的目录存在
            os.makedirs(DEFAULT_VECTOR_STORE_PATH, exist_ok=True)
            
            logger.info("向量存储管理器初始化成功")
            
        except Exception as e:
            logger.error(f"向量存储管理器初始化失败: {str(e)}")
            raise Exception(f"向量存储管理器初始化失败: {str(e)}")
    
    def create_empty_vector_store(self, device='cpu'):
        """
        创建空的向量存储
        :param device: 设备类型
        :return: 空向量存储对象
        """
        try:
            logger.info("正在创建空向量存储")
            
            if not self.embedding_model:
                raise ValueError("嵌入模型未加载")
            
            # 创建一个临时文档用于初始化FAISS
            temp_doc = Document(
                page_content="临时初始化文档",
                metadata={"temp": True}
            )
            
            # 创建向量存储
            self.vector_store = FAISS.from_documents(
                documents=[temp_doc],
                embedding=self.embedding_model
            )
            
            # 保存空向量存储到本地
            vector_store_path = os.path.join( "vector_store")
            self.save_vector_store(self.vector_store, vector_store_path)
            
            # 保存元数据
            metadata = {
                "documents_count": 0,
                "chunks_count": 0,
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
                "vector_store_path": vector_store_path,
                "created_time": datetime.now().isoformat(),
                "is_empty": True,
                "device": device
            }
            
            metadata_path = os.path.join(DEFAULT_VECTOR_STORE_PATH, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info("空向量存储创建成功")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"创建空向量存储失败: {str(e)}")
            raise Exception(f"创建空向量存储失败: {str(e)}")

    def add_documents_to_store(self, documents: List[Document], chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, device='cpu'):
        """
        向现有向量存储添加文档
        :param documents: 文档对象列表
        :param chunk_size: 分块大小
        :param chunk_overlap: 分块重叠大小
        :param device: 设备类型
        :return: 更新后的向量存储对象
        """
        try:
            logger.info(f"正在向向量存储添加 {len(documents)} 个文档")
            
            if not self.vector_store:
                logger.info("向量存储不存在，先创建空向量存储")
                self.create_empty_vector_store(device)
            
            if not self.embedding_model:
                raise ValueError("嵌入模型未加载")
            
            if not documents:
                logger.warning("没有文档需要添加")
                return self.vector_store
            
            # 分块处理文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            split_documents = []
            for doc in documents:
                try:
                    splits = text_splitter.split_documents([doc])
                    split_documents.extend(splits)
                except Exception as e:
                    logger.warning(f"文档分块失败: {str(e)}")
                    continue
            
            if not split_documents:
                logger.warning("文档分块后没有有效内容")
                return self.vector_store
            
            # 将新文档添加到现有向量存储
            self.vector_store.add_documents(split_documents)
            
            # 保存更新后的向量存储
            vector_store_path = os.path.join( DEFAULT_VECTOR_STORE_PATH, "vector_store")
            self.save_vector_store(self.vector_store, vector_store_path)
            
            # 更新元数据
            metadata_path = os.path.join( DEFAULT_VECTOR_STORE_PATH, "metadata.json")
            existing_metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
            
            # 更新计数
            original_docs = existing_metadata.get('documents_count', 0)
            original_chunks = existing_metadata.get('chunks_count', 0)
            
            updated_metadata = {
                "documents_count": original_docs + len(documents),
                "chunks_count": original_chunks + len(split_documents),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "vector_store_path": vector_store_path,
                "last_updated": datetime.now().isoformat(),
                "is_empty": False,
                "device": device
            }
            
            # 保留原有的创建时间
            if 'created_time' in existing_metadata:
                updated_metadata['created_time'] = existing_metadata['created_time']
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(updated_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功添加 {len(split_documents)} 个文档块到向量存储")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {str(e)}")
            raise Exception(f"添加文档到向量存储失败: {str(e)}")

    def load_vector_store(self, vector_store_path: str):
        """
        从指定路径加载向量存储
        :param vector_store_path: 向量存储路径
        :return: 向量存储对象
        """
        try:
            logger.info(f"正在从路径加载向量存储: {vector_store_path}")
            
            # 指定相对路径：RAG/
            if not os.path.isabs(vector_store_path):
                vector_store_path = os.path.join( DEFAULT_VECTOR_STORE_PATH, vector_store_path)
                
            index_file = os.path.join(vector_store_path, "index.faiss")
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"向量存储文件不存在: {index_file}")
            
            if not self.embedding_model:
                raise ValueError("嵌入模型未加载")
            
            # 加载向量存储
            self.vector_store = FAISS.load_local(
                vector_store_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            
            logger.info("向量存储加载成功")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"加载向量存储失败: {str(e)}")
            raise Exception(f"加载向量存储失败: {str(e)}")

    def save_vector_store(self, vector_store: FAISS, vector_store_path: str):
        """
        保存向量存储到指定路径
        :param vector_store: 向量存储对象
        :param vector_store_path: 向量存储路径
        """
        try:
            logger.info(f"正在保存向量存储到: {vector_store_path}")
            
            if not vector_store:
                raise ValueError("向量存储对象不能为空")
            
            # 保存向量存储到指定相对路径RAG/
            if not os.path.isabs(vector_store_path):
                vector_store_path = os.path.join( DEFAULT_VECTOR_STORE_PATH, vector_store_path)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
            
            # 保存向量存储
            vector_store.save_local(vector_store_path)
            
            logger.info(f"向量存储已保存到: {vector_store_path}")
            
        except Exception as e:
            logger.error(f"保存向量存储失败: {str(e)}")
            raise Exception(f"保存向量存储失败: {str(e)}")
    
    def get_vector_store_info(self, device='cpu') -> Dict[str, Any]:
        """
        获取向量存储信息
        :param device: 设备类型
        :return: 向量存储信息字典
        """
        try:
            if not self.vector_store:
                return {"status": "未加载", "count": 0, "device": device}
            
            # 获取向量存储中的文档数量
            index_to_docstore_id = self.vector_store.index_to_docstore_id
            count = len(index_to_docstore_id) if index_to_docstore_id else 0
            
            # 读取元数据
            metadata_path = os.path.join( DEFAULT_VECTOR_STORE_PATH, "metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            info = {
                "status": "已加载",
                "count": count,
                "device": device,
                "metadata": metadata
            }
            
            return info
            
        except Exception as e:
            logger.error(f"获取向量存储信息失败: {str(e)}")
            return {"status": "错误", "error": str(e), "device": device}
    
    def clear_vector_store(self, device='cpu'):
        """
        清空向量存储
        :param device: 设备类型
        """
        try:
            logger.info("正在清空向量存储")
            
            # 创建新的空向量存储
            self.create_empty_vector_store(device)
            
            logger.info("向量存储已清空")
            
        except Exception as e:
            logger.error(f"清空向量存储失败: {str(e)}")
            raise Exception(f"清空向量存储失败: {str(e)}")