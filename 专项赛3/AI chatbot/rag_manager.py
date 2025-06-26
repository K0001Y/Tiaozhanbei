"""
RAG(检索增强生成)功能管理器
负责向量存储、检索器等RAG相关功能
"""
import os
import json
import logging
import torch
from datetime import datetime
from typing import List, Optional, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from transformers import AutoModel, AutoTokenizer
from langchain.schema import Document
from utils import FileHandler, DocumentLoader
from config import (
    DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, 
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_STORE_PATH,
    SUPPORTED_ENCODINGS,
    DEFAULT_PROMPT_TEMPLATE
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGManager:
    """RAG功能管理器类"""
    
    def __init__(self, embedding_model_path: str = "model"):
        """
        初始化RAG管理器
        :param embedding_model_path: 嵌入模型路径
        """
        try:
            self.embedding_model_path = embedding_model_path
            self.embedding_model = None
            self.vector_store = None
            self.retriever = None
            self.document_loader = DocumentLoader()
            self.file_handler = FileHandler()
            self.device = self._get_device()
            
            # 确保必要的目录存在
            os.makedirs("model", exist_ok=True)
            os.makedirs("RAG", exist_ok=True)
            
            logger.info(f"RAG管理器初始化成功，使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"RAG管理器初始化失败: {str(e)}")
            raise Exception(f"RAG管理器初始化失败: {str(e)}")

    def _get_device(self):
        """
        检测并返回可用的设备
        :return: 设备类型 ('cuda' 或 'cpu')
        """
        try:
            if torch.cuda.is_available():
                device = 'cuda'
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"检测到GPU设备: {gpu_name}, 共 {gpu_count} 个GPU")
            else:
                device = 'cpu'
                logger.info("未检测到GPU设备，使用CPU")
            return device
        except Exception as e:
            logger.warning(f"设备检测失败，默认使用CPU: {str(e)}")
            return 'cpu'
        
    def load_embedding_model(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        加载嵌入模型
        :param model_name: 嵌入模型名称
        :return: 嵌入模型对象
        """
        try:
            logger.info(f"正在加载嵌入模型: {model_name}")
        
        # 构造本地模型存储路径
            local_model_path = os.path.join(self.embedding_model_path, model_name.replace("/", "_"))
        
        # 定义必需的文件
            required_files = [
                'config.json',
                'pytorch_model.bin',  # 或者 model.safetensors
                'tokenizer_config.json',
                'vocab.txt'  # 或者其他 tokenizer 文件
            ]
        
        # 检查本地模型文件是否完整
            all_files_exist = all(os.path.exists(os.path.join(local_model_path, f)) for f in required_files)
        
            if all_files_exist:
                logger.info(f"从本地加载嵌入模型: {local_model_path}")
                model_path = local_model_path
            else:
                logger.info(f"从 HuggingFace 下载嵌入模型并保存到本地: {model_name}")
            # 下载并保存模型和 tokenizer 到本地路径
                model = AutoModel.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                os.makedirs(local_model_path, exist_ok=True)
                model.save_pretrained(local_model_path)
                tokenizer.save_pretrained(local_model_path)
                model_path = local_model_path
        
        # 创建嵌入模型
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        # 保存模型信息到本地
            model_info = {
                'model_name': model_name,
                'local_path': local_model_path,
                'model_kwargs': {'device': self.device},
                'encode_kwargs': {'normalize_embeddings': True}
            }
            self._save_embedding_model_info(model_info)
        
            logger.info(f"嵌入模型加载成功，使用设备: {self.device}")
            return self.embedding_model
        
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {str(e)}")
            raise Exception(f"加载嵌入模型失败: {str(e)}")

    def _save_embedding_model_info(self, model_info: dict):
        """
        保存嵌入模型信息到文件
        :param model_info: 嵌入模型信息字典
        """
        try:
            logger.info("正在保存嵌入模型信息")
            
            model_info_path = os.path.join(self.embedding_model_path, "model_info.json")
            
            # 如果文件已存在，读取现有信息
            existing_info = {}
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    existing_info = json.load(f)
            
            # 更新模型信息
            existing_info[model_info['model_name']] = model_info
            
            # 保存到文件
            with open(model_info_path, 'w', encoding='utf-8') as f:
                json.dump(existing_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"嵌入模型信息已保存到: {model_info_path}")
            
        except Exception as e:
            logger.error(f"保存嵌入模型信息失败: {str(e)}")
            raise Exception(f"保存嵌入模型信息失败: {str(e)}")
    def _create_empty_vector_store(self):
        """
        创建空的向量存储
        :return: 空向量存储对象
        """
        try:
            logger.info("正在创建空向量存储")
            
            if not self.embedding_model:
                logger.info("嵌入模型未加载，正在加载默认模型")
                self.load_embedding_model()
            
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
            
            # 删除临时文档
            # FAISS不支持直接删除，但我们可以在后续添加真实文档时覆盖
            
            # 保存空向量存储到本地
            vector_store_path = os.path.join("RAG", "vector_store")
            self._save_vector_store(self.vector_store, vector_store_path)
            
            # 保存元数据
            metadata = {
                "documents_count": 0,
                "chunks_count": 0,
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
                "vector_store_path": vector_store_path,
                "created_time": datetime.now().isoformat(),
                "is_empty": True,
                "device": self.device
            }
            
            metadata_path = os.path.join("RAG", "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info("空向量存储创建成功")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"创建空向量存储失败: {str(e)}")
            raise Exception(f"创建空向量存储失败: {str(e)}")

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
            
            if not self.vector_store:
                logger.info("向量存储不存在，先创建空向量存储")
                self._create_empty_vector_store()
            
            if not self.embedding_model:
                logger.info("嵌入模型未加载，正在加载默认模型")
                self.load_embedding_model()
            
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
                return self.vector_store
            
            # 分块处理文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            split_documents = []
            for doc in loaded_documents:
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
            vector_store_path = os.path.join("RAG", "vector_store")
            self._save_vector_store(self.vector_store, vector_store_path)
            
            # 更新元数据
            metadata_path = os.path.join("RAG", "metadata.json")
            existing_metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
            
            # 更新计数
            original_docs = existing_metadata.get('documents_count', 0)
            original_chunks = existing_metadata.get('chunks_count', 0)
            
            updated_metadata = {
                "documents_count": original_docs + len(loaded_documents),
                "chunks_count": original_chunks + len(split_documents),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "vector_store_path": vector_store_path,
                "last_updated": datetime.now().isoformat(),
                "is_empty": False,
                "device": self.device
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

    
    def _load_vector_store(self, vector_store_path: str):
        """
        从指定路径加载向量存储
        :param vector_store_path: 向量存储路径
        :return: 向量存储对象
        """
        try:
            logger.info(f"正在从路径加载向量存储: {vector_store_path}")
            
            # 指定相对路径：RAG/
            if not os.path.isabs(vector_store_path):
                vector_store_path = os.path.join("RAG", vector_store_path)
            
            if not os.path.exists(vector_store_path + ".faiss"):
                raise FileNotFoundError(f"向量存储文件不存在: {vector_store_path}")
            
            if not self.embedding_model:
                logger.info("嵌入模型未加载，正在加载默认模型")
                self.load_embedding_model()
            
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

    def _save_vector_store(self, vector_store: FAISS, vector_store_path: str):
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
                vector_store_path = os.path.join("RAG", vector_store_path)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
            
            # 保存向量存储
            vector_store.save_local(vector_store_path)
            
            logger.info(f"向量存储已保存到: {vector_store_path}")
            
        except Exception as e:
            logger.error(f"保存向量存储失败: {str(e)}")
            raise Exception(f"保存向量存储失败: {str(e)}")
    
    def _create_retriever(self, vector_store: FAISS):
        """
        创建检索器
        :param vector_store: 向量存储对象
        :return: 检索器对象
        """
        try:
            logger.info("正在创建检索器")
            
            if not vector_store:
                raise ValueError("向量存储对象不能为空")
            
            # 创建检索器
            self.retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            logger.info("检索器创建成功")
            return self.retriever
            
        except Exception as e:
            logger.error(f"创建检索器失败: {str(e)}")
            raise Exception(f"创建检索器失败: {str(e)}")
        
    def _start_rag_manager(self, vector_store_path: str = DEFAULT_VECTOR_STORE_PATH):
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
            
            # 检查本地向量存储并加载
            full_vector_path = os.path.join("RAG", vector_store_path) if not os.path.isabs(vector_store_path) else vector_store_path
            
            if os.path.exists(full_vector_path + ".faiss"):
                logger.info("发现本地向量存储，正在加载")
                self._load_vector_store(vector_store_path)
            else:
                logger.warning("未发现本地向量存储，已创建空的存储")
                # 这里可以选择创建空的向量存储或者抛出异常
                self._create_empty_vector_store
                # raise Exception("未发现向量存储，请先创建向量存储")
            
            # 创建检索器
            if self.vector_store:
                self._create_retriever(self.vector_store)
            
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
        try:
            logger.info(f"正在执行检索操作，查询: {query}")
            
            if not query or not query.strip():
                raise ValueError("查询字符串不能为空")
            
            if not self.retriever:
                raise Exception("检索器未初始化，请先启动RAG管理器")
            
            # 执行检索操作
            retrieved_docs = self.retriever.get_relevant_documents(query)
            
            # 限制返回结果数量
            retrieved_docs = retrieved_docs[:k]
            
            # 格式化返回结果
            results = []
            for i, doc in enumerate(retrieved_docs):
                try:
                    result = {
                        "id": i,
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, 'score', None)  # 如果有相似度分数
                    }
                    results.append(result)
                except Exception as e:
                    logger.warning(f"格式化检索结果失败: {str(e)}")
                    continue
            
            logger.info(f"检索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"检索操作失败: {str(e)}")
            raise Exception(f"检索操作失败: {str(e)}")

    def get_vector_store_info(self) -> Dict[str, Any]:
        """
        获取向量存储信息
        :return: 向量存储信息字典
        """
        try:
            if not self.vector_store:
                return {"status": "未加载", "count": 0, "device": self.device}
            
            # 获取向量存储中的文档数量
            index_to_docstore_id = self.vector_store.index_to_docstore_id
            count = len(index_to_docstore_id) if index_to_docstore_id else 0
            
            # 读取元数据
            metadata_path = os.path.join("RAG", "metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            info = {
                "status": "已加载",
                "count": count,
                "device": self.device,
                "metadata": metadata
            }
            
            return info
            
        except Exception as e:
            logger.error(f"获取向量存储信息失败: {str(e)}")
            return {"status": "错误", "error": str(e), "device": self.device}

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
        try:
            logger.info("正在清空向量存储")
            
            # 创建新的空向量存储
            self._create_empty_vector_store()
            
            logger.info("向量存储已清空")
            
        except Exception as e:
            logger.error(f"清空向量存储失败: {str(e)}")
            raise Exception(f"清空向量存储失败: {str(e)}")

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