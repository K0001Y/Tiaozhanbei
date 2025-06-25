"""
RAG(检索增强生成)功能管理器
负责文档加载、向量存储、检索器等RAG相关功能
"""

import os
import json
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

from config import (
    DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, 
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_STORE_PATH,
    SUPPORTED_ENCODINGS,
    DEFAULT_PROMPT_TEMPLATE
)


class RAGManager:
    """RAG功能管理器类"""
    
    def __init__(self, embedding_model_path: str = "model"):
        self.embedding_model_path = embedding_model_path
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.documents = []
        self.embedding_model_info = None
        
        # 确保embedding模型目录存在
        os.makedirs(self.embedding_model_path, exist_ok=True)
        print(f"RAG管理器初始化完成，嵌入模型保存路径: {self.embedding_model_path}")
    
    def load_documents(self, file_paths, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
        """
        加载并处理文档
        :param file_paths: 文件路径列表或单个文件路径
        :param chunk_size: 分块大小
        :param chunk_overlap: 分块重叠大小
        :return: 文本块数量
        """
        if not isinstance(file_paths, list):
            file_paths = [file_paths]
        
        documents = []
        for file_path in file_paths:
            try:
                print(f"开始加载文档: {file_path}")
                loaded_docs = self._load_single_document(file_path)
                if loaded_docs:
                    documents.extend(loaded_docs)
                    print(f"成功加载文档: {file_path}, 文档数: {len(loaded_docs)}")
            except Exception as e:
                print(f"加载文档失败 {file_path}: {str(e)}")
        
        # 分割文档
        if documents:
            try:
                print(f"开始分割文档，共 {len(documents)} 个文档")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len
                )
                self.documents = text_splitter.split_documents(documents)
                print(f"文档分割完成，共有 {len(self.documents)} 个文本块")
                
                # 创建向量存储
                if self._create_vector_store():
                    return len(self.documents)
                else:
                    print("向量存储创建失败")
                    return 0
            except Exception as e:
                print(f"分割文档过程失败: {str(e)}")
                return 0
        else:
            print("没有成功加载任何文档")
            return 0
    
    def _load_single_document(self, file_path: str):
        """加载单个文档"""
        if file_path.endswith('.txt'):
            return self._load_text_file(file_path)
        elif file_path.endswith('.pdf'):
            print(f"检测到pdf文件，使用PyPDFLoader...")
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif os.path.isdir(file_path):
            return self._load_directory(file_path)
        else:
            print(f"不支持的文件类型: {file_path}")
            return []
    
    def _load_text_file(self, file_path: str):
        """加载文本文件，自动检测编码"""
        print(f"检测到txt文件，尝试自动检测编码...")
        
        # 尝试自动检测编码
        detected_encoding = None
        for encoding in SUPPORTED_ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # 读取一小部分验证编码
                detected_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if detected_encoding:
            print(f"检测到文件编码: {detected_encoding}")
            loader = TextLoader(file_path, encoding=detected_encoding)
            return loader.load()
        else:
            print("无法检测到正确的编码，使用latin-1编码")
            loader = TextLoader(file_path, encoding="latin-1")
            return loader.load()
    
    def _load_directory(self, dir_path: str):
        """加载目录中的所有文档"""
        print(f"检测到目录，使用DirectoryLoader...")
        
        for encoding in ["utf-8", "gbk", "gb18030"]:
            try:
                loader = DirectoryLoader(
                    dir_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": encoding}
                )
                docs = loader.load()
                print(f"成功使用{encoding}编码加载目录")
                return docs
            except Exception as e:
                print(f"使用{encoding}加载失败: {str(e)}")
                continue
        
        return []
    
    def _create_vector_store(self):
        """从文档创建向量存储"""
        if not self.documents:
            print("无法创建向量存储：缺少文档")
            return False
        
        try:
            print("开始创建向量存储...")
            embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
            
            # 保存嵌入模型信息
            self._save_embedding_info()
            
            # 确保RAG目录存在
            os.makedirs(DEFAULT_VECTOR_STORE_PATH, exist_ok=True)
            
            print("开始将文档转换为向量...")
            self.vector_store = FAISS.from_documents(self.documents, embeddings)
            
            print("创建检索器...")
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            
            print("向量存储创建成功")
            # 自动保存向量存储
            self.save_vector_store()
            return True
            
        except Exception as e:
            print(f"创建向量存储失败: {str(e)}")
            return False
    
    def _save_embedding_info(self):
        """保存嵌入模型信息"""
        try:
            self.embedding_model_info = {
                "model_name": DEFAULT_EMBEDDING_MODEL,
                "saved_path": self.embedding_model_path
            }
            
            if not os.path.exists(self.embedding_model_path):
                os.makedirs(self.embedding_model_path, exist_ok=True)
            
            with open(os.path.join(self.embedding_model_path, "embedding_info.json"), "w") as f:
                json.dump(self.embedding_model_info, f)
            
            print(f"嵌入模型信息已保存")
        except Exception as e:
            print(f"保存嵌入模型信息时出错: {str(e)}")
    
    
    def save_vector_store(self, path=DEFAULT_VECTOR_STORE_PATH):
        """保存向量数据库到本地"""
        if self.vector_store is None:
            return "向量存储为空，无法保存"
        
        try:
            os.makedirs(path, exist_ok=True)
            self.vector_store.save_local(path)
            
            # 保存嵌入模型信息到向量存储目录
            if hasattr(self, 'embedding_model_info') and self.embedding_model_info:
                embedding_info_path = os.path.join(path, "embedding_info.json")
                with open(embedding_info_path, "w") as f:
                    json.dump(self.embedding_model_info, f)
                return f"向量存储已保存到 {path}，嵌入模型信息已保存"
            
            return f"向量存储已保存到 {path}"
        except Exception as e:
            return f"保存向量存储失败: {str(e)}"
    
    def load_vector_store(self, path=DEFAULT_VECTOR_STORE_PATH, custom_embedding_model=None):
        """从本地加载向量数据库"""
        try:
            # 检查嵌入模型信息
            embedding_info_path = os.path.join(path, "embedding_info.json")
            embedding_model_name = None
            
            if os.path.exists(embedding_info_path) and not custom_embedding_model:
                try:
                    with open(embedding_info_path, "r") as f:
                        embedding_info = json.load(f)
                    embedding_model_name = embedding_info.get("model_name")
                    print(f"从保存的信息中加载嵌入模型: {embedding_model_name}")
                except Exception as e:
                    print(f"读取嵌入模型信息失败: {str(e)}")
            
            # 确定使用的嵌入模型
            if custom_embedding_model:
                embeddings = HuggingFaceEmbeddings(model_name=custom_embedding_model)
            elif embedding_model_name:
                embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            else:
                embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
            
            # 加载向量存储
            self.vector_store = FAISS.load_local(
                path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            
            return f"向量存储已从 {path} 加载成功"
        except Exception as e:
            return f"加载向量存储失败: {str(e)}"
    
    def get_relevant_documents(self, query: str, k: int = 4):
        """获取相关文档"""
        if self.retriever is None:
            print("检索器未初始化，请先加载向量存储")
            return []
        return self.retriever.get_relevant_documents(query)
    
    def is_ready(self):
        """检查R