"""
工具类模块 - 数据库管理器、文件处理和文档加载器
包含数据库操作、文件处理、文档加载、日志配置等工具函数
"""

import asyncio
import json
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import aiohttp
import requests

# 文档加载相关导入
try:
    from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain未安装，文档加载功能将受限")

# 支持的编码格式
SUPPORTED_ENCODINGS = [
    'utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 
    'big5', 'ascii', 'latin-1', 'cp1252'
]


class DatabaseManager:
    """数据库管理器 - 支持同步和异步HTTP请求"""
    
    def __init__(self, 
                 db_url: str, 
                 db_token: Optional[str] = None, 
                 use_async: bool = False,
                 timeout: int = 30,
                 max_retries: int = 3):
        """
        初始化数据库管理器
        
        Args:
            db_url: 数据库API URL
            db_token: 认证令牌（可选）
            use_async: 是否使用异步模式
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.db_url = db_url.rstrip('/')
        self.db_token = db_token
        self.use_async = use_async
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 设置请求头
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'ChatBot-Utils/1.0'
        }
        
        if self.db_token:
            self.headers['Authorization'] = f'Bearer {self.db_token}'
        
        # 初始化异步会话
        self._session = None
        
        # 设置日志
        self.logger = logging.getLogger(f"DatabaseManager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    async def _get_session(self):
        """获取异步HTTP会话"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
        return self._session
    
    async def close(self):
        """关闭异步会话"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def send_response(self, 
                     response: str, 
                     user_input: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        发送响应数据到数据库
        
        Args:
            response: AI响应内容
            user_input: 用户输入
            metadata: 元数据（可选）
            
        Returns:
            bool: 发送是否成功
        """
        data = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'ai_response': response,
            'metadata': metadata or {}
        }
        
        if self.use_async:
            # 在异步环境中运行
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self._send_response_async(data))
            except RuntimeError:
                # 如果没有运行的事件循环，创建新的
                return asyncio.run(self._send_response_async(data))
        else:
            return self._send_response_sync(data)
    
    def _send_response_sync(self, data: Dict[str, Any]) -> bool:
        """同步发送响应"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.db_url}/conversations",
                    json=data,
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                if response.status_code in [200, 201]:
                    self.logger.debug("数据发送成功")
                    return True
                else:
                    self.logger.warning(f"数据发送失败，状态码: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"发送数据失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
        
        return False
    
    async def _send_response_async(self, data: Dict[str, Any]) -> bool:
        """异步发送响应"""
        session = await self._get_session()
        
        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    f"{self.db_url}/conversations",
                    json=data
                ) as response:
                    if response.status in [200, 201]:
                        self.logger.debug("数据发送成功")
                        return True
                    else:
                        self.logger.warning(f"数据发送失败，状态码: {response.status}")
                        
            except aiohttp.ClientError as e:
                self.logger.warning(f"发送数据失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
        
        return False
    
    def get_conversations(self, 
                         limit: int = 100, 
                         offset: int = 0,
                         filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        获取对话记录
        
        Args:
            limit: 返回记录数限制
            offset: 偏移量
            filters: 过滤条件
            
        Returns:
            List[Dict]: 对话记录列表
        """
        params = {
            'limit': limit,
            'offset': offset
        }
        
        if filters:
            params.update(filters)
        
        if self.use_async:
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self._get_conversations_async(params))
            except RuntimeError:
                return asyncio.run(self._get_conversations_async(params))
        else:
            return self._get_conversations_sync(params)
    
    def _get_conversations_sync(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """同步获取对话记录"""
        try:
            response = requests.get(
                f"{self.db_url}/conversations",
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json().get('data', [])
            else:
                self.logger.error(f"获取对话记录失败，状态码: {response.status_code}")
                return []
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"获取对话记录失败: {str(e)}")
            return []
    
    async def _get_conversations_async(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """异步获取对话记录"""
        session = await self._get_session()
        
        try:
            async with session.get(
                f"{self.db_url}/conversations",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', [])
                else:
                    self.logger.error(f"获取对话记录失败，状态码: {response.status}")
                    return []
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"获取对话记录失败: {str(e)}")
            return []
    
    def test_connection(self) -> Dict[str, Any]:
        """测试数据库连接"""
        try:
            if self.use_async:
                try:
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(self._test_connection_async())
                except RuntimeError:
                    return asyncio.run(self._test_connection_async())
            else:
                return self._test_connection_sync()
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _test_connection_sync(self) -> Dict[str, Any]:
        """同步测试连接"""
        try:
            response = requests.get(
                f"{self.db_url}/health",
                headers=self.headers,
                timeout=self.timeout
            )
            
            return {
                'success': response.status_code == 200,
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _test_connection_async(self) -> Dict[str, Any]:
        """异步测试连接"""
        session = await self._get_session()
        start_time = time.time()
        
        try:
            async with session.get(f"{self.db_url}/health") as response:
                response_time = time.time() - start_time
                
                return {
                    'success': response.status == 200,
                    'status_code': response.status,
                    'response_time': response_time,
                    'timestamp': datetime.now().isoformat()
                }
                
        except aiohttp.ClientError as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class FileHandler:
    """文件处理工具类"""
    
    @staticmethod
    def read_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """
        读取文件内容
        
        Args:
            file_path: 文件路径
            encoding: 编码格式
            
        Returns:
            str: 文件内容
        """
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            for enc in ['gbk', 'gb2312', 'latin-1']:
                try:
                    with open(file_path, 'r', encoding=enc) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"无法解码文件: {file_path}")
    
    @staticmethod
    def write_file(file_path: Union[str, Path], 
                   content: str, 
                   encoding: str = 'utf-8',
                   create_dirs: bool = True) -> bool:
        """
        写入文件内容
        
        Args:
            file_path: 文件路径
            content: 文件内容
            encoding: 编码格式
            create_dirs: 是否创建目录
            
        Returns:
            bool: 写入是否成功
        """
        try:
            file_path = Path(file_path)
            
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding=encoding) as file:
                file.write(content)
            return True
        except Exception as e:
            logging.error(f"写入文件失败: {str(e)}")
            return False
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict: 文件信息
        """
        try:
            file_path = Path(file_path)
            stat = file_path.stat()
            
            return {
                'exists': file_path.exists(),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'is_file': file_path.is_file(),
                'is_dir': file_path.is_dir(),
                'extension': file_path.suffix,
                'name': file_path.name,
                'parent': str(file_path.parent)
            }
        except Exception as e:
            return {
                'exists': False,
                'error': str(e)
            }
    
    @staticmethod
    def list_files(directory: Union[str, Path], 
                   pattern: str = "*",
                   recursive: bool = False) -> List[str]:
        """
        列出目录中的文件
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            recursive: 是否递归查找
            
        Returns:
            List[str]: 文件路径列表
        """
        try:
            directory = Path(directory)
            
            if recursive:
                files = directory.rglob(pattern)
            else:
                files = directory.glob(pattern)
            
            return [str(f) for f in files if f.is_file()]
        except Exception as e:
            logging.error(f"列出文件失败: {str(e)}")
            return []


class DocumentLoader:
    """文档加载器 - 支持多种文档格式和智能编码检测"""
    
    def __init__(self, use_langchain: bool = True):
        """
        初始化文档加载器
        
        Args:
            use_langchain: 是否使用LangChain加载器（需要安装langchain）
        """
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        self.logger = logging.getLogger("DocumentLoader")
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        if use_langchain and not LANGCHAIN_AVAILABLE:
            self.logger.warning("LangChain未安装，将使用基础文档加载功能")
    
    def load_single_document(self, file_path: str) -> List[Any]:
        """
        加载单个文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            List: 文档列表（LangChain Document对象或字典）
        """
        try:
            if file_path.endswith('.txt'):
                return self._load_text_file(file_path)
            elif file_path.endswith('.pdf'):
                return self._load_pdf_file(file_path)
            elif os.path.isdir(file_path):
                return self._load_directory(file_path)
            else:
                self.logger.warning(f"不支持的文件类型: {file_path}")
                return []
        except Exception as e:
            self.logger.error(f"加载文档失败 {file_path}: {str(e)}")
            return []
    
    def _load_text_file(self, file_path: str) -> List[Any]:
        """
        加载文本文件，自动检测编码
        
        Args:
            file_path: 文件路径
            
        Returns:
            List: 文档列表
        """
        self.logger.info(f"检测到txt文件，尝试自动检测编码...")
        
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
            self.logger.info(f"检测到文件编码: {detected_encoding}")
            
            if self.use_langchain:
                try:
                    loader = TextLoader(file_path, encoding=detected_encoding)
                    return loader.load()
                except Exception as e:
                    self.logger.warning(f"LangChain加载失败，使用基础加载: {str(e)}")
            
            # 基础加载方式
            return self._load_text_basic(file_path, detected_encoding)
        else:
            self.logger.warning("无法检测到正确的编码，使用latin-1编码")
            
            if self.use_langchain:
                try:
                    loader = TextLoader(file_path, encoding="latin-1")
                    return loader.load()
                except Exception as e:
                    self.logger.warning(f"LangChain加载失败，使用基础加载: {str(e)}")
            
            return self._load_text_basic(file_path, "latin-1")
    
    def _load_text_basic(self, file_path: str, encoding: str) -> List[Dict[str, Any]]:
        """
        基础文本加载方式（不依赖LangChain）
        
        Args:
            file_path: 文件路径
            encoding: 编码格式
            
        Returns:
            List[Dict]: 文档字典列表
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return [{
                'page_content': content,
                'metadata': {
                    'source': file_path,
                    'encoding': encoding,
                    'file_size': len(content),
                    'load_time': datetime.now().isoformat()
                }
            }]
        except Exception as e:
            self.logger.error(f"基础文本加载失败: {str(e)}")
            return []
    
    def _load_pdf_file(self, file_path: str) -> List[Any]:
        """
        加载PDF文件
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            List: 文档列表
        """
        self.logger.info(f"检测到PDF文件，尝试加载...")
        
        if self.use_langchain:
            try:
                loader = PyPDFLoader(file_path)
                return loader.load()
            except Exception as e:
                self.logger.error(f"PDF加载失败: {str(e)}")
                return []
        else:
            self.logger.warning("PDF加载需要LangChain支持")
            return []
    
    def _load_directory(self, dir_path: str) -> List[Any]:
        """
        加载目录中的所有文档
        
        Args:
            dir_path: 目录路径
            
        Returns:
            List: 文档列表
        """
        self.logger.info(f"检测到目录，使用批量加载...")
        
        if self.use_langchain:
            # 尝试使用LangChain的DirectoryLoader
            for encoding in ["utf-8", "gbk", "gb18030"]:
                try:
                    loader = DirectoryLoader(
                        dir_path,
                        glob="**/*.txt",
                        loader_cls=TextLoader,
                        loader_kwargs={"encoding": encoding}
                    )
                    docs = loader.load()
                    self.logger.info(f"成功使用{encoding}编码加载目录")
                    return docs
                except Exception as e:
                    self.logger.warning(f"使用{encoding}加载失败: {str(e)}")
                    continue
            
            return []
        else:
            # 基础目录加载
            return self._load_directory_basic(dir_path)
    
    def _load_directory_basic(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        基础目录加载方式（不依赖LangChain）
        
        Args:
            dir_path: 目录路径
            
        Returns:
            List[Dict]: 文档字典列表
        """
        documents = []
        
        try:
            # 获取目录下所有txt文件
            txt_files = FileHandler.list_files(dir_path, "*.txt", recursive=True)
            
            for file_path in txt_files:
                docs = self._load_text_file(file_path)
                documents.extend(docs)
            
            self.logger.info(f"基础方式加载目录完成，共{len(documents)}个文档")
            return documents
            
        except Exception as e:
            self.logger.error(f"基础目录加载失败: {str(e)}")
            return []
    
    def load_multiple_documents(self, file_paths: List[str]) -> List[Any]:
        """
        批量加载多个文档
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            List: 所有文档列表
        """
        all_documents = []
        
        for file_path in file_paths:
            docs = self.load_single_document(file_path)
            all_documents.extend(docs)
        
        self.logger.info(f"批量加载完成，共处理{len(file_paths)}个路径，获得{len(all_documents)}个文档")
        return all_documents
    
    def get_supported_formats(self) -> List[str]:
        """
        获取支持的文件格式
        
        Returns:
            List[str]: 支持的文件格式列表
        """
        base_formats = ['.txt']
        
        if self.use_langchain:
            base_formats.extend(['.pdf'])
        
        return base_formats
    
    def validate_file_path(self, file_path: str) -> Dict[str, Any]:
        """
        验证文件路径
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict: 验证结果
        """
        file_info = FileHandler.get_file_info(file_path)
        supported_formats = self.get_supported_formats()
        
        if not file_info['exists']:
            return {
                'valid': False,
                'reason': '文件或目录不存在',
                'file_info': file_info
            }
        
        if file_info['is_file']:
            extension = file_info['extension'].lower()
            if extension not in supported_formats:
                return {
                    'valid': False,
                    'reason': f'不支持的文件格式: {extension}',
                    'supported_formats': supported_formats,
                    'file_info': file_info
                }
        
        return {
            'valid': True,
            'file_info': file_info,
            'supported_formats': supported_formats
        }


def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> None:
    """
    设置全局日志配置
    
    Args:
        level: 日志级别
        log_file: 日志文件路径（可选）
        format_string: 日志格式字符串（可选）
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def create_utils_instance(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建工具类实例的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        Dict: 包含各种工具实例的字典
    """
    instances = {}
    
    # 创建数据库管理器
    if 'database' in config:
        db_config = config['database']
        instances['db_manager'] = DatabaseManager(
            db_url=db_config.get('url'),
            db_token=db_config.get('token'),
            use_async=db_config.get('use_async', False),
            timeout=db_config.get('timeout', 30),
            max_retries=db_config.get('max_retries', 3)
        )
    
    # 创建文档加载器
    if 'document_loader' in config:
        doc_config = config['document_loader']
        instances['doc_loader'] = DocumentLoader(
            use_langchain=doc_config.get('use_langchain', True)
        )
    
    # 文件处理器（静态类，直接引用）
    instances['file_handler'] = FileHandler
    
    return instances

