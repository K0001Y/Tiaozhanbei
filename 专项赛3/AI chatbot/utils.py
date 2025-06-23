"""
工具类模块 - 数据库管理器和辅助函数
包含数据库操作、文件处理、日志配置等工具函数
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


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Union[str, Path] = "config.json"):
        self.config_file = Path(config_file)
        self._config = {}
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as file:
                    self._config = json.load(file)
            else:
                self._config = {}
            return self._config
        except Exception as e:
            logging.error(f"加载配置失败: {str(e)}")
            self._config = {}
            return self._config
    
    def save_config(self) -> bool:
        """保存配置文件"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as file:
                json.dump(self._config, file, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"保存配置失败: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def delete(self, key: str) -> bool:
        """删除配置项"""
        keys = key.split('.')
        config = self._config
        
        try:
            for k in keys[:-1]:
                config = config[k]
            del config[keys[-1]]
            return True
        except KeyError:
            return False


class LogManager:
    """日志管理器"""
    
    @staticmethod
    def setup_logger(name: str, 
                    level: int = logging.INFO,
                    log_file: Optional[str] = None,
                    format_string: Optional[str] = None) -> logging.Logger:
        """
        设置日志记录器
        
        Args:
            name: 记录器名称
            level: 日志级别
            log_file: 日志文件路径（可选）
            format_string: 格式字符串（可选）
            
        Returns:
            logging.Logger: 配置好的日志记录器
        """
        logger = logging.getLogger(name)
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 设置日志级别
        logger.setLevel(level)
        
        # 设置格式
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(format_string)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器（如果指定）
        if log_file:
            try:
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"无法创建文件处理器: {str(e)}")
        
        return logger


class ErrorHandler:
    """错误处理工具"""
    
    @staticmethod
    def format_error(error: Exception, include_traceback: bool = True) -> str:
        """
        格式化错误信息
        
        Args:
            error: 异常对象
            include_traceback: 是否包含堆栈跟踪
            
        Returns:
            str: 格式化的错误信息
        """
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        if include_traceback:
            error_info['traceback'] = traceback.format_exc()
        
        return json.dumps(error_info, indent=2, ensure_ascii=False)
    
    @staticmethod
    def log_error(logger: logging.Logger, 
                  error: Exception, 
                  context: Optional[str] = None) -> None:
        """
        记录错误日志
        
        Args:
            logger: 日志记录器
            error: 异常对象
            context: 上下文信息（可选）
        """
        error_msg = f"{type(error).__name__}: {str(error)}"
        
        if context:
            error_msg = f"{context} - {error_msg}"
        
        logger.error(error_msg, exc_info=True)


def validate_url(url: str) -> bool:
    """
    验证URL格式
    
    Args:
        url: 待验证的URL
        
    Returns:
        bool: URL是否有效
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节大小
        
    Returns:
        str: 格式化后的文件大小
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除非法字符
    
    Args:
        filename: 原始文件名
        
    Returns:
        str: 清理后的文件名
    """
    # 移除或替换非法字符
    illegal_chars = '<>:"/\\|?*'
    for char in illegal_chars:
        filename = filename.replace(char, '_')
    
    # 移除前后空格和点
    filename = filename.strip(' .')
    
    # 确保不为空
    if not filename:
        filename = "unnamed"
    
    return filename


def get_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    获取格式化的时间戳
    
    Args:
        format_string: 时间格式字符串
        
    Returns:
        str: 格式化的时间戳
    """
    return datetime.now().strftime(format_string)


def ensure_directory(directory: Union[str, Path]) -> bool:
    """
    确保目录存在，不存在则创建
    
    Args:
        directory: 目录路径
        
    Returns:
        bool: 操作是否成功
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"创建目录失败: {str(e)}")
        return False