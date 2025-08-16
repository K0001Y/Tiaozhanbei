"""
嵌入模型管理组件
负责加载和管理嵌入模型
"""
import os
import json
import logging
import torch
from typing import Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModel, AutoTokenizer
from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_PATH

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModelManager:
    """嵌入模型管理类"""
    
    def __init__(self, embedding_model_path: str = DEFAULT_EMBEDDING_PATH):
        """
        初始化嵌入模型管理器
        :param embedding_model_path: 嵌入模型路径
        """
        try:
            self.embedding_model_path = embedding_model_path
            self.embedding_model = None
            self.device = self._get_device()
            
            # 确保必要的目录存在
            os.makedirs(DEFAULT_EMBEDDING_PATH, exist_ok=True)
            
            logger.info(f"嵌入模型管理器初始化成功，使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"嵌入模型管理器初始化失败: {str(e)}")
            raise Exception(f"嵌入模型管理器初始化失败: {str(e)}")

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
                'model.safetensors',  # 或者 model.safetensors
                'tokenizer_config.json',
                'tokenizer.json'  # 或者其他 tokenizer 文件
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