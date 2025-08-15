"""
配置文件 - 包含默认配置和参数设置
"""

# API配置
DEFAULT_MODEL_NAME = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
ALI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY = "sk-ded931914889432ca04c7fc7ac922c3b"
ALI_API_KEY = "	sk-7edcf5b1583945a58545c37877e0f2f3"

# 模型配置
DEFAULT_MODEL_CONFIGS = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "presence_penalty": 0.1,
    
}

# RAG配置
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_PATH = "RAG_system/model"
DEFAULT_VECTOR_STORE_PATH = "RAG_system/RAG/"

# 支持的编码列表
SUPPORTED_ENCODINGS = ["utf-8", "gb2312", "gbk", "gb18030", "utf-16", "big5", "latin-1"]



