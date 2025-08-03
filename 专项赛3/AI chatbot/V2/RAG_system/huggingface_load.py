from langchain_huggingface import HuggingFaceEmbeddings
import os

# 设置缓存路径
os.environ["HF_HOME"] = "E:/Files/huggingface/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "E:/Files/huggingface/huggingface_cache/hub"

# 确保目录存在
os.makedirs("E:/huggingface_cache/hub", exist_ok=True)

# 加载模型
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")