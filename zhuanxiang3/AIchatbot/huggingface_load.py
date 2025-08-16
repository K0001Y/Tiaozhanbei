from langchain_community.vectorstores import FAISS
from config import DEFAULT_VECTOR_STORE_PATH
import os

# 替换为你的嵌入模型
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

vector_store_path = os.path.join(DEFAULT_VECTOR_STORE_PATH, "vector_store")
index_file = os.path.join(vector_store_path, "index.faiss")

print("Checking index file:", index_file)
print("Exists:", os.path.exists(index_file))

try:
    vector_store = FAISS.load_local(
        vector_store_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("Vector store loaded successfully!")
except Exception as e:
    print("Failed to load vector store:", str(e))