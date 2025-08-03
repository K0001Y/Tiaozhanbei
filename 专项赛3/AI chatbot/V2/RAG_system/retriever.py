"""
检索功能组件
负责创建检索器和执行检索操作
"""
import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrieverManager:
    """检索管理类"""
    
    def __init__(self):
        """初始化检索管理器"""
        try:
            self.retriever = None
            logger.info("检索管理器初始化成功")
            
        except Exception as e:
            logger.error(f"检索管理器初始化失败: {str(e)}")
            raise Exception(f"检索管理器初始化失败: {str(e)}")
    
    def create_retriever(self, vector_store: FAISS):
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
    
    def retrieve(self, query: str, k: int = 4) -> List[dict]:
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
                raise Exception("检索器未初始化，请先创建检索器")
            
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