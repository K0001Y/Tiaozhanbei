"""
测试脚本: 测试RAG检索节点功能
"""
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, TypedDict, Optional, Tuple

# 添加项目根目录到sys.path以确保导入正常工作
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_PATH = "RAG_system/model"
DEFAULT_VECTOR_STORE_PATH = "RAG_system/RAG"

# 从用户代码中导入RAGRetrieveNode和State
from config import DEFAULT_VECTOR_STORE_PATH
# 假设RAGRetrieveNode位于此模块中，如果不是，请相应调整
from nodes.c_rag_retrieval_node import RAGRetrieveNode, State

# 确保logs文件夹存在
log_dir = "V2/tests/logs"
os.makedirs(log_dir, exist_ok=True)

# 配置日志
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"rag_test_{current_time}.log")



# 测试是否可写
try:
    with open(log_file, 'w') as f:
        f.write("Log file initialized\n")
    print(f"成功创建并写入日志文件: {log_file}")
except Exception as e:
    print(f"无法写入日志文件 {log_file}: {str(e)}")
    # 使用备用位置
    log_file = f"./rag_test_{current_time}.log"
    print(f"尝试使用备用位置: {log_file}")

# 设置日志格式和级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

logger = logging.getLogger(__name__)

def create_test_state(query: str) -> State:
    """
    创建用于测试的模拟状态
    
    :param query: 用户查询
    :return: 初始化的状态对象
    """
    return {
        'user_input': query,
        'query': None,
        'messages': [],
        'memory': None,
        'documents': None,
        'response': None,
        'error': None,
        'config': {'retriever_k': 3},  # 设置检索返回的文档数
        'safety_check': None,
        'intent': None,
        'intent_details': None,
        'relevant_context': None,
        'symptoms_list': None,
        'missing_info_list': None,
        'conversation_state': None,
        'diagnosis_data': None
    }

def test_rag_node():
    """测试RAG检索节点功能"""
    logger.info("=" * 50)
    logger.info("开始测试RAG检索节点")
    logger.info("配置信息:")
    logger.info(f"- DEFAULT_CHUNK_SIZE: {DEFAULT_CHUNK_SIZE}")
    logger.info(f"- DEFAULT_CHUNK_OVERLAP: {DEFAULT_CHUNK_OVERLAP}")
    logger.info(f"- DEFAULT_EMBEDDING_MODEL: {DEFAULT_EMBEDDING_MODEL}")
    logger.info(f"- DEFAULT_EMBEDDING_PATH: {DEFAULT_EMBEDDING_PATH}")
    logger.info(f"- DEFAULT_VECTOR_STORE_PATH: {DEFAULT_VECTOR_STORE_PATH}")
    
    try:
        # 创建测试查询
        test_queries = [
            "中医如何治疗感冒？",
            "什么是阴虚火旺？",
            "针灸有哪些常用穴位？",
            ""  # 测试空查询
        ]
        
        # 初始化RAG检索节点
        logger.info("初始化RAG检索节点...")
        rag_node = RAGRetrieveNode()
        logger.info("RAG检索节点初始化成功")
        
        # 测试每个查询
        for i, query in enumerate(test_queries):
            logger.info("-" * 40)
            logger.info(f"测试查询 {i+1}: '{query}'")
            
            # 创建模拟状态
            state = create_test_state(query)
            
            # 调用节点
            logger.info("调用RAG检索节点...")
            try:
                updated_state, next_route = rag_node(state)
                
                # 记录结果
                logger.info(f"检索完成，下一路由: {next_route}")
                
                # 检查是否有错误
                if updated_state.get('error'):
                    logger.error(f"检索过程中出现错误: {updated_state['error']}")
                
                # 记录检索到的文档数
                docs = updated_state.get('documents', [])
                logger.info(f"检索到 {len(docs)} 个文档")
                
                # 记录文档信息
                if docs:
                    for j, doc in enumerate(docs):
                        metadata = doc.get('metadata', {})
                        source = metadata.get('source', '未知来源')
                        content_preview = doc.get('content', '')[:100] + '...' if len(doc.get('content', '')) > 100 else doc.get('content', '')
                        logger.info(f"  文档 {j+1}: 来源={source}, 内容预览={content_preview}")
                
                # 记录相关上下文
                context = updated_state.get('relevant_context', '')
                if context:
                    logger.info(f"提取的上下文长度: {len(context)} 字符")
                    logger.info("上下文预览 (前200字符):")
                    logger.info(context[:200] + "..." if len(context) > 200 else context)
                else:
                    logger.info("未提取到上下文")
                
            except Exception as e:
                logger.error(f"调用RAG检索节点时发生异常: {str(e)}", exc_info=True)
        
        logger.info("=" * 50)
        logger.info("RAG检索节点测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中发生异常: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_rag_node()