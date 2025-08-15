"""
RAG搜索API模块
独立的API逻辑封装，专门处理疾病搜索功能
"""
import logging
import re
import traceback
from typing import Dict, List, Any, Optional
from flask import request, jsonify

# 导入RAG检索节点
try:
    from nodes.c_rag_retrieval_node import RAGRetrieveNode, State
except ImportError:
    # 如果导入失败，提供一个简化版本用于测试
    print("警告: 无法导入RAG检索节点，使用模拟数据")
    class RAGRetrieveNode:
        def __init__(self):
            pass
        def __call__(self, state):
            return {
                'relevant_context': f"模拟检索结果: {state.get('user_input', '')}",
                'documents': [
                    {
                        'content': '高血压是一种常见的心血管疾病，主要表现为动脉血压持续升高...',
                        'metadata': {'source': '内科学(第八版)'}
                    }
                ]
            }

logger = logging.getLogger(__name__)

class RAGSearchAPI:
    """RAG搜索API类 - 专门处理疾病搜索相关的API逻辑"""
    
    def __init__(self):
        """初始化RAG搜索API"""
        try:
            logger.info("初始化RAG搜索API模块")
            self.rag_node = RAGRetrieveNode()
            self.api_name = "疾病搜索API"
            self.version = "1.0.0"
            logger.info("RAG搜索API模块初始化完成")
        except Exception as e:
            logger.error(f"RAG搜索API模块初始化失败: {str(e)}")
            raise
    
    def _validate_search_params(self, search_query: str) -> tuple[bool, str]:
        """
        验证搜索参数
        
        :param search_query: 搜索关键词
        :return: (是否有效, 错误信息)
        """
        if not search_query:
            return False, "搜索关键词不能为空"
        
        if len(search_query.strip()) == 0:
            return False, "搜索关键词不能为空"
        
        if len(search_query) > 100:
            return False, "搜索关键词过长，请限制在100字符以内"
        
        # 检查是否包含特殊字符（可根据需要调整）
        if re.search(r'[<>"\']', search_query):
            return False, "搜索关键词包含非法字符"
        
        return True, ""
    
    def _parse_disease_info(self, content: str, source: str = "未知来源") -> Dict[str, Any]:
        """
        从检索内容中解析疾病信息
        
        :param content: 检索到的文档内容
        :param source: 文档来源
        :return: 格式化的疾病信息
        """
        try:
            # 提取疾病名称（优化正则表达式）
            disease_patterns = [
                r'([^。，,.\n]{2,20}(?:病|症|综合征|疾病))',  # 以病、症结尾
                r'([^。，,.\n]*(?:高血压|糖尿病|冠心病|心脏病|肝病|肾病|癌症|肿瘤)[^。，,.\n]*)',  # 常见疾病
                r'^([^。，,.\n]{2,20})',  # 取开头的内容作为疾病名
            ]
            
            disease_name = "相关疾病"
            for pattern in disease_patterns:
                match = re.search(pattern, content)
                if match:
                    candidate_name = match.group(1).strip()
                    # 清理前缀编号和标点
                    candidate_name = re.sub(r'^[【\[\(]?[0-9]+[】\]\)]?[、.,，。]?\s*', '', candidate_name)
                    if len(candidate_name) >= 2:
                        disease_name = candidate_name[:20]  # 限制长度
                        break
            
            # 生成疾病描述
            description = content.strip()
            # 移除多余的换行和空格
            description = re.sub(r'\s+', ' ', description)
            # 截取适当长度
            if len(description) > 200:
                # 尝试在句号处截断
                sentences = description.split('。')
                truncated = ""
                for sentence in sentences:
                    if len(truncated + sentence + "。") <= 200:
                        truncated += sentence + "。"
                    else:
                        break
                description = truncated if truncated else description[:200] + "..."
            
            # 计算相关度（基于内容长度和关键词匹配）
            base_relevance = min(95, max(70, len(description) // 3 + 60))
            # 如果包含常见医学术语，提高相关度
            medical_terms = ['症状', '治疗', '诊断', '病因', '预防', '药物', '手术']
            term_bonus = sum(5 for term in medical_terms if term in content)
            relevance = min(98, base_relevance + term_bonus)
            
            return {
                "diseaseName": disease_name,
                "description": description,
                "source": source,
                "relevance": f"{relevance}%"
            }
            
        except Exception as e:
            logger.warning(f"解析疾病信息失败: {str(e)}")
            return {
                "diseaseName": "相关疾病信息",
                "description": content[:200] + "..." if len(content) > 200 else content,
                "source": source,
                "relevance": "80%"
            }
    
    def _format_search_response(self, documents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        格式化搜索响应为标准API格式
        
        :param documents: RAG检索结果
        :param query: 搜索查询
        :return: 格式化的响应数据
        """
        diseases = []
        
        if documents:
            for idx, doc in enumerate(documents):
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                source = metadata.get('source', '医学知识库')
                
                # 跳过空内容
                if not content.strip():
                    continue
                
                # 解析疾病信息
                disease_info = self._parse_disease_info(content, source)
                disease_info['diseaseId'] = idx + 1
                
                diseases.append(disease_info)
        
        # 如果没有有效结果，返回默认信息
        if not diseases:
            diseases.append({
                "diseaseId": 1,
                "diseaseName": f"关于'{query}'的相关信息",
                "description": "未找到相关的疾病信息，建议咨询专业医生进行详细诊断。如需更准确的信息，请尝试使用更具体的医学术语进行搜索。",
                "source": "系统提示",
                "relevance": "0%"
            })
        
        return {
            "success": True,
            "message": "检索成功",
            "data": {
                "diseases": diseases
            }
        }
    
    def _execute_rag_search(self, query: str) -> Dict[str, Any]:
        """
        执行RAG搜索
        
        :param query: 搜索关键词
        :return: 搜索结果状态
        """
        try:
            # 构建RAG检索状态
            state = {
                'user_input': query,
                'query': query,
                'messages': [],
                'memory': None,
                'documents': None,
                'response': None,
                'error': None,
                'config': {'retriever_k': 5},  # 获取5个结果
                'safety_check': None,
                'intent': None,
                'intent_details': None,
                'relevant_context': None,
                'symptoms_list': None,
                'missing_info_list': None,
                'conversation_state': None,
                'diagnosis_data': None
            }
            
            # 执行RAG检索
            result_state = self.rag_node(state)
            return result_state
            
        except Exception as e:
            logger.error(f"RAG搜索执行失败: {str(e)}")
            return {
                'error': f"搜索执行失败: {str(e)}",
                'documents': []
            }
    
    def search_diseases(self, search_query: str) -> tuple[int, Dict[str, Any]]:
        """
        搜索疾病信息的核心方法
        
        :param search_query: 搜索关键词
        :return: (HTTP状态码, 响应数据)
        """
        try:
            logger.info(f"执行疾病搜索: {search_query}")
            
            # 验证输入参数
            is_valid, error_msg = self._validate_search_params(search_query)
            if not is_valid:
                return 400, {
                    "success": False,
                    "message": error_msg,
                    "data": {"diseases": []}
                }
            
            # 执行RAG搜索
            result_state = self._execute_rag_search(search_query.strip())
            
            # 检查是否有错误
            if result_state.get('error'):
                logger.error(f"RAG检索出错: {result_state['error']}")
                return 500, {
                    "success": False,
                    "message": f"检索失败: {result_state['error']}",
                    "data": {"diseases": []}
                }
            
            # 获取检索结果
            documents = result_state.get('documents', [])
            
            # 格式化响应
            response_data = self._format_search_response(documents, search_query)
            
            logger.info(f"搜索完成，返回{len(response_data['data']['diseases'])}个结果")
            return 200, response_data
            
        except Exception as e:
            error_msg = f"搜索执行异常: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return 500, {
                "success": False,
                "message": error_msg,
                "data": {"diseases": []}
            }
    
    def handle_search_request(self):
        """
        处理Flask请求的入口方法
        
        :return: Flask响应对象
        """
        try:
            # 获取搜索参数
            search_query = request.args.get('search', '').strip()
            
            # 记录请求信息
            logger.info(f"收到搜索请求: {request.remote_addr} -> {search_query}")
            
            # 执行搜索
            status_code, result_data = self.search_diseases(search_query)
            
            # 返回响应
            return jsonify(result_data), status_code
            
        except Exception as e:
            error_msg = f"请求处理异常: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return jsonify({
                "success": False,
                "message": error_msg,
                "data": {"diseases": []}
            }), 500
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        获取API信息
        
        :return: API信息
        """
        return {
            "name": self.api_name,
            "version": self.version,
            "description": "基于RAG技术的疾病信息搜索API",
            "endpoints": {
                "search": {
                    "method": "GET",
                    "path": "/api/search",
                    "params": {
                        "search": "搜索关键词（必填）"
                    },
                    "description": "根据关键词搜索相关疾病信息"
                }
            },
            "status": "active"
        }