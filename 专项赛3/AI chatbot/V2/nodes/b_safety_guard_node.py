"""
安全检查节点
对用户输入进行安全协议检查，特别关注紧急医疗关键词
"""
import logging
import re
from typing import Dict, List, Any, TypedDict, Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 状态类型定义
class State(TypedDict):
    """LangGraph状态类型"""
    user_input: str  # 用户输入
    query: Optional[str]  # 查询（处理后的用户输入）
    messages: List[Any]  # 消息历史
    memory: Optional[Any]  # 对话内存
    documents: Optional[List[Dict[str, Any]]]  # RAG检索结果
    response: Optional[str]  # 最终响应
    error: Optional[str]  # 错误信息
    config: Dict[str, Any]  # 配置信息
    safety_check: Optional[Dict[str, Any]]  # 安全检查结果
    intent: Optional[str]  # 用户意图
    intent_details: Optional[Dict[str, Any]]  # 意图详细信息
    relevant_context: Optional[str]  # RAG检索相关上下文
    symptoms_list: Optional[List[Dict[str, Any]]]  # 提取的症状列表
    missing_info_list: Optional[List[str]]  # 缺失的信息列表
    conversation_state: Optional[str]  # 对话状态标记
    diagnosis_data: Optional[Dict[str, Any]]  # 辨证分析结果

class SafetyGuardNode:
    """安全检查节点类"""
    
    def __init__(self):
        """初始化安全检查节点"""
        try:
            logger.info("初始化安全检查节点")
            
            # 定义紧急关键词列表
            self.emergency_keywords = [
                "胸痛", "急救", "心脏病", "心梗", "中风", "窒息", "自杀", "自残",
                "严重出血", "无法呼吸", "昏迷", "癫痫发作", "过敏性休克", "药物过量"
            ]
            
            logger.info("安全检查节点初始化完成")
            
        except Exception as e:
            logger.error(f"安全检查节点初始化失败: {str(e)}")
            raise Exception(f"安全检查节点初始化失败: {str(e)}")
    
    def _detect_emergency_keywords(self, user_input: str) -> List[str]:
        """
        检测紧急关键词
        
        :param user_input: 用户输入
        :return: 检测到的关键词列表
        """
        detected_keywords = []
        for keyword in self.emergency_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', user_input) or keyword in user_input:
                detected_keywords.append(keyword)
        
        return detected_keywords
    
    def _create_safety_check_result(self, detected_keywords: List[str]) -> Dict[str, Any]:
        """
        创建安全检查结果
        
        :param detected_keywords: 检测到的关键词列表
        :return: 安全检查结果字典
        """
        is_emergency = len(detected_keywords) > 0
        risk_level = "high" if is_emergency else "low"
        
        return {
            "is_emergency": is_emergency,
            "detected_keywords": detected_keywords,
            "risk_level": risk_level
        }
    
    def _create_emergency_response(self, detected_keywords: List[str]) -> str:
        """
        创建紧急情况响应消息
        
        :param detected_keywords: 检测到的关键词列表
        :return: 紧急响应消息
        """
        return (
            "⚠️ 紧急提示：如果您正在经历医疗紧急情况，请立即拨打急救电话(120)或前往最近的急诊室。"
            "本系统不能提供紧急医疗帮助或建议。\n\n"
            f"检测到可能的紧急情况关键词：{', '.join(detected_keywords)}"
        )
    
    def __call__(self, state: State) -> State:
        """
        执行安全检查操作
        
        :param state: LangGraph状态
        :return: 更新后的状态和路由标识
        """
        try:
            logger.info("安全检查节点启动")
            
            # 获取用户输入
            user_input = state.get("user_input", "")
            
            # 检测紧急关键词
            try:
                detected_keywords = self._detect_emergency_keywords(user_input)
                
                # 创建安全检查结果
                safety_check = self._create_safety_check_result(detected_keywords)
                
                # 更新状态
                state['safety_check'] = safety_check
                
                # 如果检测到紧急关键词
                if detected_keywords:
                    logger.warning(f"检测到紧急关键词: {detected_keywords}")
                    
                    # 创建紧急响应消息
                    emergency_response = self._create_emergency_response(detected_keywords)
                    state['response'] = emergency_response
                    
                    # 路由到紧急处理
                    return state
                
                # 没有检测到紧急关键词，继续流程
                logger.info("安全检查通过，未检测到紧急关键词")
                
            except Exception as e:
                logger.error(f"安全检查处理失败: {str(e)}")
                state['error'] = f"安全检查处理失败: {str(e)}"
                state['response'] = "处理您的请求时发生系统错误，请重试。"
                return state, "end"
            
            # 设置下一步路由
            return state
            
        except Exception as e:
            error_msg = f"安全检查节点执行失败: {str(e)}"
            logger.error(error_msg)
            
            # 更新状态
            state['error'] = error_msg
            state['response'] = "处理您的请求时发生系统错误，请重试。"
            
            # 出错时路由到结束
            return state, "end"

# 导出节点实例以便在图中使用
safety_guard_node = SafetyGuardNode()