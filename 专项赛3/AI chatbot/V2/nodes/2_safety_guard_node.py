import logging
import re
from typing import Dict, List, Any, TypedDict, Optional, Union

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 使用与input_node相同的State类型
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

# 定义紧急关键词列表
EMERGENCY_KEYWORDS = [
    "胸痛", "急救", "心脏病", "心梗", "中风", "窒息", "自杀", "自残",
    "严重出血", "无法呼吸", "昏迷", "癫痫发作", "过敏性休克", "药物过量"
]

def safety_guard_node(state: State) -> tuple[State, str]:
    """
    安全检查节点：对用户输入进行安全协议检查，特别关注紧急医疗关键词
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态和路由指示
    """
    try:
        # 获取用户输入
        user_input = state.get("user_input", "")
        
        # 初始化安全检查结果
        safety_check = {
            "is_emergency": False,
            "detected_keywords": [],
            "risk_level": "low"
        }
        
        # 检查紧急关键词
        detected_keywords = []
        for keyword in EMERGENCY_KEYWORDS:
            if re.search(rf'\b{re.escape(keyword)}\b', user_input) or keyword in user_input:
                detected_keywords.append(keyword)
        
        # 如果检测到紧急关键词
        if detected_keywords:
            safety_check["is_emergency"] = True
            safety_check["detected_keywords"] = detected_keywords
            safety_check["risk_level"] = "high"
            
            # 创建紧急提醒消息
            emergency_response = (
                "⚠️ 紧急提示：如果您正在经历医疗紧急情况，请立即拨打急救电话(120)或前往最近的急诊室。"
                "本系统不能提供紧急医疗帮助或建议。\n\n"
                f"检测到可能的紧急情况关键词：{', '.join(detected_keywords)}"
            )
            
            logger.warning(f"检测到紧急关键词: {detected_keywords}")
            
            # 更新状态并路由到结束
            return {
                **state,
                "safety_check": safety_check,
                "response": emergency_response
            }, "emergency"
        
        # 如果没有检测到紧急关键词，继续到下一个节点
        logger.info("安全检查通过，未检测到紧急关键词")
        return {
            **state,
            "safety_check": safety_check
        }, "continue"
    
    except Exception as e:
        error_msg = f"安全检查过程中出错: {str(e)}"
        logger.error(error_msg)
        return {
            **state,
            "error": error_msg,
            "response": "处理您的请求时发生系统错误，请重试。"
        }, "end"