import logging
import re
from typing import Dict, List, Any, TypedDict, Optional, Union, Tuple, Set
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from config import DEFAULT_API_KEY,DEFAULT_BASE_URL
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

# =============== LLM配置 ===============

# 模型选择 - 直接在代码中配置
# 可选项: "openai", "azure", "local"
LLM_PROVIDER = "openai"

# OpenAI配置
OPENAI_CONFIG = {
    "model": "deepseek-chat",    # 生成追问可以使用较轻量级模型
    "temperature": 0.3,          # 适当的创造性
    "api_key": DEFAULT_API_KEY, # 你的API密钥
    "api_base": DEFAULT_BASE_URL, # API基础URL
    "timeout": 30,               # 请求超时时间
    "max_tokens": 500            # 足够生成追问的标记数
}

# Azure OpenAI配置
AZURE_CONFIG = {
    "deployment_name": "gpt-35-turbo", # Azure部署名称
    "api_version": "2023-05-15",       # API版本
    "api_key": "your-azure-key",       # Azure API密钥
    "api_base": "https://your-resource.openai.azure.com/" # Azure端点
}

# 本地模型配置
LOCAL_MODEL_CONFIG = {
    "model_path": "/path/to/chat-model.gguf", # 本地模型路径
    "n_ctx": 2048,               # 上下文窗口大小
    "n_threads": 4               # 线程数
}

# =============== 节点实现 ===============

class FollowUpQuestionNode:
    """追问节点类"""
    
    def __init__(self):
        """初始化追问节点"""
        # 初始化LLM
        self.llm = self._create_llm()
        
        # 创建追问提示模板
        self.question_prompt = self._create_question_prompt()
        
        # 创建LLM链
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.question_prompt,
            verbose=True
        )
        
        logger.info(f"追问节点初始化完成，使用LLM提供商: {LLM_PROVIDER}")
    
    def _create_llm(self):
        """根据配置创建LLM实例"""
        if LLM_PROVIDER == "openai":
            return ChatOpenAI(
                model=OPENAI_CONFIG["model"],
                temperature=OPENAI_CONFIG["temperature"],
                openai_api_key=OPENAI_CONFIG["api_key"],
                openai_api_base=OPENAI_CONFIG["api_base"],
                request_timeout=OPENAI_CONFIG["timeout"],
                max_tokens=OPENAI_CONFIG["max_tokens"]
            )
        
        elif LLM_PROVIDER == "azure":
            return AzureChatOpenAI(
                deployment_name=AZURE_CONFIG["deployment_name"],
                openai_api_version=AZURE_CONFIG["api_version"],
                openai_api_key=AZURE_CONFIG["api_key"],
                openai_api_base=AZURE_CONFIG["api_base"],
                temperature=OPENAI_CONFIG["temperature"],
                max_tokens=OPENAI_CONFIG["max_tokens"]
            )
        
        elif LLM_PROVIDER == "local":
            return LlamaCpp(
                model_path=LOCAL_MODEL_CONFIG["model_path"],
                temperature=OPENAI_CONFIG["temperature"],
                max_tokens=OPENAI_CONFIG["max_tokens"],
                n_ctx=LOCAL_MODEL_CONFIG["n_ctx"],
                n_threads=LOCAL_MODEL_CONFIG["n_threads"]
            )
        
        else:
            raise ValueError(f"不支持的LLM提供商: {LLM_PROVIDER}")
    
    def _create_question_prompt(self):
        """创建追问提示模板"""
        template = """
        你是一位专业的医疗对话助手。根据用户已提供的信息和当前缺失的信息，生成一个专业、礼貌的医疗追问。

        # 用户已知信息
        {known_symptoms}
        
        # 需要获取的信息
        {missing_info}
        
        # 之前的追问历史（避免重复这些问题）
        {previous_questions}
        
        请生成一个专业、自然且富有同理心的追问，向用户询问缺失的信息。确保:
        1. 语言专业医学准确
        2. 语气温和有礼，表达同理心
        3. 避免重复已经问过的问题
        4. 优先询问最重要的缺失信息
        5. 如果合适，可以解释为什么这些信息对诊断很重要
        6. 使用自然的对话风格，避免生硬的列表形式
        
        生成的追问:
        """
        
        return ChatPromptTemplate.from_template(template)
    
    def _format_known_symptoms(self, symptoms_list):
        """格式化已知的症状信息"""
        if not symptoms_list:
            return "用户尚未提供任何明确的症状信息。"
        
        formatted = ["用户已提供的症状信息:"]
        for i, symptom in enumerate(symptoms_list, 1):
            name = symptom.get("name", "未命名症状")
            description = symptom.get("description", "无详细描述")
            severity = symptom.get("severity", "未提及")
            duration = symptom.get("duration", "未提及")
            
            formatted.append(f"{i}. {name}: {description}")
            if severity != "未提及":
                formatted.append(f"   - 严重程度: {severity}")
            if duration != "未提及":
                formatted.append(f"   - 持续时间: {duration}")
        
        return "\n".join(formatted)
    
    def _extract_previous_questions(self, messages):
        """从历史消息中提取之前的追问"""
        previous_questions = []
        
        for msg in messages:
            # 只检查系统或助手消息
            if hasattr(msg, 'type') and msg.type in ['system', 'ai', 'assistant']:
                content = msg.content if hasattr(msg, 'content') else ""
                
                # 使用正则表达式找出问句
                questions = re.findall(r'(?:^|\n)(?:[^\n.?!]*?\?)', content)
                previous_questions.extend(questions)
        
        return previous_questions
    
    def _filter_duplicate_missing_info(self, missing_info, previous_questions):
        """过滤可能已经问过的缺失信息"""
        if not missing_info or not previous_questions:
            return missing_info
        
        filtered_info = []
        # 将之前的问题转换为小写集合，用于快速查找
        prev_questions_lower = set(q.lower().strip() for q in previous_questions)
        
        for info in missing_info:
            info_lower = info.lower()
            
            # 检查这个信息是否已经在之前的问题中被问到
            already_asked = False
            for question in prev_questions_lower:
                # 检查关键词匹配
                keywords = info_lower.split()
                significant_keywords = [k for k in keywords if len(k) > 3]  # 只考虑有意义的关键词
                
                if any(keyword in question for keyword in significant_keywords):
                    already_asked = True
                    break
            
            if not already_asked:
                filtered_info.append(info)
        
        return filtered_info
    
    def __call__(self, state: State) -> State:
        """节点主函数"""
        try:
            # 获取输入
            missing_info_list = state.get("missing_info_list", [])
            symptoms_list = state.get("symptoms_list", [])
            messages = state.get("messages", [])
            
            # 如果没有缺失信息，使用默认问题
            if not missing_info_list:
                missing_info_list = [
                    "症状的具体位置",
                    "症状的持续时间",
                    "是否有任何缓解或加重因素",
                    "相关病史"
                ]
            
            # 从历史消息中提取之前的追问
            previous_questions = self._extract_previous_questions(messages)
            
            # 过滤可能重复的缺失信息
            filtered_missing_info = self._filter_duplicate_missing_info(missing_info_list, previous_questions)
            
            # 如果过滤后没有新的缺失信息，但有症状，直接路由到诊断
            if not filtered_missing_info and symptoms_list:
                logger.info("没有新的缺失信息需要追问，直接路由到诊断")
                return {
                    **state,
                    "missing_info_list": [],
                    "conversation_state": "ready_for_diagnosis"
                }
            
            # 如果过滤后没有新的缺失信息，也没有症状，使用一个通用问题
            if not filtered_missing_info and not symptoms_list:
                filtered_missing_info = ["请描述您目前感到的不适或症状"]
            
            # 格式化已知症状和之前的追问
            known_symptoms = self._format_known_symptoms(symptoms_list)
            previous_questions_text = "\n".join(previous_questions) if previous_questions else "无之前的追问"
            
            # 准备链的输入
            chain_input = {
                "known_symptoms": known_symptoms,
                "missing_info": "\n".join([f"- {info}" for info in filtered_missing_info]),
                "previous_questions": previous_questions_text
            }
            
            # 运行链生成追问
            logger.info(f"生成追问，共有 {len(filtered_missing_info)} 个缺失信息点")
            result = self.chain.invoke(chain_input)
            follow_up_question = result.get('text', '').strip()
            
            # 如果生成的追问为空，使用默认文本
            if not follow_up_question:
                items = "\n".join([f"{i+1}. {info}" for i, info in enumerate(filtered_missing_info)])
                follow_up_question = f"为了更准确地了解您的情况，请补充以下信息：\n\n{items}"
            
            logger.info(f"生成的追问: {follow_up_question[:100]}...")
            
            # 更新状态
            updated_state = {
                **state,
                "response": follow_up_question,
                "conversation_state": "awaiting_follow_up"
            }
            
            return updated_state
        
        except Exception as e:
            error_msg = f"生成追问过程中出错: {str(e)}"
            logger.error(error_msg)
            
            # 发生错误时使用简单的默认追问
            default_question = "为了更好地了解您的情况，请告诉我更多关于您症状的详细信息，比如持续时间、严重程度和是否有任何诱因？"
            
            return {
                **state,
                "error": error_msg,
                "response": default_question,
                "conversation_state": "awaiting_follow_up"
            }

# 导出节点实例以便在图中使用
follow_up_question_node = FollowUpQuestionNode()