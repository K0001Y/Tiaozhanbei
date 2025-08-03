import logging
import json
from typing import Dict, List, Any, TypedDict, Optional, Union, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain

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

# =============== LLM配置 ===============

# 模型选择 - 直接在代码中配置
# 可选项: "openai", "azure", "local"
LLM_PROVIDER = "openai"

# OpenAI配置
OPENAI_CONFIG = {
    "model": "gpt-4",            # 使用更强大的模型进行中医辨证
    "temperature": 0.1,          # 低温度确保稳定的医疗诊断
    "api_key": "sk-your-api-key", # 你的API密钥
    "api_base": "https://api.openai.com/v1", # API基础URL
    "timeout": 90,               # 更长的超时时间用于复杂分析
    "max_tokens": 3000           # 更大的输出空间
}

# Azure OpenAI配置
AZURE_CONFIG = {
    "deployment_name": "gpt-4",  # Azure部署名称
    "api_version": "2023-05-15", # API版本
    "api_key": "your-azure-key", # Azure API密钥
    "api_base": "https://your-resource.openai.azure.com/" # Azure端点
}

# 本地模型配置
LOCAL_MODEL_CONFIG = {
    "model_path": "/path/to/medical-model.gguf", # 本地医疗专用模型路径
    "n_ctx": 8192,               # 更大的上下文窗口
    "n_threads": 8               # 更多线程以提高性能
}

# =============== 节点实现 ===============

class DiagnosisNode:
    """辨证分析节点类"""
    
    def __init__(self):
        """初始化辨证分析节点"""
        # 初始化LLM
        self.llm = self._create_llm()
        
        # 创建输出解析器
        self.output_parser = self._create_output_parser()
        
        # 创建辨证分析提示
        self.diagnosis_prompt = self._create_diagnosis_prompt()
        
        # 创建LLMChain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.diagnosis_prompt,
            output_parser=self.output_parser,
            verbose=True
        )
        
        logger.info(f"辨证分析节点初始化完成，使用LLM提供商: {LLM_PROVIDER}")
    
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
    
    def _create_output_parser(self):
        """创建结构化输出解析器"""
        response_schemas = [
            ResponseSchema(
                name="pattern_type",
                description="中医辨证分型，如'肝郁气滞'、'肾阳虚'等",
                type="string"
            ),
            ResponseSchema(
                name="pathogenesis",
                description="中医病机分析，解释症状形成的机理",
                type="string"
            ),
            ResponseSchema(
                name="analysis",
                description="详细的辨证分析过程，包括如何从症状推导出证型",
                type="string"
            ),
            ResponseSchema(
                name="confidence",
                description="对该辨证结果的置信度，0到1之间的数值",
                type="number"
            ),
            ResponseSchema(
                name="differential_diagnosis",
                description="鉴别诊断，列出其他可能的证型及其可能性",
                type="array"
            ),
            ResponseSchema(
                name="stage",
                description="当前诊断阶段，应为'diagnosis'",
                type="string"
            )
        ]
        
        return StructuredOutputParser.from_response_schemas(response_schemas)
    
    def _create_diagnosis_prompt(self):
        """创建辨证分析提示模板"""
        template = """
        你是一位经验丰富的中医辨证专家。请基于患者的症状和相关医学上下文，进行中医辨证分析。
        
        # 患者症状
        {symptoms}
        
        # 相关医学参考信息
        {relevant_context}
        
        请根据中医理论进行辨证分析，确定证型、病机并提供详细的分析依据。注意：
        1. 严格遵循中医理论体系进行辨证
        2. 分析应考虑五脏六腑、气血津液、八纲辨证等多维度
        3. 提供完整的分析思路，说明如何从症状推导出证型
        4. 给出鉴别诊断，说明为什么排除其他可能的证型
        5. 确保所有结论都有症状或理论依据支持
        
        {format_instructions}
        """
        
        return ChatPromptTemplate.from_template(template)
    
    def _format_symptoms(self, symptoms_list):
        """格式化症状列表"""
        if not symptoms_list:
            return "患者未提供明确症状信息。"
        
        formatted = []
        for i, symptom in enumerate(symptoms_list, 1):
            name = symptom.get("name", "未命名症状")
            description = symptom.get("description", "无详细描述")
            severity = symptom.get("severity", "未提及")
            duration = symptom.get("duration", "未提及")
            
            symptom_text = f"{i}. {name}: {description}"
            
            details = []
            if severity != "未提及":
                details.append(f"严重程度: {severity}")
            if duration != "未提及":
                details.append(f"持续时间: {duration}")
            
            if details:
                symptom_text += f" ({', '.join(details)})"
            
            formatted.append(symptom_text)
        
        return "\n".join(formatted)
    
    def __call__(self, state: State) -> Tuple[State, str]:
        """节点主函数"""
        try:
            # 获取输入
            symptoms_list = state.get("symptoms_list", [])
            relevant_context = state.get("relevant_context", "无相关上下文")
            
            # 格式化症状
            formatted_symptoms = self._format_symptoms(symptoms_list)
            
            # 准备链的输入
            chain_input = {
                "symptoms": formatted_symptoms,
                "relevant_context": relevant_context,
                "format_instructions": self.output_parser.get_format_instructions()
            }
            
            # 运行链
            logger.info("开始中医辨证分析...")
            result = self.chain.invoke(chain_input)
            
            # 解析结果
            parsed_output = result
            
            # 构建诊断数据
            diagnosis_data = {
                "pattern_type": parsed_output.get("pattern_type", "未能确定证型"),
                "pathogenesis": parsed_output.get("pathogenesis", "未能确定病机"),
                "analysis": parsed_output.get("analysis", ""),
                "confidence": parsed_output.get("confidence", 0.0),
                "differential_diagnosis": parsed_output.get("differential_diagnosis", []),
                "stage": "diagnosis"
            }
            
            logger.info(f"辨证分析完成，证型: {diagnosis_data['pattern_type']}, 置信度: {diagnosis_data['confidence']}")
            
            # 更新状态
            updated_state = {
                **state,
                "diagnosis_data": diagnosis_data,
                "conversation_state": "diagnosis_complete"
            }
            
            return updated_state, "to_prescription"
        
        except Exception as e:
            error_msg = f"辨证分析过程中出错: {str(e)}"
            logger.error(error_msg)
            
            # 发生错误时设置默认诊断数据
            default_diagnosis = {
                "pattern_type": "无法确定",
                "pathogenesis": "分析过程中出现错误",
                "analysis": f"在进行辨证分析时发生技术问题: {str(e)}",
                "confidence": 0.0,
                "differential_diagnosis": [],
                "stage": "diagnosis_error"
            }
            
            return {
                **state,
                "error": error_msg,
                "diagnosis_data": default_diagnosis,
                "conversation_state": "diagnosis_error"
            }, "to_prescription"

# 导出节点实例以便在图中使用
diagnosis_node = DiagnosisNode()