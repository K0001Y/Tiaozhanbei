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
    prescription_data: Optional[Dict[str, Any]]  # 处方推荐数据

# =============== LLM配置 ===============

# 模型选择 - 直接在代码中配置
# 可选项: "openai", "azure", "local"
LLM_PROVIDER = "openai"

# OpenAI配置
OPENAI_CONFIG = {
    "model": "gpt-4",            # 使用强大的模型进行中医处方
    "temperature": 0.2,          # 适当的创造性，但保持专业性
    "api_key": "sk-your-api-key", # 你的API密钥
    "api_base": "https://api.openai.com/v1", # API基础URL
    "timeout": 90,               # 较长的超时时间
    "max_tokens": 4000           # 足够生成详细处方的空间
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
    "n_ctx": 8192,               # 大上下文窗口
    "n_threads": 8               # 更多线程以提高性能
}

# =============== 节点实现 ===============

class PrescriptionNode:
    """处方推荐节点类"""
    
    def __init__(self):
        """初始化处方推荐节点"""
        # 初始化LLM
        self.llm = self._create_llm()
        
        # 创建输出解析器
        self.output_parser = self._create_output_parser()
        
        # 创建处方推荐提示
        self.prescription_prompt = self._create_prescription_prompt()
        
        # 创建响应整合提示
        self.response_prompt = self._create_response_prompt()
        
        # 创建LLM链
        self.prescription_chain = LLMChain(
            llm=self.llm,
            prompt=self.prescription_prompt,
            output_parser=self.output_parser,
            verbose=True
        )
        
        self.response_chain = LLMChain(
            llm=self.llm,
            prompt=self.response_prompt,
            verbose=True
        )
        
        logger.info(f"处方推荐节点初始化完成，使用LLM提供商: {LLM_PROVIDER}")
    
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
                name="formula_name",
                description="推荐的中药方剂名称",
                type="string"
            ),
            ResponseSchema(
                name="composition",
                description="方剂的组成，包括各个药材及其用量",
                type="array"
            ),
            ResponseSchema(
                name="preparation_method",
                description="方剂的制备方法",
                type="string"
            ),
            ResponseSchema(
                name="usage",
                description="服用方法和剂量",
                type="string"
            ),
            ResponseSchema(
                name="treatment_principle",
                description="治疗原则，如何针对证型和病机",
                type="string"
            ),
            ResponseSchema(
                name="contraindications",
                description="禁忌和注意事项",
                type="array"
            ),
            ResponseSchema(
                name="modifications",
                description="可能的加减变化，针对不同症状特点",
                type="array"
            ),
            ResponseSchema(
                name="evidence",
                description="方剂选择的依据，引用经典或现代研究",
                type="string"
            ),
            ResponseSchema(
                name="stage",
                description="当前阶段，应为'prescription'",
                type="string"
            )
        ]
        
        return StructuredOutputParser.from_response_schemas(response_schemas)
    
    def _create_prescription_prompt(self):
        """创建处方推荐提示模板"""
        template = """
        你是一位经验丰富的中医师，精通方剂学。请根据中医辨证结果，推荐合适的中药方剂。
        
        # 辨证分析结果
        证型: {pattern_type}
        病机: {pathogenesis}
        分析: {analysis}
        
        # 相关医学参考信息
        {relevant_context}
        
        请根据上述辨证结果，推荐最适合的中药方剂。确保：
        1. 方剂符合辨证论治原则，针对性强
        2. 详细列出组成药材及用量
        3. 说明制备和服用方法
        4. 提供禁忌和注意事项
        5. 说明可能的加减变化
        6. 引用经典或现代研究支持你的选择
        
        重要提示：请确保药方安全有效，特别注意有毒或有强烈副作用的药材。
        
        {format_instructions}
        """
        
        return ChatPromptTemplate.from_template(template)
    
    def _create_response_prompt(self):
        """创建响应整合提示模板"""
        template = """
        请将以下中医诊断和处方信息整合为一个专业、连贯且易于理解的回复。回复应该既体现中医专业性，又能让普通患者理解。
        
        # 患者症状
        {symptoms}
        
        # 辨证分析
        证型: {pattern_type}
        病机: {pathogenesis}
        分析: {analysis}
        
        # 处方推荐
        方剂: {formula_name}
        组成: {composition}
        用法: {usage}
        治疗原则: {treatment_principle}
        禁忌: {contraindications}
        
        请整合以上信息，创建一个结构清晰的回复，包括：
        1. 对患者症状的简要总结
        2. 中医辨证分析（用通俗语言解释专业术语）
        3. 推荐的方剂及其作用原理
        4. 服用方法和注意事项
        5. 健康生活方式建议
        
        回复应当专业但不晦涩，加入适当的解释，确保患者能够理解。最后添加一个温馨提示，说明这只是辅助建议，不能替代正规医疗诊断和治疗。
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
    
    def _format_composition(self, composition):
        """格式化方剂组成"""
        if isinstance(composition, list):
            return "\n".join([f"- {item}" for item in composition])
        return str(composition)
    
    def _format_contraindications(self, contraindications):
        """格式化禁忌事项"""
        if isinstance(contraindications, list):
            return "\n".join([f"- {item}" for item in contraindications])
        return str(contraindications)
    
    def __call__(self, state: State) -> Tuple[State, str]:
        """节点主函数"""
        try:
            # 获取输入
            diagnosis_data = state.get("diagnosis_data", {})
            symptoms_list = state.get("symptoms_list", [])
            relevant_context = state.get("relevant_context", "无相关上下文")
            
            # 如果诊断数据为空或证型为"无法确定"，则跳过处方生成
            if not diagnosis_data or diagnosis_data.get("pattern_type") == "无法确定":
                error_msg = "诊断数据不完整或未能确定证型，无法生成处方"
                logger.warning(error_msg)
                
                response = (
                    "根据您提供的信息，目前无法确定明确的中医证型。建议您提供更详细的症状描述，"
                    "或咨询专业的中医师进行面诊。\n\n"
                    "请注意：本系统仅提供初步辅助分析，不能替代专业医疗诊断和治疗。"
                )
                
                return {
                    **state,
                    "error": error_msg,
                    "response": response,
                    "conversation_state": "prescription_error"
                }, "to_safety_check"
            
            # 准备处方链的输入
            prescription_input = {
                "pattern_type": diagnosis_data.get("pattern_type", ""),
                "pathogenesis": diagnosis_data.get("pathogenesis", ""),
                "analysis": diagnosis_data.get("analysis", ""),
                "relevant_context": relevant_context,
                "format_instructions": self.output_parser.get_format_instructions()
            }
            
            # 运行处方链
            logger.info("开始生成处方推荐...")
            prescription_result = self.prescription_chain.invoke(prescription_input)
            
            # 解析结果
            prescription_data = prescription_result
            
            # 更新状态中的处方数据
            updated_state = {
                **state,
                "prescription_data": prescription_data,
                "conversation_state": "prescription_complete"
            }
            
            # 准备响应整合链的输入
            response_input = {
                "symptoms": self._format_symptoms(symptoms_list),
                "pattern_type": diagnosis_data.get("pattern_type", ""),
                "pathogenesis": diagnosis_data.get("pathogenesis", ""),
                "analysis": diagnosis_data.get("analysis", ""),
                "formula_name": prescription_data.get("formula_name", ""),
                "composition": self._format_composition(prescription_data.get("composition", [])),
                "usage": prescription_data.get("usage", ""),
                "treatment_principle": prescription_data.get("treatment_principle", ""),
                "contraindications": self._format_contraindications(prescription_data.get("contraindications", []))
            }
            
            # 运行响应整合链
            logger.info("整合最终响应...")
            response_result = self.response_chain.invoke(response_input)
            final_response = response_result.get('text', '')
            
            # 如果最终响应为空，创建一个备用响应
            if not final_response:
                final_response = self._create_fallback_response(diagnosis_data, prescription_data, symptoms_list)
            
            # 添加免责声明
            disclaimer = "\n\n【免责声明】：以上内容仅供参考，不构成医疗建议。请在专业中医师指导下使用中药，切勿自行配药或更改剂量。"
            final_response += disclaimer
            
            logger.info("处方推荐和响应整合完成")
            
            # 更新最终响应
            updated_state["response"] = final_response
            
            return updated_state, "to_safety_check"
        
        except Exception as e:
            error_msg = f"处方推荐过程中出错: {str(e)}"
            logger.error(error_msg)
            
            # 发生错误时的备用响应
            fallback_response = (
                "很抱歉，在生成处方推荐时遇到了技术问题。基于您提供的症状信息，"
                "我们已完成初步的中医辨证分析，但无法提供具体处方。\n\n"
                "建议您咨询专业的中医师获取更准确的诊断和处方。\n\n"
                "【免责声明】：本系统仅提供初步辅助分析，不能替代专业医疗诊断和治疗。"
            )
            
            return {
                **state,
                "error": error_msg,
                "response": fallback_response,
                "conversation_state": "prescription_error"
            }, "to_safety_check"
    
    def _create_fallback_response(self, diagnosis_data, prescription_data, symptoms_list):
        """创建备用响应"""
        try:
            # 症状概述
            symptoms_summary = "根据您描述的症状"
            if symptoms_list and len(symptoms_list) > 0:
                symptoms_names = [s.get("name", "症状") for s in symptoms_list if "name" in s]
                if symptoms_names:
                    symptoms_summary = f"根据您描述的{', '.join(symptoms_names)}"
            
            # 辨证分析
            pattern_type = diagnosis_data.get("pattern_type", "")
            pathogenesis = diagnosis_data.get("pathogenesis", "")
            
            diagnosis_text = "我们无法确定明确的中医证型。"
            if pattern_type and pattern_type != "无法确定":
                diagnosis_text = f"从中医角度分析，您的症状表现为【{pattern_type}】证。{pathogenesis}"
            
            # 处方推荐
            prescription_text = "目前无法提供具体的方剂推荐。"
            if prescription_data:
                formula_name = prescription_data.get("formula_name", "")
                treatment_principle = prescription_data.get("treatment_principle", "")
                usage = prescription_data.get("usage", "")
                
                if formula_name:
                    prescription_text = f"推荐使用【{formula_name}】方剂。{treatment_principle}\n\n用法：{usage}"
            
            # 组合响应
            fallback_response = f"""
{symptoms_summary}，{diagnosis_text}

{prescription_text}

建议您在专业中医师的指导下进行进一步诊治。保持良好的作息习惯，饮食规律，避免过度劳累和情绪波动，这些对您的健康恢复都有帮助。
            """
            
            return fallback_response.strip()
            
        except Exception as e:
            logger.error(f"创建备用响应时出错: {str(e)}")
            return "很抱歉，无法生成处方建议。建议您咨询专业的中医师获取诊断和治疗方案。"

# 导出节点实例以便在图中使用
prescription_node = PrescriptionNode()