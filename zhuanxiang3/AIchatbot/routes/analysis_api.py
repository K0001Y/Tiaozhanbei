"""
AI智能分析API模块 - Linus彻底重写版（修复版）
修复要点:
1. 修复MedicalRecord.from_session_aggregation调用参数问题
2. 修复请求参数验证逻辑
3. 改善错误处理和日志记录
4. 确保Mock类方法签名一致性
"""
import logging
import json
import os
import tempfile
import traceback
import subprocess
import time
from typing import Dict, Any, Optional, Tuple, List
from flask import request, jsonify
from werkzeug.datastructures import FileStorage

# 导入优化后的诊断系统和session管理器
try:
    from graph import run_tcm_graph_with_state, run_tcm_graph, get_compiled_graph
    from routes.inquiry_api import session_manager as inquiry_session_manager
    from routes.watch_api import watch_session_manager
    from routes.record_api import RecordSessionAggregator, MedicalRecord
    MOCK_MODE = False
except ImportError:
    print("警告: 无法导入优化后的系统，使用模拟版本")
    MOCK_MODE = True
    
    def run_tcm_graph(user_input, messages=None, memory=None, config=None):
        return {
            "response": f"基于AI分析: {user_input[:100]}...，建议进一步观察症状变化。",
            "diagnosis_data": {"分析结果": "初步评估完成"},
            "prescription_data": {"建议": "保持良好作息"}
        }
    
    def run_tcm_graph_with_state(user_input, existing_state=None, messages=None, memory=None, config=None):
        return run_tcm_graph(user_input, messages, memory, config)
    
    def get_compiled_graph():
        return None
    
    class MockSessionManager:
        def __init__(self):
            self._sessions = {}
            
        def _get_recent_sessions(self, time_window):
            return []
    
    inquiry_session_manager = MockSessionManager()
    watch_session_manager = MockSessionManager()
    
    class MockRecordSessionAggregator:
        @staticmethod
        def find_related_sessions(patient_info, watch_results, inquiry_results):
            return [], []
        
        @staticmethod
        def _get_recent_sessions(session_manager, time_window):
            return []
    
    class MockMedicalRecord:
        def __init__(self):
            self.symptoms = []
            self.diagnosis_findings = []
            self.prescriptions = []
            self.watch_analysis = ""
            self.inquiry_analysis = ""
        
        @classmethod
        def from_session_aggregation(cls, watch_sessions, inquiry_sessions, patient_info=""):
            """修复：确保参数签名正确"""
            record = cls()
            # 模拟从sessions提取数据
            if watch_sessions:
                record.symptoms.extend([f"望诊症状{i+1}" for i in range(len(watch_sessions))])
                record.watch_analysis = "望诊分析汇总"
            if inquiry_sessions:
                record.diagnosis_findings.extend([f"问诊发现{i+1}" for i in range(len(inquiry_sessions))])
                record.inquiry_analysis = "问诊分析汇总"
            return record
        
        def to_diagnosis_text(self):
            return "模拟诊断数据"
    
    RecordSessionAggregator = MockRecordSessionAggregator
    MedicalRecord = MockMedicalRecord

logger = logging.getLogger(__name__)

class ContextualAnalysis:
    """
    上下文分析数据 - 结合历史诊断的智能分析
    """
    
    def __init__(self):
        self.has_history = False
        self.watch_context = ""
        self.inquiry_context = ""
        self.historical_symptoms = []
        self.historical_diagnoses = []
        self.historical_prescriptions = []
        self.confidence_level = "低"  # 低/中/高
        self.analysis_mode = "单纯分析"  # 单纯分析/综合分析
    
    @classmethod
    def from_sessions(cls, watch_sessions: List[Dict], inquiry_sessions: List[Dict]) -> 'ContextualAnalysis':
        """
        从session数据创建上下文分析
        """
        context = cls()
        
        if watch_sessions or inquiry_sessions:
            context.has_history = True
            context.analysis_mode = "综合分析"
            context.confidence_level = "高"
            
            try:
                # 修复：确保调用参数正确
                medical_record = MedicalRecord.from_session_aggregation(
                    watch_sessions, inquiry_sessions, ""
                )
                
                context.historical_symptoms = medical_record.symptoms
                context.historical_diagnoses = medical_record.diagnosis_findings
                context.historical_prescriptions = medical_record.prescriptions
                
                if hasattr(medical_record, 'watch_analysis') and medical_record.watch_analysis:
                    context.watch_context = medical_record.watch_analysis
                
                if hasattr(medical_record, 'inquiry_analysis') and medical_record.inquiry_analysis:
                    context.inquiry_context = medical_record.inquiry_analysis
                    
            except Exception as e:
                logger.error(f"从sessions创建医疗记录失败: {e}")
                # 降级处理
                context.has_history = False
                context.analysis_mode = "单纯分析"
                context.confidence_level = "中"
        else:
            context.confidence_level = "中"
        
        return context
    
    def build_contextual_prompt(self, new_query: str, ocr_text: str = "") -> str:
        """
        构建包含历史上下文的分析提示
        """
        prompt_parts = []
        
        # 添加历史上下文（如果有）
        if self.has_history:
            prompt_parts.append("【历史诊断记录】")
            
            if self.historical_symptoms:
                prompt_parts.append(f"既往症状：{'; '.join(self.historical_symptoms)}")
            
            if self.watch_context:
                prompt_parts.append(f"望诊分析：{self.watch_context}")
            
            if self.inquiry_context:
                prompt_parts.append(f"问诊分析：{self.inquiry_context}")
            
            if self.historical_diagnoses:
                prompt_parts.append(f"诊断发现：{'; '.join(self.historical_diagnoses[:3])}")  # 最多3条
            
            if self.historical_prescriptions:
                prompt_parts.append(f"既往建议：{'; '.join(self.historical_prescriptions[:3])}")
        
        # 添加新的查询内容
        if new_query and new_query.strip():
            prompt_parts.append(f"【当前咨询】{new_query.strip()}")
        
        # 添加OCR内容
        if ocr_text and ocr_text.strip():
            cleaned_ocr = '\n'.join(line.strip() for line in ocr_text.split('\n') if line.strip())
            prompt_parts.append(f"【文档内容】\n{cleaned_ocr}")
        
        # 构建分析指令
        if self.has_history:
            instruction = "请结合既往诊断记录和当前咨询，提供综合的医疗健康分析建议。重点关注症状变化、诊断一致性和治疗效果评估。"
        else:
            instruction = "请根据提供的信息，进行专业的医疗健康分析并提供建议。"
        
        prompt_parts.append(f"【分析要求】{instruction}")
        
        return '\n\n'.join(prompt_parts)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        获取分析摘要信息
        """
        return {
            "analysis_mode": self.analysis_mode,
            "has_history": self.has_history,
            "confidence_level": self.confidence_level,
            "historical_data_points": {
                "symptoms_count": len(self.historical_symptoms),
                "diagnoses_count": len(self.historical_diagnoses),
                "prescriptions_count": len(self.historical_prescriptions),
                "has_watch_context": bool(self.watch_context),
                "has_inquiry_context": bool(self.inquiry_context)
            }
        }

class OCRProcessor:
    """
    OCR处理器 - 优化版本，更好的错误处理和性能
    """
    
    @staticmethod
    def process_file(file_path: str) -> Tuple[str, bool]:
        """
        处理文件进行OCR
        
        :param file_path: 文件路径
        :return: (OCR文本, 是否成功)
        """
        try:
            logger.info(f"开始OCR处理文件: {file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise Exception(f"文件不存在: {file_path}")
            
            # 模拟模式下直接返回
            if MOCK_MODE:
                logger.info("Mock模式：模拟OCR处理成功")
                return f"模拟OCR结果：这是从{os.path.basename(file_path)}中提取的文本内容", True
            
            # 检查transformer.py是否存在
            transformer_path = os.path.join(os.getcwd(), 'transformer.py')
            if not os.path.exists(transformer_path):
                raise Exception("OCR处理模块(transformer.py)不存在")
            
            # 调用OCR处理
            result = subprocess.run(
                ['python', 'transformer.py', file_path],
                capture_output=True,
                text=True,
                timeout=180,  # 3分钟超时
                cwd=os.getcwd(),
                env=os.environ.copy()  # 传递环境变量
            )
            
            if result.returncode != 0:
                error_msg = f"OCR处理失败，返回码: {result.returncode}"
                if result.stderr:
                    error_msg += f"，错误信息: {result.stderr}"
                logger.error(error_msg)
                return "", False
            
            # 解析输出
            output_lines = result.stdout.strip().split('\n')
            output_file_path = None
            
            for line in output_lines:
                if line.startswith('OUTPUT_PATH='):
                    output_file_path = line.replace('OUTPUT_PATH=', '').strip()
                    break
            
            if not output_file_path:
                logger.warning("未找到OCR输出文件路径，尝试直接从stdout获取结果")
                # 如果没有输出路径，尝试直接从stdout获取内容
                return result.stdout.strip(), True
            
            # 读取OCR结果文件
            if not os.path.exists(output_file_path):
                logger.warning(f"OCR结果文件不存在: {output_file_path}")
                return result.stdout.strip(), True
            
            with open(output_file_path, 'r', encoding='utf-8') as f:
                ocr_text = f.read()
            
            # 清理OCR结果临时文件
            try:
                os.unlink(output_file_path)
            except Exception as cleanup_error:
                logger.warning(f"清理OCR结果文件失败: {cleanup_error}")
            
            logger.info("OCR处理完成")
            return ocr_text.strip(), True
            
        except subprocess.TimeoutExpired:
            error_msg = "OCR处理超时，请尝试上传较小的文件"
            logger.error(error_msg)
            return "", False
        except Exception as e:
            error_msg = f"OCR处理异常: {str(e)}"
            logger.error(error_msg)
            return "", False

class AIAnalysisAPI:
    """AI智能分析API - Linus彻底重写版（修复版）"""
    
    def __init__(self):
        """初始化AI智能分析API"""
        try:
            logger.info("初始化AI智能分析API模块")
            
            self.api_name = "AI智能分析API"
            self.version = "2.0.1"  # 修复版
            
            # 支持的文件格式
            self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'pdf'}
            
            logger.info(f"AI智能分析API模块初始化完成 - 模式: {'Mock' if MOCK_MODE else 'Production'}")
            
        except Exception as e:
            logger.error(f"AI智能分析API模块初始化失败: {str(e)}")
            raise
    
    def _validate_analyze_params(self, query: Optional[str], file: Optional[FileStorage]) -> Tuple[bool, str]:
        """验证AI分析参数"""
        logger.debug(f"验证参数 - query: {bool(query and query.strip())}, file: {bool(file and file.filename)}")
        
        # 至少要有一个输入
        has_query = query and query.strip()
        has_file = file and file.filename and file.filename.strip()
        
        if not has_query and not has_file:
            return False, "请提供查询文本或上传文件"
        
        # 验证查询文本
        if has_query:
            query_clean = query.strip()
            if len(query_clean) > 2000:
                return False, "查询文本过长，请限制在2000字符以内"
            
            if len(query_clean) < 2:
                return False, "查询文本过于简短，请提供更详细的描述"
        
        # 验证文件
        if has_file:
            file_valid, file_error = self._validate_file(file)
            if not file_valid:
                return False, file_error
        
        return True, ""
    
    def _validate_file(self, file: FileStorage) -> Tuple[bool, str]:
        """验证上传的文件"""
        if not file:
            return False, "未上传文件"
        
        if not file.filename or file.filename.strip() == '':
            return False, "文件名为空"
        
        # 检查文件扩展名
        if '.' not in file.filename:
            return False, "文件没有扩展名"
        
        ext = file.filename.rsplit('.', 1)[1].lower()
        if ext not in self.allowed_extensions:
            return False, f"不支持的文件格式，支持的格式: {', '.join(self.allowed_extensions)}"
        
        # 检查文件大小（限制为20MB）
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 20 * 1024 * 1024:  # 20MB
            return False, "文件大小超过20MB限制"
        
        if file_size == 0:
            return False, "文件为空"
        
        return True, ""
    
    def _save_temp_file(self, file: FileStorage) -> str:
        """保存临时文件"""
        suffix = '.' + file.filename.rsplit('.', 1)[1].lower()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        try:
            file.save(temp_file.name)
            logger.debug(f"临时文件保存至: {temp_file.name}")
            return temp_file.name
        except Exception as e:
            try:
                os.unlink(temp_file.name)
            except:
                pass
            raise e
    
    def _cleanup_temp_file(self, file_path: str):
        """清理临时文件"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"临时文件已清理: {file_path}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {file_path} - {str(e)}")
    
    def _gather_historical_context(self) -> ContextualAnalysis:
        """
        收集用户的历史诊断上下文 - 核心智能功能
        """
        logger.info("收集用户历史诊断上下文")
        
        # 获取最近的session数据（2小时窗口）
        watch_sessions = []
        inquiry_sessions = []
        
        try:
            if hasattr(watch_session_manager, '_sessions'):
                watch_sessions = RecordSessionAggregator._get_recent_sessions(
                    watch_session_manager, time_window=7200
                )
                logger.debug(f"获取到{len(watch_sessions)}个望诊sessions")
        except Exception as e:
            logger.warning(f"获取望诊session失败: {e}")
        
        try:
            if hasattr(inquiry_session_manager, '_sessions'):
                inquiry_sessions = RecordSessionAggregator._get_recent_sessions(
                    inquiry_session_manager, time_window=7200
                )
                logger.debug(f"获取到{len(inquiry_sessions)}个问诊sessions")
        except Exception as e:
            logger.warning(f"获取问诊session失败: {e}")
        
        context = ContextualAnalysis.from_sessions(watch_sessions, inquiry_sessions)
        
        summary = context.get_analysis_summary()
        logger.info(f"历史上下文收集完成: {summary}")
        
        return context
    
    def ai_intelligent_analyze(self, query: Optional[str], file: Optional[FileStorage], 
                              context_mode: str = "auto") -> Tuple[int, Dict[str, Any]]:
        """
        执行AI智能分析 - 重写版，支持上下文分析
        
        :param query: 用户查询的文本内容
        :param file: 上传的图片或文档文件
        :param context_mode: 上下文模式 auto/simple/comprehensive
        :return: (HTTP状态码, 响应数据)
        """
        temp_file_path = None
        
        try:
            logger.info(f"开始AI智能分析 - query: {bool(query)}, file: {bool(file)}, mode: {context_mode}")
            
            # 验证输入参数
            is_valid, error_msg = self._validate_analyze_params(query, file)
            if not is_valid:
                logger.warning(f"参数验证失败: {error_msg}")
                return 400, {
                    "success": False,
                    "message": error_msg,
                    "data": {"solution": ""}
                }
            
            # 步骤1：处理OCR（如果有文件）
            ocr_text = ""
            if file and file.filename and file.filename.strip():
                temp_file_path = self._save_temp_file(file)
                
                ocr_result, ocr_success = OCRProcessor.process_file(temp_file_path)
                if ocr_success:
                    ocr_text = ocr_result
                    logger.info(f"OCR处理成功，提取文本长度: {len(ocr_text)}")
                else:
                    logger.warning("OCR处理失败，继续进行纯文本分析")
            
            # 步骤2：收集历史上下文（智能模式）
            contextual_analysis = None
            if context_mode in ["auto", "comprehensive"]:
                try:
                    contextual_analysis = self._gather_historical_context()
                except Exception as e:
                    logger.warning(f"收集历史上下文失败: {e}")
                    contextual_analysis = ContextualAnalysis()  # 使用空的上下文
            
            # 步骤3：构建分析输入
            if contextual_analysis and contextual_analysis.has_history:
                # 综合分析：结合历史上下文
                analysis_input = contextual_analysis.build_contextual_prompt(query or "", ocr_text)
                logger.info("使用综合分析模式（结合历史诊断记录）")
            else:
                # 简单分析：仅基于当前输入
                input_parts = []
                if query and query.strip():
                    input_parts.append(f"用户咨询：{query.strip()}")
                if ocr_text.strip():
                    input_parts.append(f"文档内容：\n{ocr_text.strip()}")
                
                analysis_input = '\n\n'.join(input_parts)
                analysis_input += "\n\n请根据以上信息，提供专业的医疗健康分析建议。如果涉及具体疾病诊断，请建议咨询专业医生。"
                logger.info("使用简单分析模式")
            
            logger.info(f"AI分析输入文本长度: {len(analysis_input)} 字符")
            
            # 步骤4：执行AI分析（使用优化后的graph）
            try:
                if contextual_analysis and contextual_analysis.has_history and not MOCK_MODE:
                    # 尝试基于已有session进行增量分析
                    latest_session = None
                    try:
                        # 获取最新的session状态
                        all_sessions = []
                        if hasattr(watch_session_manager, '_sessions'):
                            all_sessions.extend(watch_session_manager._sessions.values())
                        if hasattr(inquiry_session_manager, '_sessions'):
                            all_sessions.extend(inquiry_session_manager._sessions.values())
                        
                        if all_sessions:
                            # 按更新时间排序，获取最新的
                            all_sessions.sort(key=lambda x: x.get('updated_at', 0), reverse=True)
                            latest_session_data = all_sessions[0]
                            if 'state' in latest_session_data:
                                latest_session = latest_session_data['state']
                            elif 'graph_state' in latest_session_data:
                                latest_session = latest_session_data['graph_state']
                    except Exception as e:
                        logger.warning(f"获取最新session状态失败: {e}")
                    
                    if latest_session:
                        logger.info("基于最新session状态进行增量AI分析")
                        analysis_result = run_tcm_graph_with_state(
                            user_input=analysis_input,
                            existing_state=latest_session,
                            config={"retriever_k": 5}
                        )
                    else:
                        logger.info("使用完整AI分析")
                        analysis_result = run_tcm_graph(
                            user_input=analysis_input,
                            config={"retriever_k": 5}
                        )
                else:
                    logger.info("使用标准AI分析")
                    analysis_result = run_tcm_graph(
                        user_input=analysis_input,
                        config={"retriever_k": 4}
                    )
                    
            except Exception as graph_error:
                logger.error(f"Graph执行失败: {graph_error}")
                # 降级处理：返回简单的分析结果
                analysis_result = {
                    "response": f"AI分析完成。基于您提供的信息，建议您注意观察症状变化并适时咨询专业医生。",
                    "diagnosis_data": {"状态": "分析完成"},
                    "prescription_data": {"建议": "如有需要请咨询专业医生"}
                }
            
            # 步骤5：提取和格式化结果
            solution = analysis_result.get("response", "")
            
            # 如果主要响应为空，构建备用响应
            if not solution.strip():
                diagnosis_data = analysis_result.get("diagnosis_data", {})
                prescription_data = analysis_result.get("prescription_data", {})
                
                result_parts = []
                
                if diagnosis_data:
                    if isinstance(diagnosis_data, dict):
                        for key, value in diagnosis_data.items():
                            if value:
                                result_parts.append(f"{key}：{value}")
                    else:
                        result_parts.append(str(diagnosis_data))
                
                if prescription_data:
                    if isinstance(prescription_data, dict):
                        for key, value in prescription_data.items():
                            if value:
                                result_parts.append(f"{key}：{value}")
                    else:
                        result_parts.append(str(prescription_data))
                
                solution = "。".join(result_parts) if result_parts else "AI分析完成，建议咨询专业医生获取更详细的建议。"
            
            # 构建响应
            response_data = {
                "success": True,
                "message": "AI分析完成",
                "data": {
                    "solution": solution
                }
            }
            
            # 添加分析元数据（调试用，生产环境可移除）
            if contextual_analysis:
                response_data["_analysis_metadata"] = contextual_analysis.get_analysis_summary()
                if MOCK_MODE:
                    response_data["_analysis_metadata"]["mock_mode"] = True
            
            logger.info("AI智能分析完成")
            return 200, response_data
            
        except Exception as e:
            error_msg = f"AI智能分析失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return 500, {
                "success": False,
                "message": error_msg,
                "data": {"solution": ""}
            }
        
        finally:
            # 清理临时文件
            if temp_file_path:
                self._cleanup_temp_file(temp_file_path)
    
    def handle_ai_analyze_request(self):
        """
        处理AI智能分析请求 - 保持向后兼容
        POST /api/ai/analyze
        """
        try:
            logger.info(f"收到AI智能分析请求: {request.remote_addr}")
            logger.debug(f"Form数据: {dict(request.form)}")
            logger.debug(f"Files数据: {list(request.files.keys())}")
            
            # 获取参数
            query = request.form.get('query', '').strip()
            if not query:
                query = None
            else:
                logger.debug(f"查询文本长度: {len(query)}")
            
            # 获取上下文模式（新功能）
            context_mode = request.form.get('contextMode', 'auto')
            if context_mode not in ['auto', 'simple', 'comprehensive']:
                context_mode = 'auto'
            
            # 获取上传的文件
            file = None
            if 'file' in request.files:
                uploaded_file = request.files['file']
                if uploaded_file.filename and uploaded_file.filename.strip():
                    file = uploaded_file
                    logger.debug(f"上传文件: {file.filename}")
            
            # 执行AI智能分析
            status_code, result_data = self.ai_intelligent_analyze(query, file, context_mode)
            
            logger.info(f"AI分析完成，状态码: {status_code}")
            return jsonify(result_data), status_code
            
        except Exception as e:
            error_msg = f"AI智能分析请求处理异常: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return jsonify({
                "success": False,
                "message": error_msg,
                "data": {"solution": ""}
            }), 500
    
    def get_api_info(self) -> Dict[str, Any]:
        """获取API信息"""
        return {
            "name": self.api_name,
            "version": self.version,
            "description": "AI智能分析API - Linus重写版（修复版），集成Session系统和上下文分析",
            "mode": "Mock" if MOCK_MODE else "Production",
            "features": [
                "智能上下文分析，结合用户历史诊断记录",
                "使用优化后的graph系统，避免重复编译",
                "多种分析模式：简单分析/综合分析",
                "优化的OCR处理流程",
                "增量分析，基于已有session状态",
                "完全向后兼容原有接口",
                "修复了参数验证和方法调用问题"
            ],
            "endpoints": {
                "ai_analyze": {
                    "method": "POST",
                    "path": "/api/ai/analyze",
                    "content_type": "multipart/form-data",
                    "params": {
                        "query": "用户查询的文本内容（可选）",
                        "file": "上传的图片或文档文件（可选）",
                        "contextMode": "上下文模式：auto(自动)/simple(简单)/comprehensive(综合)（可选）"
                    },
                    "description": "AI智能分析，接口编号7.1",
                    "note": "query和file参数至少提供一个"
                }
            },
            "analysis_modes": {
                "simple": "仅基于当前输入进行分析",
                "comprehensive": "强制结合历史诊断记录进行综合分析",
                "auto": "自动选择：有历史记录时使用综合分析，否则使用简单分析（推荐）"
            },
            "session_integration": {
                "inquiry_sessions": "已集成" if hasattr(inquiry_session_manager, '_sessions') else "未集成",
                "watch_sessions": "已集成" if hasattr(watch_session_manager, '_sessions') else "未集成",
                "context_window": "2小时"
            },
            "supported_file_formats": list(self.allowed_extensions),
            "max_file_size": "20MB",
            "status": "active"
        }