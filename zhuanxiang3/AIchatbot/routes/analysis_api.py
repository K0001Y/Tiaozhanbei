"""
AI智能分析API模块
实现AI智能分析文本或图片内容，提供医疗建议
"""
import logging
import json
import os
import tempfile
import traceback
import subprocess
from typing import Dict, Any, Optional, Tuple
from flask import request, jsonify
from werkzeug.datastructures import FileStorage

# 导入您的诊断系统
try:
    from graph import run_tcm_graph  # 诊断系统
except ImportError:
    # 如果导入失败，提供模拟函数用于测试
    print("警告: 无法导入诊断系统，使用模拟数据")
    
    def run_tcm_graph(user_input, messages=None, memory=None, config=None):
        return {
            "response": f"基于AI分析: {user_input[:100]}...，建议进一步观察症状变化，如有疑虑请及时就医。",
            "diagnosis_data": {"分析结果": "初步评估完成", "建议": "密切观察"},
            "prescription_data": {"建议": "保持良好作息，适当运动"}
        }

logger = logging.getLogger(__name__)

class AIAnalysisAPI:
    """AI智能分析API类"""
    
    def __init__(self):
        """初始化AI智能分析API"""
        try:
            logger.info("初始化AI智能分析API模块")
            
            self.api_name = "AI智能分析API"
            self.version = "1.0.0"
            
            # 支持的文件格式（用于OCR处理）
            self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'pdf'}
            
            logger.info("AI智能分析API模块初始化完成")
            
        except Exception as e:
            logger.error(f"AI智能分析API模块初始化失败: {str(e)}")
            raise
    
    def _validate_analyze_params(self, query: Optional[str], file: Optional[FileStorage]) -> Tuple[bool, str]:
        """
        验证AI分析参数
        
        :param query: 用户查询文本
        :param file: 上传的文件
        :return: (是否有效, 错误信息)
        """
        # 至少要有一个输入
        if not query and not file:
            return False, "请提供查询文本或上传文件"
        
        # 验证查询文本
        if query:
            query = query.strip()
            if len(query) > 2000:
                return False, "查询文本过长，请限制在2000字符以内"
            
            if len(query) < 2:
                return False, "查询文本过于简短，请提供更详细的描述"
        
        # 验证文件
        if file:
            file_valid, file_error = self._validate_file(file)
            if not file_valid:
                return False, file_error
        
        return True, ""
    
    def _validate_file(self, file: FileStorage) -> Tuple[bool, str]:
        """
        验证上传的文件
        
        :param file: 上传的文件
        :return: (是否有效, 错误信息)
        """
        if not file:
            return False, "未上传文件"
        
        if file.filename == '':
            return False, "文件名为空"
        
        # 检查文件扩展名
        if '.' not in file.filename:
            return False, "文件没有扩展名"
        
        ext = file.filename.rsplit('.', 1)[1].lower()
        if ext not in self.allowed_extensions:
            return False, f"不支持的文件格式，支持的格式: {', '.join(self.allowed_extensions)}"
        
        # 检查文件大小（限制为20MB）
        file.seek(0, 2)  # 移动到文件末尾
        file_size = file.tell()
        file.seek(0)  # 重置文件指针
        
        if file_size > 20 * 1024 * 1024:  # 20MB
            return False, "文件大小超过20MB限制"
        
        if file_size == 0:
            return False, "文件为空"
        
        return True, ""
    
    def _save_temp_file(self, file: FileStorage) -> str:
        """
        保存临时文件并返回文件路径
        
        :param file: 上传的文件
        :return: 临时文件路径
        """
        # 创建临时文件
        suffix = '.' + file.filename.rsplit('.', 1)[1].lower()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        try:
            # 保存文件
            file.save(temp_file.name)
            return temp_file.name
        except Exception as e:
            # 如果保存失败，删除临时文件
            try:
                os.unlink(temp_file.name)
            except:
                pass
            raise e
    
    def _cleanup_temp_file(self, file_path: str):
        """
        清理临时文件
        
        :param file_path: 文件路径
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"清理临时文件失败: {file_path} - {str(e)}")
    
    def _process_file_with_ocr(self, file_path: str) -> str:
        """
        使用OCR处理文件
        
        :param file_path: 文件路径
        :return: OCR提取的文本内容
        """
        try:
            logger.info(f"开始OCR处理文件: {file_path}")
            
            # 调用transformer.py进行OCR处理
            result = subprocess.run(
                ['python', 'transformer.py', file_path],
                capture_output=True,
                text=True,
                timeout=120,  # 2分钟超时
                cwd=os.getcwd()  # 确保在正确的工作目录
            )
            
            if result.returncode != 0:
                error_msg = f"OCR处理失败，返回码: {result.returncode}"
                if result.stderr:
                    error_msg += f"，错误信息: {result.stderr}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # 解析输出，寻找OCR结果文件路径
            output_lines = result.stdout.strip().split('\n')
            output_file_path = None
            
            for line in output_lines:
                if line.startswith('OUTPUT_PATH='):
                    output_file_path = line.replace('OUTPUT_PATH=', '').strip()
                    break
            
            if not output_file_path:
                raise Exception("未能从OCR处理结果中获取输出文件路径")
            
            # 检查OCR结果文件是否存在
            if not os.path.exists(output_file_path):
                raise Exception(f"OCR结果文件不存在: {output_file_path}")
            
            # 读取OCR结果
            with open(output_file_path, 'r', encoding='utf-8') as f:
                ocr_text = f.read()
            
            # 清理OCR结果临时文件
            try:
                os.unlink(output_file_path)
            except Exception as cleanup_error:
                logger.warning(f"清理OCR结果文件失败: {cleanup_error}")
            
            logger.info("OCR处理完成")
            return ocr_text.strip()
            
        except subprocess.TimeoutExpired:
            raise Exception("OCR处理超时，请尝试上传较小的文件")
        except Exception as e:
            logger.error(f"OCR处理异常: {str(e)}")
            raise Exception(f"文件OCR处理失败: {str(e)}")
    
    def _build_analysis_input(self, query: Optional[str], ocr_text: Optional[str]) -> str:
        """
        构建AI分析的输入文本
        
        :param query: 用户查询文本
        :param ocr_text: OCR提取的文本
        :return: 格式化的分析输入文本
        """
        input_parts = []
        
        # 添加用户查询
        if query and query.strip():
            input_parts.append(f"用户咨询：{query.strip()}")
        
        # 添加OCR提取的内容
        if ocr_text and ocr_text.strip():
            # 清理OCR文本，移除多余的空行和空格
            cleaned_ocr = '\n'.join(line.strip() for line in ocr_text.split('\n') if line.strip())
            input_parts.append(f"文档内容：\n{cleaned_ocr}")
        
        # 如果没有任何输入，提供默认文本
        if not input_parts:
            input_parts.append("请提供健康医疗方面的专业分析和建议。")
        
        # 合并所有输入
        combined_input = '\n\n'.join(input_parts)
        
        # 添加AI分析指令
        combined_input += "\n\n请根据以上信息，提供专业的医疗健康分析建议。如果涉及具体疾病诊断，请建议咨询专业医生。"
        
        return combined_input
    
    def ai_intelligent_analyze(self, query: Optional[str], file: Optional[FileStorage]) -> Tuple[int, Dict[str, Any]]:
        """
        执行AI智能分析
        
        :param query: 用户查询的文本内容
        :param file: 上传的图片或文档文件
        :return: (HTTP状态码, 响应数据)
        """
        temp_file_path = None
        
        try:
            logger.info("开始AI智能分析")
            
            # 验证输入参数
            is_valid, error_msg = self._validate_analyze_params(query, file)
            if not is_valid:
                return 400, {
                    "success": False,
                    "message": error_msg,
                    "data": {"solution": ""}
                }
            
            ocr_text = None
            
            # 处理文件（如果有）
            if file:
                # 保存临时文件
                temp_file_path = self._save_temp_file(file)
                
                # 进行OCR处理
                try:
                    ocr_text = self._process_file_with_ocr(temp_file_path)
                except Exception as ocr_error:
                    logger.error(f"OCR处理失败: {str(ocr_error)}")
                    return 500, {
                        "success": False,
                        "message": f"文件处理失败: {str(ocr_error)}",
                        "data": {"solution": ""}
                    }
            
            # 构建分析输入文本
            analysis_input = self._build_analysis_input(query, ocr_text)
            
            logger.info(f"AI分析输入文本: {analysis_input[:200]}...")
            
            # 调用诊断系统进行AI分析
            logger.info("调用诊断系统进行AI智能分析")
            analysis_result = run_tcm_graph(
                user_input=analysis_input,
                config={"retriever_k": 4}
            )
            
            # 提取分析结果
            solution = analysis_result.get("response", "")
            
            # 如果主要响应为空，尝试从其他字段构建响应
            if not solution.strip():
                diagnosis_data = analysis_result.get("diagnosis_data", {})
                prescription_data = analysis_result.get("prescription_data", {})
                
                result_parts = []
                
                if diagnosis_data:
                    if isinstance(diagnosis_data, dict):
                        for key, value in diagnosis_data.items():
                            result_parts.append(f"{key}：{value}")
                    else:
                        result_parts.append(str(diagnosis_data))
                
                if prescription_data:
                    if isinstance(prescription_data, dict):
                        for key, value in prescription_data.items():
                            result_parts.append(f"{key}：{value}")
                    else:
                        result_parts.append(str(prescription_data))
                
                solution = "。".join(result_parts) if result_parts else "AI分析完成，建议咨询专业医生获取更详细的建议。"
            
            logger.info("AI智能分析完成")
            
            return 200, {
                "success": True,
                "message": "AI分析完成",
                "data": {
                    "solution": solution
                }
            }
            
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
        处理AI智能分析请求 (接口7.1)
        POST /api/ai/analyze
        """
        try:
            logger.info(f"收到AI智能分析请求: {request.remote_addr}")
            
            # 获取multipart/form-data参数
            query = request.form.get('query', '').strip()
            if not query:
                query = None
            
            # 获取上传的文件
            file = None
            if 'file' in request.files:
                file = request.files['file']
                # 如果文件名为空，视为未上传
                if file.filename == '':
                    file = None
            
            # 执行AI智能分析
            status_code, result_data = self.ai_intelligent_analyze(query, file)
            
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
        """
        获取API信息
        
        :return: API信息
        """
        return {
            "name": self.api_name,
            "version": self.version,
            "description": "AI智能分析文本或图片内容，提供医疗建议",
            "endpoints": {
                "ai_analyze": {
                    "method": "POST",
                    "path": "/api/ai/analyze",
                    "content_type": "multipart/form-data",
                    "params": {
                        "query": "用户查询的文本内容（可选）",
                        "file": "上传的图片或文档文件（可选）"
                    },
                    "description": "AI智能分析，接口编号7.1",
                    "note": "query和file参数至少提供一个"
                }
            },
            "supported_file_formats": list(self.allowed_extensions),
            "max_file_size": "20MB",
            "status": "active"
        }