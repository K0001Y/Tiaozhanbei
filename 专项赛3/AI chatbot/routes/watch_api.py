"""
医学图像分析API模块
实现中医望诊图像分析和补充分析功能
"""
import logging
import json
import os
import tempfile
import traceback
from typing import Dict, Any, Optional
from flask import request, jsonify
from werkzeug.datastructures import FileStorage

# 导入您的图像识别和诊断系统
try:
    from others.watch import TCMDiagnosisSystem  # 图像识别系统
    from graph import run_tcm_graph  # 诊断系统
except ImportError:
    # 如果导入失败，提供模拟类用于测试
    print("警告: 无法导入图像识别或诊断系统，使用模拟数据")
    
    class TCMDiagnosisSystem:
        def __init__(self, api_key, base_url=None):
            pass
        def comprehensive_diagnosis(self, image_url):
            return {
                "图像识别": {"类型": "舌诊", "置信度": 0.95},
                "分析结果": json.dumps({
                    "诊断类型": "舌诊",
                    "舌质": {"颜色": {"主色": "淡红"}, "形态": {"整体形状": "正常"}},
                    "辨证提示": ["脾胃虚弱(85%)"],
                    "健康建议": ["注意饮食规律"]
                }, ensure_ascii=False)
            }
    
    def run_tcm_graph(user_input, messages=None, memory=None, config=None):
        return {
            "response": f"基于图像分析: {user_input}，建议注意饮食调理。",
            "diagnosis_data": {"证型": "脾胃虚弱", "建议": "调理脾胃"}
        }

logger = logging.getLogger(__name__)

class MedicalImageAPI:
    """医学图像分析API类"""
    
    def __init__(self):
        """初始化医学图像分析API"""
        try:
            logger.info("初始化医学图像分析API模块")
            
            # 初始化图像识别系统
            # 请替换为您的实际API密钥
            api_key = os.environ.get('OPENAI_API_KEY', 'sk-xxx')
            base_url = os.environ.get('OPENAI_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
            
            self.tcm_system = TCMDiagnosisSystem(api_key=api_key, base_url=base_url)
            
            self.api_name = "医学图像分析API"
            self.version = "1.0.0"
            
            # 支持的图像格式
            self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
            
            logger.info("医学图像分析API模块初始化完成")
            
        except Exception as e:
            logger.error(f"医学图像分析API模块初始化失败: {str(e)}")
            raise
    
    def _validate_image_file(self, file: FileStorage) -> tuple[bool, str]:
        """
        验证上传的图像文件
        
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
        
        # 检查文件大小（限制为10MB）
        file.seek(0, 2)  # 移动到文件末尾
        file_size = file.tell()
        file.seek(0)  # 重置文件指针
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            return False, "文件大小超过10MB限制"
        
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
    
    def _extract_diagnosis_text(self, image_analysis_result: Dict[str, Any]) -> str:
        """
        从图像分析结果中提取诊断描述文本
        
        :param image_analysis_result: 图像分析结果
        :return: 诊断描述文本
        """
        try:
            # 获取分析结果
            analysis_result = image_analysis_result.get("分析结果", "")
            image_type = image_analysis_result.get("图像识别", {}).get("类型", "未知")
            confidence = image_analysis_result.get("图像识别", {}).get("置信度", 0)
            
            # 尝试解析JSON格式的分析结果
            try:
                if isinstance(analysis_result, str):
                    analysis_data = json.loads(analysis_result)
                else:
                    analysis_data = analysis_result
                
                # 构建诊断描述
                description_parts = []
                
                # 添加图像类型信息
                description_parts.append(f"图像类型：{image_type}（置信度：{confidence:.1%}）")
                
                # 提取关键诊断信息
                if "辨证提示" in analysis_data:
                    syndrome_tips = analysis_data["辨证提示"]
                    if syndrome_tips:
                        description_parts.append(f"辨证提示：{'; '.join(syndrome_tips)}")
                
                # 提取健康建议
                if "健康建议" in analysis_data:
                    health_advice = analysis_data["健康建议"]
                    if health_advice:
                        description_parts.append(f"健康建议：{'; '.join(health_advice)}")
                
                # 提取中医理论依据
                if "中医理论依据" in analysis_data:
                    theory_basis = analysis_data["中医理论依据"]
                    if theory_basis:
                        description_parts.append(f"中医理论依据：{theory_basis}")
                
                # 如果没有提取到具体信息，使用原始分析结果
                if len(description_parts) <= 1:
                    # 尝试提取其他有用信息
                    for key, value in analysis_data.items():
                        if key not in ["诊断类型", "图像质量评估"] and value:
                            if isinstance(value, str):
                                description_parts.append(f"{key}：{value}")
                            elif isinstance(value, (list, dict)):
                                description_parts.append(f"{key}：{json.dumps(value, ensure_ascii=False)}")
                
                return "。".join(description_parts)
                
            except json.JSONDecodeError:
                # 如果不是JSON格式，直接返回文本结果
                return f"图像类型：{image_type}（置信度：{confidence:.1%}）。{analysis_result}"
                
        except Exception as e:
            logger.warning(f"提取诊断文本失败: {str(e)}")
            return f"图像分析完成，但提取诊断信息时出现问题：{str(e)}"
    
    def analyze_image(self, image_file: FileStorage, description: str = "") -> tuple[int, Dict[str, Any]]:
        """
        图片望诊分析
        
        :param image_file: 上传的图像文件
        :param description: 可选的图像描述
        :return: (HTTP状态码, 响应数据)
        """
        temp_file_path = None
        
        try:
            logger.info("开始图片望诊分析")
            
            # 验证图像文件
            is_valid, error_msg = self._validate_image_file(image_file)
            if not is_valid:
                return 400, {
                    "success": False,
                    "message": error_msg,
                    "data": {"results": ""}
                }
            
            # 保存临时文件
            temp_file_path = self._save_temp_file(image_file)
            
            # 转换为图像URL（这里需要根据您的实际情况调整）
            # 如果您的图像识别系统需要URL，可能需要将文件上传到云存储
            # 这里假设使用本地文件路径
            image_url = f"file://{temp_file_path}"
            
            # 调用图像识别系统进行分析
            logger.info(f"调用图像识别系统分析: {image_url}")
            image_analysis_result = self.tcm_system.comprehensive_diagnosis(image_url)
            
            # 提取诊断描述文本
            diagnosis_text = self._extract_diagnosis_text(image_analysis_result)
            
            # 添加可选描述信息
            if description:
                diagnosis_text = f"图像描述：{description}。{diagnosis_text}"
            
            # 调用诊断系统进行进一步分析
            logger.info("调用诊断系统进行分析")
            diagnosis_result = run_tcm_graph(
                user_input=diagnosis_text,
                config={"retriever_k": 3}
            )
            
            # 获取最终诊断结果
            final_results = diagnosis_result.get("response", diagnosis_text)
            
            logger.info("图片望诊分析完成")
            
            return 200, {
                "success": True,
                "message": "望诊分析成功",
                "data": {
                    "results": final_results
                }
            }
            
        except Exception as e:
            error_msg = f"图片望诊分析失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return 500, {
                "success": False,
                "message": error_msg,
                "data": {"results": ""}
            }
            
        finally:
            # 清理临时文件
            if temp_file_path:
                self._cleanup_temp_file(temp_file_path)
    
    def analyze_with_supplement(self, prev_analysis: str, additional_info: str, 
                               additional_file: Optional[FileStorage] = None) -> tuple[int, Dict[str, Any]]:
        """
        望诊补充分析
        
        :param prev_analysis: 之前的望诊分析结果
        :param additional_info: 补充信息
        :param additional_file: 可选的补充图像文件
        :return: (HTTP状态码, 响应数据)
        """
        temp_file_path = None
        
        try:
            logger.info("开始望诊补充分析")
            
            # 构建补充分析的输入文本
            analysis_input_parts = []
            
            # 添加之前的分析结果
            if prev_analysis:
                analysis_input_parts.append(f"之前的望诊分析：{prev_analysis}")
            
            # 添加补充信息
            if additional_info:
                analysis_input_parts.append(f"补充信息：{additional_info}")
            
            # 处理补充图像文件（如果有）
            if additional_file:
                # 验证图像文件
                is_valid, error_msg = self._validate_image_file(additional_file)
                if not is_valid:
                    return 400, {
                        "success": False,
                        "message": f"补充图像文件错误: {error_msg}",
                        "data": {"results": ""}
                    }
                
                # 保存临时文件并分析
                temp_file_path = self._save_temp_file(additional_file)
                image_url = f"file://{temp_file_path}"
                
                # 分析补充图像
                logger.info("分析补充图像")
                additional_image_result = self.tcm_system.comprehensive_diagnosis(image_url)
                additional_diagnosis_text = self._extract_diagnosis_text(additional_image_result)
                
                analysis_input_parts.append(f"补充图像分析：{additional_diagnosis_text}")
            
            # 合并所有输入信息
            combined_input = "。".join(analysis_input_parts)
            
            if not combined_input.strip():
                return 400, {
                    "success": False,
                    "message": "补充分析输入信息为空",
                    "data": {"results": ""}
                }
            
            # 调用诊断系统进行综合分析
            logger.info("调用诊断系统进行补充分析")
            diagnosis_result = run_tcm_graph(
                user_input=combined_input,
                config={"retriever_k": 4}  # 补充分析使用更多检索结果
            )
            
            # 获取最终诊断结果
            final_results = diagnosis_result.get("response", combined_input)
            
            logger.info("望诊补充分析完成")
            
            return 200, {
                "success": True,
                "message": "补充望诊信息成功",
                "data": {
                    "results": final_results
                }
            }
            
        except Exception as e:
            error_msg = f"望诊补充分析失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return 500, {
                "success": False,
                "message": error_msg,
                "data": {"results": ""}
            }
            
        finally:
            # 清理临时文件
            if temp_file_path:
                self._cleanup_temp_file(temp_file_path)
    
    def handle_watch_request(self):
        """
        处理图片望诊请求 (接口4.1)
        POST /api/watch
        """
        try:
            logger.info(f"收到图片望诊请求: {request.remote_addr}")
            
            # 获取上传的图像文件
            if 'image' not in request.files:
                return jsonify({
                    "success": False,
                    "message": "未找到上传的图像文件",
                    "data": {"results": ""}
                }), 400
            
            image_file = request.files['image']
            description = request.form.get('description', '')
            
            # 执行图片望诊分析
            status_code, result_data = self.analyze_image(image_file, description)
            
            return jsonify(result_data), status_code
            
        except Exception as e:
            error_msg = f"图片望诊请求处理异常: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return jsonify({
                "success": False,
                "message": error_msg,
                "data": {"results": ""}
            }), 500
    
    def handle_watch_complete_request(self):
        """
        处理望诊补充请求 (接口4.2)
        POST /api/watch/complete
        """
        try:
            logger.info(f"收到望诊补充请求: {request.remote_addr}")
            
            # 获取必要参数
            prev_analysis = request.form.get('prevAnalysis', '').strip()
            additional_info = request.form.get('additionalInfo', '').strip()
            
            # 检查必要参数
            if not prev_analysis and not additional_info:
                return jsonify({
                    "success": False,
                    "message": "必须提供之前的分析结果或补充信息",
                    "data": {"results": ""}
                }), 400
            
            # 获取可选的补充图像文件
            additional_file = None
            if 'additionalFile' in request.files:
                additional_file = request.files['additionalFile']
                # 如果文件名为空，视为未上传
                if additional_file.filename == '':
                    additional_file = None
            
            # 执行望诊补充分析
            status_code, result_data = self.analyze_with_supplement(
                prev_analysis, additional_info, additional_file
            )
            
            return jsonify(result_data), status_code
            
        except Exception as e:
            error_msg = f"望诊补充请求处理异常: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return jsonify({
                "success": False,
                "message": error_msg,
                "data": {"results": ""}
            }), 500
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        获取API信息
        
        :return: API信息
        """
        return {
            "name": self.api_name,
            "version": self.version,
            "description": "基于中医理论的医学图像分析API，支持望诊分析和补充分析",
            "endpoints": {
                "watch": {
                    "method": "POST",
                    "path": "/api/watch",
                    "content_type": "multipart/form-data",
                    "params": {
                        "image": "图像文件（必填）",
                        "description": "图像描述（可选）"
                    },
                    "description": "图片望诊分析，接口编号4.1"
                },
                "watch_complete": {
                    "method": "POST", 
                    "path": "/api/watch/complete",
                    "content_type": "multipart/form-data",
                    "params": {
                        "prevAnalysis": "之前的望诊分析结果",
                        "additionalInfo": "补充信息",
                        "additionalFile": "补充图像文件（可选）"
                    },
                    "description": "望诊补充分析，接口编号4.2"
                }
            },
            "supported_formats": list(self.allowed_extensions),
            "max_file_size": "10MB",
            "status": "active"
        }