"""
中医问诊分析API模块
实现中医问诊分析和补充问诊功能
"""
import logging
import json
import os
import tempfile
import traceback
from typing import Dict, Any, Optional
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
            "response": f"基于问诊分析: {user_input}，建议调理脾胃，注意作息规律。",
            "diagnosis_data": {"证型": "气血两虚", "建议": "益气养血"},
            "prescription_data": {"方剂": "四君子汤加减"}
        }

logger = logging.getLogger(__name__)

class MedicalInquiryAPI:
    """中医问诊分析API类"""
    
    def __init__(self):
        """初始化中医问诊分析API"""
        try:
            logger.info("初始化中医问诊分析API模块")
            
            self.api_name = "中医问诊分析API"
            self.version = "1.0.0"
            
            # 支持的图像格式（用于补充问诊的检查报告）
            self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'pdf'}
            
            logger.info("中医问诊分析API模块初始化完成")
            
        except Exception as e:
            logger.error(f"中医问诊分析API模块初始化失败: {str(e)}")
            raise
    
    def _validate_inquiry_params(self, age: Any, gender: str, symptoms: str) -> tuple[bool, str]:
        """
        验证问诊参数
        
        :param age: 年龄
        :param gender: 性别
        :param symptoms: 症状描述
        :return: (是否有效, 错误信息)
        """
        # 验证年龄
        if age is None:
            return False, "年龄不能为空"
        
        try:
            age_int = int(age)
            if age_int < 0 or age_int > 150:
                return False, "年龄必须在0-150之间"
        except (ValueError, TypeError):
            return False, "年龄必须是有效的数字"
        
        # 验证性别
        if not gender:
            return False, "性别不能为空"
        
        gender = gender.strip()
        valid_genders = ['男', '女', '男性', '女性', 'male', 'female', 'M', 'F']
        if gender not in valid_genders:
            return False, f"性别必须是以下之一: {', '.join(['男', '女', '男性', '女性'])}"
        
        # 验证症状描述
        if not symptoms or not symptoms.strip():
            return False, "症状描述不能为空"
        
        if len(symptoms.strip()) < 2:
            return False, "症状描述过于简短，请提供更详细的描述"
        
        if len(symptoms) > 1000:
            return False, "症状描述过长，请限制在1000字符以内"
        
        return True, ""
    
    def _validate_image_file(self, file: FileStorage) -> tuple[bool, str]:
        """
        验证上传的检查报告文件
        
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
    
    def _normalize_gender(self, gender: str) -> str:
        """
        标准化性别表示
        
        :param gender: 输入的性别
        :return: 标准化的性别
        """
        gender = gender.strip().lower()
        if gender in ['男', '男性', 'male', 'm']:
            return '男'
        elif gender in ['女', '女性', 'female', 'f']:
            return '女'
        else:
            return gender  # 保持原样，让验证函数处理
    
    def _build_inquiry_text(self, age: int, gender: str, symptoms: str) -> str:
        """
        构建问诊输入文本
        
        :param age: 年龄
        :param gender: 性别
        :param symptoms: 症状描述
        :return: 格式化的问诊文本
        """
        # 标准化性别
        normalized_gender = self._normalize_gender(gender)
        
        # 构建基本信息
        inquiry_text = f"患者基本信息：{age}岁，{normalized_gender}性。"
        
        # 添加症状描述
        inquiry_text += f"主要症状：{symptoms.strip()}。"
        
        # 添加问诊提示
        inquiry_text += "请根据中医理论进行辨证分析，提供诊疗建议。"
        
        return inquiry_text
    
    def initial_inquiry(self, age: Any, gender: str, symptoms: str) -> tuple[int, Dict[str, Any]]:
        """
        初步问诊分析
        
        :param age: 年龄
        :param gender: 性别
        :param symptoms: 症状描述
        :return: (HTTP状态码, 响应数据)
        """
        try:
            logger.info("开始初步问诊分析")
            
            # 验证输入参数
            is_valid, error_msg = self._validate_inquiry_params(age, gender, symptoms)
            if not is_valid:
                return 400, {
                    "success": False,
                    "message": error_msg,
                    "data": {"results": ""}
                }
            
            # 构建问诊输入文本
            inquiry_text = self._build_inquiry_text(int(age), gender, symptoms)
            
            logger.info(f"问诊输入文本: {inquiry_text}")
            
            # 调用诊断系统进行分析
            logger.info("调用诊断系统进行问诊分析")
            diagnosis_result = run_tcm_graph(
                user_input=inquiry_text,
                config={"retriever_k": 4}
            )
            
            # 获取分析结果
            final_results = diagnosis_result.get("response", "")
            
            # 如果结果为空，尝试从其他字段获取信息
            if not final_results.strip():
                diagnosis_data = diagnosis_result.get("diagnosis_data", {})
                prescription_data = diagnosis_result.get("prescription_data", {})
                
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
                
                final_results = "。".join(result_parts) if result_parts else inquiry_text
            
            logger.info("初步问诊分析完成")
            
            return 200, {
                "success": True,
                "message": "问诊分析成功",
                "data": {
                    "results": final_results
                }
            }
            
        except Exception as e:
            error_msg = f"初步问诊分析失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return 500, {
                "success": False,
                "message": error_msg,
                "data": {"results": ""}
            }
    
    def complete_inquiry(self, prev_inquiry: str, additional_info: str, 
                        additional_file: Optional[FileStorage] = None) -> tuple[int, Dict[str, Any]]:
        """
        补充问诊分析
        
        :param prev_inquiry: 之前的问诊分析结果
        :param additional_info: 补充信息
        :param additional_file: 可选的检查报告文件
        :return: (HTTP状态码, 响应数据)
        """
        temp_file_path = None
        
        try:
            logger.info("开始补充问诊分析")
            
            # 验证必要参数
            if not prev_inquiry and not additional_info:
                return 400, {
                    "success": False,
                    "message": "必须提供之前的问诊结果或补充信息",
                    "data": {"results": ""}
                }
            
            # 构建补充问诊的输入文本
            inquiry_parts = []
            
            # 添加之前的问诊结果
            if prev_inquiry and prev_inquiry.strip():
                inquiry_parts.append(f"之前的问诊分析：{prev_inquiry.strip()}")
            
            # 添加补充信息
            if additional_info and additional_info.strip():
                inquiry_parts.append(f"补充信息：{additional_info.strip()}")
            
            # 处理检查报告文件（如果有）
            if additional_file:
                # 验证文件
                is_valid, error_msg = self._validate_image_file(additional_file)
                if not is_valid:
                    return 400, {
                        "success": False,
                        "message": f"检查报告文件错误: {error_msg}",
                        "data": {"results": ""}
                    }
                
                # 保存临时文件
                temp_file_path = self._save_temp_file(additional_file)
                
                # 根据文件类型添加描述
                file_ext = additional_file.filename.rsplit('.', 1)[1].lower()
                if file_ext == 'pdf':
                    inquiry_parts.append("已上传PDF格式的检查报告，请结合报告内容进行综合分析。")
                else:
                    inquiry_parts.append("已上传检查报告图片，请结合图像信息进行综合分析。")
            
            # 合并所有输入信息
            combined_inquiry = "。".join(inquiry_parts)
            
            if not combined_inquiry.strip():
                return 400, {
                    "success": False,
                    "message": "补充问诊输入信息为空",
                    "data": {"results": ""}
                }
            
            # 添加补充问诊的特殊指令
            combined_inquiry += "。请在之前分析的基础上，结合新的补充信息，提供更完善的中医辨证分析和诊疗建议。"
            
            logger.info(f"补充问诊输入文本: {combined_inquiry}")
            
            # 调用诊断系统进行综合分析
            logger.info("调用诊断系统进行补充问诊分析")
            diagnosis_result = run_tcm_graph(
                user_input=combined_inquiry,
                config={"retriever_k": 5}  # 补充分析使用更多检索结果
            )
            
            # 获取分析结果
            final_results = diagnosis_result.get("response", "")
            
            # 如果结果为空，尝试从其他字段获取信息
            if not final_results.strip():
                diagnosis_data = diagnosis_result.get("diagnosis_data", {})
                prescription_data = diagnosis_result.get("prescription_data", {})
                
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
                
                final_results = "。".join(result_parts) if result_parts else combined_inquiry
            
            logger.info("补充问诊分析完成")
            
            return 200, {
                "success": True,
                "message": "补充问诊信息成功",
                "data": {
                    "results": final_results
                }
            }
            
        except Exception as e:
            error_msg = f"补充问诊分析失败: {str(e)}"
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
    
    def handle_inquiry_request(self):
        """
        处理初步问诊请求 (接口5.1)
        POST /api/inquiry
        """
        try:
            logger.info(f"收到初步问诊请求: {request.remote_addr}")
            
            # 获取JSON数据
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "success": False,
                    "message": "请求数据格式错误，需要JSON格式",
                    "data": {"results": ""}
                }), 400
            
            # 提取参数
            age = data.get('age')
            gender = data.get('gender', '')
            symptoms = data.get('symptoms', '')
            
            # 执行初步问诊分析
            status_code, result_data = self.initial_inquiry(age, gender, symptoms)
            
            return jsonify(result_data), status_code
            
        except Exception as e:
            error_msg = f"初步问诊请求处理异常: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return jsonify({
                "success": False,
                "message": error_msg,
                "data": {"results": ""}
            }), 500
    
    def handle_inquiry_complete_request(self):
        """
        处理补充问诊请求 (接口5.2)
        POST /api/inquiry/complete
        """
        try:
            logger.info(f"收到补充问诊请求: {request.remote_addr}")
            
            # 获取form-data参数
            prev_inquiry = request.form.get('prevInquiry', '').strip()
            additional_info = request.form.get('additionalInfo', '').strip()
            
            # 检查必要参数
            if not prev_inquiry and not additional_info:
                return jsonify({
                    "success": False,
                    "message": "必须提供之前的问诊结果或补充信息",
                    "data": {"results": ""}
                }), 400
            
            # 获取可选的检查报告文件
            additional_file = None
            if 'additionalFile' in request.files:
                additional_file = request.files['additionalFile']
                # 如果文件名为空，视为未上传
                if additional_file.filename == '':
                    additional_file = None
            
            # 执行补充问诊分析
            status_code, result_data = self.complete_inquiry(
                prev_inquiry, additional_info, additional_file
            )
            
            return jsonify(result_data), status_code
            
        except Exception as e:
            error_msg = f"补充问诊请求处理异常: {str(e)}"
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
            "description": "基于中医理论的问诊分析API，支持初步问诊和补充问诊",
            "endpoints": {
                "inquiry": {
                    "method": "POST",
                    "path": "/api/inquiry",
                    "content_type": "application/json",
                    "body": {
                        "age": "年龄（数字）",
                        "gender": "性别（男/女）",
                        "symptoms": "症状描述"
                    },
                    "description": "初步问诊分析，接口编号5.1"
                },
                "inquiry_complete": {
                    "method": "POST",
                    "path": "/api/inquiry/complete",
                    "content_type": "multipart/form-data",
                    "params": {
                        "prevInquiry": "之前的问诊分析结果",
                        "additionalInfo": "补充信息",
                        "additionalFile": "检查报告文件（可选）"
                    },
                    "description": "补充问诊分析，接口编号5.2"
                }
            },
            "supported_file_formats": list(self.allowed_extensions),
            "max_file_size": "20MB",
            "status": "active"
        }