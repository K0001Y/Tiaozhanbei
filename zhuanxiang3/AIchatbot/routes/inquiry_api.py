"""
中医问诊分析API模块 - Linus修复版
实现中医问诊分析和补充问诊功能，带智能状态管理
"""
import logging
import json
import os
import tempfile
import traceback
import hashlib
import pickle
import time
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
            "prescription_data": {"方剂": "四君子汤加减"},
            "user_input": user_input,
            "messages": messages or [],
            "memory": memory,
            "config": config or {"retriever_k": 4}
        }

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Linus式Session管理器 - 基于内容Hash的智能状态管理
    用户不知道有session，但我们在后台维护状态
    """
    
    def __init__(self, ttl_seconds=3600):  # 1小时TTL
        self._sessions = {}  # session_id -> state
        self._ttl = ttl_seconds
        self._last_cleanup = time.time()
    
    def _cleanup_expired_sessions(self):
        """清理过期的session"""
        current_time = time.time()
        if current_time - self._last_cleanup < 300:  # 5分钟清理一次
            return
            
        expired_sessions = []
        for session_id, data in self._sessions.items():
            if current_time - data['created_at'] > self._ttl:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self._sessions[session_id]
        
        self._last_cleanup = current_time
        logger.info(f"清理了 {len(expired_sessions)} 个过期session")
    
    def _generate_session_id(self, base_info: str) -> str:
        """基于基础信息生成session ID"""
        # 使用内容hash作为session标识，相同内容=相同session
        return hashlib.md5(base_info.encode('utf-8')).hexdigest()[:16]
    
    def get_or_create_session(self, base_info: str, initial_state: Dict = None) -> str:
        """获取或创建session"""
        self._cleanup_expired_sessions()
        
        session_id = self._generate_session_id(base_info)
        
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                'state': initial_state or {},
                'created_at': time.time(),
                'updated_at': time.time()
            }
            logger.info(f"创建新session: {session_id}")
        else:
            logger.info(f"复用已存在session: {session_id}")
        
        return session_id
    
    def get_session_state(self, session_id: str) -> Optional[Dict]:
        """获取session状态"""
        if session_id in self._sessions:
            return self._sessions[session_id]['state']
        return None
    
    def update_session_state(self, session_id: str, new_state: Dict):
        """更新session状态"""
        if session_id in self._sessions:
            self._sessions[session_id]['state'] = new_state
            self._sessions[session_id]['updated_at'] = time.time()
        else:
            logger.warning(f"尝试更新不存在的session: {session_id}")
    
    def extract_session_from_prev_inquiry(self, prev_inquiry: str) -> Optional[str]:
        """从之前的问诊结果中提取可能的session标识"""
        # 这是个聪明的hack：从之前的分析结果中反推可能的session
        # 我们寻找特征性的内容来匹配已有session
        
        for session_id, data in self._sessions.items():
            state = data['state']
            if 'response' in state:
                # 如果当前输入包含之前的响应内容，很可能是同一个session
                if prev_inquiry in state['response'] or state['response'] in prev_inquiry:
                    logger.info(f"从prev_inquiry中识别出session: {session_id}")
                    return session_id
        
        return None

# 全局session管理器
session_manager = SessionManager()

class MedicalInquiryAPI:
    """中医问诊分析API类 - Linus修复版"""
    
    def __init__(self):
        """初始化中医问诊分析API"""
        try:
            logger.info("初始化中医问诊分析API模块")
            
            self.api_name = "中医问诊分析API"
            self.version = "2.0.0"  # Linus修复版
            
            # 支持的图像格式（用于补充问诊的检查报告）
            self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'pdf'}
            
            logger.info("中医问诊分析API模块初始化完成")
            
        except Exception as e:
            logger.error(f"中医问诊分析API模块初始化失败: {str(e)}")
            raise
    
    def _validate_inquiry_params(self, age: Any, gender: str, symptoms: str) -> tuple[bool, str]:
        """
        验证问诊参数 - 保持原有逻辑
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
        """验证上传的检查报告文件 - 保持原有逻辑"""
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
        """保存临时文件并返回文件路径 - 保持原有逻辑"""
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
        """清理临时文件 - 保持原有逻辑"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"清理临时文件失败: {file_path} - {str(e)}")
    
    def _normalize_gender(self, gender: str) -> str:
        """标准化性别表示 - 保持原有逻辑"""
        gender = gender.strip().lower()
        if gender in ['男', '男性', 'male', 'm']:
            return '男'
        elif gender in ['女', '女性', 'female', 'f']:
            return '女'
        else:
            return gender  # 保持原样，让验证函数处理
    
    def _build_inquiry_text(self, age: int, gender: str, symptoms: str) -> str:
        """构建问诊输入文本 - 保持原有逻辑"""
        # 标准化性别
        normalized_gender = self._normalize_gender(gender)
        
        # 构建基本信息
        inquiry_text = f"患者基本信息：{age}岁，{normalized_gender}性。"
        
        # 添加症状描述
        inquiry_text += f"主要症状：{symptoms.strip()}。"
        
        # 添加问诊提示
        inquiry_text += "请根据中医理论进行辨证分析，提供诊疗建议。"
        
        return inquiry_text
    
    def _extract_result_from_state(self, state: Dict) -> str:
        """从graph state中提取最终结果"""
        # 首先尝试获取response
        final_results = state.get("response", "")
        
        # 如果结果为空，尝试从其他字段获取信息
        if not final_results.strip():
            diagnosis_data = state.get("diagnosis_data", {})
            prescription_data = state.get("prescription_data", {})
            
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
            
            final_results = "。".join(result_parts) if result_parts else ""
        
        return final_results
    
    def initial_inquiry(self, age: Any, gender: str, symptoms: str) -> tuple[int, Dict[str, Any]]:
        """
        初步问诊分析 - Linus修复版：引入智能状态管理
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
            
            # 生成session标识 - 基于基础信息
            base_info = f"{age}_{gender}_{symptoms[:50]}"  # 使用前50字符避免过长
            session_id = session_manager.get_or_create_session(base_info)
            
            # 检查是否已有计算结果
            existing_state = session_manager.get_session_state(session_id)
            if existing_state and existing_state.get("response"):
                logger.info(f"复用已有计算结果，session: {session_id}")
                final_results = self._extract_result_from_state(existing_state)
                
                return 200, {
                    "success": True,
                    "message": "问诊分析成功",
                    "data": {
                        "results": final_results
                    }
                }
            
            # 调用诊断系统进行分析
            logger.info("调用诊断系统进行问诊分析")
            diagnosis_result = run_tcm_graph(
                user_input=inquiry_text,
                config={"retriever_k": 4}
            )
            
            # 保存完整的graph state
            session_manager.update_session_state(session_id, diagnosis_result)
            
            # 提取结果
            final_results = self._extract_result_from_state(diagnosis_result)
            
            if not final_results.strip():
                final_results = inquiry_text  # 兜底策略
            
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
        补充问诊分析 - Linus修复版：智能状态恢复，避免重复计算
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
            
            # 智能识别已有session
            session_id = None
            existing_state = None
            
            if prev_inquiry and prev_inquiry.strip():
                # 尝试从之前的问诊结果中识别session
                session_id = session_manager.extract_session_from_prev_inquiry(prev_inquiry.strip())
                if session_id:
                    existing_state = session_manager.get_session_state(session_id)
                    logger.info(f"识别到已有session: {session_id}")
            
            # 构建补充信息
            additional_parts = []
            if additional_info and additional_info.strip():
                additional_parts.append(f"补充信息：{additional_info.strip()}")
            
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
                    additional_parts.append("已上传PDF格式的检查报告，请结合报告内容进行综合分析。")
                else:
                    additional_parts.append("已上传检查报告图片，请结合图像信息进行综合分析。")
            
            if not additional_parts:
                return 400, {
                    "success": False,
                    "message": "补充信息为空",
                    "data": {"results": ""}
                }
            
            # 构建增量输入
            incremental_input = "。".join(additional_parts)
            incremental_input += "。请在之前分析的基础上，结合新的补充信息，提供更完善的中医辨证分析和诊疗建议。"
            
            logger.info(f"补充问诊增量输入: {incremental_input}")
            
            if existing_state:
                # 【Linus式核心修复】使用已有state进行增量计算，避免重复劳动
                logger.info("基于已有state进行增量分析")
                
                # 将补充信息添加到messages中
                updated_messages = existing_state.get("messages", [])
                updated_messages.append({"role": "user", "content": incremental_input})
                
                # 使用已有状态进行增量计算
                diagnosis_result = run_tcm_graph(
                    user_input=incremental_input,
                    messages=updated_messages,
                    memory=existing_state.get("memory"),
                    config=existing_state.get("config", {"retriever_k": 5})
                )
            else:
                # 如果没有找到已有状态，只能重新计算（但记录警告）
                logger.warning("未找到已有session，执行完整计算")
                
                combined_inquiry = prev_inquiry + "。" + incremental_input if prev_inquiry else incremental_input
                
                diagnosis_result = run_tcm_graph(
                    user_input=combined_inquiry,
                    config={"retriever_k": 5}
                )
                
                # 为新的计算创建session
                session_id = session_manager.get_or_create_session(
                    f"complete_{hash(combined_inquiry)}", 
                    diagnosis_result
                )
            
            # 更新session状态
            if session_id:
                session_manager.update_session_state(session_id, diagnosis_result)
            
            # 提取最终结果
            final_results = self._extract_result_from_state(diagnosis_result)
            
            if not final_results.strip():
                final_results = prev_inquiry + "。" + incremental_input  # 兜底策略
            
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
        处理初步问诊请求 (接口5.1) - 保持接口不变
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
        处理补充问诊请求 (接口5.2) - 保持接口不变
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
        获取API信息 - 保持原有接口
        """
        return {
            "name": self.api_name,
            "version": self.version,
            "description": "基于中医理论的问诊分析API，支持初步问诊和补充问诊 - Linus修复版",
            "features": [
                "智能状态管理，避免重复计算",
                "基于内容Hash的session识别",
                "增量计算，复用已有结果", 
                "自动过期清理，防止内存泄漏"
            ],
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