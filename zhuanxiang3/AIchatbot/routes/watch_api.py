"""
医学图像分析API模块 - 简化版望诊补充逻辑
直接操作state字典，采用问诊补充的合并策略
"""
import logging
import json
import os
import tempfile
import traceback
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from flask import request, jsonify
from werkzeug.datastructures import FileStorage
from config import ALI_BASE_URL, ALI_API_KEY

# 导入优化后的图像识别和诊断系统
try:
    from others.watch import TCMDiagnosisSystem
    from graph import run_tcm_graph
except ImportError:
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
            "diagnosis_data": {"证型": "脾胃虚弱", "建议": "调理脾胃"},
            "user_input": user_input,
            "messages": messages or [],
            "memory": memory,
            "config": config or {"retriever_k": 4}
        }

logger = logging.getLogger(__name__)

class WatchSessionManager:
    """
    望诊专用Session管理器 - 简化版，与问诊逻辑一致
    """
    
    def __init__(self, ttl_seconds=3600):
        self._sessions = {}  # session_id -> session_data
        self._ttl = ttl_seconds
        self._last_cleanup = time.time()
    
    def _cleanup_expired_sessions(self):
        """清理过期session"""
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
        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期望诊session")
    
    def _generate_image_session_id(self, image_path: str, description: str = "") -> str:
        """基于图像内容生成session ID"""
        hasher = hashlib.md5()
        try:
            with open(image_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        except:
            # 如果读取失败，使用文件路径
            hasher.update(image_path.encode('utf-8'))
        
        # 加入描述信息
        if description:
            hasher.update(description.encode('utf-8'))
        
        return f"watch_{hasher.hexdigest()[:16]}"
    
    def get_or_create_image_session(self, image_path: str, description: str = "") -> str:
        """获取或创建图像session"""
        self._cleanup_expired_sessions()
        
        session_id = self._generate_image_session_id(image_path, description)
        
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                'state': {},  # 改为直接存储state字典
                'created_at': time.time(),
                'updated_at': time.time(),
                'type': 'image'
            }
            logger.info(f"创建新图像session: {session_id}")
        else:
            logger.info(f"复用已存在图像session: {session_id}")
        
        return session_id
    
    def extract_session_from_analysis(self, prev_analysis: str) -> Optional[str]:
        """从分析结果中提取session - 简化版，类似问诊逻辑"""
        for session_id, data in self._sessions.items():
            state = data.get('state', {})
            if 'response' in state:
                # 检查分析结果是否匹配（简化的包含关系判断）
                if prev_analysis in state['response'] or state['response'] in prev_analysis:
                    logger.info(f"从分析结果中识别出session: {session_id}")
                    return session_id
        return None
    
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

# 全局望诊session管理器
watch_session_manager = WatchSessionManager()

class MedicalImageAPI:
    """医学图像分析API类 - 简化版"""
    
    def __init__(self):
        """初始化医学图像分析API"""
        try:
            logger.info("初始化医学图像分析API模块")
            
            # 初始化图像识别系统
            api_key = ALI_API_KEY
            base_url = ALI_BASE_URL
            
            self.tcm_system = TCMDiagnosisSystem(api_key=api_key, base_url=base_url)
            
            self.api_name = "医学图像分析API"
            self.version = "2.1.0"  # 简化版
            
            # 支持的图像格式
            self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
            
            logger.info("医学图像分析API模块初始化完成")
            
        except Exception as e:
            logger.error(f"医学图像分析API模块初始化失败: {str(e)}")
            raise
    
    def _validate_image_file(self, file: FileStorage) -> tuple[bool, str]:
        """验证上传的图像文件"""
        if not file:
            return False, "未上传文件"
        
        if file.filename == '':
            return False, "文件名为空"
        
        if '.' not in file.filename:
            return False, "文件没有扩展名"
        
        ext = file.filename.rsplit('.', 1)[1].lower()
        if ext not in self.allowed_extensions:
            return False, f"不支持的文件格式，支持的格式: {', '.join(self.allowed_extensions)}"
        
        # 检查文件大小（限制为10MB）
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            return False, "文件大小超过10MB限制"
        
        if file_size == 0:
            return False, "文件为空"
        
        return True, ""
    
    def _save_temp_file(self, file: FileStorage) -> str:
        """保存临时文件"""
        suffix = '.' + file.filename.rsplit('.', 1)[1].lower()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        try:
            file.save(temp_file.name)
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
        except Exception as e:
            logger.warning(f"清理临时文件失败: {file_path} - {str(e)}")
    
    def _analyze_image_and_get_basic_result(self, image_path: str) -> str:
        """分析图像并返回基础结果文本 - 简化版"""
        logger.info(f"调用图像识别系统分析: {image_path}")
        
        try:
            # 使用base64编码传递图像
            import base64
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
                base64_url = f"data:image/jpeg;base64,{image_data}"
                
            logger.info("使用base64编码传递图像")
            image_analysis_result = self.tcm_system.comprehensive_diagnosis(base64_url)
            
        except Exception as e:
            logger.warning(f"base64方式失败: {e}")
            try:
                logger.info("尝试直接文件路径")
                image_analysis_result = self.tcm_system.comprehensive_diagnosis(image_path)
            except Exception as e2:
                logger.error(f"所有图像传递方式都失败: {e2}")
                return f"图像识别失败: {str(e2)}"
        
        # 简单的文本组装，不使用复杂的DiagnosisData类
        image_info = image_analysis_result.get("图像识别", {})
        image_type = image_info.get("类型", "未知")
        confidence = image_info.get("置信度", 0.0)
        
        # 解析分析结果
        analysis_content = image_analysis_result.get("分析结果", "")
        try:
            if isinstance(analysis_content, str):
                parsed_content = json.loads(analysis_content)
            else:
                parsed_content = analysis_content
        except (json.JSONDecodeError, TypeError):
            parsed_content = {"原始结果": str(analysis_content)}
        
        # 简单文本组装
        result_parts = [f"图像类型：{image_type}（置信度：{confidence:.1%}）"]
        
        if "辨证提示" in parsed_content:
            syndromes = parsed_content["辨证提示"]
            if syndromes:
                result_parts.append(f"辨证提示：{'; '.join(syndromes)}")
        
        if "健康建议" in parsed_content:
            advice = parsed_content["健康建议"]
            if advice:
                result_parts.append(f"健康建议：{'; '.join(advice)}")
        
        return "。".join(result_parts)
    
    def _extract_result_from_state(self, state: Dict) -> str:
        """从graph state中提取最终结果 - 与问诊逻辑一致"""
        # 首先尝试获取response
        final_results = state.get("response", "")
        
        # 如果结果为空，尝试从其他字段获取信息
        if not final_results.strip():
            diagnosis_data = state.get("diagnosis_data", {})
            
            result_parts = []
            if diagnosis_data:
                if isinstance(diagnosis_data, dict):
                    for key, value in diagnosis_data.items():
                        result_parts.append(f"{key}：{value}")
                else:
                    result_parts.append(str(diagnosis_data))
            
            final_results = "。".join(result_parts) if result_parts else ""
        
        return final_results
    
    def analyze_image(self, image_file: FileStorage, description: str = "") -> tuple[int, Dict[str, Any]]:
        """
        图片望诊分析 - 简化版
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
            
            # 生成session ID
            session_id = watch_session_manager.get_or_create_image_session(temp_file_path, description)
            
            # 检查是否已有计算结果
            existing_state = watch_session_manager.get_session_state(session_id)
            if existing_state and existing_state.get('response'):
                logger.info(f"复用已有望诊分析结果，session: {session_id}")
                final_results = self._extract_result_from_state(existing_state)
                
                return 200, {
                    "success": True,
                    "message": "望诊分析成功",
                    "data": {"results": final_results}
                }
            
            # 分析图像获取基础结果
            basic_result = self._analyze_image_and_get_basic_result(temp_file_path)
            
            # 构建完整输入
            full_input = basic_result
            if description.strip():
                full_input = f"图像描述：{description}。{basic_result}"
            
            # 使用graph进行增强诊断
            logger.info("调用graph进行增强诊断")
            diagnosis_result = run_tcm_graph(
                user_input=full_input,
                config={"retriever_k": 3}
            )
            
            # 保存session状态
            watch_session_manager.update_session_state(session_id, diagnosis_result)
            
            # 提取最终结果
            final_results = self._extract_result_from_state(diagnosis_result)
            
            logger.info("图片望诊分析完成")
            
            return 200, {
                "success": True,
                "message": "望诊分析成功",
                "data": {"results": final_results}
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
            if temp_file_path:
                self._cleanup_temp_file(temp_file_path)
    
    def analyze_with_supplement(self, prev_analysis: str, additional_info: str, 
                               additional_file: Optional[FileStorage] = None) -> tuple[int, Dict[str, Any]]:
        """
        望诊补充分析 - 简化版，采用问诊补充的合并策略
        """
        temp_file_path = None
        
        try:
            logger.info("开始望诊补充分析")
            
            # 智能识别已有session
            session_id = None
            existing_state = None
            
            if prev_analysis and prev_analysis.strip():
                # 尝试从之前的分析结果中识别session
                session_id = watch_session_manager.extract_session_from_analysis(prev_analysis.strip())
                if session_id:
                    existing_state = watch_session_manager.get_session_state(session_id)
                    logger.info(f"识别到已有session: {session_id}")
            
            # 构建补充信息
            additional_parts = []
            if additional_info and additional_info.strip():
                additional_parts.append(f"补充信息：{additional_info.strip()}")
            
            # 处理补充图像（如果有）
            if additional_file:
                is_valid, error_msg = self._validate_image_file(additional_file)
                if not is_valid:
                    return 400, {
                        "success": False,
                        "message": f"补充图像文件错误: {error_msg}",
                        "data": {"results": ""}
                    }
                
                temp_file_path = self._save_temp_file(additional_file)
                
                # 分析补充图像
                additional_image_result = self._analyze_image_and_get_basic_result(temp_file_path)
                additional_parts.append(f"补充图像分析：{additional_image_result}")
            
            if not additional_parts:
                return 400, {
                    "success": False,
                    "message": "补充信息为空",
                    "data": {"results": ""}
                }
            
            # 构建增量输入 - 与问诊逻辑一致
            incremental_input = "。".join(additional_parts)
            incremental_input += "。请在之前分析的基础上，结合新的补充信息，提供更完善的中医辨证分析和诊疗建议。"
            
            logger.info(f"补充望诊增量输入: {incremental_input}")
            
            if existing_state:
                # 【与问诊逻辑一致】基于已有state进行增量计算
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
                # 如果没有找到已有状态，重新计算
                logger.warning("未找到已有session，执行完整计算")
                
                combined_inquiry = prev_analysis + "。" + incremental_input if prev_analysis else incremental_input
                
                diagnosis_result = run_tcm_graph(
                    user_input=combined_inquiry,
                    config={"retriever_k": 5}
                )
                
                # 为新的计算创建session
                session_id = watch_session_manager.get_or_create_image_session(
                    f"supplement_{hash(combined_inquiry)}", 
                    ""
                )
            
            # 更新session状态
            if session_id:
                watch_session_manager.update_session_state(session_id, diagnosis_result)
            
            # 提取最终结果
            final_results = self._extract_result_from_state(diagnosis_result)
            
            if not final_results.strip():
                final_results = prev_analysis + "。" + incremental_input  # 兜底策略
            
            logger.info("望诊补充分析完成")
            
            return 200, {
                "success": True,
                "message": "补充望诊信息成功",
                "data": {"results": final_results}
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
            if temp_file_path:
                self._cleanup_temp_file(temp_file_path)
    
    def handle_watch_request(self):
        """处理图片望诊请求 (接口4.1)"""
        try:
            logger.info(f"收到图片望诊请求: {request.remote_addr}")
            
            if 'image' not in request.files:
                return jsonify({
                    "success": False,
                    "message": "未找到上传的图像文件",
                    "data": {"results": ""}
                }), 400
            
            image_file = request.files['image']
            description = request.form.get('description', '')
            
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
        """处理望诊补充请求 (接口4.2)"""
        try:
            logger.info(f"收到望诊补充请求: {request.remote_addr}")
            
            prev_analysis = request.form.get('prevAnalysis', '').strip()
            additional_info = request.form.get('additionalInfo', '').strip()
            
            if not prev_analysis and not additional_info:
                return jsonify({
                    "success": False,
                    "message": "必须提供之前的分析结果或补充信息",
                    "data": {"results": ""}
                }), 400
            
            if not additional_info:
                return jsonify({
                    "success": False,
                    "message": "additionalInfo参数不能为空", 
                    "data": {"results": ""}
                }), 400
            
            additional_file = None
            if 'additionalFile' in request.files:
                additional_file = request.files['additionalFile']
                if additional_file.filename == '':
                    additional_file = None
            
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
        """获取API信息"""
        return {
            "name": self.api_name,
            "version": self.version,
            "description": "基于中医理论的医学图像分析API - 简化版，与问诊逻辑一致",
            "features": [
                "简化的Session管理，与问诊逻辑一致",
                "直接操作state字典，无复杂数据抽象",
                "统一的增量分析策略",
                "简化的文本合并逻辑"
            ],
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