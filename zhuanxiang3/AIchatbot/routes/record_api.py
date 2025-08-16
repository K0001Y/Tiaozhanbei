"""
病历生成API模块 - 严格匹配API文档版本
修改要点:
1. 只接收 watchResults 和 inquiryResults 两个参数
2. 移除 patientInfo 和 smartMode 功能
3. 保持智能Session聚合的核心逻辑
"""
import logging
import json
import traceback
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
from flask import request, jsonify

# 导入session管理器（从之前修复的模块）
try:
    # 导入全局session管理器
    from routes.inquiry_api import session_manager as inquiry_session_manager
    from routes.watch_api import watch_session_manager
except ImportError:
    # 如果导入失败，创建模拟的session管理器
    print("警告: 无法导入session管理器，使用模拟版本")
    
    class MockSessionManager:
        def __init__(self):
            self._sessions = {}
        
        def extract_session_from_prev_inquiry(self, text):
            return None
        
        def get_session_state(self, session_id):
            return None
        
        def get_all_sessions(self):
            return []
    
    inquiry_session_manager = MockSessionManager()
    watch_session_manager = MockSessionManager()

logger = logging.getLogger(__name__)

class MedicalRecord:
    """
    结构化病历数据 - 消除字符串拼接的垃圾做法
    """
    
    def __init__(self):
        self.symptoms = []  # 症状列表
        self.diagnosis_findings = []  # 诊断发现
        self.prescriptions = []  # 处方建议
        self.watch_analysis = ""  # 望诊分析
        self.inquiry_analysis = ""  # 问诊分析
        self.confidence_score = 0.0  # 综合置信度
        self.data_sources = []  # 数据来源记录
    
    @classmethod
    def from_session_aggregation(cls, watch_sessions: List[Dict], inquiry_sessions: List[Dict]) -> 'MedicalRecord':
        """
        从session数据聚合创建病历 - 核心智能聚合逻辑（移除patientInfo参数）
        """
        record = cls()
        
        # 聚合望诊数据
        for session_data in watch_sessions:
            if 'diagnosis_data' in session_data and session_data['diagnosis_data']:
                diagnosis_data = session_data['diagnosis_data']
                
                # 提取症状信息
                if hasattr(diagnosis_data, 'syndromes') and diagnosis_data.syndromes:
                    record.symptoms.extend(diagnosis_data.syndromes)
                
                # 提取诊断信息
                if hasattr(diagnosis_data, 'theory_basis') and diagnosis_data.theory_basis:
                    record.diagnosis_findings.append(f"望诊发现：{diagnosis_data.theory_basis}")
                
                # 提取建议
                if hasattr(diagnosis_data, 'health_advice') and diagnosis_data.health_advice:
                    record.prescriptions.extend(diagnosis_data.health_advice)
                
                # 记录望诊分析
                if hasattr(diagnosis_data, 'to_diagnosis_text'):
                    record.watch_analysis = diagnosis_data.to_diagnosis_text()
                
                # 更新置信度
                if hasattr(diagnosis_data, 'confidence'):
                    record.confidence_score = max(record.confidence_score, diagnosis_data.confidence)
                
                record.data_sources.append("望诊Session")
        
        # 聚合问诊数据
        for session_data in inquiry_sessions:
            if 'state' in session_data:
                state = session_data['state']
                
                # 从graph state中提取数据
                if 'symptoms_list' in state and state['symptoms_list']:
                    symptoms_list = state['symptoms_list']
                    if isinstance(symptoms_list, list):
                        for symptom in symptoms_list:
                            if isinstance(symptom, dict):
                                symptom_text = symptom.get('name', str(symptom))
                            else:
                                symptom_text = str(symptom)
                            record.symptoms.append(symptom_text)
                    else:
                        record.symptoms.append(str(symptoms_list))
                
                # 提取诊断数据
                if 'diagnosis_data' in state and state['diagnosis_data']:
                    diagnosis_data = state['diagnosis_data']
                    if isinstance(diagnosis_data, dict):
                        for key, value in diagnosis_data.items():
                            if value:
                                record.diagnosis_findings.append(f"问诊{key}：{value}")
                    else:
                        record.diagnosis_findings.append(f"问诊诊断：{diagnosis_data}")
                
                # 提取处方数据
                if 'prescription_data' in state and state['prescription_data']:
                    prescription_data = state['prescription_data']
                    if isinstance(prescription_data, dict):
                        for key, value in prescription_data.items():
                            if value:
                                if isinstance(value, list):
                                    record.prescriptions.extend(value)
                                else:
                                    record.prescriptions.append(str(value))
                    else:
                        record.prescriptions.append(str(prescription_data))
                
                # 记录问诊分析
                if 'response' in state and state['response']:
                    record.inquiry_analysis = state['response']
                
                record.data_sources.append("问诊Session")
        
        # 去重和清理
        record.symptoms = list(dict.fromkeys([s for s in record.symptoms if s.strip()]))
        record.diagnosis_findings = list(dict.fromkeys([d for d in record.diagnosis_findings if d.strip()]))
        record.prescriptions = list(dict.fromkeys([p for p in record.prescriptions if p.strip()]))
        
        return record
    
    @classmethod
    def from_user_input(cls, watch_results: str, inquiry_results: str) -> 'MedicalRecord':
        """
        从用户输入创建病历 - 向后兼容的降级方案（移除patientInfo参数）
        """
        record = cls()
        record.watch_analysis = watch_results
        record.inquiry_analysis = inquiry_results
        
        # 简单解析用户提供的数据
        if watch_results:
            record.diagnosis_findings.append(f"望诊分析：{watch_results}")
            record.data_sources.append("用户提供的望诊结果")
        
        if inquiry_results:
            record.diagnosis_findings.append(f"问诊分析：{inquiry_results}")
            record.data_sources.append("用户提供的问诊结果")
        
        return record
    
    def to_api_response(self) -> Dict[str, str]:
        """
        转换为API响应格式 - 保持向后兼容
        """
        # 构建症状描述
        if self.symptoms:
            symptoms_text = "主诉：" + "、".join(self.symptoms) + "。"
        else:
            symptoms_text = "主诉：暂无明确症状。"
        
        # 构建疾病描述
        if self.diagnosis_findings:
            disease_text = "。".join(self.diagnosis_findings) + "。"
        else:
            disease_text = "暂无明确诊断。"
        
        # 构建处方描述
        if self.prescriptions:
            prescription_text = "建议：" + "、".join(self.prescriptions) + "。"
        else:
            prescription_text = "暂无特殊建议。"
        
        return {
            "symptoms": symptoms_text,
            "disease": disease_text,
            "prescription": prescription_text
        }

class RecordSessionAggregator:
    """
    Session数据聚合器 - 智能搜索和聚合相关session
    """
    
    @staticmethod
    def find_related_sessions(watch_results: str, inquiry_results: str) -> Tuple[List[Dict], List[Dict]]:
        """
        查找相关的望诊和问诊session（移除patientInfo参数）
        """
        watch_sessions = []
        inquiry_sessions = []
        
        # 1. 尝试从inquiry_results中识别问诊session
        if inquiry_results and hasattr(inquiry_session_manager, 'extract_session_from_prev_inquiry'):
            inquiry_session_id = inquiry_session_manager.extract_session_from_prev_inquiry(inquiry_results)
            if inquiry_session_id:
                inquiry_session_data = inquiry_session_manager.get_session_state(inquiry_session_id)
                if inquiry_session_data:
                    inquiry_sessions.append(inquiry_session_data)
                    logger.info(f"找到相关问诊session: {inquiry_session_id}")
        
        # 2. 尝试从watch_results中识别望诊session
        if watch_results and hasattr(watch_session_manager, 'extract_session_from_analysis'):
            watch_session_id = watch_session_manager.extract_session_from_analysis(watch_results)
            if watch_session_id:
                watch_session_data = watch_session_manager.get_session_data(watch_session_id)
                if watch_session_data:
                    watch_sessions.append(watch_session_data)
                    logger.info(f"找到相关望诊session: {watch_session_id}")
        
        # 3. 如果没找到特定session，尝试获取最近的session（时间窗口内）
        if not watch_sessions:
            recent_watch_sessions = RecordSessionAggregator._get_recent_sessions(
                watch_session_manager, time_window=3600  # 1小时内
            )
            watch_sessions.extend(recent_watch_sessions[:2])  # 最多取2个最近的
        
        if not inquiry_sessions:
            recent_inquiry_sessions = RecordSessionAggregator._get_recent_sessions(
                inquiry_session_manager, time_window=3600
            )
            inquiry_sessions.extend(recent_inquiry_sessions[:2])
        
        logger.info(f"聚合到 {len(watch_sessions)} 个望诊session 和 {len(inquiry_sessions)} 个问诊session")
        return watch_sessions, inquiry_sessions
    
    @staticmethod
    def _get_recent_sessions(session_manager, time_window: int) -> List[Dict]:
        """
        获取最近的session
        """
        if not hasattr(session_manager, '_sessions'):
            return []
        
        current_time = time.time()
        recent_sessions = []
        
        for session_id, session_data in session_manager._sessions.items():
            if 'updated_at' in session_data:
                time_diff = current_time - session_data['updated_at']
                if time_diff <= time_window:
                    recent_sessions.append(session_data)
        
        # 按更新时间排序，最新的在前
        recent_sessions.sort(key=lambda x: x.get('updated_at', 0), reverse=True)
        return recent_sessions

class RecordAPI:
    """
    病历生成API - 严格匹配文档版本
    """
    
    def __init__(self):
        """初始化病历生成API"""
        self.api_name = "病历生成API"
        self.version = "2.1.0"  # 严格匹配文档版本
        logger.info("病历生成API模块初始化完成 - 严格匹配API文档")
    
    def generate_medical_record(self, watch_results: str, inquiry_results: str) -> tuple[int, Dict[str, Any]]:
        """
        生成病历报告 - 简化版本，严格按照API文档
        
        参数验证：
        - watchResults 和 inquiryResults 至少需要提供一个
        """
        try:
            logger.info("开始病历生成")
            
            # 验证输入参数
            if not watch_results.strip() and not inquiry_results.strip():
                return 400, {
                    "success": False,
                    "message": "watchResults 和 inquiryResults 至少需要提供一个",
                    "data": {
                        "symptoms": "",
                        "disease": "",
                        "prescription": ""
                    }
                }
            
            # 步骤1：尝试从session系统聚合数据
            watch_sessions, inquiry_sessions = RecordSessionAggregator.find_related_sessions(
                watch_results, inquiry_results
            )
            
            # 步骤2：选择数据源和生成策略
            if watch_sessions or inquiry_sessions:
                # 智能聚合方案：从session数据生成病历
                logger.info("使用session聚合数据生成病历")
                medical_record = MedicalRecord.from_session_aggregation(
                    watch_sessions, inquiry_sessions
                )
                
                # 如果session数据不足，用用户提供的数据补充
                if not medical_record.watch_analysis and watch_results:
                    medical_record.watch_analysis = watch_results
                    medical_record.diagnosis_findings.append(f"补充望诊：{watch_results}")
                
                if not medical_record.inquiry_analysis and inquiry_results:
                    medical_record.inquiry_analysis = inquiry_results
                    medical_record.diagnosis_findings.append(f"补充问诊：{inquiry_results}")
                
            else:
                # 降级方案：从用户输入生成病历
                logger.info("未找到相关session，使用用户提供数据生成病历")
                medical_record = MedicalRecord.from_user_input(watch_results, inquiry_results)
            
            # 步骤3：生成API响应
            api_response = medical_record.to_api_response()
            
            result = {
                "success": True,
                "message": "病历生成成功",
                "data": api_response
            }
            
            logger.info("病历生成成功")
            return 200, result
            
        except Exception as e:
            error_msg = f"病历生成失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return 500, {
                "success": False,
                "message": error_msg,
                "data": {
                    "symptoms": "病历生成失败",
                    "disease": "病历生成失败",
                    "prescription": "病历生成失败"
                }
            }
    
    def handle_record_request(self):
        """
        处理病历生成请求 - 严格按照API文档接收参数
        POST /api/record
        """
        try:
            logger.info(f"收到病历生成请求: {request.remote_addr}")
            
            # 获取JSON数据
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "success": False,
                    "message": "请求数据格式错误，需要JSON格式",
                    "data": {
                        "symptoms": "",
                        "disease": "",
                        "prescription": ""
                    }
                }), 400
            
            # 严格按照API文档只接收这两个参数
            watch_results = data.get('watchResults', '')
            inquiry_results = data.get('inquiryResults', '')
            
            # 执行病历生成
            status_code, result = self.generate_medical_record(watch_results, inquiry_results)
            
            return jsonify(result), status_code
            
        except Exception as e:
            error_msg = f"病历生成请求处理失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return jsonify({
                "success": False,
                "message": error_msg,
                "data": {
                    "symptoms": "",
                    "disease": "",
                    "prescription": ""
                }
            }), 500
    
    def get_api_info(self) -> Dict[str, Any]:
        """获取API信息 - 更新为严格匹配文档版本"""
        return {
            "name": self.api_name,
            "version": self.version,
            "description": "病历生成API - 严格匹配API文档版本",
            "features": [
                "智能Session数据聚合，自动查找相关诊断记录",
                "优先使用session数据，降级到用户输入",
                "结构化病历数据处理，消除字符串拼接",
                "严格按照API文档接收参数"
            ],
            "endpoints": {
                "record": {
                    "method": "POST",
                    "path": "/api/record",
                    "content_type": "application/json",
                    "body": {
                        "watchResults": "望诊分析结果（可选，但与inquiryResults至少提供一个）",
                        "inquiryResults": "问诊分析结果（可选，但与watchResults至少提供一个）"
                    },
                    "description": "根据session聚合数据生成病历报告，严格匹配API文档格式",
                    "response_format": {
                        "success": "bool",
                        "message": "string",
                        "data": {
                            "symptoms": "主诉症状信息",
                            "disease": "疾病诊断信息",
                            "prescription": "处方建议信息"
                        }
                    }
                }
            },
            "parameter_validation": {
                "required": "watchResults 和 inquiryResults 至少需要提供一个",
                "removed_features": ["patientInfo参数", "smartMode智能模式"]
            },
            "session_integration": {
                "inquiry_sessions": "已集成" if hasattr(inquiry_session_manager, '_sessions') else "未集成",
                "watch_sessions": "已集成" if hasattr(watch_session_manager, '_sessions') else "未集成"
            },
            "status": "active"
        }