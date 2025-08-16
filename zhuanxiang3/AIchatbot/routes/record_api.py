"""
病历生成API模块
专门处理从graph状态生成病历报告的功能
对应接口 6.1 病理报告生成
"""
import logging
from typing import Dict, Any, List, Optional
from flask import request, jsonify

logger = logging.getLogger(__name__)

class RecordAPI:
    """
    病历生成API模块
    专门从graph状态信息生成病历报告
    """
    
    def __init__(self, graph=None):
        """
        初始化病历生成API
        
        :param graph: 图状态管理器实例，用于获取分析结果
        """
        self.api_name = "病历生成API"
        self.version = "1.0.0"
        self.graph = graph  # 保存graph实例的引用
        logger.info("病历生成API模块初始化完成")
    
    def set_graph(self, graph):
        """
        设置graph实例
        
        :param graph: 图状态管理器实例
        """
        self.graph = graph
        logger.info("Graph实例已设置")
    
    def extract_data_from_state(self, state: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        """
        从graph state中提取symptoms_list、diagnosis_data、prescription_data
        
        :param state: graph运行后的状态信息
        :return: (HTTP状态码, 提取结果)
        """
        try:
            # 提取症状列表
            symptoms_list = state.get('symptoms_list', [])
            if isinstance(symptoms_list, list):
                symptoms_text = "主诉" + "、".join(symptoms_list) + "。" if symptoms_list else "暂无明确症状。"
            else:
                symptoms_text = str(symptoms_list) if symptoms_list else "暂无明确症状。"
            
            # 提取诊断数据
            diagnosis_data = state.get('diagnosis_data', {})
            if isinstance(diagnosis_data, dict):
                # 构建诊断文本
                diagnosis_parts = []
                
                # 加入望诊结果
                watch_info = diagnosis_data.get('watch_analysis', '')
                if watch_info:
                    diagnosis_parts.append(f"经望诊分析，{watch_info}")
                
                # 加入问诊结果
                inquiry_info = diagnosis_data.get('inquiry_analysis', '')
                if inquiry_info:
                    diagnosis_parts.append(f"问诊补充信息显示{inquiry_info}")
                
                # 如果有其他诊断信息
                other_diagnosis = diagnosis_data.get('final_diagnosis', '')
                if other_diagnosis:
                    diagnosis_parts.append(other_diagnosis)
                
                disease_text = "。".join(diagnosis_parts) + "。" if diagnosis_parts else "暂无明确诊断。"
            else:
                disease_text = str(diagnosis_data) if diagnosis_data else "暂无明确诊断。"
            
            # 提取处方数据
            prescription_data = state.get('prescription_data', {})
            if isinstance(prescription_data, dict):
                prescription_parts = []
                
                # 生活建议
                lifestyle_advice = prescription_data.get('lifestyle_advice', [])
                if isinstance(lifestyle_advice, list):
                    prescription_parts.extend(lifestyle_advice)
                elif lifestyle_advice:
                    prescription_parts.append(str(lifestyle_advice))
                
                # 药物建议
                medication_advice = prescription_data.get('medication', [])
                if isinstance(medication_advice, list):
                    prescription_parts.extend(medication_advice)
                elif medication_advice:
                    prescription_parts.append(str(medication_advice))
                
                # 其他建议
                other_advice = prescription_data.get('recommendations', [])
                if isinstance(other_advice, list):
                    prescription_parts.extend(other_advice)
                elif other_advice:
                    prescription_parts.append(str(other_advice))
                
                prescription_text = "建议" + "、".join(prescription_parts) + "。" if prescription_parts else "暂无特殊建议。"
            else:
                prescription_text = str(prescription_data) if prescription_data else "暂无特殊建议。"
            
            extracted_data = {
                "symptoms": symptoms_text,
                "disease": disease_text,
                "prescription": prescription_text
            }
            
            logger.info("成功从state中提取数据")
            return 200, extracted_data
            
        except Exception as e:
            logger.error(f"从state提取数据失败: {str(e)}")
            return 500, {
                "symptoms": "数据提取失败",
                "disease": "数据提取失败", 
                "prescription": "数据提取失败"
            }
    
    def generate_medical_record(self, patient_info: str, watch_results: str, inquiry_results: str) -> tuple[int, Dict[str, Any]]:
        """
        生成病历报告核心方法
        
        :param patient_info: 患者信息
        :param watch_results: 望诊结果
        :param inquiry_results: 问诊结果
        :return: (HTTP状态码, 响应数据)
        """
        try:
            if not self.graph:
                logger.error("Graph实例未设置")
                return 500, {
                    "success": False,
                    "message": "系统配置错误：Graph实例未初始化",
                    "data": {
                        "symptoms": "",
                        "disease": "",
                        "prescription": ""
                    }
                }
            
            # 从graph获取最终状态
            try:
                final_state = self.graph.get_final_state()
                if not final_state:
                    logger.warning("无法获取graph最终状态")
                    return 404, {
                        "success": False,
                        "message": "未找到分析结果，请先完成望诊和问诊流程",
                        "data": {
                            "symptoms": "",
                            "disease": "",
                            "prescription": ""
                        }
                    }
            except AttributeError:
                # 如果graph没有get_final_state方法，尝试其他可能的方法
                logger.warning("Graph实例没有get_final_state方法，尝试其他方式获取state")
                if hasattr(self.graph, 'state'):
                    final_state = self.graph.state
                elif hasattr(self.graph, 'get_state'):
                    final_state = self.graph.get_state()
                else:
                    logger.error("无法从Graph获取状态信息")
                    return 500, {
                        "success": False,
                        "message": "无法获取系统状态",
                        "data": {
                            "symptoms": "",
                            "disease": "",
                            "prescription": ""
                        }
                    }
            
            # 提取数据
            status_code, extracted_data = self.extract_data_from_state(final_state)
            
            if status_code == 200:
                result = {
                    "success": True,
                    "message": "病历生成成功",
                    "data": extracted_data
                }
                logger.info("病历生成成功")
                return 200, result
            else:
                return status_code, {
                    "success": False,
                    "message": "病历生成过程中出现错误",
                    "data": extracted_data
                }
                
        except Exception as e:
            logger.error(f"病历生成失败: {str(e)}")
            return 500, {
                "success": False,
                "message": f"病历生成失败: {str(e)}",
                "data": {
                    "symptoms": "",
                    "disease": "",
                    "prescription": ""
                }
            }
    
    def handle_record_request(self):
        """
        处理病历生成请求的入口方法
        接口 6.1: POST /api/record
        """
        try:
            # 获取POST请求的JSON数据
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "success": False,
                    "message": "请求数据格式错误",
                    "data": {
                        "symptoms": "",
                        "disease": "",
                        "prescription": ""
                    }
                }), 400
            
            patient_info = data.get('patientInfo', '')
            watch_results = data.get('watchResults', '')
            inquiry_results = data.get('inquiryResults', '')
            
            # 执行病历生成
            status_code, result = self.generate_medical_record(
                patient_info, watch_results, inquiry_results
            )
            
            return jsonify(result), status_code
            
        except Exception as e:
            logger.error(f"病历生成请求处理失败: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"请求处理失败: {str(e)}",
                "data": {
                    "symptoms": "",
                    "disease": "",
                    "prescription": ""
                }
            }), 500
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        获取API信息
        """
        return {
            "name": self.api_name,
            "version": self.version,
            "description": "病历生成API，根据graph状态信息生成完整病历报告",
            "endpoints": {
                "record": {
                    "method": "POST",
                    "path": "/api/record",
                    "body": {
                        "patientInfo": "患者基本信息",
                        "watchResults": "望诊分析结果",
                        "inquiryResults": "问诊分析结果"
                    },
                    "description": "根据望诊和问诊结果从graph状态生成完整病历报告",
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
            "status": "active",
            "graph_status": "已连接" if self.graph else "未连接"
        }