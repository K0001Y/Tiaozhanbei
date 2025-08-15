"""
新API模块模板示例
演示如何创建新的API模块并集成到主服务器中
"""
import logging
from typing import Dict, Any, List
from flask import request, jsonify

logger = logging.getLogger(__name__)

class DiagnosisAPI:
    """
    诊断API模块示例
    演示如何创建符合规范的API模块
    """
    
    def __init__(self):
        """初始化诊断API"""
        self.api_name = "智能诊断API"
        self.version = "1.0.0"
        logger.info("诊断API模块初始化完成")
    
    def diagnose_symptoms(self, symptoms: List[str], patient_info: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        """
        症状诊断核心方法
        
        :param symptoms: 症状列表
        :param patient_info: 患者信息
        :return: (HTTP状态码, 响应数据)
        """
        try:
            # 这里实现您的诊断逻辑
            # 示例响应格式
            diagnosis_result = {
                "success": True,
                "message": "诊断完成",
                "data": {
                    "diagnosisId": "DIAG_001",
                    "possibleDiseases": [
                        {
                            "diseaseId": 1,
                            "diseaseName": "感冒",
                            "probability": "85%",
                            "reason": "症状匹配度高"
                        }
                    ],
                    "recommendations": [
                        "建议多休息",
                        "多喝温水",
                        "必要时就医"
                    ]
                }
            }
            
            return 200, diagnosis_result
            
        except Exception as e:
            logger.error(f"诊断失败: {str(e)}")
            return 500, {
                "success": False,
                "message": f"诊断失败: {str(e)}",
                "data": {}
            }
    
    def handle_diagnosis_request(self):
        """
        处理Flask请求的入口方法
        必须实现这个方法以符合API模块规范
        """
        try:
            # 获取POST请求的JSON数据
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "success": False,
                    "message": "请求数据格式错误",
                    "data": {}
                }), 400
            
            symptoms = data.get('symptoms', [])
            patient_info = data.get('patientInfo', {})
            
            # 执行诊断
            status_code, result = self.diagnose_symptoms(symptoms, patient_info)
            
            return jsonify(result), status_code
            
        except Exception as e:
            logger.error(f"请求处理失败: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"请求处理失败: {str(e)}",
                "data": {}
            }), 500
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        获取API信息
        必须实现这个方法以符合API模块规范
        """
        return {
            "name": self.api_name,
            "version": self.version,
            "description": "基于症状的智能诊断API",
            "endpoints": {
                "diagnosis": {
                    "method": "POST",
                    "path": "/api/diagnosis",
                    "body": {
                        "symptoms": ["症状1", "症状2"],
                        "patientInfo": {
                            "age": "年龄",
                            "gender": "性别",
                            "medicalHistory": "病史"
                        }
                    },
                    "description": "根据症状和患者信息进行智能诊断"
                }
            },
            "status": "active"
        }

class SymptomAnalysisAPI:
    """
    症状分析API模块示例
    另一个API模块的实现示例
    """
    
    def __init__(self):
        """初始化症状分析API"""
        self.api_name = "症状分析API"
        self.version = "1.0.0"
        logger.info("症状分析API模块初始化完成")
    
    def analyze_symptoms(self, symptom_text: str) -> tuple[int, Dict[str, Any]]:
        """
        症状分析核心方法
        
        :param symptom_text: 症状描述文本
        :return: (HTTP状态码, 响应数据)
        """
        try:
            # 这里实现您的症状分析逻辑
            analysis_result = {
                "success": True,
                "message": "症状分析完成",
                "data": {
                    "extractedSymptoms": [
                        {
                            "symptom": "头痛",
                            "severity": "中等",
                            "confidence": "90%"
                        },
                        {
                            "symptom": "发热",
                            "severity": "轻微",
                            "confidence": "85%"
                        }
                    ],
                    "suggestedQuestions": [
                        "头痛持续多长时间了？",
                        "体温是多少？",
                        "是否伴有其他症状？"
                    ]
                }
            }
            
            return 200, analysis_result
            
        except Exception as e:
            logger.error(f"症状分析失败: {str(e)}")
            return 500, {
                "success": False,
                "message": f"症状分析失败: {str(e)}",
                "data": {}
            }
    
    def handle_analysis_request(self):
        """处理Flask请求的入口方法"""
        try:
            # 获取GET请求参数
            symptom_text = request.args.get('symptoms', '').strip()
            
            if not symptom_text:
                return jsonify({
                    "success": False,
                    "message": "症状描述不能为空",
                    "data": {}
                }), 400
            
            # 执行分析
            status_code, result = self.analyze_symptoms(symptom_text)
            
            return jsonify(result), status_code
            
        except Exception as e:
            logger.error(f"请求处理失败: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"请求处理失败: {str(e)}",
                "data": {}
            }), 500
    
    def get_api_info(self) -> Dict[str, Any]:
        """获取API信息"""
        return {
            "name": self.api_name,
            "version": self.version,
            "description": "从文本中提取和分析症状信息",
            "endpoints": {
                "analysis": {
                    "method": "GET", 
                    "path": "/api/symptom-analysis",
                    "params": {
                        "symptoms": "症状描述文本（必填）"
                    },
                    "description": "分析症状描述文本，提取结构化症状信息"
                }
            },
            "status": "active"
        }

# ===========================================
# 如何将新API模块集成到主服务器的示例代码
# ===========================================

def example_add_new_apis():
    """
    示例：如何添加新的API模块到主服务器
    在 api_server_main.py 中的 _initialize_api_modules 方法里添加
    """
    
    # 在 APIServer 的 _initialize_api_modules 方法中添加：
    
    # # 注册诊断API
    # logger.info("初始化诊断API模块")
    # self.api_modules['diagnosis'] = DiagnosisAPI()
    # 
    # # 注册症状分析API
    # logger.info("初始化症状分析API模块")
    # self.api_modules['symptom_analysis'] = SymptomAnalysisAPI()
    
    # 在 _register_routes 方法中添加对应的路由：
    
    # # 诊断API路由
    # @self.app.route('/api/diagnosis', methods=['POST'])
    # def diagnosis():
    #     """智能诊断接口"""
    #     if 'diagnosis' not in self.api_modules:
    #         return jsonify({
    #             "success": False,
    #             "message": "诊断API模块未初始化",
    #             "data": {}
    #         }), 500
    #     
    #     return self.api_modules['diagnosis'].handle_diagnosis_request()
    # 
    # # 症状分析API路由
    # @self.app.route('/api/symptom-analysis', methods=['GET'])
    # def symptom_analysis():
    #     """症状分析接口"""
    #     if 'symptom_analysis' not in self.api_modules:
    #         return jsonify({
    #             "success": False,
    #             "message": "症状分析API模块未初始化",
    #             "data": {}
    #         }), 500
    #     
    #     return self.api_modules['symptom_analysis'].handle_analysis_request()
    
    pass

# ===========================================
# API模块开发规范
# ===========================================

"""
API模块开发规范：

1. 每个API模块必须实现以下方法：
   - __init__(): 初始化模块
   - handle_*_request(): 处理Flask请求的入口方法
   - get_api_info(): 返回API信息

2. 核心业务方法应该返回 (status_code, response_data) 格式

3. 使用logging记录关键操作和错误

4. 响应格式应该统一：
   {
     "success": bool,
     "message": str,
     "data": object
   }

5. 错误处理要完善，避免未捕获的异常

6. API信息要包含完整的接口文档

7. 模块名称要有意义，便于管理

使用这个模板可以快速开发新的API模块！
"""