"""
医学图像分析API模块 - Linus式优化版本
实现中医望诊图像分析和补充分析功能

优化要点:
1. 修复垃圾数据结构，引入结构化诊断数据
2. 实现真正的增量补充分析逻辑
3. 消除无意义的字符串拼接特殊情况
4. 保持API接口完全向后兼容
"""
import logging
import json
import os
import tempfile
import traceback
from typing import Dict, Any, Optional, List, Tuple
from flask import request, jsonify
from werkzeug.datastructures import FileStorage
from config import ALI_BASE_URL, ALI_API_KEY

# 导入您的图像识别和诊断系统
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
            "diagnosis_data": {"证型": "脾胃虚弱", "建议": "调理脾胃"}
        }

logger = logging.getLogger(__name__)


class DiagnosisData:
    """结构化诊断数据 - 消除字符串粥的核心数据结构"""
    
    def __init__(self):
        self.image_type = ""
        self.confidence = 0.0
        self.syndromes = []  # 辨证提示列表
        self.health_advice = []  # 健康建议列表
        self.theory_basis = ""  # 中医理论依据
        self.raw_analysis = {}  # 原始分析数据
        self.additional_findings = []  # 补充发现
    
    @classmethod
    def from_analysis_result(cls, analysis_result: Dict[str, Any]) -> 'DiagnosisData':
        """从图像分析结果创建结构化诊断数据"""
        data = cls()
        
        # 提取基础信息
        image_info = analysis_result.get("图像识别", {})
        data.image_type = image_info.get("类型", "未知")
        data.confidence = image_info.get("置信度", 0.0)
        
        # 解析分析结果
        analysis_content = analysis_result.get("分析结果", "")
        try:
            if isinstance(analysis_content, str):
                parsed_content = json.loads(analysis_content)
            else:
                parsed_content = analysis_content
            
            data.syndromes = parsed_content.get("辨证提示", [])
            data.health_advice = parsed_content.get("健康建议", [])
            data.theory_basis = parsed_content.get("中医理论依据", "")
            data.raw_analysis = parsed_content
            
        except (json.JSONDecodeError, TypeError):
            # 如果解析失败，将原始内容作为理论依据
            data.theory_basis = str(analysis_content)
            data.raw_analysis = {"原始结果": analysis_content}
        
        return data
    
    @classmethod 
    def from_text(cls, text: str) -> 'DiagnosisData':
        """从文本分析结果创建结构化诊断数据（用于解析历史分析）"""
        data = cls()
        
        # 简单解析文本中的关键信息
        lines = text.split('。')
        for line in lines:
            if "图像类型" in line:
                parts = line.split('：')
                if len(parts) > 1:
                    type_info = parts[1].strip()
                    if '（置信度' in type_info:
                        data.image_type = type_info.split('（')[0]
                        # 提取置信度
                        conf_part = type_info.split('置信度：')[1].split('）')[0] if '置信度：' in type_info else ""
                        try:
                            data.confidence = float(conf_part.replace('%', '')) / 100 if conf_part else 0.0
                        except:
                            data.confidence = 0.0
            elif "辨证提示" in line:
                advice_part = line.split('：')[1] if '：' in line else line
                data.syndromes = [advice_part.strip()]
            elif "健康建议" in line:
                advice_part = line.split('：')[1] if '：' in line else line
                data.health_advice = [advice_part.strip()]
        
        data.raw_analysis = {"解析文本": text}
        return data
    
    def merge_with_additional(self, additional_data: 'DiagnosisData', additional_info: str = "") -> 'DiagnosisData':
        """智能合并补充诊断数据 - 真正的增量分析"""
        merged = DiagnosisData()
        
        # 合并图像类型（以置信度更高的为准）
        if additional_data.confidence > self.confidence:
            merged.image_type = additional_data.image_type
            merged.confidence = additional_data.confidence
        else:
            merged.image_type = self.image_type
            merged.confidence = self.confidence
        
        # 智能合并辨证提示（去重并保持逻辑关系）
        merged.syndromes = self._merge_syndromes(self.syndromes, additional_data.syndromes)
        
        # 合并健康建议（去重）
        merged.health_advice = self._merge_advice(self.health_advice, additional_data.health_advice)
        
        # 合并理论依据
        theory_parts = [t for t in [self.theory_basis, additional_data.theory_basis] if t.strip()]
        merged.theory_basis = "；".join(theory_parts)
        
        # 记录补充信息
        if additional_info.strip():
            merged.additional_findings.append(additional_info.strip())
        
        # 合并原始分析数据
        merged.raw_analysis = {
            "原始分析": self.raw_analysis,
            "补充分析": additional_data.raw_analysis,
            "补充信息": additional_info
        }
        
        return merged
    
    def _merge_syndromes(self, old_syndromes: List[str], new_syndromes: List[str]) -> List[str]:
        """智能合并辨证提示，处理冲突和重复"""
        if not old_syndromes:
            return new_syndromes
        if not new_syndromes:
            return old_syndromes
        
        # 简单去重合并（实际项目中可以添加更复杂的中医逻辑）
        merged = []
        all_syndromes = old_syndromes + new_syndromes
        
        for syndrome in all_syndromes:
            # 提取证型名称（去掉百分比）
            syndrome_name = syndrome.split('(')[0].strip()
            
            # 检查是否已存在类似证型
            exists = False
            for existing in merged:
                if syndrome_name in existing or existing.split('(')[0].strip() in syndrome_name:
                    exists = True
                    break
            
            if not exists:
                merged.append(syndrome)
        
        return merged
    
    def _merge_advice(self, old_advice: List[str], new_advice: List[str]) -> List[str]:
        """合并健康建议，去重"""
        all_advice = old_advice + new_advice
        # 简单去重
        seen = set()
        merged = []
        for advice in all_advice:
            advice_key = advice.strip().lower()
            if advice_key not in seen:
                seen.add(advice_key)
                merged.append(advice)
        return merged
    
    def to_diagnosis_text(self, include_image_info: bool = True) -> str:
        """转换为诊断文本（仅在最终输出时使用）"""
        parts = []
        
        if include_image_info and self.image_type:
            parts.append(f"图像类型：{self.image_type}（置信度：{self.confidence:.1%}）")
        
        if self.syndromes:
            parts.append(f"辨证提示：{'; '.join(self.syndromes)}")
        
        if self.health_advice:
            parts.append(f"健康建议：{'; '.join(self.health_advice)}")
        
        if self.theory_basis:
            parts.append(f"中医理论依据：{self.theory_basis}")
        
        if self.additional_findings:
            parts.append(f"补充发现：{'; '.join(self.additional_findings)}")
        
        return "。".join(parts) if parts else "诊断分析完成"


class MedicalImageAPI:
    """医学图像分析API类 - Linus式优化版本"""
    
    def __init__(self):
        """初始化医学图像分析API"""
        try:
            logger.info("初始化医学图像分析API模块")
            
            # 初始化图像识别系统
            api_key = ALI_API_KEY
            base_url = ALI_BASE_URL
            
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
        """保存临时文件并返回文件路径"""
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
    
    def _analyze_image_to_structured_data(self, image_path: str) -> DiagnosisData:
        """分析图像并返回结构化诊断数据"""
        logger.info(f"调用图像识别系统分析: {image_path}")
        
        # 尝试不同的图像传递方式
        try:
            # 方案1: 尝试base64编码（大多数API支持）
            import base64
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
                base64_url = f"data:image/jpeg;base64,{image_data}"
                
            logger.info("使用base64编码传递图像")
            image_analysis_result = self.tcm_system.comprehensive_diagnosis(base64_url)
            
        except Exception as e:
            logger.warning(f"base64方式失败: {e}")
            try:
                # 方案2: 尝试直接传递文件路径（某些API支持）
                logger.info("尝试直接文件路径")
                image_analysis_result = self.tcm_system.comprehensive_diagnosis(image_path)
            except Exception as e2:
                logger.error(f"所有图像传递方式都失败: {e2}")
                # 返回一个默认的分析结果
                image_analysis_result = {
                    "图像识别": {"类型": "处理失败", "置信度": 0.0},
                    "分析结果": json.dumps({
                        "错误": f"图像识别失败: {str(e2)}",
                        "建议": "请检查图像文件格式和网络连接"
                    }, ensure_ascii=False)
                }
        
        return DiagnosisData.from_analysis_result(image_analysis_result)
    
    def _enhanced_diagnosis_with_structure(self, diagnosis_data: DiagnosisData, description: str = "") -> str:
        """使用结构化数据进行增强诊断"""
        # 构建更智能的诊断输入
        diagnosis_input = diagnosis_data.to_diagnosis_text()
        
        if description.strip():
            diagnosis_input = f"图像描述：{description}。{diagnosis_input}"
        
        # 调用诊断系统
        logger.info("调用诊断系统进行结构化分析")
        diagnosis_result = run_tcm_graph(
            user_input=diagnosis_input,
            config={"retriever_k": 3}
        )
        
        return diagnosis_result.get("response", diagnosis_input)
    
    def analyze_image(self, image_file: FileStorage, description: str = "") -> tuple[int, Dict[str, Any]]:
        """
        图片望诊分析 - 保持原有API接口
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
            
            # 分析图像获取结构化数据（直接传递文件路径）
            diagnosis_data = self._analyze_image_to_structured_data(temp_file_path)
            
            # 进行增强诊断
            final_results = self._enhanced_diagnosis_with_structure(diagnosis_data, description)
            
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
            if temp_file_path:
                self._cleanup_temp_file(temp_file_path)
    
    def analyze_with_supplement(self, prev_analysis: str, additional_info: str, 
                               additional_file: Optional[FileStorage] = None) -> tuple[int, Dict[str, Any]]:
        """
        望诊补充分析 - 真正的增量分析实现，保持原有API接口
        
        优化要点：
        1. 解析历史分析为结构化数据
        2. 处理新图像（如果有）为结构化数据  
        3. 智能合并而不是简单字符串拼接
        4. 只在必要时进行完整重新诊断
        """
        temp_file_path = None
        
        try:
            logger.info("开始望诊补充分析")
            
            # 第一步：将历史分析转换为结构化数据
            prev_diagnosis_data = DiagnosisData.from_text(prev_analysis) if prev_analysis.strip() else DiagnosisData()
            
            # 第二步：处理补充图像（如果有）
            additional_diagnosis_data = DiagnosisData()
            if additional_file:
                is_valid, error_msg = self._validate_image_file(additional_file)
                if not is_valid:
                    return 400, {
                        "success": False,
                        "message": f"补充图像文件错误: {error_msg}",
                        "data": {"results": ""}
                    }
                
                temp_file_path = self._save_temp_file(additional_file)
                
                logger.info("分析补充图像")
                additional_diagnosis_data = self._analyze_image_to_structured_data(temp_file_path)
            
            # 第三步：智能合并诊断数据
            merged_diagnosis_data = prev_diagnosis_data.merge_with_additional(
                additional_diagnosis_data, additional_info
            )
            
            # 第四步：判断是否需要重新诊断
            needs_full_rediagnosis = self._should_perform_full_rediagnosis(
                prev_diagnosis_data, additional_diagnosis_data, additional_info
            )
            
            if needs_full_rediagnosis:
                # 需要完整重新诊断
                logger.info("执行完整重新诊断")
                final_results = self._enhanced_diagnosis_with_structure(
                    merged_diagnosis_data, f"补充分析：{additional_info}"
                )
            else:
                # 增量更新即可
                logger.info("执行增量诊断更新")
                final_results = self._incremental_diagnosis_update(
                    prev_diagnosis_data, merged_diagnosis_data, additional_info
                )
            
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
            if temp_file_path:
                self._cleanup_temp_file(temp_file_path)
    
    def _should_perform_full_rediagnosis(self, prev_data: DiagnosisData, 
                                        additional_data: DiagnosisData, additional_info: str) -> bool:
        """判断是否需要完整重新诊断 - 避免不必要的重新计算"""
        
        # 如果有新图像且置信度显著不同，需要重新诊断
        if additional_data.image_type and abs(additional_data.confidence - prev_data.confidence) > 0.3:
            return True
        
        # 如果补充信息包含关键症状词汇，需要重新诊断
        critical_keywords = ["疼痛", "发热", "恶心", "头晕", "胸闷", "气短", "失眠", "腹泻", "便秘"]
        if any(keyword in additional_info for keyword in critical_keywords):
            return True
        
        # 如果新图像类型与原图像类型不同，需要重新诊断
        if (additional_data.image_type and prev_data.image_type and 
            additional_data.image_type != prev_data.image_type):
            return True
        
        # 否则使用增量更新
        return False
    
    def _incremental_diagnosis_update(self, prev_data: DiagnosisData, 
                                     merged_data: DiagnosisData, additional_info: str) -> str:
        """增量诊断更新 - 不重新分析，只更新结论"""
        
        base_text = merged_data.to_diagnosis_text()
        
        # 添加增量更新的说明
        update_parts = []
        
        if additional_info.strip():
            update_parts.append(f"根据补充信息'{additional_info}'")
        
        if len(merged_data.syndromes) > len(prev_data.syndromes):
            new_syndromes = [s for s in merged_data.syndromes if s not in prev_data.syndromes]
            if new_syndromes:
                update_parts.append(f"新增辨证考虑：{'; '.join(new_syndromes)}")
        
        if len(merged_data.health_advice) > len(prev_data.health_advice):
            new_advice = [a for a in merged_data.health_advice if a not in prev_data.health_advice]
            if new_advice:
                update_parts.append(f"补充建议：{'; '.join(new_advice)}")
        
        if update_parts:
            return f"{base_text}。{'; '.join(update_parts)}。"
        else:
            return f"{base_text}。综合补充信息，原诊断结论仍然适用。"
    
    def handle_watch_request(self):
        """处理图片望诊请求 (接口4.1) - 保持原有接口"""
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
        """处理望诊补充请求 (接口4.2) - 保持原有接口"""
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
        """获取API信息 - 保持原有接口"""
        return {
            "name": self.api_name,
            "version": self.version,
            "description": "基于中医理论的医学图像分析API，支持望诊分析和补充分析（Linus式优化版本）",
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
            "status": "active",
            "optimizations": [
                "结构化诊断数据处理",
                "智能增量分析",
                "减少不必要的重新计算",
                "消除字符串拼接特殊情况"
            ]
        }