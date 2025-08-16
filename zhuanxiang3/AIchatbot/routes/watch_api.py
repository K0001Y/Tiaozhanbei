"""
医学图像分析API模块 - Linus修复版
使用优化后的graph.py，实现智能状态管理
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
    # 使用修复后的graph接口
    from graph import run_tcm_graph_with_state, run_tcm_graph
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
    
    def run_tcm_graph_with_state(user_input, existing_state=None, messages=None, memory=None, config=None):
        return run_tcm_graph(user_input, messages, memory, config)

logger = logging.getLogger(__name__)

class WatchSessionManager:
    """
    望诊专用Session管理器 - 基于图像内容Hash
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
                'graph_state': None,
                'diagnosis_data': None,
                'created_at': time.time(),
                'updated_at': time.time(),
                'type': 'image'
            }
            logger.info(f"创建新图像session: {session_id}")
        else:
            logger.info(f"复用已存在图像session: {session_id}")
        
        return session_id
    
    def extract_session_from_analysis(self, prev_analysis: str) -> Optional[str]:
        """从分析结果中提取session"""
        for session_id, data in self._sessions.items():
            if 'diagnosis_data' in data and data['diagnosis_data']:
                # 检查分析结果是否匹配
                diagnosis_text = data['diagnosis_data'].to_diagnosis_text()
                if (prev_analysis in diagnosis_text or 
                    diagnosis_text in prev_analysis or
                    self._text_similarity(prev_analysis, diagnosis_text) > 0.7):
                    logger.info(f"从分析结果中识别出session: {session_id}")
                    return session_id
        return None
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get_session_data(self, session_id: str) -> Optional[Dict]:
        """获取session数据"""
        if session_id in self._sessions:
            return self._sessions[session_id]
        return None
    
    def update_session(self, session_id: str, graph_state: Dict = None, diagnosis_data=None):
        """更新session"""
        if session_id in self._sessions:
            if graph_state:
                self._sessions[session_id]['graph_state'] = graph_state
            if diagnosis_data:
                self._sessions[session_id]['diagnosis_data'] = diagnosis_data
            self._sessions[session_id]['updated_at'] = time.time()

# 全局望诊session管理器
watch_session_manager = WatchSessionManager()

class DiagnosisData:
    """结构化诊断数据"""
    
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
            data.theory_basis = str(analysis_content)
            data.raw_analysis = {"原始结果": analysis_content}
        
        return data
    
    @classmethod 
    def from_text(cls, text: str) -> 'DiagnosisData':
        """从文本分析结果创建结构化诊断数据"""
        data = cls()
        
        # 智能解析文本
        lines = [line.strip() for line in text.split('。') if line.strip()]
        
        for line in lines:
            if "图像类型" in line or "类型：" in line:
                data._parse_image_type(line)
            elif "辨证提示" in line or "证型" in line:
                data._parse_syndromes(line)
            elif "健康建议" in line or "建议" in line:
                data._parse_health_advice(line)
            elif "理论依据" in line:
                data._parse_theory_basis(line)
        
        data.raw_analysis = {"解析文本": text}
        return data
    
    def _parse_image_type(self, line: str):
        """解析图像类型"""
        parts = line.split('：')
        if len(parts) > 1:
            type_info = parts[1].strip()
            if '（置信度' in type_info:
                self.image_type = type_info.split('（')[0]
                conf_part = type_info.split('置信度：')[1].split('）')[0] if '置信度：' in type_info else ""
                try:
                    self.confidence = float(conf_part.replace('%', '')) / 100 if conf_part else 0.0
                except:
                    self.confidence = 0.0
            else:
                self.image_type = type_info
    
    def _parse_syndromes(self, line: str):
        """解析辨证提示"""
        content = line.split('：')[1] if '：' in line else line
        syndromes = [s.strip() for s in content.split(';') if s.strip()]
        if not syndromes:
            syndromes = [s.strip() for s in content.split('，') if s.strip()]
        self.syndromes.extend(syndromes)
    
    def _parse_health_advice(self, line: str):
        """解析健康建议"""
        content = line.split('：')[1] if '：' in line else line
        advice_list = [a.strip() for a in content.split(';') if a.strip()]
        if not advice_list:
            advice_list = [a.strip() for a in content.split('，') if a.strip()]
        self.health_advice.extend(advice_list)
    
    def _parse_theory_basis(self, line: str):
        """解析理论依据"""
        content = line.split('：')[1] if '：' in line else line
        self.theory_basis = content.strip()
    
    def merge_with_additional(self, additional_data: 'DiagnosisData', additional_info: str = "") -> 'DiagnosisData':
        """智能合并诊断数据"""
        merged = DiagnosisData()
        
        # 合并图像类型（优先选择置信度高的）
        if additional_data.confidence > self.confidence:
            merged.image_type = additional_data.image_type
            merged.confidence = additional_data.confidence
        else:
            merged.image_type = self.image_type
            merged.confidence = self.confidence
        
        # 合并辨证提示（去重）
        all_syndromes = self.syndromes + additional_data.syndromes
        merged.syndromes = list(dict.fromkeys(all_syndromes))  # 保持顺序的去重
        
        # 合并健康建议（去重）
        all_advice = self.health_advice + additional_data.health_advice
        merged.health_advice = list(dict.fromkeys(all_advice))
        
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
    
    def to_diagnosis_text(self, include_image_info: bool = True) -> str:
        """转换为诊断文本"""
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
    
    def has_meaningful_data(self) -> bool:
        """检查是否有有意义的诊断数据"""
        return bool(self.syndromes or self.health_advice or self.theory_basis)

class MedicalImageAPI:
    """医学图像分析API类 - Linus修复版"""
    
    def __init__(self):
        """初始化医学图像分析API"""
        try:
            logger.info("初始化医学图像分析API模块")
            
            # 初始化图像识别系统
            api_key = ALI_API_KEY
            base_url = ALI_BASE_URL
            
            self.tcm_system = TCMDiagnosisSystem(api_key=api_key, base_url=base_url)
            
            self.api_name = "医学图像分析API"
            self.version = "2.0.0"  # Linus修复版
            
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
    
    def _analyze_image_to_structured_data(self, image_path: str) -> DiagnosisData:
        """分析图像并返回结构化诊断数据"""
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
                image_analysis_result = {
                    "图像识别": {"类型": "处理失败", "置信度": 0.0},
                    "分析结果": json.dumps({
                        "错误": f"图像识别失败: {str(e2)}",
                        "建议": "请检查图像文件格式和网络连接"
                    }, ensure_ascii=False)
                }
        
        return DiagnosisData.from_analysis_result(image_analysis_result)
    
    def _enhanced_diagnosis_with_graph(self, diagnosis_data: DiagnosisData, 
                                     description: str = "", existing_graph_state: Dict = None) -> Tuple[str, Dict]:
        """
        使用优化后的graph进行增强诊断
        """
        # 构建诊断输入
        diagnosis_input = diagnosis_data.to_diagnosis_text()
        if description.strip():
            diagnosis_input = f"图像描述：{description}。{diagnosis_input}"
        
        if existing_graph_state:
            # 【使用修复后的graph接口】基于已有状态进行增量计算
            logger.info("基于已有graph状态进行增量诊断")
            result = run_tcm_graph_with_state(
                user_input=diagnosis_input,
                existing_state=existing_graph_state,
                config={"retriever_k": 3}
            )
        else:
            # 新的完整计算
            logger.info("执行新的完整诊断")
            result = run_tcm_graph(
                user_input=diagnosis_input,
                config={"retriever_k": 3}
            )
        
        final_response = result.get("response", diagnosis_input)
        return final_response, result
    
    def analyze_image(self, image_file: FileStorage, description: str = "") -> tuple[int, Dict[str, Any]]:
        """
        图片望诊分析 - 使用优化后的graph和session管理
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
            session_data = watch_session_manager.get_session_data(session_id)
            if (session_data and 
                session_data.get('diagnosis_data') and 
                session_data['diagnosis_data'].has_meaningful_data()):
                
                logger.info(f"复用已有望诊分析结果，session: {session_id}")
                final_results = session_data['diagnosis_data'].to_diagnosis_text()
                
                return 200, {
                    "success": True,
                    "message": "望诊分析成功",
                    "data": {"results": final_results}
                }
            
            # 分析图像获取结构化数据
            diagnosis_data = self._analyze_image_to_structured_data(temp_file_path)
            
            # 使用优化后的graph进行增强诊断
            final_results, graph_result = self._enhanced_diagnosis_with_graph(diagnosis_data, description)
            
            # 保存session状态
            watch_session_manager.update_session(session_id, graph_result, diagnosis_data)
            
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
        望诊补充分析 - 使用优化后的graph进行增量计算
        """
        temp_file_path = None
        
        try:
            logger.info("开始望诊补充分析")
            
            # 智能识别已有session
            session_id = watch_session_manager.extract_session_from_analysis(prev_analysis) if prev_analysis else None
            existing_session_data = watch_session_manager.get_session_data(session_id) if session_id else None
            
            # 获取已有诊断数据
            prev_diagnosis_data = DiagnosisData.from_text(prev_analysis) if prev_analysis.strip() else DiagnosisData()
            
            # 处理补充图像（如果有）
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
                
                # 为补充图像创建session
                additional_session_id = watch_session_manager.get_or_create_image_session(temp_file_path, additional_info)
                additional_session_data = watch_session_manager.get_session_data(additional_session_id)
                
                if additional_session_data and additional_session_data.get('diagnosis_data'):
                    # 复用已有分析
                    logger.info("复用补充图像的已有分析结果")
                    additional_diagnosis_data = additional_session_data['diagnosis_data']
                else:
                    # 新分析
                    logger.info("分析补充图像")
                    additional_diagnosis_data = self._analyze_image_to_structured_data(temp_file_path)
            
            # 智能合并诊断数据
            merged_diagnosis_data = prev_diagnosis_data.merge_with_additional(
                additional_diagnosis_data, additional_info
            )
            
            # 判断是否需要重新诊断
            needs_full_rediagnosis = self._should_perform_full_rediagnosis(
                prev_diagnosis_data, additional_diagnosis_data, additional_info
            )
            
            if needs_full_rediagnosis and existing_session_data and existing_session_data.get('graph_state'):
                # 【核心优化】基于已有session的graph state进行增量计算
                logger.info("基于已有session进行增量重新诊断")
                final_results, new_graph_result = self._enhanced_diagnosis_with_graph(
                    merged_diagnosis_data, 
                    f"补充分析：{additional_info}",
                    existing_session_data['graph_state']
                )
                
                # 更新session
                if session_id:
                    watch_session_manager.update_session(session_id, new_graph_result, merged_diagnosis_data)
            elif needs_full_rediagnosis:
                # 没有已有session，需要完整重新诊断
                logger.info("执行完整重新诊断")
                final_results, new_graph_result = self._enhanced_diagnosis_with_graph(
                    merged_diagnosis_data, f"补充分析：{additional_info}"
                )
            else:
                # 【性能优化】增量更新，无需重新诊断
                logger.info("执行轻量级增量更新")
                final_results = self._incremental_diagnosis_update(
                    prev_diagnosis_data, merged_diagnosis_data, additional_info
                )
            
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
    
    def _should_perform_full_rediagnosis(self, prev_data: DiagnosisData, 
                                        additional_data: DiagnosisData, additional_info: str) -> bool:
        """判断是否需要完整重新诊断"""
        
        # 如果有新图像且置信度显著不同，需要重新诊断
        if additional_data.image_type and abs(additional_data.confidence - prev_data.confidence) > 0.3:
            return True
        
        # 如果补充信息包含关键症状词汇，需要重新诊断
        critical_keywords = ["疼痛", "发热", "恶心", "头晕", "胸闷", "气短", "失眠", "腹泻", "便秘", 
                           "出血", "肿胀", "皮疹", "咳嗽", "乏力", "食欲不振"]
        if any(keyword in additional_info for keyword in critical_keywords):
            return True
        
        # 如果新图像类型与原图像类型不同，需要重新诊断
        if (additional_data.image_type and prev_data.image_type and 
            additional_data.image_type != prev_data.image_type):
            return True
        
        # 如果没有原有诊断数据，需要完整诊断
        if not prev_data.has_meaningful_data():
            return True
        
        # 否则使用增量更新
        return False
    
    def _incremental_diagnosis_update(self, prev_data: DiagnosisData, 
                                     merged_data: DiagnosisData, additional_info: str) -> str:
        """增量诊断更新"""
        
        base_text = merged_data.to_diagnosis_text()
        
        # 构建增量更新说明
        update_parts = []
        
        if additional_info.strip():
            update_parts.append(f"根据补充信息'{additional_info}'")
        
        # 检查新增的辨证提示
        new_syndromes = [s for s in merged_data.syndromes if s not in prev_data.syndromes]
        if new_syndromes:
            update_parts.append(f"新增辨证考虑：{'; '.join(new_syndromes)}")
        
        # 检查新增的健康建议
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
            "description": "基于中医理论的医学图像分析API - Linus修复版，使用优化后的graph",
            "features": [
                "智能望诊Session管理，避免重复图像分析",
                "使用优化后的graph，消除重复编译开销",
                "真正的增量分析，避免不必要的重新计算",
                "结构化诊断数据处理，消除字符串拼接",
                "基于图像内容Hash的智能缓存"
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