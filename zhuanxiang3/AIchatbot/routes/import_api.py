"""
文档导入分析API模块 - 严格匹配API文档版本
修改要点:
1. 路径从 /api/import 改为 /api/record/import
2. 参数名从 document 改为 recordImage
3. 只处理文件上传，移除JSON文本内容处理
4. 专门处理病历图片导入
"""
import logging
import os
import tempfile
import re
from typing import Dict, Any, List, Optional
from flask import request, jsonify
from paddleocr import PaddleOCR, draw_ocr
import fitz
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logger = logging.getLogger(__name__)

class ImportAPI:
    """
    文档导入分析API模块 - 严格匹配API文档版本
    专门处理病历图片的OCR识别和分析功能
    """
    
    def __init__(self):
        """
        初始化文档导入分析API
        """
        self.api_name = "病历导入分析API"
        self.version = "1.1.0"  # 严格匹配文档版本
        
        # OCR处理参数
        self.page_num = 10  # 最多处理页数
        
        # 支持的文件格式（专门针对病历图片）
        self.allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'pdf'}
        
        # 初始化OCR引擎
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=self.page_num, det=True)
            logger.info("OCR引擎初始化成功")
        except Exception as e:
            logger.error(f"OCR引擎初始化失败: {str(e)}")
            self.ocr = None
        
        # 确保输出目录存在
        self.output_dir = 'data_system/ocr_results'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        logger.info("病历导入分析API模块初始化完成")
    
    def _validate_record_image(self, file) -> tuple[bool, str]:
        """
        验证上传的病历图片文件
        
        :param file: 上传的文件对象
        :return: (是否有效, 错误消息)
        """
        if not file:
            return False, "未上传病历图片文件"
        
        if file.filename == '':
            return False, "文件名为空"
        
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
    
    def process_pdf_ocr(self, pdf_path: str) -> str:
        """
        处理PDF文件的OCR识别
        
        :param pdf_path: PDF文件路径
        :return: 识别出的文本内容
        """
        if not self.ocr:
            return "OCR引擎未初始化"
        
        all_text = []  # 存储所有文本
        
        try:
            with fitz.open(pdf_path) as pdf:
                total_pages = pdf.page_count
                logger.info(f"病历PDF总页数: {total_pages}")
                
                # 对每一页单独进行OCR处理
                for pg in range(min(self.page_num, total_pages)):
                    try:
                        page = pdf[pg]
                        mat = fitz.Matrix(2, 2)
                        pm = page.get_pixmap(matrix=mat, alpha=False)
                        
                        # 限制图像大小
                        if pm.width > 2000 or pm.height > 2000:
                            pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                        
                        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                        img_array = np.array(img)
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        # OCR识别
                        result = self.ocr.ocr(img_bgr, cls=True, det=True)
                        
                        # 添加页码信息
                        page_text = [f"==== 病历第 {pg + 1} 页 ===="]
                        
                        # 提取文本
                        if result:
                            for line in result:
                                if line:
                                    for text_info in line:
                                        if len(text_info) >= 2:
                                            text = text_info[1][0]
                                            page_text.append(text)
                        else:
                            page_text.append("此页未识别到文本")
                        
                        all_text.append("\n".join(page_text))
                        
                    except Exception as e:
                        error_msg = f"处理病历第{pg + 1}页时出错: {str(e)}"
                        logger.warning(error_msg)
                        all_text.append(f"==== 病历第 {pg + 1} 页 ====\n{error_msg}")
                        continue
                        
        except Exception as e:
            error_message = f"处理病历PDF时出错: {str(e)}"
            logger.error(error_message)
            all_text.append(error_message)
        
        return "\n\n".join(all_text)
    
    def process_image_ocr(self, image_path: str) -> str:
        """
        处理病历图像文件的OCR识别
        
        :param image_path: 图像文件路径
        :return: 识别出的文本内容
        """
        if not self.ocr:
            return "OCR引擎未初始化"
        
        try:
            result = self.ocr.ocr(image_path, cls=True, det=True)
            text_lines = []
            
            if result:
                for line in result:
                    if line:
                        for text_info in line:
                            if len(text_info) >= 2:
                                text_lines.append(text_info[1][0])
            else:
                logger.info("病历图像中未识别到文本")
                text_lines.append("未识别到文本")
            
            return "\n".join(text_lines)
            
        except Exception as e:
            error_message = f"处理病历图像时出错: {str(e)}"
            logger.error(error_message)
            return error_message
    
    def save_ocr_result(self, text_content: str, filename: str) -> str:
        """
        保存OCR结果到文件
        
        :param text_content: OCR识别的文本内容
        :param filename: 原始文件名
        :return: 保存的文件路径
        """
        try:
            # 构建输出文件路径
            base_name = os.path.splitext(filename)[0]
            output_txt_path = os.path.join(self.output_dir, f"record_{base_name}.txt")
            
            # 保存文本内容
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"病历OCR识别结果\n")
                f.write(f"原始文件: {filename}\n")
                f.write("=" * 50 + "\n")
                f.write(text_content)
            
            logger.info(f"病历OCR结果已保存到: {output_txt_path}")
            return output_txt_path
            
        except Exception as e:
            logger.error(f"保存病历OCR结果失败: {str(e)}")
            return ""
    
    def extract_medical_info(self, text_content: str) -> Dict[str, str]:
        """
        从病历文本内容中提取医学信息
        
        :param text_content: OCR识别的病历文本内容
        :return: 提取的医学信息字典
        """
        try:
            # 初始化结果
            symptoms = ""
            disease = ""
            prescription = ""
            
            # 将文本按行分割并清理
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            text_lower = text_content.lower()
            
            # 症状关键词（针对病历）
            symptom_keywords = ['主诉', '现病史', '症状', '不适', '疼痛', '头痛', '头晕', '发热', 
                              '咳嗽', '胸痛', '腹痛', '恶心', '呕吐', '乏力', '失眠', '心悸']
            
            # 诊断关键词（针对病历）
            diagnosis_keywords = ['诊断', '印象', '病情', '分析', '检查结果', '发现', '考虑', 
                                '可能', '倾向', '辨证', '证型']
            
            # 处方关键词（针对病历）
            prescription_keywords = ['处方', '治疗方案', '用药', '服用', '注意事项', '复查', 
                                   '休息', '饮食', '建议', '医嘱']
            
            # 提取症状信息
            symptom_lines = []
            for line in lines:
                if any(keyword in line for keyword in symptom_keywords):
                    # 清理和格式化症状描述
                    clean_line = re.sub(r'(主诉[:：]?|现病史[:：]?|症状[:：]?)', '', line).strip()
                    if clean_line and len(clean_line) > 2:
                        symptom_lines.append(clean_line)
            
            if symptom_lines:
                symptoms = "主诉：" + "，".join(symptom_lines) + "。"
            else:
                # 如果没有明确的症状描述，尝试从文本中提取常见症状词汇
                common_symptoms = []
                symptom_patterns = ['头痛', '头晕', '发热', '咳嗽', '胸痛', '腹痛', '恶心', 
                                  '呕吐', '乏力', '心悸', '失眠', '食欲不振', '便秘', '腹泻']
                for symptom in symptom_patterns:
                    if symptom in text_content:
                        common_symptoms.append(symptom)
                
                if common_symptoms:
                    symptoms = "主诉：" + "、".join(common_symptoms) + "。"
                else:
                    symptoms = "主诉：根据病历记录，暂无明确症状描述。"
            
            # 提取疾病/诊断信息
            diagnosis_lines = []
            for line in lines:
                if any(keyword in line for keyword in diagnosis_keywords):
                    # 清理诊断描述
                    clean_line = re.sub(r'(诊断[:：]?|印象[:：]?|分析[:：]?|考虑[:：]?)', '', line).strip()
                    if clean_line and len(clean_line) > 3:  # 过滤太短的内容
                        diagnosis_lines.append(clean_line)
            
            if diagnosis_lines:
                disease = "根据病历记录，" + "；".join(diagnosis_lines) + "。"
            else:
                # 尝试寻找常见疾病名称
                disease_patterns = ['高血压', '糖尿病', '冠心病', '感冒', '发热', '肺炎', 
                                  '胃炎', '肝炎', '肾炎', '关节炎', '支气管炎', '贫血']
                found_diseases = []
                for disease_name in disease_patterns:
                    if disease_name in text_content:
                        found_diseases.append(disease_name)
                
                if found_diseases:
                    disease = "根据病历记录，患者可能存在：" + "、".join(found_diseases) + "等疾病。"
                else:
                    disease = "根据病历记录，需要进一步检查以明确诊断。"
            
            # 提取处方/治疗信息
            prescription_lines = []
            for line in lines:
                if any(keyword in line for keyword in prescription_keywords):
                    # 清理处方描述
                    clean_line = re.sub(r'(处方[:：]?|治疗[:：]?|建议[:：]?|医嘱[:：]?)', '', line).strip()
                    if clean_line and len(clean_line) > 2:
                        prescription_lines.append(clean_line)
            
            if prescription_lines:
                prescription = "治疗建议：" + "；".join(prescription_lines) + "。"
            else:
                # 提供基于病历的通用建议
                prescription = "治疗建议：根据病历记录，请遵医嘱用药，注意休息，定期复查。"
            
            return {
                "symptoms": symptoms,
                "disease": disease,
                "prescription": prescription
            }
            
        except Exception as e:
            logger.error(f"提取病历医学信息失败: {str(e)}")
            return {
                "symptoms": "病历信息提取失败",
                "disease": "病历信息提取失败",
                "prescription": "病历信息提取失败"
            }
    
    def import_record_image(self, record_image_file) -> tuple[int, Dict[str, Any]]:
        """
        导入病历图片的核心方法
        
        :param record_image_file: 上传的病历图片文件
        :return: (HTTP状态码, 响应数据)
        """
        temp_file_path = None
        
        try:
            logger.info("开始处理病历图片导入")
            
            # 验证文件
            is_valid, error_msg = self._validate_record_image(record_image_file)
            if not is_valid:
                return 400, {
                    "success": False,
                    "message": error_msg,
                    "data": {
                        "symptoms": "",
                        "disease": "",
                        "prescription": ""
                    }
                }
            
            # 保存上传的文件到临时目录
            original_filename = record_image_file.filename
            file_extension = os.path.splitext(original_filename)[1].lower()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                record_image_file.save(temp_file.name)
                temp_file_path = temp_file.name
            
            # 根据文件类型进行OCR处理
            if file_extension == '.pdf':
                logger.info(f"处理病历PDF文件: {original_filename}")
                text_content = self.process_pdf_ocr(temp_file_path)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                logger.info(f"处理病历图像文件: {original_filename}")
                text_content = self.process_image_ocr(temp_file_path)
            else:
                return 400, {
                    "success": False,
                    "message": f"不支持的病历文件类型: {file_extension}",
                    "data": {
                        "symptoms": "",
                        "disease": "",
                        "prescription": ""
                    }
                }
            
            # 保存OCR结果
            self.save_ocr_result(text_content, original_filename)
            
            # 提取医学信息
            medical_info = self.extract_medical_info(text_content)
            
            result = {
                "success": True,
                "message": "病历导入成功",
                "data": medical_info
            }
            
            logger.info(f"病历图片导入完成: {original_filename}")
            return 200, result
            
        except Exception as e:
            error_msg = f"病历导入失败: {str(e)}"
            logger.error(error_msg)
            return 500, {
                "success": False,
                "message": error_msg,
                "data": {
                    "symptoms": "",
                    "disease": "",
                    "prescription": ""
                }
            }
        
        finally:
            # 清理临时文件
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
    
    def handle_record_import_request(self):
        """
        处理病历导入请求的入口方法
        接口 6.2: POST /api/record/import
        """
        try:
            logger.info(f"收到病历导入请求: {request.remote_addr}")
            
            # 严格按照API文档检查参数
            if 'recordImage' not in request.files:
                return jsonify({
                    "success": False,
                    "message": "未找到上传的病历图片文件，参数名应为'recordImage'",
                    "data": {
                        "symptoms": "",
                        "disease": "",
                        "prescription": ""
                    }
                }), 400
            
            record_image_file = request.files['recordImage']
            
            if record_image_file.filename == '':
                return jsonify({
                    "success": False,
                    "message": "未选择病历图片文件",
                    "data": {
                        "symptoms": "",
                        "disease": "",
                        "prescription": ""
                    }
                }), 400
            
            # 导入病历图片
            status_code, result = self.import_record_image(record_image_file)
            
            return jsonify(result), status_code
            
        except Exception as e:
            error_msg = f"病历导入请求处理失败: {str(e)}"
            logger.error(error_msg)
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
        """
        获取API信息 - 更新为严格匹配文档版本
        """
        return {
            "name": self.api_name,
            "version": self.version,
            "description": "病历导入分析API，专门处理病历图片的OCR识别和分析",
            "endpoints": {
                "record_import": {
                    "method": "POST",
                    "path": "/api/record/import",
                    "content_type": "multipart/form-data",
                    "params": {
                        "recordImage": "病历图片文件（必填）"
                    },
                    "description": "导入已有病历图片，使用OCR识别并提取医学信息生成新病历",
                    "supported_formats": list(self.allowed_extensions),
                    "max_file_size": "20MB",
                    "response_format": {
                        "success": "bool",
                        "message": "string",
                        "data": {
                            "symptoms": "主诉症状信息",
                            "disease": "疾病分析信息",
                            "prescription": "处方建议信息"
                        }
                    }
                }
            },
            "features": [
                "专门针对病历图片的OCR识别",
                "支持PDF和图像格式的病历文件",
                "自动提取症状、诊断、处方信息",
                "严格按照API文档规范实现"
            ],
            "status": "active",
            "ocr_engine": "PaddleOCR" if self.ocr else "未初始化",
            "max_pages": self.page_num,
            "output_directory": self.output_dir
        }