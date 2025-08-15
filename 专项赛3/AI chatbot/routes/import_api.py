"""
文档导入分析API模块
专门处理医学文档的OCR识别和分析功能
对应接口 6.2 病理文档分析
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
    文档导入分析API模块
    专门处理医学文档的OCR识别和信息提取功能
    """
    
    def __init__(self):
        """
        初始化文档导入分析API
        """
        self.api_name = "文档导入分析API"
        self.version = "1.0.0"
        
        # OCR处理参数
        self.page_num = 10  # 最多处理页数
        
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
        
        logger.info("文档导入分析API模块初始化完成")
    
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
                logger.info(f"PDF总页数: {total_pages}")
                
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
                        page_text = [f"==== 第 {pg + 1} 页 ===="]
                        
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
                        error_msg = f"处理第{pg + 1}页时出错: {str(e)}"
                        logger.warning(error_msg)
                        all_text.append(f"==== 第 {pg + 1} 页 ====\n{error_msg}")
                        continue
                        
        except Exception as e:
            error_message = f"处理PDF时出错: {str(e)}"
            logger.error(error_message)
            all_text.append(error_message)
        
        return "\n\n".join(all_text)
    
    def process_image_ocr(self, image_path: str) -> str:
        """
        处理图像文件的OCR识别
        
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
                logger.info("图像中未识别到文本")
                text_lines.append("未识别到文本")
            
            return "\n".join(text_lines)
            
        except Exception as e:
            error_message = f"处理图像时出错: {str(e)}"
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
            output_txt_path = os.path.join(self.output_dir, base_name + ".txt")
            
            # 保存文本内容
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            logger.info(f"OCR结果已保存到: {output_txt_path}")
            return output_txt_path
            
        except Exception as e:
            logger.error(f"保存OCR结果失败: {str(e)}")
            return ""
    
    def extract_medical_info(self, text_content: str) -> Dict[str, str]:
        """
        从文本内容中提取医学信息
        
        :param text_content: OCR识别的文本内容
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
            
            # 症状关键词
            symptom_keywords = ['主诉', '症状', '不适', '疼痛', '头痛', '头晕', '发热', '咳嗽', '胸痛', '腹痛', '恶心', '呕吐']
            
            # 诊断关键词
            diagnosis_keywords = ['诊断', '病情', '分析', '检查', '发现', '考虑', '可能', '倾向']
            
            # 处方关键词
            prescription_keywords = ['建议', '治疗', '用药', '服用', '注意', '复查', '休息', '饮食']
            
            # 提取症状信息
            symptom_lines = []
            for line in lines:
                if any(keyword in line for keyword in symptom_keywords):
                    # 清理和格式化症状描述
                    clean_line = re.sub(r'(主诉[:：]?|症状[:：]?)', '', line).strip()
                    if clean_line:
                        symptom_lines.append(clean_line)
            
            if symptom_lines:
                symptoms = "主诉" + "、".join(symptom_lines) + "。"
            else:
                # 如果没有明确的症状描述，尝试从文本中提取常见症状词汇
                common_symptoms = []
                symptom_patterns = ['头痛', '头晕', '发热', '咳嗽', '胸痛', '腹痛', '恶心', '呕吐', '乏力', '心悸']
                for symptom in symptom_patterns:
                    if symptom in text_content:
                        common_symptoms.append(symptom)
                
                if common_symptoms:
                    symptoms = "主诉" + "、".join(common_symptoms) + "。"
                else:
                    symptoms = "主诉暂无明确症状记录。"
            
            # 提取疾病/诊断信息
            diagnosis_lines = []
            for line in lines:
                if any(keyword in line for keyword in diagnosis_keywords):
                    # 清理诊断描述
                    clean_line = re.sub(r'(诊断[:：]?|分析[:：]?|考虑[:：]?)', '', line).strip()
                    if clean_line and len(clean_line) > 3:  # 过滤太短的内容
                        diagnosis_lines.append(clean_line)
            
            if diagnosis_lines:
                disease = "经分析，" + "。".join(diagnosis_lines) + "。"
            else:
                # 尝试寻找疾病名称
                disease_patterns = ['高血压', '糖尿病', '冠心病', '感冒', '发热', '肺炎', '胃炎', '肝炎']
                found_diseases = []
                for disease_name in disease_patterns:
                    if disease_name in text_content:
                        found_diseases.append(disease_name)
                
                if found_diseases:
                    disease = "经分析，可能存在" + "、".join(found_diseases) + "等疾病。"
                else:
                    disease = "经分析，需要进一步检查以明确诊断。"
            
            # 提取处方/建议信息
            prescription_lines = []
            for line in lines:
                if any(keyword in line for keyword in prescription_keywords):
                    # 清理建议描述
                    clean_line = re.sub(r'(建议[:：]?|治疗[:：]?|注意[:：]?)', '', line).strip()
                    if clean_line and len(clean_line) > 2:
                        prescription_lines.append(clean_line)
            
            if prescription_lines:
                prescription = "建议" + "，".join(prescription_lines) + "。"
            else:
                # 提供通用建议
                prescription = "建议注意休息，合理饮食，必要时及时就医。"
            
            return {
                "symptoms": symptoms,
                "disease": disease,
                "prescription": prescription
            }
            
        except Exception as e:
            logger.error(f"提取医学信息失败: {str(e)}")
            return {
                "symptoms": "信息提取失败",
                "disease": "信息提取失败",
                "prescription": "信息提取失败"
            }
    
    def analyze_document(self, document_content: str = None, file_path: str = None, filename: str = None) -> tuple[int, Dict[str, Any]]:
        """
        分析医学文档核心方法
        
        :param document_content: 直接提供的文档内容
        :param file_path: 需要OCR处理的文件路径
        :param filename: 原始文件名（用于保存结果）
        :return: (HTTP状态码, 响应数据)
        """
        try:
            text_content = ""
            
            # 如果提供了文件路径，使用OCR处理
            if file_path:
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.pdf':
                    logger.info(f"处理PDF文件: {file_path}")
                    text_content = self.process_pdf_ocr(file_path)
                elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    logger.info(f"处理图像文件: {file_path}")
                    text_content = self.process_image_ocr(file_path)
                else:
                    return 400, {
                        "success": False,
                        "message": f"不支持的文件类型: {file_extension}",
                        "data": {
                            "symptoms": "",
                            "disease": "",
                            "prescription": ""
                        }
                    }
                
                # 保存OCR结果到文件
                if filename:
                    self.save_ocr_result(text_content, filename)
            
            # 如果直接提供了文档内容
            elif document_content:
                text_content = document_content
            else:
                return 400, {
                    "success": False,
                    "message": "未提供文档内容或文件",
                    "data": {
                        "symptoms": "",
                        "disease": "",
                        "prescription": ""
                    }
                }
            
            # 提取医学信息
            medical_info = self.extract_medical_info(text_content)
            
            result = {
                "success": True,
                "message": "病历导入成功",
                "data": medical_info
            }
            
            logger.info("文档分析完成")
            return 200, result
            
        except Exception as e:
            logger.error(f"文档分析失败: {str(e)}")
            return 500, {
                "success": False,
                "message": f"文档分析失败: {str(e)}",
                "data": {
                    "symptoms": "",
                    "disease": "",
                    "prescription": ""
                }
            }
    
    def handle_import_request(self):
        """
        处理文档导入分析请求的入口方法
        接口 6.2: POST /api/import
        """
        try:
            # 检查是否为文件上传
            if 'document' in request.files:
                # 处理文件上传
                file = request.files['document']
                if file.filename == '':
                    return jsonify({
                        "success": False,
                        "message": "未选择文件",
                        "data": {
                            "symptoms": "",
                            "disease": "",
                            "prescription": ""
                        }
                    }), 400
                
                # 保存上传的文件到临时目录
                original_filename = file.filename
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(original_filename)[1]) as temp_file:
                    file.save(temp_file.name)
                    temp_file_path = temp_file.name
                
                try:
                    # 使用OCR处理文件
                    status_code, result = self.analyze_document(
                        file_path=temp_file_path, 
                        filename=original_filename
                    )
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                
            else:
                # 处理JSON数据
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
                
                document_content = data.get('content', '')
                
                if not document_content:
                    return jsonify({
                        "success": False,
                        "message": "文档内容不能为空",
                        "data": {
                            "symptoms": "",
                            "disease": "",
                            "prescription": ""
                        }
                    }), 400
                
                # 直接分析文档内容
                status_code, result = self.analyze_document(document_content=document_content)
            
            return jsonify(result), status_code
            
        except Exception as e:
            logger.error(f"文档导入请求处理失败: {str(e)}")
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
            "description": "文档导入分析API，使用OCR识别和分析医学文档",
            "endpoints": {
                "import": {
                    "method": "POST",
                    "path": "/api/import",
                    "body": "multipart/form-data（文件上传）或 JSON（文本内容）",
                    "description": "使用OCR分析上传的医学文档并提取症状、疾病、处方信息",
                    "supported_formats": ["PDF", "JPG", "JPEG", "PNG", "BMP", "GIF"],
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
            "status": "active",
            "ocr_engine": "PaddleOCR" if self.ocr else "未初始化",
            "max_pages": self.page_num,
            "output_directory": self.output_dir
        }