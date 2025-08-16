import os
import sys
from paddleocr import PaddleOCR
import fitz
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置OCR参数
PAGE_NUM = 10

def process_pdf(pdf_path, ocr):
    # 使用fitz渲染PDF
    all_text = []  # 存储所有文本
    
    try:
        with fitz.open(pdf_path) as pdf:
            total_pages = pdf.page_count  # 获取PDF总页数
            print(f"[DEBUG] Total pages in PDF: {total_pages}")
            
            # 对每一页单独进行OCR处理，而不是批量处理
            for pg in range(min(PAGE_NUM, total_pages)):  # 避免超出实际页数
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
                    
                    # 修复: 使用一致的参数
                    result = ocr.ocr(img_bgr, cls=True, det=True)
                    
                    # 添加页码信息
                    page_text = [f"==== 第 {pg + 1} 页 ===="]
                    
                    # 提取文本 - 修复结果处理逻辑
                    if result:  # 确保结果不为空
                        for line in result:  # PaddleOCR处理单张图像时返回的是一个列表
                            if line:  # 确保行不为空
                                for text_info in line:
                                    if len(text_info) >= 2:  # 确保包含文本信息
                                        text = text_info[1][0]  # 获取文本内容
                                        page_text.append(text)
                    else:
                        page_text.append("此页未识别到文本")
                    
                    # 将页面文本添加到总文本
                    all_text.append("\n".join(page_text))
                    
                except Exception as e:
                    error_msg = f"[WARNING] Error processing page {pg}: {str(e)}"
                    print(error_msg)
                    all_text.append(f"==== 第 {pg + 1} 页 ====\n{error_msg}")
                    continue
    except Exception as e:
        error_message = f"Error processing PDF: {str(e)}"
        print(f"[ERROR] {error_message}")
        all_text.append(error_message)
    
    return "\n\n".join(all_text)

def process_image(image_path, ocr):
    # 处理图像的OCR识别
    try:
        # 确保参数一致
        result = ocr.ocr(image_path, cls=True, det=True)
        text_lines = []
        
        # 修复结果处理逻辑
        if result:
            for line in result:
                if line:
                    for text_info in line:
                        if len(text_info) >= 2:
                            text_lines.append(text_info[1][0])  # 获取文本内容
        else:
            print(f"[DEBUG] Empty result detected for image.")
            text_lines.append("未识别到文本")
        
        return "\n".join(text_lines)
    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        print(f"[ERROR] {error_message}")
        return error_message

def process_file(file_path, ocr):
    # 获取文件扩展名并处理不同类型的文件
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))
    
    # 指定输出目录
    output_dir = 'data_system/ocr_results'
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 构建输出文件路径
    output_txt_path = os.path.join(output_dir, file_name + ".txt")
    
    try:
        if file_extension.lower() == '.pdf':
            print(f"[DEBUG] Detected PDF file: {file_path}")
            text_content = process_pdf(file_path, ocr)
        elif file_extension.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            print(f"[DEBUG] Detected image file: {file_path}")
            text_content = process_image(file_path, ocr)
        else:
            print(f"[ERROR] Unsupported file type: {file_extension}")
            text_content = f"Unsupported file type: {file_extension}"
        
        # 保存文本内容到txt文件
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        print(f"[INFO] OCR results saved to: {output_txt_path}")
        return output_txt_path
    except Exception as e:
        error_message = f"Error processing file: {str(e)}"
        print(f"[ERROR] {error_message}")
        
        # 即使出错也创建txt文件，包含错误信息
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(error_message)
        
        return output_txt_path

def main():
    # 如果有命令行参数，使用第一个参数作为文件路径
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"[INFO] 处理文件: {file_path}")
    else:
        print(f"[ERROR] 未提供文件路径")
        return
    
    # 初始化OCR引擎 - 修复: 使用一致的参数
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=PAGE_NUM, det=True)
    
    # 处理文件并获取生成的txt文件路径
    output_txt_path = process_file(file_path, ocr)
    
    # 打印结果路径用于调试
    print(f"[DEBUG] Text file generated: {output_txt_path}")
    # 返回生成的文本文件路径
    print(f"OUTPUT_PATH={output_txt_path}")

if __name__ == "__main__":
    main()