from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
import glob
import re
from keras.models import load_model
import tempfile

app = Flask(__name__)
CORS(app)

# 配置上传和输出文件夹
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 设置目标尺寸
TARGET_WIDTH, TARGET_HEIGHT = 224, 448

# 疾病标签映射（中文）
disease_labels = [
    '正常', '糖尿病', '青光眼', '白内障', 'AMD', '高血压', '近视', '其他疾病或异常'
]

# 疾病标签英文映射
disease_labels_en = [
    'Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other'
]

# 设置每个标签的阈值
thresholds = [0.5, 0.2, 0.5, 0.3, 0.3, 0.4, 0.1, 0.7]

# 模型路径
MODEL_PATH = 'model_fold_1.h5'

def allowed_file(filename):
    """检查文件是否为允许的扩展名"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_classification_model(model_path):
    """加载模型"""
    return load_model(model_path)

def process_image(image_data):
    """处理单张图片"""
    # 调整图像大小并归一化
    img_resized = cv2.resize(image_data, (TARGET_WIDTH, TARGET_HEIGHT))
    img_resized = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_resized, axis=0)  # 扩展批次维度
    return img_input

def resize_image(image, target_height):
    """调整图像大小，保持纵横比"""
    h, w = image.shape[:2]
    scale = target_height / h
    new_width = int(w * scale)
    return cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_LINEAR)

def stitch_two_images(image1, image2):
    """拼接两张图像"""
    h1, _ = image1.shape[:2]
    h2, _ = image2.shape[:2]
    if h1 != h2:
        if h1 < h2:
            image1 = resize_image(image1, h2)
        else:
            image2 = resize_image(image2, h1)
    return np.hstack((image1, image2))

def convert_prediction_to_diseases(prediction):
    """根据阈值将预测概率转换成疾病结果"""
    disease_result = []
    for i in range(len(prediction)):
        # 根据每个标签的阈值转换为 0 或 1
        disease_result.append(1 if prediction[i] > thresholds[i] else 0)
    
    # 获取预测的疾病
    diseases = []
    for i, result in enumerate(disease_result):
        if result == 1:
            diseases.append({
                "label": disease_labels[i],
                "label_en": disease_labels_en[i],
                "probability": float(prediction[i])
            })
    
    return {
        "binary_result": disease_result,
        "diseases": diseases
    }

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    """处理眼部图像并进行疾病诊断"""

    try:
        app.logger.info(f"收到请求数据: {request.files.keys()}")
        
        # 修复：检查以'leftEyeFiles'和'rightEyeFiles'开头的字段
        left_files = []
        right_files = []
        
        for key in request.files:
            if key.startswith('leftEyeFiles'):
                left_files.append(request.files[key])
            elif key.startswith('rightEyeFiles'):
                right_files.append(request.files[key])
        
        if not left_files or not right_files:
            app.logger.error(f"缺少必要的文件字段，收到的字段: {list(request.files.keys())}")
            return jsonify({'status': 'error', 'message': '缺少必要的文件'}), 400
        
        if len(left_files) != len(right_files):
            return jsonify({'status': 'error', 'message': '左右眼图片数量不匹配'}), 400
        
        # 确保模型已加载
        global model
        if 'model' not in globals():
            app.logger.info(f"Loading model from {MODEL_PATH}")
            model = load_classification_model(MODEL_PATH)
            app.logger.info("Model loaded successfully")
        
        results = []
        
        for i in range(len(left_files)):
            left_file = left_files[i]
            right_file = right_files[i]
            
            if left_file.filename == '' or right_file.filename == '':
                continue
                
            if allowed_file(left_file.filename) and allowed_file(right_file.filename):
                # 生成唯一ID
                patient_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                
                # 保存文件
                left_filename = secure_filename(f'left_{patient_id}.jpg')
                right_filename = secure_filename(f'right_{patient_id}.jpg')
                
                left_path = os.path.join(app.config['UPLOAD_FOLDER'], left_filename)
                right_path = os.path.join(app.config['UPLOAD_FOLDER'], right_filename)
                
                left_file.save(left_path)
                right_file.save(right_path)
                
                try:
                    # 读取图像文件
                    left_image = cv2.imread(left_path)
                    right_image = cv2.imread(right_path)
                    
                    if left_image is None or right_image is None:
                        app.logger.error(f"无法读取图像: {left_path} 或 {right_path}")
                        results.append({
                            'id': patient_id,
                            'name': f'诊断结果_{i+1}',
                            'disease': '错误',
                            'disease_en': 'Error',
                            'message': '无法读取图像文件',
                            'confidence': 0.0
                        })
                        continue
                    
                    # 拼接图像
                    stitched_image = stitch_two_images(left_image, right_image)
                    
                    # 保存拼接图像
                    stitch_output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{patient_id}_stitched.jpg")
                    cv2.imwrite(stitch_output_path, stitched_image)
                    
                    # 处理图像
                    processed_img = process_image(stitched_image)
                    
                    # 模型预测
                    prediction = model.predict(processed_img)[0]
                    
                    # 转换预测结果
                    result = convert_prediction_to_diseases(prediction)
                    
                    # 提取主要的疾病（概率最高的一个）
                    primary_disease = "正常"  # 默认为正常
                    primary_disease_en = "Normal"  # 默认英文为正常
                    max_prob = 0
                    
                    # 首先检查是否有检测到的疾病
                    detected_diseases = result.get("diseases", [])
                    if detected_diseases:
                        # 从检测到的疾病中找出概率最高的
                        primary_disease = detected_diseases[0]["label"]
                        primary_disease_en = detected_diseases[0]["label_en"]
                        max_prob = detected_diseases[0]["probability"]
                        
                        for disease in detected_diseases[1:]:
                            if disease["probability"] > max_prob:
                                max_prob = disease["probability"]
                                primary_disease = disease["label"]
                                primary_disease_en = disease["label_en"]
                    else:
                        # 如果没有检测到疾病，找出概率最高的标签
                        probabilities = prediction.tolist()
                        max_index = np.argmax(probabilities)
                        primary_disease = disease_labels[max_index]
                        primary_disease_en = disease_labels_en[max_index]
                        max_prob = probabilities[max_index]
                    
                    # 添加到诊断结果
                    results.append({
                        'id': patient_id,
                        'name': f'诊断结果_{i+1}',
                        'disease': primary_disease,
                        'disease_en': primary_disease_en,
                        'confidence': float(max_prob),
                        'image_path': stitch_output_path,
                        'all_probabilities': prediction.tolist(),
                        'detected_diseases': [{'name': d["label"], 'name_en': d["label_en"], 'probability': d["probability"]} for d in detected_diseases]
                    })
                    
                except Exception as e:
                    app.logger.error(f"处理眼部图像时出错 {patient_id}: {str(e)}")
                    results.append({
                        'id': patient_id,
                        'name': f'诊断结果_{i+1}',
                        'disease': '错误',
                        'disease_en': 'Error',
                        'message': str(e),
                        'confidence': 0.0
                    })
        
        # 保存诊断结果到CSV文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"diagnosis_results_{timestamp}.csv")
        with open(csv_output_path, 'w', encoding='utf-8') as f:
            # 写入CSV头
            f.write("ID,疾病,疾病(英文),置信度\n")
            # 写入每条诊断结果
            for result in results:
                disease = result.get("disease", "错误")
                disease_en = result.get("disease_en", "Error")
                conf = result.get("confidence", 0.0)
                f.write(f"{result['id']},{disease},{disease_en},{conf:.4f}\n")
        
        return jsonify({
            'status': 'success',
            'diagnosis': results,
            'csv_output_path': csv_output_path,
            'total_processed': len(results)
        })
    
    except Exception as e:
        app.logger.error(f"诊断处理中发生错误: {str(e)}")
        return jsonify({'status': 'error', 'message': f'处理请求时发生错误: {str(e)}'}), 500

@app.route('/api/batch-diagnose', methods=['POST'])
def batch_diagnose():
    """批量处理左右眼文件夹中的图像"""
    try:
        # 检查请求格式
        if not request.json:
            return jsonify({"status": "error", "message": "未提供JSON数据"}), 400
        
        request_data = request.json
        
        # 检查必要的字段
        if 'leftEyeFolder' not in request_data or 'rightEyeFolder' not in request_data:
            return jsonify({"status": "error", "message": "请求中缺少leftEyeFolder或rightEyeFolder"}), 400
        
        left_eye_folder = request_data.get('leftEyeFolder')
        right_eye_folder = request_data.get('rightEyeFolder')
        output_folder = request_data.get('outputFolder', app.config['OUTPUT_FOLDER'])
        
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 匹配左右眼文件
        matched_pairs = match_eye_files(left_eye_folder, right_eye_folder)
        
        if not matched_pairs:
            return jsonify({"status": "error", "message": "在文件夹中未找到匹配的眼睛对"}), 400
        
        # 确保模型已加载
        global model
        if 'model' not in globals():
            app.logger.info(f"从 {MODEL_PATH} 加载模型")
            model = load_classification_model(MODEL_PATH)
            app.logger.info("模型加载成功")
        
        # 处理结果
        diagnosis_results = []
        
        # 处理每对眼睛图片
        for left_path, right_path, patient_id in matched_pairs:
            try:
                # 读取图像文件
                left_image = cv2.imread(left_path)
                right_image = cv2.imread(right_path)
                
                if left_image is None or right_image is None:
                    app.logger.error(f"无法读取图像: {left_path} 或 {right_path}")
                    continue
                
                # 拼接图像
                stitched_image = stitch_two_images(left_image, right_image)
                
                # 保存拼接图像
                stitch_output_path = os.path.join(output_folder, f"{patient_id}_stitched.jpg")
                cv2.imwrite(stitch_output_path, stitched_image)
                
                # 处理图像
                processed_img = process_image(stitched_image)
                
                # 模型预测
                prediction = model.predict(processed_img)[0]
                
                # 转换预测结果
                result = convert_prediction_to_diseases(prediction)
                
                # 提取主要的疾病（概率最高的一个）
                primary_disease = "正常"  # 默认为正常
                primary_disease_en = "Normal"  # 默认英文为正常
                max_prob = 0
                
                # 首先检查是否有检测到的疾病
                detected_diseases = result.get("diseases", [])
                if detected_diseases:
                    # 从检测到的疾病中找出概率最高的
                    primary_disease = detected_diseases[0]["label"]
                    primary_disease_en = detected_diseases[0]["label_en"]
                    max_prob = detected_diseases[0]["probability"]
                    
                    for disease in detected_diseases[1:]:
                        if disease["probability"] > max_prob:
                            max_prob = disease["probability"]
                            primary_disease = disease["label"]
                            primary_disease_en = disease["label_en"]
                else:
                    # 如果没有检测到疾病，找出概率最高的标签
                    probabilities = prediction.tolist()
                    max_index = np.argmax(probabilities)
                    primary_disease = disease_labels[max_index]
                    primary_disease_en = disease_labels_en[max_index]
                    max_prob = probabilities[max_index]
                
                # 添加到诊断结果
                diagnosis_results.append({
                    "id": patient_id,
                    "disease": primary_disease,
                    "disease_en": primary_disease_en,
                    "probability": float(max_prob),
                    "left_eye_path": left_path,
                    "right_eye_path": right_path,
                    "stitched_path": stitch_output_path,
                    "all_probabilities": prediction.tolist(),
                    "detected_diseases": [{'name': d["label"], 'name_en': d["label_en"], 'probability': d["probability"]} for d in detected_diseases]
                })
                
                app.logger.info(f"处理完成ID {patient_id} - 诊断为: {primary_disease}")
            
            except Exception as e:
                app.logger.error(f"处理眼睛对 {patient_id} 时出错: {str(e)}")
                # 出错时添加错误信息
                diagnosis_results.append({
                    "id": patient_id,
                    "disease": "错误",
                    "disease_en": "Error",
                    "error_message": str(e),
                    "left_eye_path": left_path,
                    "right_eye_path": right_path
                })
        
        # 保存诊断结果到CSV文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_output_path = os.path.join(output_folder, f"diagnosis_results_{timestamp}.csv")
        with open(csv_output_path, 'w', encoding='utf-8') as f:
            # 写入CSV头
            f.write("ID,疾病,疾病(英文),置信度\n")
            # 写入每条诊断结果
            for result in diagnosis_results:
                disease = result.get("disease", "错误")
                disease_en = result.get("disease_en", "Error")
                prob = result.get("probability", 0.0)
                f.write(f"{result['id']},{disease},{disease_en},{prob:.4f}\n")
        
        # 返回结果
        return jsonify({
            "status": "success",
            "diagnosis": diagnosis_results,
            "csv_output_path": csv_output_path,
            "total_processed": len(diagnosis_results)
        })
    
    except Exception as e:
        app.logger.error(f"批量诊断中发生错误: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

def extract_id_from_filename(filename):
    """从文件名中提取ID号码"""
    # 尝试找出文件名中的数字部分作为ID
    basename = os.path.basename(filename)
    # 方法1: 直接查找数字序列
    numbers = re.findall(r'\d+', basename)
    if numbers:
        return numbers[0]
    
    # 方法2: 如果没有数字，使用文件名的前缀部分（去掉左右眼和扩展名标识）
    name_parts = basename.lower().split('_')
    for part in name_parts:
        if not any(keyword in part for keyword in ['left', 'right', 'l', 'r', 'eye', '左', '右', '眼']):
            return part
    
    # 如果都没找到合适的ID，返回文件名（不含扩展名）
    return os.path.splitext(basename)[0]

def match_eye_files(left_folder, right_folder):
    """匹配左右眼文件"""
    # 获取两个文件夹中的所有图像文件
    left_eye_files = glob.glob(os.path.join(left_folder, '*.jpg')) + \
                    glob.glob(os.path.join(left_folder, '*.jpeg')) + \
                    glob.glob(os.path.join(left_folder, '*.png'))
    
    right_eye_files = glob.glob(os.path.join(right_folder, '*.jpg')) + \
                     glob.glob(os.path.join(right_folder, '*.jpeg')) + \
                     glob.glob(os.path.join(right_folder, '*.png'))
    
    # 提取每个文件的ID
    left_ids = {extract_id_from_filename(f): f for f in left_eye_files}
    right_ids = {extract_id_from_filename(f): f for f in right_eye_files}
    
    # 找到匹配的文件对
    matched_pairs = []
    
    # 查找左右眼图像的匹配对
    for id_key in left_ids:
        if id_key in right_ids:
            matched_pairs.append((left_ids[id_key], right_ids[id_key], id_key))
    
    return matched_pairs

# 初始化模型（在应用启动时加载）
@app.before_first_request
def initialize_model():
    global model
    app.logger.info(f"从 {MODEL_PATH} 加载模型")
    try:
        model = load_classification_model(MODEL_PATH)
        app.logger.info("模型加载成功")
    except Exception as e:
        app.logger.error(f"加载模型时出错: {str(e)}")
        # 模型加载失败时，初始化为None
        model = None

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=5000)