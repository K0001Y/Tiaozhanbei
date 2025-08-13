from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import base64
import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import requests
import json
import os
from typing import Dict, List, Any

# 导入你的模型类
from models.main import ImprovedSwinUNet

app = Flask(__name__)

# 设置请求频率限制
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# 配置
MODEL_PATH = "models\swinunet_lesion.pth"  # 模型文件路径
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
THRESHOLD = 0.3

# 全局变量
model = None
device = None

def init_model():
    """初始化模型"""
    global model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImprovedSwinUNet().to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"✅ 模型已加载到 {device}")
    else:
        print(f"❌ 模型文件 {MODEL_PATH} 不存在")

def decode_base64_image(base64_str: str) -> Image.Image:
    """解码base64图片"""
    try:
        # 移除data:image/xxx;base64,前缀
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # 解码
        image_data = base64.b64decode(base64_str)
        
        # 检查文件大小
        if len(image_data) > MAX_IMAGE_SIZE:
            raise ValueError("图片文件过大")
            
        # 转换为PIL Image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"图片解码失败: {str(e)}")

def predict_mask(image: Image.Image) -> np.ndarray:
    """预测图片mask"""
    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    tensor_img = transform_img(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_img, target_size=(224, 224))
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    return (pred_mask > THRESHOLD).astype(np.uint8)

def calculate_area(mask: np.ndarray) -> int:
    """计算mask面积"""
    return np.sum(mask)

def diagnose_vitiligo(normal_image: Image.Image, uv_image: Image.Image) -> str:
    """诊断白癜风状态"""
    try:
        # 预测两张图片的mask
        mask_normal = predict_mask(normal_image)
        mask_uv = predict_mask(uv_image)
        
        # 计算面积
        area_normal = calculate_area(mask_normal)
        area_uv = calculate_area(mask_uv)
        
        # 判断状态：紫外线图片面积大于普通图片表示发展期
        status = "发展期" if area_uv > area_normal else "稳定期"
        return status
        
    except Exception as e:
        raise RuntimeError(f"诊断失败: {str(e)}")

def call_deepseek_api(message: str, context: str = None) -> str:
    """调用DeepSeek API进行对话"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # 构建系统提示
        system_prompt = """你是一个专业的白癜风疾病咨询助手。请注意以下几点：
1. 只回答与白癜风相关的问题
2. 提供准确、专业的医疗信息
3. 提醒用户仅供参考，不能替代专业医疗诊断
4. 如果问题与白癜风无关，请礼貌地引导用户咨询相关问题
5. 回答要简洁明了，通俗易懂"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API调用失败: {str(e)}")
    except KeyError:
        raise RuntimeError("API响应格式错误")

@app.route("/api/diagnose", methods=["POST"])
@limiter.limit("10 per minute")
def diagnose():
    """图像诊断API"""
    try:
        # 检查模型是否加载
        if model is None:
            return jsonify({
                "success": False,
                "message": "模型未加载"
            }), 500
            
        # 获取请求数据
        data = request.get_json()
        if not data or "groups" not in data:
            return jsonify({
                "success": False,
                "message": "缺少必要参数"
            }), 400
            
        groups = data["groups"]
        if not isinstance(groups, list) or len(groups) == 0:
            return jsonify({
                "success": False,
                "message": "groups参数格式错误"
            }), 400
            
        results = []
        
        # 处理每个图像组
        for group in groups:
            try:
                # 验证必要字段
                required_fields = ["groupId", "normalImage", "uvImage"]
                for field in required_fields:
                    if field not in group:
                        return jsonify({
                            "success": False,
                            "message": f"缺少必要参数: {field}"
                        }), 400
                
                # 解码图片
                normal_image = decode_base64_image(group["normalImage"])
                uv_image = decode_base64_image(group["uvImage"])
                
                # 进行诊断
                result = diagnose_vitiligo(normal_image, uv_image)
                
                results.append({
                    "groupId": group["groupId"],
                    "result": result
                })
                
            except ValueError as e:
                if "图片文件过大" in str(e):
                    return jsonify({
                        "success": False,
                        "message": "上传的图片文件过大，请压缩后重试"
                    }), 413
                else:
                    return jsonify({
                        "success": False,
                        "message": "缺少必要参数或图片格式不正确"
                    }), 400
            except Exception as e:
                return jsonify({
                    "success": False,
                    "message": f"处理图像组 {group.get('groupId', 'unknown')} 时发生错误"
                }), 500
        
        return jsonify({
            "success": True,
            "results": results
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": "服务器内部错误，请稍后重试"
        }), 500

@app.route("/api/dialog", methods=["POST"])
@limiter.limit("20 per minute")
def dialog():
    """对话交互API"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({
                "success": False,
                "message": "缺少必要参数"
            }), 400
            
        message = data["message"]
        if not message or len(message.strip()) == 0:
            return jsonify({
                "success": False,
                "message": "消息内容不能为空"
            }), 400
            
        if len(message) > 1000:
            return jsonify({
                "success": False,
                "message": "消息长度超过限制（1000字符）"
            }), 400
            
        context = data.get("context", "")
        
        # 调用DeepSeek API
        reply = call_deepseek_api(message, context)
        
        return jsonify({
            "success": True,
            "reply": reply
        }), 200
        
    except RuntimeError as e:
        if "API调用失败" in str(e):
            return jsonify({
                "success": False,
                "message": "AI服务暂时不可用，请稍后重试"
            }), 500
        else:
            return jsonify({
                "success": False,
                "message": str(e)
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "message": "AI服务暂时不可用，请稍后重试"
        }), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    """请求频率限制处理"""
    return jsonify({
        "success": False,
        "message": "请求过于频繁，请稍后再试"
    }), 429

@app.errorhandler(413)
def payload_too_large(e):
    """请求体过大处理"""
    return jsonify({
        "success": False,
        "message": "上传的图片文件过大，请压缩后重试"
    }), 413

@app.route("/health", methods=["GET"])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }), 200

if __name__ == "__main__":
    # 初始化模型
    init_model()
    
    # 启动服务
    app.run(host="0.0.0.0", port=5000, debug=False)