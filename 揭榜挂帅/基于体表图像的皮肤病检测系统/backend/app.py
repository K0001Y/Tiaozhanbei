from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import base64
import io
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from datetime import datetime
import traceback

# 导入你的模型类（需要确保main.py在同一目录下）
try:
    from models.main import ImprovedSwinUNet
except ImportError:
    print("警告: 无法导入ImprovedSwinUNet，请确保main.py文件存在")
    ImprovedSwinUNet = None

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置限流器
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# 全局配置
CONFIG = {
    'model_path': 'models/swinunet_lesion.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_image_size': 10 * 1024 * 1024,  # 10MB
    'threshold': 0.3,
    'image_size': (224, 224)
}

# 全局变量
model = None
transform_img = None

def init_model():
    """初始化模型"""
    global model, transform_img
    
    try:
        # 图像预处理
        transform_img = transforms.Compose([
            transforms.Resize(CONFIG['image_size']),
            transforms.ToTensor()
        ])
        
        # 加载模型
        if ImprovedSwinUNet is None:
            raise Exception("模型类未找到")
            
        model = ImprovedSwinUNet().to(CONFIG['device'])
        
        if os.path.exists(CONFIG['model_path']):
            model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
            model.eval()
            logger.info(f"模型加载成功，使用设备: {CONFIG['device']}")
        else:
            logger.error(f"模型文件不存在: {CONFIG['model_path']}")
            raise FileNotFoundError(f"模型文件不存在: {CONFIG['model_path']}")
            
    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")
        model = None

def decode_base64_image(base64_string):
    """解码base64图像"""
    try:
        # 处理data URL格式
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # 解码base64
        image_data = base64.b64decode(base64_string)
        
        # 检查文件大小
        if len(image_data) > CONFIG['max_image_size']:
            raise ValueError("图片文件过大")
        
        # 转换为PIL图像
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    
    except Exception as e:
        logger.error(f"图像解码失败: {str(e)}")
        raise ValueError(f"图像解码失败: {str(e)}")

def predict_mask(image, threshold=None):
    """预测图像掩码"""
    if threshold is None:
        threshold = CONFIG['threshold']
    
    try:
        if model is None:
            raise Exception("模型未初始化")
        
        # 图像预处理
        tensor_img = transform_img(image).unsqueeze(0).to(CONFIG['device'])
        
        # 预测
        with torch.no_grad():
            output = model(tensor_img, target_size=CONFIG['image_size'])
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # 二值化
        binary_mask = (pred_mask > threshold).astype(np.uint8)
        return binary_mask
    
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise Exception(f"预测失败: {str(e)}")

def calculate_area(mask):
    """计算掩码面积"""
    return int(np.sum(mask))

def diagnose_vitiligo(normal_image, uv_image):
    """诊断白癜风状态"""
    try:
        # 预测两张图片的掩码
        mask_normal = predict_mask(normal_image)
        mask_uv = predict_mask(uv_image)
        
        # 计算面积
        area_normal = calculate_area(mask_normal)
        area_uv = calculate_area(mask_uv)
        
        # 判断状态
        status = "发展期" if area_uv > area_normal else "稳定期"
        
        logger.info(f"诊断结果 - 普通光面积: {area_normal}, 紫外线面积: {area_uv}, 状态: {status}")
        
        return {
            'status': status,
            'normal_area': area_normal,
            'uv_area': area_uv
        }
    
    except Exception as e:
        logger.error(f"诊断失败: {str(e)}")
        raise Exception(f"诊断失败: {str(e)}")

@app.route('/api/diagnose', methods=['POST'])
@limiter.limit("10 per minute")
def diagnose():
    """图像诊断接口"""
    try:
        # 检查模型是否可用
        if model is None:
            return jsonify({
                'success': False,
                'message': '模型服务暂时不可用，请稍后重试'
            }), 500
        
        # 获取请求数据
        data = request.get_json()
        if not data or 'groups' not in data:
            return jsonify({
                'success': False,
                'message': '缺少必要参数groups'
            }), 400
        
        groups = data['groups']
        if not isinstance(groups, list) or len(groups) == 0:
            return jsonify({
                'success': False,
                'message': 'groups必须是非空数组'
            }), 400
        
        results = []
        
        for group in groups:
            # 验证必要字段
            if not all(key in group for key in ['groupId', 'normalImage', 'uvImage']):
                return jsonify({
                    'success': False,
                    'message': '每个图像组必须包含groupId、normalImage、uvImage字段'
                }), 400
            
            try:
                # 解码图像
                normal_image = decode_base64_image(group['normalImage'])
                uv_image = decode_base64_image(group['uvImage'])
                
                # 诊断
                diagnosis = diagnose_vitiligo(normal_image, uv_image)
                
                results.append({
                    'groupId': group['groupId'],
                    'result': diagnosis['status']
                })
                
            except ValueError as ve:
                if "图片文件过大" in str(ve):
                    return jsonify({
                        'success': False,
                        'message': '上传的图片文件过大，请压缩后重试'
                    }), 413
                else:
                    return jsonify({
                        'success': False,
                        'message': '缺少必要参数或图片格式不正确'
                    }), 400
            
            except Exception as e:
                logger.error(f"处理图像组 {group['groupId']} 失败: {str(e)}")
                return jsonify({
                    'success': False,
                    'message': '图像处理失败，请检查图片格式'
                }), 400
        
        return jsonify({
            'success': True,
            'results': results
        }), 200
    
    except Exception as e:
        logger.error(f"诊断接口异常: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': '服务器内部错误，请稍后重试'
        }), 500

@app.route('/api/dialog', methods=['POST'])
@limiter.limit("20 per minute")
def dialog():
    """对话交互接口"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'message': '消息内容不能为空'
            }), 400
        
        message = data['message'].strip()
        if not message:
            return jsonify({
                'success': False,
                'message': '消息内容不能为空'
            }), 400
        
        if len(message) > 1000:
            return jsonify({
                'success': False,
                'message': '消息长度不能超过1000字符'
            }), 400
        
        # 简单的白癜风知识问答（实际项目中可接入AI模型）
        context = data.get('context', '')
        
        # 这里使用简单的规则回复，实际项目中应该接入专业的AI对话模型
        reply = generate_reply(message, context)
        
        return jsonify({
            'success': True,
            'reply': reply
        }), 200
    
    except Exception as e:
        logger.error(f"对话接口异常: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'AI服务暂时不可用，请稍后重试'
        }), 500

def generate_reply(message, context):
    """生成回复（简单版本，实际项目中应接入AI模型）"""
    message_lower = message.lower()
    
    # 简单的关键词匹配回复
    if any(keyword in message_lower for keyword in ['白癜风', '症状', '诊断']):
        return """白癜风是一种常见的色素脱失性皮肤病，主要特征包括：

1. 皮肤出现白色斑块，边界清楚
2. 白斑部位毛发可能变白
3. 通常无自觉症状，不痛不痒
4. 好发于面部、手部、足部等暴露部位

诊断主要依据：
- 临床表现
- 伍德灯检查（紫外线灯）
- 皮肤镜检查
- 必要时进行皮肤活检

建议及时就医，进行专业诊断和治疗。本系统提供的AI分析仅供参考，不能替代医生的专业诊断。"""
    
    elif any(keyword in message_lower for keyword in ['治疗', '怎么办', '药物']):
        return """白癜风的治疗方案需要根据病情制定，常见治疗方法包括：

1. 外用药物治疗：
   - 糖皮质激素类药膏
   - 钙调磷酸酶抑制剂
   - 维生素D3衍生物

2. 光疗：
   - 窄谱UVB治疗
   - 308nm准分子激光

3. 口服药物：
   - 免疫调节剂
   - 维生素类

4. 手术治疗：
   - 自体表皮移植
   - 黑素细胞移植

重要提醒：
- 治疗方案必须由专业医生制定
- 不同患者适合的治疗方法不同
- 需要长期坚持治疗
- 避免自行用药

请务必到正规医院皮肤科就诊！"""
    
    elif any(keyword in message_lower for keyword in ['预防', '注意事项', '护理']):
        return """白癜风患者的日常护理和预防要点：

1. 避免外伤：
   - 避免皮肤外伤、摩擦
   - 选择宽松、柔软的衣物

2. 防晒措施：
   - 避免强烈阳光直射
   - 使用SPF30+的防晒霜
   - 外出时穿长袖衣物、戴帽子

3. 饮食调理：
   - 多吃富含酪氨酸的食物
   - 补充铜、锌等微量元素
   - 避免过量维生素C

4. 心理调节：
   - 保持积极乐观的心态
   - 适当运动，增强体质
   - 必要时寻求心理支持

5. 规律生活：
   - 保证充足睡眠
   - 避免熬夜
   - 戒烟限酒

定期复查，配合医生治疗是关键！"""
    
    else:
        return """您好！我是白癜风诊断系统的AI助手。

我可以为您提供以下帮助：
- 白癜风相关知识介绍
- 症状识别和诊断说明
- 治疗方法科普
- 日常护理建议
- 预防措施指导

请注意：本系统提供的信息仅供参考学习，不能替代专业医疗诊断。如有疑似症状，请及时就医咨询专业皮肤科医生。

您可以询问关于白癜风的任何问题，我会尽力为您解答。"""

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    status = "healthy" if model is not None else "unhealthy"
    return jsonify({
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'device': CONFIG['device']
    })

@app.errorhandler(429)
def ratelimit_handler(e):
    """处理限流错误"""
    return jsonify({
        'success': False,
        'message': '请求过于频繁，请稍后再试'
    }), 429

@app.errorhandler(413)
def request_entity_too_large(error):
    """处理请求体过大错误"""
    return jsonify({
        'success': False,
        'message': '上传的图片文件过大，请压缩后重试'
    }), 413

if __name__ == '__main__':
    # 启动时初始化模型
    print("正在初始化模型...")
    init_model()
    
    if model is None:
        print("警告: 模型初始化失败，服务器将在受限模式下运行")
    
    # 启动Flask应用
    print(f"服务器启动中...")
    print(f"使用设备: {CONFIG['device']}")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # 生产环境设为False
        threaded=True
    )