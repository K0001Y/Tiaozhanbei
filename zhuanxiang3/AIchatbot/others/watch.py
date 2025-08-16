import os
import json
import re
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from enum import Enum
from config import ALI_API_KEY, ALI_BASE_URL

# 初始化客户端 - 保持不变
client = OpenAI(
    api_key=ALI_API_KEY,
    base_url=ALI_BASE_URL
)

class ImageType(Enum):
    """图像类型枚举 - 保持不变"""
    TONGUE = "舌诊"
    FACE = "面诊" 
    HAND = "手诊"
    EYE = "眼诊"
    EAR = "耳诊"
    BODY = "体诊"
    UNKNOWN = "未知"

class TCMDiagnosisSystem:
    """中医望诊AI系统 - 核心修改在这里"""
    
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """初始化系统 - 保持不变"""
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "qwen-vl-max"
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """
        Linus式解决方案：robust JSON提取
        "做一件事，做好它"
        """
        if not text or not text.strip():
            return None
            
        text = text.strip()
        
        # 策略1：直接解析(最常见情况)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 策略2：提取JSON块(处理带解释文字的情况)
        # 找最外层的 { ... }
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # 策略3：寻找嵌套JSON
        brace_count = 0
        start_idx = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        json_str = text[start_idx:i+1]
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        start_idx = -1
                        continue
        
        return None
    
    def _safe_api_call(self, messages: List[Dict], default_response: Dict) -> str:
        """
        Linus式API调用：永远不崩溃
        "错误处理应该是boring的"
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1
            )
            
            if not response.choices or not response.choices[0].message.content:
                return json.dumps(default_response, ensure_ascii=False)
                
            content = response.choices[0].message.content.strip()
            
            # 尝试提取JSON
            extracted_json = self._extract_json_from_text(content)
            if extracted_json:
                return json.dumps(extracted_json, ensure_ascii=False)
            else:
                # 如果完全无法解析，返回原始内容包装
                fallback = {
                    **default_response,
                    "原始响应": content,
                    "解析状态": "JSON提取失败，返回原始内容"
                }
                return json.dumps(fallback, ensure_ascii=False)
                
        except Exception as e:
            error_response = {
                **default_response,
                "错误": f"API调用失败: {str(e)}",
                "建议": "请检查网络连接和API配置"
            }
            return json.dumps(error_response, ensure_ascii=False)
    
    def identify_image_type(self, image_url: str) -> Tuple[ImageType, float]:
        """
        识别图像类型 - 保持接口不变，内部robust化
        """
        identification_prompt = """
        你是专业的医学图像识别AI。识别图片的中医望诊类型：舌诊、面诊、手诊、眼诊、耳诊、体诊、未知。

        严格按此JSON格式返回：
        {
            "image_type": "类型名称",  
            "confidence": 0.95,
            "description": "简要描述"
        }
        """
        
        messages = [
            {"role": "system", "content": identification_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "识别图像类型，返回JSON"}
                ]
            }
        ]
        
        # 默认响应
        default_response = {
            "image_type": "未知",
            "confidence": 0.0,
            "description": "识别失败"
        }
        
        # 安全API调用
        result_str = self._safe_api_call(messages, default_response)
        
        try:
            result = json.loads(result_str)
            
            # 类型映射 - 保持不变
            image_type_map = {
                "舌诊": ImageType.TONGUE,
                "面诊": ImageType.FACE,
                "手诊": ImageType.HAND,
                "眼诊": ImageType.EYE,
                "耳诊": ImageType.EAR,
                "体诊": ImageType.BODY,
                "未知": ImageType.UNKNOWN
            }
            
            image_type = image_type_map.get(result.get("image_type", "未知"), ImageType.UNKNOWN)
            confidence = float(result.get("confidence", 0.0))
            
            return image_type, confidence
            
        except Exception as e:
            print(f"图像识别失败: {e}")
            return ImageType.UNKNOWN, 0.0
    
    def _make_diagnosis_request(self, system_prompt: str, user_prompt: str, image_url: str) -> str:
        """
        发送诊断请求 - 保持接口，增强robust性
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        default_response = {
            "错误": "分析失败",
            "建议": "请检查图像质量或重试"
        }
        
        return self._safe_api_call(messages, default_response)
    
    # 保持所有其他方法完全不变
    def analyze_tongue(self, image_url: str) -> str:
        """舌诊分析 - 保持不变"""
        system_prompt = """
        你是一名专业的中医舌诊AI助手，请对用户提供的舌头图像进行多维度特征分析。
        要求分析结果客观、准确、符合中医舌诊理论，严格按以下维度提取特征：

        特征提取维度：
        1. 舌质分析：颜色/形态/质地
        2. 舌苔分析：苔色/苔质/润燥
        3. 分区特征：
           - 舌尖（心肺区）
           - 舌中（脾胃区）
           - 舌根（肾区）
           - 舌边（肝胆区）

        输出要求：
        1. 必须使用严格JSON格式输出
        2. 包含中医理论依据和西医关联提示
        3. 评估图像诊断适用性
        """

        user_prompt = """
        请输出结构化JSON数据，格式如下：
        {
          "诊断类型": "舌诊",
          "舌质": {
            "颜色": {"主色": "", "异常色斑": ""},
            "形态": {"整体形状": "", "分区特征": "", "特殊特征": []},
            "湿度评分": "1-5级"
          },
          "舌苔": {
            "厚度": "mm级估算",
            "覆盖率": "百分比",
            "质地描述": []
          },
          "辨证提示": [],
          "中医理论依据": "",
          "西医可能关联提示": "",
          "图像质量评估": "",
          "健康建议": []
        }

        注意：
        1. 湿度评分：1=干燥龟裂 → 5=水滑湿润
        2. 辨证提示格式：["证型名称(置信度%)", ...]
        3. 图像质量需评估：清晰度/光照/角度/遮挡物
        """

        return self._make_diagnosis_request(system_prompt, user_prompt, image_url)
    
    # 其他analyze_*方法保持完全不变...
    def analyze_face(self, image_url: str) -> str:
        """面诊分析"""
        system_prompt = """
        你是一名专业的中医面诊AI助手，请对用户提供的面部图像进行中医面诊分析。
        根据中医面诊理论，分析面部五官、气色、形态等特征。

        分析维度：
        1. 面部气色：红、黄、白、青、黑五色分析
        2. 五官特征：眼、鼻、口、耳、眉的形态和色泽
        3. 面部分区：
           - 额部（心区）
           - 鼻部（脾胃区）
           - 左颊（肝区）
           - 右颊（肺区）
           - 下颌（肾区）
        4. 皮肤状态：润燥、光泽、纹理
        """

        user_prompt = """
        请输出结构化JSON数据：
        {
          "诊断类型": "面诊",
          "面部气色": {
            "主色调": "",
            "异常色斑": [],
            "光泽度": ""
          },
          "五官分析": {
            "眼部": {"形态": "", "色泽": "", "神采": ""},
            "鼻部": {"形态": "", "色泽": ""},
            "口唇": {"形态": "", "色泽": "", "润燥": ""},
            "耳部": {"形态": "", "色泽": ""},
            "眉毛": {"形态": "", "色泽": ""}
          },
          "面部分区": {
            "额部": "",
            "鼻部": "",
            "左颊": "",
            "右颊": "",
            "下颌": ""
          },
          "辨证提示": [],
          "中医理论依据": "",
          "西医可能关联提示": "",
          "图像质量评估": "",
          "健康建议": []
        }
        """

        return self._make_diagnosis_request(system_prompt, user_prompt, image_url)
    
    def analyze_hand(self, image_url: str) -> str:
        """手诊分析"""
        system_prompt = """
        你是一名专业的中医手诊AI助手，请对用户提供的手部图像进行多维度特征分析。
        要求分析结果客观、准确、符合中医手诊理论，严格按以下维度提取特征：

        特征提取维度（结合手心手背）：
        1. 手掌整体分析（手心）：
           - 颜色分区：大小鱼际/掌心/指根
           - 温度/湿度/弹性评估
        2. 手背特征分析：
           - 静脉分布/浮肿程度/皮肤纹理
        3. 手指特征（手心手背结合）：
           - 形态/长度比例/关节状态
           - 指腹饱满度/指节纹路
        4. 指甲分析（手背为主）：
           - 甲色/甲形/月牙/纹理
        5. 掌纹分析（手心为主）：
           - 生命线（长度、清晰度、断裂）
           - 智慧线（走势、分叉）
           - 感情线（连贯性、岛纹）
           - 健康线（有无、深浅）
        6. 分区反射区：
           - 心区（手掌大鱼际）
           - 肝区（食指下方）
           - 脾区（小指下方）
           - 肺区（无名指下方）
           - 肾区（手掌根部）

        输出要求：
        1. 必须使用严格JSON格式输出
        2. 包含中医理论依据和西医关联提示
        3. 分别评估手心手背图像质量
        """

        user_prompt = """
        请输出结构化JSON数据，格式如下：
        {
          "诊断类型": "手诊",
          "手掌整体分析": {
            "颜色分区": {
              "大鱼际": {"颜色": "", "异常": ""},
              "小鱼际": {"颜色": "", "异常": ""},
              "掌心": {"颜色": "", "异常": ""},
              "指根": {"颜色": "", "异常": ""}
            },
            "质地评估": {
              "温度感": "1-5级评估",
              "湿度感": "1-5级评估", 
              "弹性感": "1-5级评估"
            }
          },
          "手背特征分析": {
            "静脉分布": {"明显程度": "", "分布特点": "", "异常表现": []},
            "浮肿程度": "0-3级",
            "皮肤纹理": {"粗糙度": "", "光泽度": "", "特殊纹理": []}
          },
          "手指特征": {
            "形态分析": {
              "拇指": {"长度比例": "", "形态": "", "关节状态": ""},
              "食指": {"长度比例": "", "形态": "", "关节状态": ""},
              "中指": {"长度比例": "", "形态": "", "关节状态": ""},
              "无名指": {"长度比例": "", "形态": "", "关节状态": ""},
              "小指": {"长度比例": "", "形态": "", "关节状态": ""}
            },
            "指腹特征": {
              "饱满度": "1-5级",
              "弹性": "1-5级",
              "纹路清晰度": "1-5级"
            }
          },
          "指甲分析": {
            "甲色": {"整体色调": "", "异常颜色": []},
            "甲形": {"形状": "", "厚薄": "", "生长状态": ""},
            "月牙": {"有无": "", "大小": "", "颜色": ""},
            "甲面纹理": {"光滑度": "", "纵纹": "", "横纹": "", "斑点": []}
          },
          "掌纹分析": {
            "生命线": {
              "长度": "短/中/长",
              "清晰度": "1-5级",
              "深浅": "浅/中/深",
              "断裂情况": "",
              "起止位置": ""
            },
            "智慧线": {
              "走势": "直线/弧线/下垂",
              "长度": "短/中/长",
              "分叉": "有无及位置",
              "清晰度": "1-5级"
            },
            "感情线": {
              "连贯性": "连贯/断续",
              "深浅": "浅/中/深",
              "岛纹": "有无及位置",
              "终点位置": ""
            },
            "健康线": {
              "存在": "有/无",
              "清晰度": "1-5级",
              "走向": "",
              "异常": []
            }
          },
          "分区反射区": {
            "心区_大鱼际": {"色泽": "", "丰满度": "", "异常": ""},
            "肝区_食指下": {"色泽": "", "丰满度": "", "异常": ""},
            "脾区_小指下": {"色泽": "", "丰满度": "", "异常": ""},
            "肺区_无名指下": {"色泽": "", "丰满度": "", "异常": ""},
            "肾区_手掌根": {"色泽": "", "丰满度": "", "异常": ""}
          },
          "辨证提示": [],
          "中医理论依据": "",
          "西医可能关联提示": "",
          "图像质量评估": {
            "手心图像": {"清晰度": "1-5级", "光照": "1-5级", "角度": "1-5级"},
            "手背图像": {"清晰度": "1-5级", "光照": "1-5级", "角度": "1-5级"},
            "整体评价": ""
          },
          "健康建议": []
        }

        注意事项：
        1. 各项评分说明：
           - 温度感：1=凉寒 → 5=温热
           - 湿度感：1=干燥 → 5=潮湿
           - 弹性感：1=松弛 → 5=紧实
           - 浮肿程度：0=无 1=轻度 2=中度 3=重度
           - 清晰度：1=模糊不清 → 5=清晰可见
        2. 辨证提示格式：["证型名称(置信度%)", "脏腑异常(置信度%)", ...]
        3. 分别评估手心和手背的图像质量
        4. 结合手心手背特征进行综合分析
        """

        return self._make_diagnosis_request(system_prompt, user_prompt, image_url)
    
    def analyze_eye(self, image_url: str) -> str:
        """眼诊分析"""
        system_prompt = """
        你是一名专业的中医眼诊AI助手，请对用户提供的眼部图像进行中医眼诊分析。
        根据中医眼诊理论，分析眼部形态、色泽、神采等特征。

        分析维度：
        1. 眼神：有神/无神、明亮度
        2. 眼睑：上下眼睑形态、色泽
        3. 眼白：色泽、血丝、黄疸征象
        4. 瞳孔：大小、反应
        5. 眼周：黑眼圈、眼袋、皱纹
        """

        user_prompt = """
        请输出结构化JSON数据：
        {
          "诊断类型": "眼诊",
          "眼神评估": {
            "神采": "",
            "明亮度": "",
            "专注度": ""
          },
          "眼部结构": {
            "上眼睑": "",
            "下眼睑": "",
            "眼白": "",
            "瞳孔": ""
          },
          "眼周特征": {
            "黑眼圈": "",
            "眼袋": "",
            "细纹": ""
          },
          "辨证提示": [],
          "中医理论依据": "",
          "西医可能关联提示": "",
          "图像质量评估": "",
          "健康建议": []
        }
        """

        return self._make_diagnosis_request(system_prompt, user_prompt, image_url)
    
    def analyze_ear(self, image_url: str) -> str:
        """耳诊分析"""
        system_prompt = """
        你是一名专业的中医耳诊AI助手，请对用户提供的耳部图像进行中医耳诊分析。
        根据中医耳诊理论，分析耳部形态、色泽等特征。

        分析维度：
        1. 耳郭形态：大小、厚薄、形状
        2. 耳部色泽：红润度、异常颜色
        3. 耳穴对应：各个耳穴区域的状态
        4. 耳垂特征：厚薄、褶皱
        """

        user_prompt = """
        请输出结构化JSON数据：
        {
          "诊断类型": "耳诊",
          "耳郭形态": {
            "大小": "",
            "厚薄": "",
            "整体形状": ""
          },
          "耳部色泽": {
            "整体色调": "",
            "异常区域": []
          },
          "耳穴分析": {
            "上耳轮": "",
            "中耳轮": "",
            "下耳轮": "",
            "耳甲": ""
          },
          "耳垂特征": "",
          "辨证提示": [],
          "中医理论依据": "",
          "西医可能关联提示": "",
          "图像质量评估": "",
          "健康建议": []
        }
        """

        return self._make_diagnosis_request(system_prompt, user_prompt, image_url)
    
    def analyze_body(self, image_url: str) -> str:
        """体诊分析"""
        system_prompt = """
        你是一名专业的中医体诊AI助手，请对用户提供的身体图像进行中医体诊分析。
        根据中医体诊理论，分析体型、姿态、皮肤等特征。

        分析维度：
        1. 体型特征：胖瘦、高矮、比例
        2. 姿态观察：站姿、坐姿、精神状态
        3. 皮肤状态：色泽、纹理、异常
        4. 肌肉状态：结实度、对称性
        """

        user_prompt = """
        请输出结构化JSON数据：
        {
          "诊断类型": "体诊",
          "体型特征": {
            "整体评估": "",
            "胖瘦程度": "",
            "比例协调": ""
          },
          "姿态观察": {
            "站姿": "",
            "精神状态": ""
          },
          "皮肤状态": {
            "色泽": "",
            "纹理": "",
            "异常表现": []
          },
          "辨证提示": [],
          "中医理论依据": "",
          "西医可能关联提示": "",
          "图像质量评估": "",
          "健康建议": []
        }
        """

        return self._make_diagnosis_request(system_prompt, user_prompt, image_url)
    
    def comprehensive_diagnosis(self, image_url: str) -> Dict:
        """
        综合诊断：自动识别图像类型并进行相应分析
        保持接口完全不变
        """
        # 1. 识别图像类型
        image_type, confidence = self.identify_image_type(image_url)
        
        print(f"检测到图像类型: {image_type.value} (置信度: {confidence:.2f})")
        
        # 2. 根据类型选择分析方法 - 保持不变
        analysis_methods = {
            ImageType.TONGUE: self.analyze_tongue,
            ImageType.FACE: self.analyze_face,
            ImageType.HAND: self.analyze_hand,
            ImageType.EYE: self.analyze_eye,
            ImageType.EAR: self.analyze_ear,
            ImageType.BODY: self.analyze_body,
        }
        
        if image_type in analysis_methods:
            result = analysis_methods[image_type](image_url)
        else:
            result = json.dumps({
                "错误": "无法识别的图像类型",
                "建议": "请上传清晰的中医望诊相关图像（舌头、面部、手部、眼部、耳部或身体）"
            }, ensure_ascii=False)
        
        # 3. 构建完整诊断结果 - 保持不变
        diagnosis_result = {
            "图像识别": {
                "类型": image_type.value,
                "置信度": confidence
            },
            "分析结果": result,
            "分析时间": self._get_current_time()
        }
        
        return diagnosis_result
    
    def _get_current_time(self) -> str:
        """获取当前时间 - 保持不变"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 使用示例和测试功能 - 保持完全不变
def main():
    """主函数 - 使用示例"""
    # 初始化系统
    tcm_system = TCMDiagnosisSystem(api_key=ALI_API_KEY)
    
    # 测试图像URLs（请替换为实际图像）
    test_images = {
        "舌头图像": "http://www.zhongyijinnang.com/wp-content/uploads/2019/02/20-%E7%99%BD%E6%BB%91%E8%85%BB%E8%8B%94.jpg",
        # "面部图像": "your_face_image_url",
        # "手部图像": "your_hand_image_url",
    }
    
    for description, image_url in test_images.items():
        print(f"\n=== 分析 {description} ===")
        try:
            # 综合诊断
            result = tcm_system.comprehensive_diagnosis(image_url)
            
            # 格式化输出
            print(f"图像类型: {result['图像识别']['类型']}")
            print(f"识别置信度: {result['图像识别']['置信度']:.2f}")
            print(f"分析时间: {result['分析时间']}")
            print("\n诊断结果:")
            
            # 尝试解析JSON结果
            try:
                analysis_json = json.loads(result['分析结果'])
                print(json.dumps(analysis_json, ensure_ascii=False, indent=2))
            except:
                print(result['分析结果'])
                
        except Exception as e:
            print(f"分析失败: {e}")

if __name__ == "__main__":
    main()