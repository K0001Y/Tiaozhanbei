import os
import json
import re
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from enum import Enum
from config import ALI_API_KEY, ALI_BASE_URL

# 初始化客户端
client = OpenAI(
    api_key=ALI_API_KEY,
    base_url=ALI_BASE_URL
)

class ImageType(Enum):
    """图像类型枚举"""
    TONGUE = "舌诊"
    FACE = "面诊" 
    HAND = "手诊"
    EYE = "眼诊"
    EAR = "耳诊"
    BODY = "体诊"
    UNKNOWN = "未知"

class TCMDiagnosisSystem:
    """中医望诊AI系统 - 修复版本"""
    
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """初始化系统"""
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "qwen-vl-max"
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """
        增强版JSON提取 - 专门处理阿里云markdown格式
        """
        if not text or not text.strip():
            return None
            
        text = text.strip()
        
        # 策略1：移除markdown标记后直接解析
        # 处理 ```json ... ``` 格式
        if text.startswith('```'):
            # 找到第一个换行符和最后一个```
            lines = text.split('\n')
            if len(lines) > 2:
                # 移除第一行的```json和最后一行的```
                json_content = '\n'.join(lines[1:-1])
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    pass
        
        # 策略2：直接解析(最常见情况)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 策略3：正则提取JSON块
        # 匹配 ```json 和 ``` 之间的内容
        json_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 策略4：提取大括号内容
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(brace_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # 策略5：逐字符扫描
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
    
    def _safe_api_call(self, messages: List[Dict], default_response: Dict) -> Dict:
        """
        安全API调用 - 修复版本，直接返回字典而不是JSON字符串
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1
            )
            
            if not response.choices or not response.choices[0].message.content:
                return default_response
                
            content = response.choices[0].message.content.strip()
            
            # 尝试提取JSON
            extracted_json = self._extract_json_from_text(content)
            if extracted_json:
                return extracted_json
            else:
                # 如果完全无法解析，返回包装的原始内容
                return {
                    **default_response,
                    "原始响应": content,
                    "解析状态": "JSON提取失败，返回原始内容"
                }
                
        except Exception as e:
            return {
                **default_response,
                "错误": f"API调用失败: {str(e)}",
                "建议": "请检查网络连接和API配置"
            }
    
    def identify_image_type(self, image_url: str) -> Tuple[ImageType, float]:
        """识别图像类型"""
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
        
        default_response = {
            "image_type": "未知",
            "confidence": 0.0,
            "description": "识别失败"
        }
        
        # 直接获取字典结果
        result = self._safe_api_call(messages, default_response)
        
        try:
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
    
    def _make_diagnosis_request(self, system_prompt: str, user_prompt: str, image_url: str) -> Dict:
        """
        发送诊断请求 - 修复版本，直接返回字典
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
    
    def analyze_tongue(self, image_url: str) -> Dict:
        """舌诊分析 - 返回字典而不是JSON字符串"""
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
    
    def analyze_face(self, image_url: str) -> Dict:
        """面诊分析 - 返回字典"""
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
    
    def analyze_hand(self, image_url: str) -> Dict:
        """手诊分析 - 返回字典"""
        # 这里为了简洁，使用简化版的手诊分析
        system_prompt = """
        你是专业的中医手诊AI助手。请分析手部图像的中医特征。
        """
        
        user_prompt = """
        请返回JSON格式的手诊分析结果：
        {
          "诊断类型": "手诊",
          "手部特征": "描述手部整体特征",
          "辨证提示": [],
          "健康建议": []
        }
        """
        
        return self._make_diagnosis_request(system_prompt, user_prompt, image_url)
    
    def analyze_eye(self, image_url: str) -> Dict:
        """眼诊分析 - 返回字典"""
        system_prompt = """你是专业的中医眼诊AI助手。"""
        user_prompt = """
        请返回JSON格式的眼诊分析：
        {
          "诊断类型": "眼诊",
          "眼部特征": "描述眼部特征",
          "辨证提示": [],
          "健康建议": []
        }
        """
        return self._make_diagnosis_request(system_prompt, user_prompt, image_url)
    
    def analyze_ear(self, image_url: str) -> Dict:
        """耳诊分析 - 返回字典"""
        system_prompt = """你是专业的中医耳诊AI助手。"""
        user_prompt = """
        请返回JSON格式的耳诊分析：
        {
          "诊断类型": "耳诊",
          "耳部特征": "描述耳部特征",
          "辨证提示": [],
          "健康建议": []
        }
        """
        return self._make_diagnosis_request(system_prompt, user_prompt, image_url)
    
    def analyze_body(self, image_url: str) -> Dict:
        """体诊分析 - 返回字典"""
        system_prompt = """你是专业的中医体诊AI助手。"""
        user_prompt = """
        请返回JSON格式的体诊分析：
        {
          "诊断类型": "体诊",
          "体型特征": "描述体型特征",
          "辨证提示": [],
          "健康建议": []
        }
        """
        return self._make_diagnosis_request(system_prompt, user_prompt, image_url)
    
    def comprehensive_diagnosis(self, image_url: str) -> Dict:
        """
        综合诊断：自动识别图像类型并进行相应分析
        返回完整的字典结构而不是嵌套的JSON字符串
        """
        # 1. 识别图像类型
        image_type, confidence = self.identify_image_type(image_url)
        
        print(f"检测到图像类型: {image_type.value} (置信度: {confidence:.2f})")
        
        # 2. 根据类型选择分析方法
        analysis_methods = {
            ImageType.TONGUE: self.analyze_tongue,
            ImageType.FACE: self.analyze_face,
            ImageType.HAND: self.analyze_hand,
            ImageType.EYE: self.analyze_eye,
            ImageType.EAR: self.analyze_ear,
            ImageType.BODY: self.analyze_body,
        }
        
        if image_type in analysis_methods:
            analysis_result = analysis_methods[image_type](image_url)
        else:
            analysis_result = {
                "错误": "无法识别的图像类型",
                "建议": "请上传清晰的中医望诊相关图像（舌头、面部、手部、眼部、耳部或身体）"
            }
        
        # 3. 构建完整诊断结果
        diagnosis_result = {
            "图像识别": {
                "类型": image_type.value,
                "置信度": confidence
            },
            "分析结果": analysis_result,  # 直接使用字典，不转换为JSON字符串
            "分析时间": self._get_current_time()
        }
        
        return diagnosis_result
    
    def _get_current_time(self) -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 测试函数
def main():
    """主函数 - 使用示例"""
    # 初始化系统
    tcm_system = TCMDiagnosisSystem(api_key=ALI_API_KEY)
    
    # 测试图像URL
    test_image_url = "http://www.zhongyijinnang.com/wp-content/uploads/2019/02/20-%E7%99%BD%E6%BB%91%E8%85%BB%E8%8B%94.jpg"
    
    print(f"=== 分析测试图像 ===")
    try:
        # 综合诊断
        result = tcm_system.comprehensive_diagnosis(test_image_url)
        
        # 格式化输出
        print(f"图像类型: {result['图像识别']['类型']}")
        print(f"识别置信度: {result['图像识别']['置信度']:.2f}")
        print(f"分析时间: {result['分析时间']}")
        
        print("\n=== 详细诊断结果 ===")
        analysis_result = result['分析结果']
        
        # 现在analysis_result直接是字典，不需要再解析JSON
        if isinstance(analysis_result, dict):
            print(json.dumps(analysis_result, ensure_ascii=False, indent=2))
        else:
            print("结果格式异常:", analysis_result)
            
    except Exception as e:
        print(f"分析失败: {e}")

if __name__ == "__main__":
    main()