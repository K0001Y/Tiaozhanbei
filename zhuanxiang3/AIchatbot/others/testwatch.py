import os
import json
import re
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from enum import Enum


# 初始化客户端
client = OpenAI(
    api_key="sk-7edcf5b1583945a58545c37877e0f2f3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
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
    """中医望诊AI系统 - 增强调试版本"""
    
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """初始化系统"""
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "qwen-vl-max"
        print(f"系统初始化完成 - API Base URL: {base_url}")
        print(f"使用模型: {self.model}")
    
    def _debug_api_response(self, response) -> None:
        """调试API响应的详细信息"""
        print(f"\n=== API响应调试信息 ===")
        print(f"响应类型: {type(response)}")
        
        if hasattr(response, 'choices'):
            print(f"choices数量: {len(response.choices) if response.choices else 0}")
            if response.choices:
                choice = response.choices[0]
                print(f"第一个choice类型: {type(choice)}")
                if hasattr(choice, 'message'):
                    print(f"message类型: {type(choice.message)}")
                    if hasattr(choice.message, 'content'):
                        content = choice.message.content
                        print(f"content类型: {type(content)}")
                        print(f"content长度: {len(content) if content else 0}")
                        if content:
                            print(f"content前200字符: {content[:200]}")
                        else:
                            print("⚠️ content为空!")
                    else:
                        print("⚠️ message没有content属性!")
                else:
                    print("⚠️ choice没有message属性!")
        else:
            print("⚠️ 响应没有choices属性!")
        
        # 打印完整响应对象的属性
        print(f"响应对象属性: {dir(response)}")
        print("=== 调试信息结束 ===\n")
    
    def _test_basic_api_call(self) -> bool:
        """测试基础API调用是否正常"""
        print("\n=== 测试基础API调用 ===")
        try:
            # 简单的文本对话测试
            test_messages = [
                {"role": "system", "content": "你是一个测试助手"},
                {"role": "user", "content": "请回复'测试成功'"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=test_messages,
                temperature=0.1
            )
            
            self._debug_api_response(response)
            
            if response.choices and response.choices[0].message.content:
                print(f"✅ 基础API调用成功: {response.choices[0].message.content}")
                return True
            else:
                print("❌ 基础API调用失败: 响应为空")
                return False
                
        except Exception as e:
            print(f"❌ 基础API调用异常: {e}")
            return False
    
    def _test_image_api_call(self, image_url: str) -> bool:
        """测试图像API调用是否正常"""
        print(f"\n=== 测试图像API调用 ===")
        print(f"测试图像URL: {image_url}")
        
        try:
            # 简单的图像描述测试
            test_messages = [
                {"role": "system", "content": "你是一个图像描述助手"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "请简单描述这张图片"}
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=test_messages,
                temperature=0.1
            )
            
            self._debug_api_response(response)
            
            if response.choices and response.choices[0].message.content:
                print(f"✅ 图像API调用成功")
                print(f"响应内容: {response.choices[0].message.content[:200]}...")
                return True
            else:
                print("❌ 图像API调用失败: 响应为空")
                return False
                
        except Exception as e:
            print(f"❌ 图像API调用异常: {e}")
            return False
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """从文本中提取JSON - 增强调试版本"""
        print(f"\n=== JSON提取调试 ===")
        print(f"输入文本类型: {type(text)}")
        print(f"输入文本长度: {len(text) if text else 0}")
        
        if not text or not text.strip():
            print("⚠️ 输入文本为空或只包含空白字符")
            return None
            
        text = text.strip()
        print(f"清理后文本前100字符: {text[:100]}")
        
        # 策略1：直接解析
        try:
            result = json.loads(text)
            print("✅ 策略1成功: 直接JSON解析")
            return result
        except json.JSONDecodeError as e:
            print(f"策略1失败: {e}")
        
        # 策略2：提取JSON块
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        print(f"策略2: 找到{len(matches)}个JSON候选块")
        
        for i, match in enumerate(matches):
            try:
                result = json.loads(match)
                print(f"✅ 策略2成功: 第{i+1}个JSON块解析成功")
                return result
            except json.JSONDecodeError:
                print(f"策略2: 第{i+1}个JSON块解析失败")
        
        # 策略3：嵌套JSON搜索
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
                        result = json.loads(json_str)
                        print("✅ 策略3成功: 嵌套JSON解析成功")
                        return result
                    except json.JSONDecodeError:
                        start_idx = -1
                        continue
        
        print("❌ 所有JSON提取策略都失败")
        return None
    
    def _safe_api_call(self, messages: List[Dict], default_response: Dict) -> str:
        """安全API调用 - 增强调试版本"""
        print(f"\n=== API调用详情 ===")
        print(f"消息数量: {len(messages)}")
        print(f"模型: {self.model}")
        
        # 打印消息结构（不包含具体图像URL）
        for i, msg in enumerate(messages):
            print(f"消息{i+1}: role={msg.get('role')}")
            content = msg.get('content')
            if isinstance(content, list):
                print(f"  内容类型: list, 长度={len(content)}")
                for j, item in enumerate(content):
                    if item.get('type') == 'image_url':
                        print(f"    项目{j+1}: 图像URL")
                    else:
                        print(f"    项目{j+1}: {item.get('type')} - {str(item.get('text', ''))[:50]}")
            else:
                print(f"  内容: {str(content)[:100]}")
        
        try:
            print("正在发送API请求...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1
            )
            
            print("✅ API请求发送成功")
            self._debug_api_response(response)
            
            if not response.choices or not response.choices[0].message.content:
                print("⚠️ API响应为空，返回默认响应")
                return json.dumps(default_response, ensure_ascii=False)
                
            content = response.choices[0].message.content.strip()
            print(f"✅ 获得API响应，长度: {len(content)}")
            
            # 尝试提取JSON
            extracted_json = self._extract_json_from_text(content)
            if extracted_json:
                print("✅ JSON提取成功")
                return json.dumps(extracted_json, ensure_ascii=False)
            else:
                print("⚠️ JSON提取失败，返回包装的原始内容")
                fallback = {
                    **default_response,
                    "原始响应": content,
                    "解析状态": "JSON提取失败，返回原始内容"
                }
                return json.dumps(fallback, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ API调用异常: {str(e)}")
            error_response = {
                **default_response,
                "错误": f"API调用失败: {str(e)}",
                "建议": "请检查网络连接和API配置"
            }
            return json.dumps(error_response, ensure_ascii=False)
    
    def identify_image_type(self, image_url: str) -> Tuple[ImageType, float]:
        """识别图像类型 - 增强调试版本"""
        print(f"\n=== 开始图像类型识别 ===")
        print(f"图像URL: {image_url}")
        
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
        
        result_str = self._safe_api_call(messages, default_response)
        
        try:
            result = json.loads(result_str)
            print(f"✅ 图像识别结果: {result}")
            
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
            print(f"❌ 图像识别结果解析失败: {e}")
            print(f"原始结果: {result_str}")
            return ImageType.UNKNOWN, 0.0
    
    def comprehensive_diagnosis_debug(self, image_url: str) -> Dict:
        """综合诊断 - 调试版本"""
        print(f"\n{'='*50}")
        print(f"开始综合诊断调试")
        print(f"图像URL: {image_url}")
        print(f"{'='*50}")
        
        # 步骤1: 测试基础API
        if not self._test_basic_api_call():
            return {"错误": "基础API调用失败", "建议": "检查API配置"}
        
        # 步骤2: 测试图像API
        if not self._test_image_api_call(image_url):
            return {"错误": "图像API调用失败", "建议": "检查图像URL或模型支持"}
        
        # 步骤3: 识别图像类型
        image_type, confidence = self.identify_image_type(image_url)
        print(f"✅ 图像类型识别完成: {image_type.value} (置信度: {confidence:.2f})")
        
        # 步骤4: 进行具体分析（这里简化为舌诊示例）
        if image_type == ImageType.TONGUE:
            analysis_result = self.analyze_tongue_debug(image_url)
        else:
            analysis_result = json.dumps({
                "诊断类型": image_type.value,
                "状态": "类型识别成功但未进行详细分析",
                "说明": "当前调试版本仅支持舌诊详细分析"
            }, ensure_ascii=False)
        
        return {
            "图像识别": {
                "类型": image_type.value,
                "置信度": confidence
            },
            "分析结果": analysis_result,
            "分析时间": self._get_current_time()
        }
    
    def analyze_tongue_debug(self, image_url: str) -> str:
        """舌诊分析 - 调试版本"""
        print(f"\n=== 开始舌诊分析 ===")
        
        system_prompt = """你是专业的中医舌诊AI助手。请分析舌头图像并返回JSON格式结果。

        返回格式：
        {
          "诊断类型": "舌诊",
          "舌质": "描述舌质特征",
          "舌苔": "描述舌苔特征",
          "分析状态": "成功"
        }"""

        user_prompt = "请分析这张舌头图像，返回JSON格式的诊断结果。"

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
            "诊断类型": "舌诊",
            "错误": "分析失败",
            "建议": "请检查图像质量或重试"
        }
        
        return self._safe_api_call(messages, default_response)
    
    def _get_current_time(self) -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 调试测试函数
def debug_main():
    """调试主函数"""
    print("开始TCM诊断系统调试...")
    
    # 初始化系统
    tcm_system = TCMDiagnosisSystem(api_key="sk-7edcf5b1583945a58545c37877e0f2f3")
    
    # 测试图像URL（请替换为你的实际图像）
    test_image_url = "http://www.zhongyijinnang.com/wp-content/uploads/2019/02/20-%E7%99%BD%E6%BB%91%E8%85%BB%E8%8B%94.jpg"
    
    print(f"\n使用测试图像: {test_image_url}")
    
    # 运行调试诊断
    result = tcm_system.comprehensive_diagnosis_debug(test_image_url)
    
    print(f"\n{'='*50}")
    print("最终诊断结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"{'='*50}")

if __name__ == "__main__":
    debug_main()