import os
from openai import OpenAI
from typing import List

# 初始化客户端
client = OpenAI(
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def analyze_hand_images(palm_url: str, dorsum_url: str):
    """
    专业中医手诊分析（需手心手背两张图片）
    :param palm_url: 手心(掌面)图像URL
    :param dorsum_url: 手背(背面)图像URL
    """
    # 专业中医手诊提示词
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

    response = client.chat.completions.create(
        model="qwen-vl-max",  # 使用千问VL Max多模态模型
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "第一张：手心图像（掌面）"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": palm_url}
                    },
                    {
                        "type": "text",
                        "text": "第二张：手背图像（背面）"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": dorsum_url}
                    },
                    {
                        "type": "text",
                        "text": """
                        请输出结构化JSON数据，综合两张图像进行分析：
                        {
                          "手心分析": {
                            "颜色分区": {
                              "大小鱼际": "",
                              "掌心": "",
                              "指根": ""
                            },
                            "湿度评估": "1-5级 (1=干燥脱屑, 5=潮湿多汗)",
                            "弹性评估": "1-5级 (1=僵硬, 5=柔软)",
                            "掌纹特征": {
                              "生命线": {"长度": "短/中/长", "形态": ""},
                              "智慧线": {"走势": "", "分叉": ""},
                              "感情线": {"连贯性": "", "岛纹": ""},
                              "健康线": {"明显度": "1-5级"}
                            },
                            "反射区异常": ["", ""]
                          },
                          "手背分析": {
                            "静脉特征": {"明显度": "1-5级", "分布": ""},
                            "浮肿程度": "1-5级 (1=无, 5=明显浮肿)",
                            "皮肤纹理": "",
                            "指甲特征": {
                              "甲色": "",
                              "月牙": {"数量": "", "大小": "", "清晰度": ""},
                              "纵纹/横沟": ""
                            }
                          },
                          "手指综合分析": {
                            "形态异常": ["", ""],
                            "长度比例": "描述",
                            "关节肿胀": "是/否",
                            "指腹饱满度": "1-5级"
                          },
                          "辨证提示": [],
                          "中医理论依据": "",
                          "西医可能关联提示": "",
                          "图像质量评估": {
                            "手心质量": "",
                            "手背质量": ""
                          }
                        }

                        注意：
                        1. 辨证提示格式：["证型名称(置信度%)", ...]
                        2. 图像质量需评估：清晰度/光照/角度/遮挡物
                        3. 所有评估等级均为1-5分制
                        4. 反射区异常示例：["肝区青暗", "心区红赤"]
                        """
                    }
                ]
            }
        ],
        temperature=0.1  # 降低随机性保证专业性
    )

    # 返回结构化JSON数据
    return response.choices[0].message.content


# 使用示例
if __name__ == "__main__":
    # 替换为实际手部图像URL（手心+手背）
    palm_url = "https://example.com/palm.jpg"  # 手心图像
    dorsum_url = "https://example.com/dorsum.jpg"  # 手背图像

    # 获取分析结果
    result = analyze_hand_images(palm_url, dorsum_url)
    print("手诊分析结果：")
    print(result)

    # 实际应用中可直接返回给前端：
    # import json
    # return json.loads(result)