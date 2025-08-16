#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试图像描述为空的问题
"""

import json
from watch import TCMDiagnosisSystem  # 替换为实际的模块名
from config import ALI_API_KEY, ALI_BASE_URL


def debug_image_description():
    """调试图像描述问题"""
    
    print("=== 调试图像描述问题 ===\n")
    
    # 初始化系统
    try:
        tcm_system = TCMDiagnosisSystem(api_key=ALI_API_KEY, base_url=ALI_BASE_URL)
        print("✅ TCM系统初始化成功")
    except Exception as e:
        print(f"❌ TCM系统初始化失败: {e}")
        return
    
    # 测试图像URL
    test_image_url = "http://www.zhongyijinnang.com/wp-content/uploads/2019/02/20-%E7%99%BD%E6%BB%91%E8%85%BB%E8%8B%94.jpg"
    
    print(f"\n=== 第一步：测试图像类型识别 ===")
    
    try:
        image_type, confidence = tcm_system.identify_image_type(test_image_url)
        print(f"✅ 图像类型识别成功")
        print(f"   类型: {image_type.value}")
        print(f"   置信度: {confidence}")
    except Exception as e:
        print(f"❌ 图像类型识别失败: {e}")
        return
    
    print(f"\n=== 第二步：直接调用API查看原始响应 ===")
    
    # 直接测试API调用，看原始响应
    identification_prompt = """
    你是专业的医学图像识别AI。请详细描述这张图片：
    1. 你看到了什么？
    2. 这是什么类型的医学图像？
    3. 图像的质量如何？
    4. 有什么特征值得注意？
    
    请用自然语言详细描述，不需要JSON格式。
    """
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=ALI_API_KEY, base_url=ALI_BASE_URL)
        
        response = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {"role": "system", "content": identification_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": test_image_url}},
                        {"type": "text", "text": "请详细描述这张图片"}
                    ]
                }
            ],
            temperature=0.1
        )
        
        raw_content = response.choices[0].message.content
        print(f"✅ API调用成功")
        print(f"📝 原始API响应:")
        print(f"   长度: {len(raw_content)} 字符")
        print(f"   内容: {raw_content[:200]}...")
        
        if not raw_content or raw_content.strip() == "":
            print("❌ API返回空内容！")
        elif "补充分析" in raw_content:
            print("❌ API返回的是模板内容，不是真实分析！")
        else:
            print("✅ API返回了真实的分析内容")
            
    except Exception as e:
        print(f"❌ 直接API调用失败: {e}")
        return
    
    print(f"\n=== 第三步：测试JSON格式API调用 ===")
    
    json_prompt = """
    请分析这张医学图像，返回JSON格式：
    {
        "image_type": "图像类型",
        "confidence": 0.95,
        "description": "详细的图像描述，说明你看到了什么具体特征"
    }
    """
    
    try:
        response = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {"role": "system", "content": json_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": test_image_url}},
                        {"type": "text", "text": "请分析图像并返回JSON"}
                    ]
                }
            ],
            temperature=0.1
        )
        
        json_content = response.choices[0].message.content
        print(f"✅ JSON格式API调用成功")
        print(f"📝 JSON API响应:")
        print(f"   内容: {json_content}")
        
        # 尝试解析JSON
        try:
            parsed_json = json.loads(json_content)
            description = parsed_json.get("description", "")
            
            if not description or description.strip() == "":
                print("❌ JSON中description字段为空！")
            elif "补充分析" in description:
                print("❌ description包含模板内容！")
            else:
                print(f"✅ description有内容: {description[:100]}...")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            
            # 尝试提取JSON
            extracted = tcm_system._extract_json_from_text(json_content)
            if extracted:
                print(f"✅ JSON提取成功: {extracted}")
                description = extracted.get("description", "")
                if description:
                    print(f"✅ 提取到的description: {description[:100]}...")
                else:
                    print("❌ 提取到的JSON中description为空")
            else:
                print("❌ 无法提取有效JSON")
        
    except Exception as e:
        print(f"❌ JSON API调用失败: {e}")
    
    print(f"\n=== 第四步：测试完整诊断流程 ===")
    
    try:
        result = tcm_system.comprehensive_diagnosis(test_image_url)
        print(f"✅ 完整诊断成功")
        print(f"📝 诊断结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 检查分析结果
        analysis_result = result.get("分析结果", "")
        if isinstance(analysis_result, str):
            try:
                analysis_json = json.loads(analysis_result)
                if "description" in str(analysis_json):
                    print("✅ 分析结果包含description字段")
                else:
                    print("❌ 分析结果不包含description字段")
            except:
                print("❌ 分析结果不是有效JSON")
        
    except Exception as e:
        print(f"❌ 完整诊断失败: {e}")
    
    print(f"\n=== 调试总结 ===")
    print("请检查以上输出，找出问题所在：")
    print("1. API是否返回真实内容？")
    print("2. JSON解析是否正确？")
    print("3. description字段是否有值？")
    print("4. 是否在某个环节被替换为模板内容？")

if __name__ == "__main__":
    debug_image_description()