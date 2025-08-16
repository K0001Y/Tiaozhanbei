#!/usr/bin/env python3
"""
独立的API测试脚本 - 用于调试问诊补充问题
"""
import requests
import json

# 配置
BASE_URL = "http://localhost:8080"  # 修改为您的服务器地址

def test_inquiry_complete_debug():
    """调试问诊补充功能"""
    
    print("=" * 60)
    print("开始调试问诊补充功能")
    print("=" * 60)
    
    # 步骤1: 先进行初步问诊
    print("\n步骤1: 初步问诊")
    inquiry_data = {
        "age": 35,
        "gender": "男", 
        "symptoms": "头痛、发烧、咳嗽，持续3天，伴有乏力"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/inquiry",
            json=inquiry_data,
            headers={'Content-Type': 'application/json'},
            timeout=240
        )
        
        print(f"初步问诊响应状态码: {response.status_code}")
        print(f"初步问诊响应: {response.text}")
        
        if response.status_code == 200:
            inquiry_result = response.json()
            prev_inquiry = inquiry_result.get('data', {}).get('results', '')
            print(f"✅ 初步问诊成功")
            print(f"获取到的问诊结果长度: {len(prev_inquiry)}")
            print(f"问诊结果前200字符: {prev_inquiry[:200]}")
        else:
            print(f"❌ 初步问诊失败")
            prev_inquiry = "患者基本信息：35岁，男性。主要症状：头痛、发烧、咳嗽，持续3天，伴有乏力。根据中医辨证，考虑为外感风寒证，建议温阳散寒治疗。"
            print(f"使用默认问诊结果: {prev_inquiry}")
            
    except Exception as e:
        print(f"❌ 初步问诊异常: {e}")
        prev_inquiry = "患者基本信息：35岁，男性。主要症状：头痛、发烧、咳嗽，持续3天，伴有乏力。根据中医辨证，考虑为外感风寒证，建议温阳散寒治疗。"
        print(f"使用默认问诊结果: {prev_inquiry}")
    
    # 步骤2: 测试不同格式的补充问诊
    additional_info = "患者还有肠胃不适，食欲不振，大便偏稀"
    
    # 测试1: application/x-www-form-urlencoded 格式
    print(f"\n步骤2a: 测试form-urlencoded格式")
    test_form_urlencoded(prev_inquiry, additional_info)
    
    # 测试2: application/json 格式  
    print(f"\n步骤2b: 测试JSON格式")
    test_json_format(prev_inquiry, additional_info)
    
    # 测试3: multipart/form-data 格式
    print(f"\n步骤2c: 测试multipart格式")
    test_multipart_format(prev_inquiry, additional_info)

def test_form_urlencoded(prev_inquiry, additional_info):
    """测试application/x-www-form-urlencoded格式"""
    data = {
        'prevInquiry': prev_inquiry,
        'additionalInfo': additional_info
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/inquiry/complete",
            data=data,  # 使用data参数，自动设置为form-urlencoded
            timeout=30
        )
        
        print(f"Form-urlencoded响应状态码: {response.status_code}")
        print(f"Form-urlencoded响应: {response.text}")
        
        if response.status_code == 200:
            print("✅ Form-urlencoded格式成功")
        else:
            print("❌ Form-urlencoded格式失败")
            
    except Exception as e:
        print(f"❌ Form-urlencoded异常: {e}")

def test_json_format(prev_inquiry, additional_info):
    """测试application/json格式"""
    data = {
        'prevInquiry': prev_inquiry,
        'additionalInfo': additional_info
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/inquiry/complete",
            json=data,  # 使用json参数，自动设置Content-Type
            timeout=30
        )
        
        print(f"JSON响应状态码: {response.status_code}")
        print(f"JSON响应: {response.text}")
        
        if response.status_code == 200:
            print("✅ JSON格式成功")
        else:
            print("❌ JSON格式失败")
            
    except Exception as e:
        print(f"❌ JSON异常: {e}")

def test_multipart_format(prev_inquiry, additional_info):
    """测试multipart/form-data格式"""
    # 使用files参数会自动设置为multipart/form-data
    data = {
        'prevInquiry': prev_inquiry,
        'additionalInfo': additional_info
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/inquiry/complete",
            data=data,
            files={'dummy': (None, '')},  # 添加一个空文件触发multipart
            timeout=30
        )
        
        print(f"Multipart响应状态码: {response.status_code}")
        print(f"Multipart响应: {response.text}")
        
        if response.status_code == 200:
            print("✅ Multipart格式成功")
        else:
            print("❌ Multipart格式失败")
            
    except Exception as e:
        print(f"❌ Multipart异常: {e}")

if __name__ == "__main__":
    test_inquiry_complete_debug()