#!/usr/bin/env python3
"""
正确匹配问诊API的测试代码
根据实际API实现调整参数名称和数据结构
"""

import requests
import json
import time
from io import BytesIO
from PIL import Image

class CorrectInquiryTester:
    """匹配实际API实现的问诊测试器"""
    
    def __init__(self, base_url: str = 'http://localhost:8080'):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
        
    def test_inquiry_api_correct(self):
        """测试初步问诊API - 使用正确的参数"""
        print("\n🔍 测试初步问诊API (正确参数)")
        print("-" * 50)
        
        # 根据你的API实现，正确的参数应该是:
        # age: 年龄（数字）
        # gender: 性别（男/女）  
        # symptoms: 症状描述
        
        correct_data = {
            "age": 35,
            "gender": "男",
            "symptoms": "头痛、发烧、咳嗽，持续3天，晚上症状加重"
        }
        
        print(f"发送正确格式数据:")
        print(json.dumps(correct_data, ensure_ascii=False, indent=2))
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/inquiry",
                json=correct_data,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"响应状态码: {response.status_code}")
            
            try:
                resp_data = response.json()
                print(f"响应内容:")
                print(json.dumps(resp_data, ensure_ascii=False, indent=2))
                
                if response.status_code == 200:
                    print("✅ 初步问诊API测试成功")
                else:
                    print(f"❌ 初步问诊API测试失败: {resp_data.get('message', '未知错误')}")
                    
            except json.JSONDecodeError:
                print(f"响应不是JSON格式: {response.text}")
                
        except Exception as e:
            print(f"❌ 请求失败: {e}")
    
    def test_inquiry_api_edge_cases(self):
        """测试初步问诊API的边界情况"""
        print("\n🧪 测试初步问诊API边界情况")
        print("-" * 50)
        
        test_cases = [
            {
                "name": "缺少age参数",
                "data": {"gender": "女", "symptoms": "头痛"}
            },
            {
                "name": "缺少gender参数", 
                "data": {"age": 30, "symptoms": "头痛"}
            },
            {
                "name": "缺少symptoms参数",
                "data": {"age": 30, "gender": "女"}
            },
            {
                "name": "年龄为负数",
                "data": {"age": -1, "gender": "男", "symptoms": "头痛"}
            },
            {
                "name": "年龄过大",
                "data": {"age": 200, "gender": "女", "symptoms": "头痛"}
            },
            {
                "name": "性别格式错误",
                "data": {"age": 25, "gender": "未知", "symptoms": "头痛"}
            },
            {
                "name": "症状描述过短",
                "data": {"age": 25, "gender": "男", "symptoms": "a"}
            },
            {
                "name": "症状描述为空",
                "data": {"age": 25, "gender": "女", "symptoms": ""}
            }
        ]
        
        for test_case in test_cases:
            print(f"\n测试用例: {test_case['name']}")
            
            try:
                response = self.session.post(
                    f"{self.base_url}/api/inquiry",
                    json=test_case['data'],
                    headers={'Content-Type': 'application/json'}
                )
                
                print(f"状态码: {response.status_code}")
                
                if response.status_code == 400:
                    try:
                        resp_data = response.json()
                        print(f"✅ 正确返回400: {resp_data.get('message', '')}")
                    except:
                        print(f"✅ 正确返回400")
                else:
                    print(f"⚠️  意外状态码: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ 请求失败: {e}")
    
    def test_inquiry_complete_correct(self):
        """测试补充问诊API - 使用正确的参数"""
        print("\n🔍 测试补充问诊API (正确参数)")
        print("-" * 50)
        
        # 根据你的API实现，正确的参数应该是:
        # prevInquiry: 之前的问诊分析结果
        # additionalInfo: 补充信息
        # additionalFile: 检查报告文件（可选）
        
        correct_form_data = {
            'prevInquiry': '初步诊断为气血两虚，建议调理脾胃',
            'additionalInfo': '患者补充信息：最近还出现流鼻涕、打喷嚏症状，晚上睡眠质量差'
        }
        
        print(f"发送正确格式的form数据:")
        for key, value in correct_form_data.items():
            print(f"  {key}: {value}")
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/inquiry/complete",
                data=correct_form_data
            )
            
            print(f"响应状态码: {response.status_code}")
            
            try:
                resp_data = response.json()
                print(f"响应内容:")
                print(json.dumps(resp_data, ensure_ascii=False, indent=2))
                
                if response.status_code == 200:
                    print("✅ 补充问诊API测试成功")
                else:
                    print(f"❌ 补充问诊API测试失败: {resp_data.get('message', '未知错误')}")
                    
            except json.JSONDecodeError:
                print(f"响应不是JSON格式: {response.text}")
                
        except Exception as e:
            print(f"❌ 请求失败: {e}")
    
    def test_inquiry_complete_with_file(self):
        """测试补充问诊API - 带文件上传"""
        print("\n📎 测试补充问诊API (带文件上传)")
        print("-" * 50)
        
        # 创建测试图像文件
        img = Image.new('RGB', (200, 200), color='white')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        form_data = {
            'prevInquiry': '初步诊断为感冒',
            'additionalInfo': '患者上传了检查报告，请结合报告分析'
        }
        
        files = {
            'additionalFile': ('test_report.jpg', img_bytes, 'image/jpeg')
        }
        
        print(f"发送form数据和文件:")
        for key, value in form_data.items():
            print(f"  {key}: {value}")
        print(f"  文件: test_report.jpg (JPEG格式)")
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/inquiry/complete",
                data=form_data,
                files=files
            )
            
            print(f"响应状态码: {response.status_code}")
            
            try:
                resp_data = response.json()
                print(f"响应内容:")
                print(json.dumps(resp_data, ensure_ascii=False, indent=2))
                
                if response.status_code == 200:
                    print("✅ 带文件的补充问诊API测试成功")
                else:
                    print(f"❌ 带文件的补充问诊API测试失败: {resp_data.get('message', '未知错误')}")
                    
            except json.JSONDecodeError:
                print(f"响应不是JSON格式: {response.text}")
                
        except Exception as e:
            print(f"❌ 请求失败: {e}")
    
    def test_inquiry_complete_edge_cases(self):
        """测试补充问诊API的边界情况"""
        print("\n🧪 测试补充问诊API边界情况")
        print("-" * 50)
        
        test_cases = [
            {
                "name": "两个参数都为空",
                "data": {'prevInquiry': '', 'additionalInfo': ''}
            },
            {
                "name": "只有prevInquiry",
                "data": {'prevInquiry': '初步诊断为感冒', 'additionalInfo': ''}
            },
            {
                "name": "只有additionalInfo", 
                "data": {'prevInquiry': '', 'additionalInfo': '补充症状：流鼻涕'}
            },
            {
                "name": "完全缺少参数",
                "data": {}
            }
        ]
        
        for test_case in test_cases:
            print(f"\n测试用例: {test_case['name']}")
            
            try:
                response = self.session.post(
                    f"{self.base_url}/api/inquiry/complete",
                    data=test_case['data']
                )
                
                print(f"状态码: {response.status_code}")
                
                if test_case['name'] in ["两个参数都为空", "完全缺少参数"] and response.status_code == 400:
                    print(f"✅ 正确返回400错误")
                elif test_case['name'] in ["只有prevInquiry", "只有additionalInfo"] and response.status_code == 200:
                    print(f"✅ 正确处理单参数情况")
                else:
                    try:
                        resp_data = response.json()
                        print(f"响应: {resp_data.get('message', '')}")
                    except:
                        print(f"响应: {response.text[:100]}")
                        
            except Exception as e:
                print(f"❌ 请求失败: {e}")
    
    def run_all_correct_tests(self):
        """运行所有正确的测试"""
        print("🚀 运行正确匹配API的测试")
        print("=" * 60)
        
        # 等待服务器
        print("等待服务器启动...")
        for i in range(10):
            try:
                response = self.session.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    print(f"✅ 服务器已启动")
                    break
            except:
                time.sleep(1)
                continue
        
        # 运行测试
        self.test_inquiry_api_correct()
        self.test_inquiry_api_edge_cases()
        self.test_inquiry_complete_correct()
        self.test_inquiry_complete_with_file()
        self.test_inquiry_complete_edge_cases()
        
        print("\n" + "=" * 60)
        print("📊 测试完成")
        print("=" * 60)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='正确匹配问诊API的测试工具')
    parser.add_argument('--url', default='http://localhost:8080', help='服务器地址')
    
    args = parser.parse_args()
    
    tester = CorrectInquiryTester(args.url)
    tester.run_all_correct_tests()

if __name__ == '__main__':
    main()