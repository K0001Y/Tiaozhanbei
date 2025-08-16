#!/usr/bin/env python3
"""
医学图像分析API测试脚本
测试图片望诊分析和补充分析接口

使用方法：
1. 确保你的服务器已启动
2. 修改BASE_URL为你的实际服务器地址
3. 运行: python test_medical_api.py
"""

import requests
import json
import os
import time
from typing import Dict, Any, Optional

# 配置
BASE_URL = "http://localhost:8080"  # 修改为你的实际服务器地址
IMAGE_PATH = r"D:\ROG\Documents\微信图片_20250816172059.jpg"
TIMEOUT = 240  # 请求超时时间（秒）

class MedicalAPITester:
    """医学图像分析API测试器"""
    
    def __init__(self, base_url: str, image_path: str):
        self.base_url = base_url.rstrip('/')
        self.image_path = image_path
        self.session = requests.Session()
        
        # 验证图片文件存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"测试图片不存在: {image_path}")
        
        print(f"✅ 初始化API测试器")
        print(f"   服务器地址: {self.base_url}")
        print(f"   测试图片: {self.image_path}")
        print(f"   图片大小: {os.path.getsize(self.image_path)} bytes")
        print("-" * 60)
    
    def test_server_health(self) -> bool:
        """测试服务器是否可访问"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            print(f"✅ 服务器连接正常 (状态码: {response.status_code})")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ 服务器连接失败: {e}")
            return False
    
    def test_watch_api(self, description: str = "测试图片") -> Optional[Dict[str, Any]]:
        """
        测试图片望诊分析接口 (POST /api/watch)
        
        :param description: 图片描述
        :return: API响应结果，失败时返回None
        """
        print(f"\n🔍 测试接口: POST /api/watch")
        print(f"   图片描述: {description}")
        
        try:
            # 准备文件上传
            with open(self.image_path, 'rb') as f:
                files = {
                    'image': ('test_image.jpg', f, 'image/jpeg')
                }
                data = {
                    'description': description
                }
                
                print(f"   📤 发送请求...")
                start_time = time.time()
                
                response = self.session.post(
                    f"{self.base_url}/api/watch",
                    files=files,
                    data=data,
                    timeout=TIMEOUT
                )
                
                elapsed_time = time.time() - start_time
                print(f"   ⏱️  请求耗时: {elapsed_time:.2f}秒")
                print(f"   📨 响应状态码: {response.status_code}")
                
                # 解析响应
                try:
                    result = response.json()
                    print(f"   📋 响应格式: JSON ✅")
                except json.JSONDecodeError:
                    print(f"   ❌ 响应格式错误，非JSON格式")
                    print(f"   原始响应: {response.text[:200]}...")
                    return None
                
                # 验证响应结构
                if self._validate_response_structure(result):
                    print(f"   ✅ 接口测试成功")
                    
                    # 输出关键信息
                    success = result.get('success', False)
                    message = result.get('message', '')
                    results = result.get('data', {}).get('results', '')
                    
                    print(f"   成功状态: {success}")
                    print(f"   响应消息: {message}")
                    print(f"   分析结果: {results[:100]}..." if len(results) > 100 else f"   分析结果: {results}")
                    
                    return result
                else:
                    print(f"   ❌ 响应结构验证失败")
                    return None
                    
        except requests.exceptions.Timeout:
            print(f"   ❌ 请求超时 (超过{TIMEOUT}秒)")
            return None
        except requests.exceptions.RequestException as e:
            print(f"   ❌ 网络请求失败: {e}")
            return None
        except Exception as e:
            print(f"   ❌ 测试异常: {e}")
            return None
    
    def test_watch_complete_api(self, prev_analysis: str, additional_info: str = "补充测试信息", 
                               test_additional_image: bool = False) -> Optional[Dict[str, Any]]:
        """
        测试望诊补充分析接口 (POST /api/watch/complete)
        
        :param prev_analysis: 之前的分析结果
        :param additional_info: 补充信息
        :param test_additional_image: 是否测试补充图片功能
        :return: API响应结果，失败时返回None
        """
        print(f"\n🔍 测试接口: POST /api/watch/complete")
        print(f"   补充信息: {additional_info}")
        print(f"   补充图片: {'是' if test_additional_image else '否'}")
        
        try:
            # 准备请求数据
            data = {
                'prevAnalysis': prev_analysis,
                'additionalInfo': additional_info
            }
            
            files = {}
            if test_additional_image:
                # 使用同一张图片作为补充图片（实际使用中应该是不同的图片）
                with open(self.image_path, 'rb') as f:
                    files['additionalFile'] = ('additional_image.jpg', f, 'image/jpeg')
                    
                    print(f"   📤 发送请求（包含补充图片）...")
                    start_time = time.time()
                    
                    response = self.session.post(
                        f"{self.base_url}/api/watch/complete",
                        files=files,
                        data=data,
                        timeout=TIMEOUT
                    )
            else:
                print(f"   📤 发送请求（仅补充信息）...")
                start_time = time.time()
                
                response = self.session.post(
                    f"{self.base_url}/api/watch/complete",
                    data=data,
                    timeout=TIMEOUT
                )
            
            elapsed_time = time.time() - start_time
            print(f"   ⏱️  请求耗时: {elapsed_time:.2f}秒")
            print(f"   📨 响应状态码: {response.status_code}")
            
            # 解析响应
            try:
                result = response.json()
                print(f"   📋 响应格式: JSON ✅")
            except json.JSONDecodeError:
                print(f"   ❌ 响应格式错误，非JSON格式")
                print(f"   原始响应: {response.text[:200]}...")
                return None
            
            # 验证响应结构
            if self._validate_response_structure(result):
                print(f"   ✅ 接口测试成功")
                
                # 输出关键信息
                success = result.get('success', False)
                message = result.get('message', '')
                results = result.get('data', {}).get('results', '')
                
                print(f"   成功状态: {success}")
                print(f"   响应消息: {message}")
                print(f"   补充分析结果: {results[:100]}..." if len(results) > 100 else f"   补充分析结果: {results}")
                
                return result
            else:
                print(f"   ❌ 响应结构验证失败")
                return None
                
        except requests.exceptions.Timeout:
            print(f"   ❌ 请求超时 (超过{TIMEOUT}秒)")
            return None
        except requests.exceptions.RequestException as e:
            print(f"   ❌ 网络请求失败: {e}")
            return None
        except Exception as e:
            print(f"   ❌ 测试异常: {e}")
            return None
    
    def _validate_response_structure(self, response: Dict[str, Any]) -> bool:
        """验证API响应结构是否符合预期"""
        try:
            # 检查必要字段
            if 'success' not in response:
                print(f"   ❌ 缺少'success'字段")
                return False
            
            if 'message' not in response:
                print(f"   ❌ 缺少'message'字段")
                return False
            
            if 'data' not in response:
                print(f"   ❌ 缺少'data'字段")
                return False
            
            data = response['data']
            if not isinstance(data, dict):
                print(f"   ❌ 'data'字段不是字典类型")
                return False
            
            if 'results' not in data:
                print(f"   ❌ 'data'中缺少'results'字段")
                return False
            
            return True
            
        except Exception as e:
            print(f"   ❌ 响应结构验证异常: {e}")
            return False
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("🚀 开始医学图像分析API综合测试")
        print("=" * 60)
        
        # 1. 测试服务器连通性
        if not self.test_server_health():
            print("\n❌ 服务器连接失败，测试终止")
            return
        
        # 2. 测试图片望诊分析接口
        print(f"\n📍 第一阶段：测试基础望诊分析")
        watch_result = self.test_watch_api("这是一张图片，请分析")
        
        if not watch_result or not watch_result.get('success'):
            print(f"\n❌ 基础望诊分析测试失败，跳过补充分析测试")
            return
        
        # 获取基础分析结果用于补充分析
        prev_analysis = watch_result.get('data', {}).get('results', '')
        
        # 3. 测试望诊补充分析接口（仅补充信息）
        print(f"\n📍 第二阶段：测试补充分析（仅文本补充）")
        complete_result1 = self.test_watch_complete_api(
            prev_analysis=prev_analysis,
            additional_info="患者还有口苦、咽干的症状",
            test_additional_image=False
        )
        
        # 4. 测试望诊补充分析接口（包含补充图片）
        print(f"\n📍 第三阶段：测试补充分析（包含补充图片）")
        complete_result2 = self.test_watch_complete_api(
            prev_analysis=prev_analysis,
            additional_info="请结合新的图片进一步分析",
            test_additional_image=True
        )
        
        # 5. 输出测试总结
        print(f"\n" + "=" * 60)
        print(f"📊 测试总结")
        print(f"=" * 60)
        
        tests = [
            ("服务器连通性", True),
            ("基础望诊分析", watch_result and watch_result.get('success')),
            ("补充分析（文本）", complete_result1 and complete_result1.get('success')),
            ("补充分析（图片）", complete_result2 and complete_result2.get('success'))
        ]
        
        passed = sum(1 for _, result in tests if result)
        total = len(tests)
        
        for test_name, result in tests:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"   {test_name}: {status}")
        
        print(f"\n🏆 测试结果: {passed}/{total} 通过")
        
        if passed == total:
            print(f"🎉 所有测试通过！API工作正常")
        else:
            print(f"⚠️  部分测试失败，请检查服务器日志")
    
    def test_error_cases(self):
        """测试错误情况"""
        print(f"\n🧪 测试错误处理能力")
        print("=" * 60)
        
        # 测试无文件上传
        print(f"\n🔍 测试错误情况1: 无图片文件")
        try:
            response = self.session.post(f"{self.base_url}/api/watch", data={'description': 'test'})
            result = response.json()
            success = result.get('success', True)  # 期望失败
            print(f"   结果: {'❌ 正确拒绝' if not success else '⚠️ 应该拒绝但没有'}")
        except Exception as e:
            print(f"   异常: {e}")
        
        # 测试空的补充分析
        print(f"\n🔍 测试错误情况2: 空的补充信息")
        try:
            response = self.session.post(f"{self.base_url}/api/watch/complete", data={})
            result = response.json()
            success = result.get('success', True)  # 期望失败
            print(f"   结果: {'❌ 正确拒绝' if not success else '⚠️ 应该拒绝但没有'}")
        except Exception as e:
            print(f"   异常: {e}")


def main():
    """主函数"""
    try:
        # 创建测试器
        tester = MedicalAPITester(BASE_URL, IMAGE_PATH)
        
        # 运行综合测试
        tester.run_comprehensive_test()
        
        # 运行错误处理测试
        tester.test_error_cases()
        
    except FileNotFoundError as e:
        print(f"❌ 文件错误: {e}")
        print(f"请检查图片路径是否正确: {IMAGE_PATH}")
    except Exception as e:
        print(f"❌ 测试程序异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🏥 医学图像分析API测试程序")
    print("作者: Linus式优化版本")
    print("=" * 60)
    
    # 提示用户确认配置
    print(f"📋 当前配置:")
    print(f"   服务器地址: {BASE_URL}")
    print(f"   测试图片: {IMAGE_PATH}")
    print(f"   请求超时: {TIMEOUT}秒")
    
    input(f"\n按回车键开始测试...")
    
    main()