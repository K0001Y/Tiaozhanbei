#!/usr/bin/env python3
"""
API服务器集成测试
Linus原则：消除重复，测试真实场景，快速失败
"""

import requests
import json
import time
import sys
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Optional, Tuple


class APITester:
    """简洁的API测试器 - 消除所有重复代码"""
    
    def __init__(self, base_url: str = 'http://localhost:8080'):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 10
        self.failed_tests = 0
        self.total_tests = 0
    
    def _test_request(self, method: str, endpoint: str, expected_status: int = 200,
                     data: Any = None, files: Dict = None, params: Dict = None) -> bool:
        """统一的请求测试函数 - 消除重复代码"""
        self.total_tests += 1
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params)
            elif method == 'POST':
                if files:
                    response = self.session.post(url, data=data, files=files)
                elif isinstance(data, dict) and not files:
                    response = self.session.post(url, json=data)
                else:
                    response = self.session.post(url, data=data)
            
            success = response.status_code == expected_status
            
            if success:
                print(f"✅ {method} {endpoint}")
            else:
                print(f"❌ {method} {endpoint} - 期望{expected_status}, 实际{response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   错误: {error_data.get('message', 'unknown')}")
                except:
                    print(f"   响应: {response.text[:100]}")
                self.failed_tests += 1
            
            return success
            
        except Exception as e:
            print(f"❌ {method} {endpoint} - 异常: {e}")
            self.failed_tests += 1
            return False
    
    def _create_test_image(self) -> BytesIO:
        """创建测试图片"""
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    def test_basic_endpoints(self):
        """测试基础接口"""
        print("\n🔍 测试基础接口")
        print("-" * 30)
        
        # 基础接口应该总是工作
        self._test_request('GET', '/')
        self._test_request('GET', '/health')
        self._test_request('GET', '/nonexistent', 404)
    
    def test_search_api(self):
        """测试搜索API"""
        print("\n🔍 测试搜索API")
        print("-" * 30)
        
        # 正常搜索
        self._test_request('GET', '/api/search', params={'search': '头痛'})
        
        # 边界情况
        self._test_request('GET', '/api/search', params={'search': ''})
        self._test_request('GET', '/api/search')  # 缺少参数
    
    def test_watch_api(self):
        """测试望诊API"""
        print("\n🔍 测试望诊API")
        print("-" * 30)
        
        # 带图片的望诊
        img = self._create_test_image()
        files = {'image': ('test.jpg', img, 'image/jpeg')}
        self._test_request('POST', '/api/watch', files=files)
        
        # 望诊补充
        data = {
            'prevWatch': '初步观察结果',
            'additionalInfo': '补充信息'
        }
        self._test_request('POST', '/api/watch/complete', data=data)
    
    def test_inquiry_api(self):
        """测试问诊API"""
        print("\n🔍 测试问诊API")
        print("-" * 30)
        
        # 正常问诊
        data = {
            "age": 35,
            "gender": "男", 
            "symptoms": "头痛、发烧、咳嗽"
        }
        self._test_request('POST', '/api/inquiry', data=data)
        
        # 问诊补充
        data = {
            'prevInquiry': '初步诊断',
            'additionalInfo': '补充症状'
        }
        self._test_request('POST', '/api/inquiry/complete', data=data)
        
        # 边界情况：缺少必要参数
        self._test_request('POST', '/api/inquiry', data={'age': 30}, expected_status=400)
    
    def test_record_api(self):
        """测试病历API"""
        print("\n🔍 测试病历API")
        print("-" * 30)
        
        data = {
            "patientInfo": "张三，35岁，男性",
            "symptoms": "头痛、发烧",
            "diagnosis": "感冒"
        }
        self._test_request('POST', '/api/record', data=data)
    
    def test_import_api(self):
        """测试导入API"""
        print("\n🔍 测试导入API")
        print("-" * 30)
        
        # 文件导入
        img = self._create_test_image()
        files = {'file': ('report.jpg', img, 'image/jpeg')}
        self._test_request('POST', '/api/import', files=files)
        
        # JSON导入
        data = {"document": "检查报告内容"}
        self._test_request('POST', '/api/import', data=data)
    
    def test_ai_analysis_api(self):
        """测试AI分析API"""
        print("\n🔍 测试AI分析API") 
        print("-" * 30)
        
        data = {
            'symptoms': '头痛、发烧',
            'history': '无特殊病史'
        }
        self._test_request('POST', '/api/ai/analyze', data=data)
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n🔍 测试错误处理")
        print("-" * 30)
        
        # 错误的Content-Type
        self._test_request('POST', '/api/inquiry', 
                          data="invalid json", expected_status=400)
        
        # 超大请求
        huge_data = {"symptoms": "x" * 10000}
        self._test_request('POST', '/api/inquiry', data=huge_data)
    
    def wait_for_server(self) -> bool:
        """等待服务器启动"""
        print("等待服务器启动...")
        
        for i in range(30):  # 最多等30秒
            try:
                response = self.session.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    print(f"✅ 服务器已启动 ({i+1}s)")
                    return True
            except:
                time.sleep(1)
                if i % 5 == 4:  # 每5秒提示一次
                    print(f"   仍在等待... ({i+1}s)")
        
        print("❌ 服务器启动超时")
        return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 API服务器集成测试")
        print("=" * 50)
        
        if not self.wait_for_server():
            sys.exit(1)
        
        # 按顺序运行所有测试
        test_methods = [
            self.test_basic_endpoints,
            self.test_search_api,
            self.test_watch_api, 
            self.test_inquiry_api,
            self.test_record_api,
            self.test_import_api,
            self.test_ai_analysis_api,
            self.test_error_handling
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"❌ 测试方法 {test_method.__name__} 异常: {e}")
                self.failed_tests += 1
        
        # 测试结果
        print("\n" + "=" * 50)
        print("📊 测试结果")
        print("=" * 50)
        print(f"总测试数: {self.total_tests}")
        print(f"失败数: {self.failed_tests}")
        print(f"成功率: {(self.total_tests - self.failed_tests) / self.total_tests * 100:.1f}%")
        
        if self.failed_tests == 0:
            print("🎉 所有测试通过！")
            return True
        else:
            print(f"💥 {self.failed_tests} 个测试失败")
            return False


class LoadTester:
    """简单的负载测试器"""
    
    def __init__(self, base_url: str = 'http://localhost:8080'):
        self.base_url = base_url
    
    def concurrent_test(self, num_requests: int = 50):
        """并发测试"""
        import threading
        import concurrent.futures
        
        print(f"\n🔥 并发测试 ({num_requests}个请求)")
        print("-" * 30)
        
        def make_request():
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                return response.status_code == 200
            except:
                return False
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        
        success_count = sum(results)
        total_time = end_time - start_time
        
        print(f"总时间: {total_time:.2f}s")
        print(f"成功请求: {success_count}/{num_requests}")
        print(f"QPS: {num_requests/total_time:.1f}")
        
        if success_count == num_requests:
            print("✅ 并发测试通过")
        else:
            print(f"❌ 并发测试失败 ({num_requests - success_count}个请求失败)")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='API服务器集成测试')
    parser.add_argument('--url', default='http://localhost:8080', help='服务器地址')
    parser.add_argument('--load', action='store_true', help='运行负载测试')
    parser.add_argument('--requests', type=int, default=50, help='负载测试请求数')
    
    args = parser.parse_args()
    
    # 运行功能测试
    tester = APITester(args.url)
    success = tester.run_all_tests()
    
    # 运行负载测试
    if args.load and success:
        load_tester = LoadTester(args.url)
        load_tester.concurrent_test(args.requests)
    
    # 退出代码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()