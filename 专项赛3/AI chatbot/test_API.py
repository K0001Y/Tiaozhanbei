"""
API功能测试脚本
用于验证Web API的各个端点是否正常工作
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_api_status():
    """测试API状态"""
    try:
        print("🧪 测试API状态...")
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ API状态正常")
            print(f"   版本: {data.get('data', {}).get('version', 'N/A')}")
            return True
        else:
            print(f"❌ API状态异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 连接API失败: {str(e)}")
        return False

def test_chat():
    """测试聊天功能"""
    try:
        print("\n🧪 测试聊天功能...")
        
        chat_data = {
            "message": "你好，请介绍一下自己",
            "session_id": "test-session"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=chat_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 聊天功能正常")
            print(f"   响应: {data['response'][:100]}...")
            print(f"   响应时间: {data['response_time']:.2f}秒")
            print(f"   RAG状态: {data['used_rag']}")
            return True
        else:
            print(f"❌ 聊天功能异常: {response.status_code}")
            print(f"   错误: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 聊天测试失败: {str(e)}")
        return False

def test_config():
    """测试配置获取"""
    try:
        print("\n🧪 测试配置获取...")
        
        response = requests.get(f"{API_BASE_URL}/config?session_id=test-session")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 配置获取正常")
            print(f"   模型: {data['model_name']}")
            print(f"   RAG启用: {data['rag_enabled']}")
            return True
        else:
            print(f"❌ 配置获取异常: {response.status_code}")
            print(f"   错误: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 配置测试失败: {str(e)}")
        return False

def test_sessions():
    """测试会话管理"""
    try:
        print("\n🧪 测试会话管理...")
        
        # 获取会话列表
        response = requests.get(f"{API_BASE_URL}/sessions")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 会话管理正常")
            print(f"   活跃会话数: {data['active_sessions']}")
            
            # 测试清除会话历史
            clear_response = requests.post(f"{API_BASE_URL}/sessions/test-session/clear")
            if clear_response.status_code == 200:
                print("✅ 会话历史清除正常")
                return True
            else:
                print(f"⚠️ 会话历史清除异常: {clear_response.status_code}")
                return False
        else:
            print(f"❌ 会话管理异常: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 会话测试失败: {str(e)}")
        return False

def test_stats():
    """测试统计信息"""
    try:
        print("\n🧪 测试统计信息...")
        
        response = requests.get(f"{API_BASE_URL}/stats")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 统计信息正常")
            print(f"   总请求数: {data['performance'].get('总请求数', 0)}")
            print(f"   活跃会话: {data['sessions']['total']}")
            return True
        else:
            print(f"❌ 统计信息异常: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 统计测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🔍 LangChain聊天机器人 - API功能测试")
    print("=" * 60)
    
    # 等待服务器启动
    print("⏳ 等待服务器启动...")
    time.sleep(2)
    
    # 运行所有测试
    tests = [
        ("API状态", test_api_status),
        ("聊天功能", test_chat),
        ("配置获取", test_config),
        ("会话管理", test_sessions),
        ("统计信息", test_stats),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        print("-" * 40)
    
    print(f"\n📊 测试结果: {passed}/{len(tests)} 通过")
    
    if passed == len(tests):
        print("🎉 所有API测试通过！")
        print("💡 你可以现在打开前端界面测试完整功能")
        print("   - 打开 frontend_example.html")
        print("   - 或访问 http://localhost:8000/docs")
    elif passed >= len(tests) // 2:
        print("⚠️ 部分API功能正常，可以继续使用")
        print("💡 有些功能可能需要进一步调试")
    else:
        print("❌ 大部分API功能异常，请检查服务器日志")

if __name__ == "__main__":
    main()