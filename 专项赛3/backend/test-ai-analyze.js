// 测试7.1 AI智能分析接口
const fetch = require('node-fetch');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const BASE_URL = 'http://localhost:5001';

// 测试用的token - 实际使用时需要登录获取
const TEST_TOKEN = 'your_test_token_here';

async function testAIAnalyze() {
  console.log('🧪 开始测试7.1 AI智能分析接口...\n');

  // 测试1: 纯文本查询
  try {
    console.log('📝 测试1: 纯文本查询');
    const response1 = await fetch(`${BASE_URL}/api/ai/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${TEST_TOKEN}`
      },
      body: JSON.stringify({
        query: '我最近经常头痛，还伴有恶心的症状，请帮我分析一下可能的原因'
      })
    });

    const result1 = await response1.json();
    console.log('✅ 文本查询结果:', result1);
    console.log('📋 Solution内容预览:', result1.data?.solution?.substring(0, 200) + '...\n');
  } catch (error) {
    console.error('❌ 文本查询测试失败:', error.message);
  }

  // 测试2: 文件上传（如果有测试图片的话）
  const testImagePath = path.join(__dirname, 'test-image.jpg');
  if (fs.existsSync(testImagePath)) {
    try {
      console.log('🖼️ 测试2: 文件上传分析');
      const formData = new FormData();
      formData.append('file', fs.createReadStream(testImagePath));
      formData.append('query', '请分析这张医学图片');

      const response2 = await fetch(`${BASE_URL}/api/ai/analyze`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${TEST_TOKEN}`
        },
        body: formData
      });

      const result2 = await response2.json();
      console.log('✅ 文件分析结果:', result2);
      console.log('📋 Solution内容预览:', result2.data?.solution?.substring(0, 200) + '...\n');
    } catch (error) {
      console.error('❌ 文件上传测试失败:', error.message);
    }
  } else {
    console.log('⚠️ 跳过文件测试 - 未找到测试图片文件\n');
  }

  // 测试3: 错误情况 - 空参数
  try {
    console.log('🚫 测试3: 错误情况测试（空参数）');
    const response3 = await fetch(`${BASE_URL}/api/ai/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${TEST_TOKEN}`
      },
      body: JSON.stringify({})
    });

    const result3 = await response3.json();
    console.log('✅ 错误处理结果:', result3);
  } catch (error) {
    console.error('❌ 错误测试失败:', error.message);
  }

  console.log('\n🎉 测试完成!');
}

// 测试服务器连接
async function testServerConnection() {
  try {
    const response = await fetch(`${BASE_URL}/health`);
    const result = await response.json();
    console.log('🏥 服务器状态:', result);
    return true;
  } catch (error) {
    console.error('❌ 服务器连接失败:', error.message);
    console.log('💡 请确保后端服务器运行在端口5001');
    return false;
  }
}

async function main() {
  const serverOk = await testServerConnection();
  if (serverOk) {
    await testAIAnalyze();
  }
}

// 如果直接运行此文件
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { testAIAnalyze };
