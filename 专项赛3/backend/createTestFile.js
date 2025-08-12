const fs = require('fs');
const path = require('path');

// 创建测试文件
const testFile = path.join(__dirname, 'uploads', 'library', 'test-document.txt');
const testContent = `
# 测试医学文档

这是一个测试用的医学知识库文档。

## 症状描述
- 头痛
- 头晕
- 恶心

## 可能的病因
1. 高血压
2. 偏头痛
3. 感冒

## 治疗建议
- 充分休息
- 适量饮水
- 必要时就医
`;

// 确保目录存在
const uploadDir = path.dirname(testFile);
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// 写入测试文件
fs.writeFileSync(testFile, testContent, 'utf8');
console.log('测试文件创建成功:', testFile);
