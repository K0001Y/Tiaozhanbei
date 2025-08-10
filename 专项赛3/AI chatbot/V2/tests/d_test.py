#!/usr/bin/env python
import sys
import os
import logging
import json
import datetime
import argparse
from typing import Dict, List, Any, TypedDict, Optional

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the node - modify this import based on your project structure

from nodes.d_recognize_intent_node import recognize_intent_node


# Define State type similar to the original code
class State(TypedDict):
    """LangGraph状态类型"""
    user_input: str  # 用户输入
    query: Optional[str]  # 查询（处理后的用户输入）
    messages: List[Any]  # 消息历史
    memory: Optional[Any]  # 对话内存
    documents: Optional[List[Dict[str, Any]]]  # RAG检索结果
    response: Optional[str]  # 最终响应
    error: Optional[str]  # 错误信息
    config: Dict[str, Any]  # 配置信息
    safety_check: Optional[Dict[str, Any]]  # 安全检查结果
    intent: Optional[str]  # 用户意图
    intent_details: Optional[Dict[str, Any]]  # 意图详细信息

def setup_logging():
    """设置日志记录"""
    # 创建日志目录
    log_dir = "tests/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建时间戳用于日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/intent_recognition_test_{timestamp}.log"
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("IntentRecognitionTest")
    logger.info(f"日志将保存到: {log_file}")
    
    return logger, log_dir, timestamp

def run_test_case(test_case, logger):
    """运行单个测试用例"""
    logger.info(f"测试用例: {test_case['description']}")
    logger.info(f"输入: {test_case['user_input']}")
    
    # 创建状态
    state: State = {
        "user_input": test_case["user_input"],
        "messages": [],
        "query": None,
        "memory": None,
        "documents": None,
        "response": None,
        "error": None,
        "config": {},
        "safety_check": None,
        "intent": None,
        "intent_details": None
    }
    
    # 记录测试开始时间
    start_time = datetime.datetime.now()
    
    try:
        # 调用意图识别节点
        updated_state, route = recognize_intent_node(state)
        
        # 计算响应时间
        response_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # 获取结果
        intent = updated_state.get("intent", "unknown")
        details = updated_state.get("intent_details", {})
        
        # 记录日志
        logger.info(f"检测到的意图: {intent}")
        logger.info(f"置信度: {details.get('confidence', 0)}")
        logger.info(f"关键词: {details.get('keywords', [])}")
        logger.info(f"推理过程: {details.get('reasoning', '')}")
        logger.info(f"路由: {route}")
        logger.info(f"响应时间: {response_time:.2f}秒")
        
        # 检查是否符合预期
        matches_expected = intent == test_case["expected_intent"]
        logger.info(f"是否符合预期意图: {matches_expected}")
        
        result = {
            "test_case": test_case,
            "detected_intent": intent,
            "confidence": details.get("confidence", 0),
            "keywords": details.get("keywords", []),
            "reasoning": details.get("reasoning", ""),
            "matches_expected": matches_expected,
            "route": route,
            "response_time": response_time,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"处理测试用例时出错: {str(e)}", exc_info=True)
        result = {
            "test_case": test_case,
            "error": str(e),
            "matches_expected": False,
            "response_time": (datetime.datetime.now() - start_time).total_seconds()
        }
    
    logger.info("-" * 50)
    return result

def test_intent_recognition(logger, log_dir, timestamp):
    """测试意图识别节点"""
    logger.info("开始意图识别测试")
    
    # 测试用例
    test_cases = [
        {
            "user_input": "我最近头疼得厉害，是怎么回事？",
            "expected_intent": "diagnosis",
            "description": "明确的医疗症状（头痛）"
        },
        {
            "user_input": "如何缓解感冒症状？",
            "expected_intent": "diagnosis",
            "description": "寻求治疗建议"
        },
        {
            "user_input": "什么是高血压？",
            "expected_intent": "question",
            "description": "一般医疗问题"
        },
        {
            "user_input": "今天天气真好",
            "expected_intent": "unclear",
            "description": "非医疗相关陈述"
        },
        {
            "user_input": "我的病历上显示血压偏高",
            "expected_intent": "diagnosis",
            "description": "包含医疗关键词'病历'"
        },
        {
            "user_input": "我想知道心脏CT检查有什么风险",
            "expected_intent": "diagnosis",
            "description": "包含医疗关键词'CT'"
        },
        {
            "user_input": "我最近做了B超检查，医生说肝脏有点问题",
            "expected_intent": "diagnosis",
            "description": "包含医疗关键词'B超'和描述病情"
        },
        {
            "user_input": "请帮我解释一下这个化验单的结果",
            "expected_intent": "diagnosis",
            "description": "包含医疗关键词'化验'"
        },
        {
            "user_input": "我可以吃什么水果？",
            "expected_intent": "question",
            "description": "一般饮食问题，不是明确的诊断请求"
        },
        {
            "user_input": "你好，请问你是谁？",
            "expected_intent": "unclear",
            "description": "询问系统身份，非医疗相关"
        }
    ]
    
    # 运行测试
    results = []
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"测试 {i+1}/{len(test_cases)}")
        result = run_test_case(test_case, logger)
        results.append(result)
    
    # 计算成功率
    success_count = sum(1 for r in results if r.get("matches_expected", False))
    success_rate = (success_count / len(test_cases)) * 100
    
    logger.info(f"测试摘要: {success_count}/{len(test_cases)} 测试通过 ({success_rate:.2f}%)")
    
    # 计算平均响应时间
    response_times = [r.get("response_time", 0) for r in results if "response_time" in r]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    logger.info(f"平均响应时间: {avg_response_time:.2f}秒")
    
    # 保存详细结果到JSON
    results_file = f"{log_dir}/intent_recognition_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"详细结果已保存到 {results_file}")
    
    return results, success_rate, avg_response_time

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试意图识别节点')
    parser.add_argument('--filter', type=str, default=None, help='按描述筛选测试用例')
    args = parser.parse_args()
    
    # 设置日志
    logger, log_dir, timestamp = setup_logging()
    
    logger.info("=" * 60)
    logger.info("意图识别节点测试")
    logger.info("=" * 60)
    
    # 运行测试
    results, success_rate, avg_response_time = test_intent_recognition(logger, log_dir, timestamp)
    
    # 创建测试报告
    report = {
        "timestamp": timestamp,
        "total_tests": len(results),
        "success_count": sum(1 for r in results if r.get("matches_expected", False)),
        "success_rate": success_rate,
        "avg_response_time": avg_response_time,
        "results": results
    }
    
    # 保存报告
    report_file = f"{log_dir}/intent_recognition_report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"测试报告已保存到 {report_file}")
    
    # 检查是否所有测试都通过
    all_passed = all(r.get("matches_expected", False) for r in results)
    
    if all_passed:
        logger.info("所有测试均通过！✅")
        sys.exit(0)
    else:
        logger.info("部分测试失败。请查看日志了解详情。❌")
        # 打印失败的测试
        failed_tests = [r for r in results if not r.get("matches_expected", False)]
        logger.info(f"失败的测试 ({len(failed_tests)}):")
        for i, test in enumerate(failed_tests):
            logger.info(f"  {i+1}. 输入: '{test['test_case']['user_input']}'")
            logger.info(f"     预期: '{test['test_case']['expected_intent']}', 实际: '{test.get('detected_intent', 'ERROR')}'")
        sys.exit(1)

if __name__ == "__main__":
    main()