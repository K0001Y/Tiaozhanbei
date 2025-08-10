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
from nodes.f_follow_up_question_node import follow_up_question_node


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
    relevant_context: Optional[str]  # RAG检索相关上下文
    symptoms_list: Optional[List[Dict[str, Any]]]  # 提取的症状列表
    missing_info_list: Optional[List[str]]  # 缺失的信息列表
    conversation_state: Optional[str]  # 对话状态标记


# 模拟消息类
class MockMessage:
    def __init__(self, content: str, message_type: str = "ai"):
        self.content = content
        self.type = message_type


def setup_logging():
    """设置日志记录"""
    # 创建日志目录
    log_dir = "tests/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建时间戳用于日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/follow_up_question_test_{timestamp}.log"
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("FollowUpQuestionTest")
    logger.info(f"日志将保存到: {log_file}")
    
    return logger, log_dir, timestamp


def evaluate_question_quality(question: str, test_case: Dict) -> Dict[str, Any]:
    """评估追问质量"""
    evaluation = {
        "has_content": bool(question and question.strip()),
        "is_question": "?" in question,
        "is_polite": any(word in question.lower() for word in ["请", "您", "能否", "可以", "谢谢"]),
        "is_medical": any(word in question.lower() for word in ["症状", "疼痛", "不适", "病史", "时间", "位置", "严重"]),
        "avoids_repetition": True,  # 默认为True，在有历史消息时会检查
        "length_appropriate": 10 < len(question) < 500,
        "contains_empathy": any(word in question.lower() for word in ["了解", "帮助", "更好", "准确"])
    }
    
    # 检查是否避免重复（如果有历史追问）
    if test_case.get("previous_questions"):
        prev_questions_text = " ".join(test_case["previous_questions"])
        # 简单检查是否包含相同的关键词
        question_words = set(question.lower().split())
        prev_words = set(prev_questions_text.lower().split())
        overlap_ratio = len(question_words & prev_words) / len(question_words) if question_words else 0
        evaluation["avoids_repetition"] = overlap_ratio < 0.7  # 如果重叠度小于70%认为避免了重复
    
    # 计算总分
    total_score = sum(evaluation.values())
    max_score = len(evaluation)
    evaluation["quality_score"] = (total_score / max_score) * 100
    
    return evaluation


def run_test_case(test_case, logger):
    """运行单个测试用例"""
    logger.info(f"测试用例: {test_case['description']}")
    logger.info(f"症状列表: {test_case.get('symptoms_list', [])}")
    logger.info(f"缺失信息: {test_case.get('missing_info_list', [])}")
    
    # 创建状态
    state: State = {
        "user_input": test_case.get("user_input", ""),
        "messages": test_case.get("messages", []),
        "query": None,
        "memory": None,
        "documents": None,
        "response": None,
        "error": None,
        "config": {},
        "safety_check": None,
        "intent": None,
        "intent_details": None,
        "relevant_context": None,
        "symptoms_list": test_case.get("symptoms_list"),
        "missing_info_list": test_case.get("missing_info_list"),
        "conversation_state": test_case.get("conversation_state")
    }
    
    # 记录测试开始时间
    start_time = datetime.datetime.now()
    
    try:
        # 调用追问节点
        updated_state, route = follow_up_question_node(state)
        
        # 计算响应时间
        response_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # 获取结果
        generated_question = updated_state.get("response", "")
        conversation_state = updated_state.get("conversation_state", "")
        error = updated_state.get("error")
        
        # 记录日志
        logger.info(f"生成的追问: {generated_question}")
        logger.info(f"对话状态: {conversation_state}")
        logger.info(f"路由: {route}")
        logger.info(f"响应时间: {response_time:.2f}秒")
        
        # 评估追问质量
        quality_eval = evaluate_question_quality(generated_question, test_case)
        logger.info(f"质量评分: {quality_eval['quality_score']:.1f}/100")
        
        # 检查是否符合预期路由
        expected_route = test_case.get("expected_route", "to_output")
        route_matches = route == expected_route
        logger.info(f"路由是否符合预期: {route_matches} (预期: {expected_route}, 实际: {route})")
        
        # 检查是否有错误
        has_error = error is not None
        if has_error:
            logger.warning(f"节点报告了错误: {error}")
        
        result = {
            "test_case": test_case,
            "generated_question": generated_question,
            "conversation_state": conversation_state,
            "route": route,
            "route_matches_expected": route_matches,
            "quality_evaluation": quality_eval,
            "has_error": has_error,
            "error": error,
            "response_time": response_time,
            "success": not has_error and route_matches and quality_eval["quality_score"] >= 50
        }
        
    except Exception as e:
        logger.error(f"处理测试用例时出错: {str(e)}", exc_info=True)
        result = {
            "test_case": test_case,
            "error": str(e),
            "success": False,
            "response_time": (datetime.datetime.now() - start_time).total_seconds()
        }
    
    logger.info("-" * 50)
    return result


def test_follow_up_question_node(logger, log_dir, timestamp):
    """测试追问节点"""
    logger.info("开始追问节点测试")
    
    # 测试用例
    test_cases = [
        {
            "description": "基础情况 - 有症状但缺少详细信息",
            "symptoms_list": [
                {
                    "name": "头痛",
                    "description": "头部疼痛",
                    "severity": "中等",
                    "duration": "未提及"
                }
            ],
            "missing_info_list": [
                "症状的具体位置",
                "持续时间",
                "诱发因素"
            ],
            "messages": [],
            "expected_route": "to_output",
            "user_input": "我头痛"
        },
        {
            "description": "复杂情况 - 多个症状，部分信息缺失",
            "symptoms_list": [
                {
                    "name": "胸痛",
                    "description": "胸部闷痛",
                    "severity": "严重",
                    "duration": "2小时"
                },
                {
                    "name": "气短",
                    "description": "呼吸困难",
                    "severity": "未提及",
                    "duration": "未提及"
                }
            ],
            "missing_info_list": [
                "胸痛的具体位置",
                "气短的诱发因素",
                "相关病史"
            ],
            "messages": [],
            "expected_route": "to_output"
        },
        {
            "description": "无症状情况 - 需要引导用户描述症状",
            "symptoms_list": [],
            "missing_info_list": [],
            "messages": [],
            "expected_route": "to_output",
            "user_input": "我感觉不舒服"
        },
        {
            "description": "有历史追问 - 应避免重复问题",
            "symptoms_list": [
                {
                    "name": "腹痛",
                    "description": "肚子痛",
                    "severity": "未提及",
                    "duration": "未提及"
                }
            ],
            "missing_info_list": [
                "疼痛的具体位置",
                "疼痛的性质",
                "持续时间",
                "相关症状"
            ],
            "messages": [
                MockMessage("请问您的腹痛是在哪个具体位置？", "ai"),
                MockMessage("肚子中间", "human"),
                MockMessage("这种疼痛持续多长时间了？", "ai")
            ],
            "previous_questions": [
                "请问您的腹痛是在哪个具体位置？",
                "这种疼痛持续多长时间了？"
            ],
            "expected_route": "to_output"
        },
        {
            "description": "信息充足情况 - 应路由到诊断",
            "symptoms_list": [
                {
                    "name": "发热",
                    "description": "体温38.5度",
                    "severity": "中等",
                    "duration": "3天"
                }
            ],
            "missing_info_list": [],
            "messages": [],
            "expected_route": "to_diagnosis"
        },
        {
            "description": "边界情况 - 空的缺失信息列表但有症状",
            "symptoms_list": [
                {
                    "name": "咳嗽",
                    "description": "干咳",
                    "severity": "轻微",
                    "duration": "1周"
                }
            ],
            "missing_info_list": [],
            "messages": [],
            "expected_route": "to_diagnosis"
        },
        {
            "description": "大量历史消息 - 测试性能",
            "symptoms_list": [
                {
                    "name": "关节痛",
                    "description": "膝盖疼痛",
                    "severity": "未提及",
                    "duration": "未提及"
                }
            ],
            "missing_info_list": [
                "疼痛的具体位置",
                "活动时的疼痛情况"
            ],
            "messages": [MockMessage(f"历史消息{i}", "human") for i in range(20)],
            "expected_route": "to_output"
        },
        {
            "description": "特殊字符处理 - 测试鲁棒性",
            "symptoms_list": [
                {
                    "name": "皮疹",
                    "description": "皮肤出现红点...",
                    "severity": "轻微",
                    "duration": "未知"
                }
            ],
            "missing_info_list": [
                "皮疹的分布范围",
                "是否伴有瘙痒?"
            ],
            "messages": [],
            "expected_route": "to_output"
        }
    ]
    
    # 运行测试
    results = []
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"测试 {i+1}/{len(test_cases)}")
        result = run_test_case(test_case, logger)
        results.append(result)
    
    # 计算成功率
    success_count = sum(1 for r in results if r.get("success", False))
    success_rate = (success_count / len(test_cases)) * 100
    
    logger.info(f"测试摘要: {success_count}/{len(test_cases)} 测试通过 ({success_rate:.2f}%)")
    
    # 计算平均响应时间
    response_times = [r.get("response_time", 0) for r in results if "response_time" in r]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    logger.info(f"平均响应时间: {avg_response_time:.2f}秒")
    
    # 计算平均质量评分
    quality_scores = [
        r.get("quality_evaluation", {}).get("quality_score", 0) 
        for r in results 
        if r.get("quality_evaluation")
    ]
    avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    logger.info(f"平均质量评分: {avg_quality_score:.1f}/100")
    
    # 保存详细结果到JSON
    results_file = f"{log_dir}/follow_up_question_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"详细结果已保存到 {results_file}")
    
    return results, success_rate, avg_response_time, avg_quality_score


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试追问节点')
    parser.add_argument('--filter', type=str, default=None, help='按描述筛选测试用例')
    parser.add_argument('--quality-threshold', type=float, default=50.0, help='质量评分阈值')
    args = parser.parse_args()
    
    # 设置日志
    logger, log_dir, timestamp = setup_logging()
    
    logger.info("=" * 60)
    logger.info("追问节点测试")
    logger.info("=" * 60)
    
    # 运行测试
    results, success_rate, avg_response_time, avg_quality_score = test_follow_up_question_node(logger, log_dir, timestamp)
    
    # 创建测试报告
    report = {
        "timestamp": timestamp,
        "total_tests": len(results),
        "success_count": sum(1 for r in results if r.get("success", False)),
        "success_rate": success_rate,
        "avg_response_time": avg_response_time,
        "avg_quality_score": avg_quality_score,
        "quality_threshold": args.quality_threshold,
        "results": results
    }
    
    # 保存报告
    report_file = f"{log_dir}/follow_up_question_report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"测试报告已保存到 {report_file}")
    
    # 检查测试结果
    all_passed = all(r.get("success", False) for r in results)
    quality_threshold_met = avg_quality_score >= args.quality_threshold
    
    if all_passed and quality_threshold_met:
        logger.info("所有测试均通过且质量达标！✅")
        sys.exit(0)
    else:
        if not all_passed:
            logger.info("部分测试失败。❌")
            # 打印失败的测试
            failed_tests = [r for r in results if not r.get("success", False)]
            logger.info(f"失败的测试 ({len(failed_tests)}):")
            for i, test in enumerate(failed_tests):
                logger.info(f"  {i+1}. {test['test_case']['description']}")
                if test.get('error'):
                    logger.info(f"     错误: {test['error']}")
                if not test.get('route_matches_expected', True):
                    logger.info(f"     路由不匹配: 预期 {test['test_case'].get('expected_route')}, 实际 {test.get('route')}")
        
        if not quality_threshold_met:
            logger.info(f"平均质量评分 ({avg_quality_score:.1f}) 未达到阈值 ({args.quality_threshold})。❌")
        
        sys.exit(1)


if __name__ == "__main__":
    main()