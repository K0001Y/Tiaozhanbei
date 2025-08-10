#!/usr/bin/env python
import sys
import os
import logging
import json
import datetime
import argparse
import re
from typing import Dict, List, Any, TypedDict, Optional

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the node - modify this import based on your project structure
from nodes.i_conversation_chain_node import conversation_chain_node


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


# 模拟消息类
class MockMessage:
    def __init__(self, content: str, message_type: str = "human"):
        self.content = content
        self.type = message_type


def setup_logging():
    """设置日志记录"""
    # 创建日志目录
    log_dir = "tests/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建时间戳用于日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/conversation_chain_test_{timestamp}.log"
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("ConversationChainTest")
    logger.info(f"日志将保存到: {log_file}")
    
    return logger, log_dir, timestamp


def validate_tcm_knowledge(response: str) -> Dict[str, Any]:
    """验证中医知识的准确性"""
    # 中医基础概念关键词
    tcm_concepts = {
        "基础理论": ["阴阳", "五行", "脏腑", "经络", "气血", "津液", "精气神"],
        "诊断方法": ["望闻问切", "望诊", "闻诊", "问诊", "切诊", "辨证论治"],
        "治疗方法": ["中药", "针灸", "推拿", "拔罐", "刮痧", "气功", "食疗"],
        "病理概念": ["风寒暑湿燥火", "六淫", "七情", "痰湿", "血瘀", "气滞"],
        "脏腑理论": ["心肝脾肺肾", "胆胃大肠小肠膀胱三焦"]
    }
    
    validation = {
        "contains_tcm_concepts": False,
        "concept_categories": [],
        "professional_terms_count": 0,
        "explains_concepts": False,
        "mentions_limitations": False,
        "balanced_perspective": False
    }
    
    response_lower = response.lower()
    
    # 检查是否包含中医概念
    for category, concepts in tcm_concepts.items():
        found_concepts = [concept for concept in concepts if concept in response]
        if found_concepts:
            validation["contains_tcm_concepts"] = True
            validation["concept_categories"].append(category)
            validation["professional_terms_count"] += len(found_concepts)
    
    # 检查是否有解释说明
    explanation_indicators = ["就是", "指的是", "意思是", "也就是说", "简单来说", "通俗地讲"]
    validation["explains_concepts"] = any(indicator in response for indicator in explanation_indicators)
    
    # 检查是否提及局限性或建议专业咨询
    limitation_indicators = ["建议咨询", "专业医师", "仅供参考", "不能替代", "具体诊断", "个人情况"]
    validation["mentions_limitations"] = any(indicator in response for indicator in limitation_indicators)
    
    # 检查是否保持平衡观点
    balance_indicators = ["传统观点", "现代医学", "科学证据", "临床研究", "需要验证"]
    validation["balanced_perspective"] = any(indicator in response for indicator in balance_indicators)
    
    return validation


def evaluate_conversation_quality(response: str, test_case: Dict) -> Dict[str, Any]:
    """评估对话质量"""
    evaluation = {
        "has_response": bool(response and response.strip()),
        "adequate_length": len(response) > 50,
        "professional_tone": False,
        "user_friendly": False,
        "contextually_relevant": False,
        "avoids_diagnosis": True,
        "provides_knowledge": False,
        "appropriate_disclaimers": False
    }
    
    if response:
        # 检查专业性
        professional_indicators = ["中医", "传统医学", "理论", "学说", "医学"]
        evaluation["professional_tone"] = any(indicator in response for indicator in professional_indicators)
        
        # 检查用户友好性
        friendly_indicators = ["希望", "帮助", "了解", "您", "建议", "可以"]
        evaluation["user_friendly"] = any(indicator in response for indicator in friendly_indicators)
        
        # 检查上下文相关性
        user_input = test_case.get("user_input", "").lower()
        key_words = [word for word in user_input.split() if len(word) > 1]
        if key_words:
            evaluation["contextually_relevant"] = any(word in response.lower() for word in key_words)
        
        # 检查是否避免直接诊断
        diagnosis_indicators = ["您患有", "诊断为", "确诊", "肯定是", "一定是"]
        evaluation["avoids_diagnosis"] = not any(indicator in response for indicator in diagnosis_indicators)
        
        # 检查是否提供知识
        knowledge_indicators = ["原理", "机制", "作用", "功效", "特点", "方法"]
        evaluation["provides_knowledge"] = any(indicator in response for indicator in knowledge_indicators)
        
        # 检查免责声明
        disclaimer_indicators = ["仅供参考", "不构成医疗建议", "建议咨询专业", "个人情况不同"]
        evaluation["appropriate_disclaimers"] = any(indicator in response for indicator in disclaimer_indicators)
    
    # 验证中医知识
    tcm_validation = validate_tcm_knowledge(response)
    evaluation.update(tcm_validation)
    
    # 计算总分
    score_items = [
        "has_response", "adequate_length", "professional_tone", "user_friendly",
        "contextually_relevant", "avoids_diagnosis", "provides_knowledge",
        "contains_tcm_concepts", "explains_concepts", "mentions_limitations"
    ]
    
    score = sum(evaluation.get(item, False) for item in score_items)
    max_score = len(score_items)
    evaluation["overall_score"] = (score / max_score) * 100
    
    return evaluation


def run_test_case(test_case, logger):
    """运行单个测试用例"""
    logger.info(f"测试用例: {test_case['description']}")
    logger.info(f"用户输入: {test_case['user_input']}")
    
    # 创建状态
    state: State = {
        "user_input": test_case["user_input"],
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
        "relevant_context": test_case.get("relevant_context")
    }
    
    # 记录测试开始时间
    start_time = datetime.datetime.now()
    
    try:
        # 调用对话链节点
        updated_state, route = conversation_chain_node(state)
        
        # 计算响应时间
        response_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # 获取结果
        response = updated_state.get("response", "")
        error = updated_state.get("error")
        
        # 记录日志
        logger.info(f"生成的回复: {response}")
        logger.info(f"路由: {route}")
        logger.info(f"响应时间: {response_time:.2f}秒")
        
        # 评估回复质量
        quality_eval = evaluate_conversation_quality(response, test_case)
        logger.info(f"质量评分: {quality_eval['overall_score']:.1f}/100")
        logger.info(f"包含中医概念: {quality_eval['contains_tcm_concepts']}")
        logger.info(f"概念类别: {quality_eval['concept_categories']}")
        
        # 检查是否符合预期路由
        expected_route = test_case.get("expected_route", "to_safety_check")
        route_matches = route == expected_route
        logger.info(f"路由是否符合预期: {route_matches} (预期: {expected_route}, 实际: {route})")
        
        # 检查是否符合预期主题（如果有）
        expected_topic = test_case.get("expected_topic")
        topic_matches = True
        if expected_topic:
            topic_matches = expected_topic.lower() in response.lower()
            logger.info(f"主题是否符合预期: {topic_matches} (预期包含: {expected_topic})")
        
        # 检查是否有错误
        has_error = error is not None
        if has_error:
            logger.warning(f"节点报告了错误: {error}")
        
        result = {
            "test_case": test_case,
            "response": response,
            "route": route,
            "route_matches_expected": route_matches,
            "topic_matches_expected": topic_matches,
            "quality_evaluation": quality_eval,
            "has_error": has_error,
            "error": error,
            "response_time": response_time,
            "success": (not has_error and route_matches and topic_matches and 
                       quality_eval["overall_score"] >= 60)
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


def test_conversation_chain_node(logger, log_dir, timestamp):
    """测试对话链节点"""
    logger.info("开始对话链节点测试")
    
    # 测试用例
    test_cases = [
        {
            "description": "基础中医理论问题 - 阴阳理论",
            "user_input": "什么是阴阳理论？",
            "relevant_context": "阴阳理论是中医的基础理论之一，认为宇宙万物都可以用阴阳来概括。",
            "expected_route": "to_safety_check",
            "expected_topic": "阴阳",
            "messages": []
        },
        {
            "description": "中医诊断方法询问",
            "user_input": "中医的四诊是什么？",
            "relevant_context": "四诊指望闻问切，是中医诊断疾病的基本方法。",
            "expected_route": "to_safety_check",
            "expected_topic": "望闻问切",
            "messages": []
        },
        {
            "description": "中药知识询问",
            "user_input": "人参有什么功效？",
            "relevant_context": "人参是常用的补气药，具有大补元气、复脉固脱、补脾益肺、生津安神的功效。",
            "expected_route": "to_safety_check",
            "expected_topic": "人参",
            "messages": []
        },
        {
            "description": "针灸相关问题",
            "user_input": "针灸的原理是什么？",
            "relevant_context": "针灸通过刺激特定穴位，调节经络气血，达到治疗疾病的目的。",
            "expected_route": "to_safety_check",
            "expected_topic": "针灸",
            "messages": []
        },
        {
            "description": "有历史对话的连续问题",
            "user_input": "那五行理论呢？",
            "relevant_context": "五行理论是中医另一个重要理论，包括木火土金水五种基本元素。",
            "expected_route": "to_safety_check",
            "expected_topic": "五行",
            "messages": [
                MockMessage("什么是阴阳理论？", "human"),
                MockMessage("阴阳理论是中医的基础理论...", "ai")
            ]
        },
        {
            "description": "现代医学与中医结合的问题",
            "user_input": "中医和西医有什么区别？",
            "relevant_context": "中医注重整体观念和辨证论治，西医注重局部病理和循证医学。",
            "expected_route": "to_safety_check",
            "expected_topic": "中医",
            "messages": []
        },
        {
            "description": "模糊或非中医问题",
            "user_input": "今天天气怎么样？",
            "relevant_context": "",
            "expected_route": "to_safety_check",
            "messages": []
        },
        {
            "description": "无相关上下文的问题",
            "user_input": "什么是经络？",
            "relevant_context": None,
            "expected_route": "to_safety_check",
            "expected_topic": "经络",
            "messages": []
        },
        {
            "description": "复杂的中医理论问题",
            "user_input": "请解释一下脏腑辨证的基本原理",
            "relevant_context": """
            脏腑辨证是以脏腑的生理功能、病理特点为基础，分析病证症状，
            判断脏腑虚实寒热，为确定治疗原则提供依据的辨证方法。
            包括心、肝、脾、肺、肾等脏器的辨证。
            """,
            "expected_route": "to_safety_check",
            "expected_topic": "脏腑",
            "messages": []
        },
        {
            "description": "养生保健类问题",
            "user_input": "中医有哪些养生方法？",
            "relevant_context": "中医养生包括调摄精神、节制饮食、适度运动、起居有常等方面。",
            "expected_route": "to_safety_check",
            "expected_topic": "养生",
            "messages": []
        },
        {
            "description": "边界测试 - 空输入",
            "user_input": "",
            "relevant_context": "中医基础知识",
            "expected_route": "to_safety_check",
            "messages": []
        },
        {
            "description": "性能测试 - 长文本输入",
            "user_input": "请详细介绍中医的发展历史，从古代到现代，包括重要的医学家和著作，以及中医理论的演变过程，还有现代中医的发展现状和未来趋势。",
            "relevant_context": "中医有数千年历史，经历了多个发展阶段...",
            "expected_route": "to_safety_check",
            "expected_topic": "中医",
            "messages": []
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
        r.get("quality_evaluation", {}).get("overall_score", 0) 
        for r in results 
        if r.get("quality_evaluation")
    ]
    avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    logger.info(f"平均质量评分: {avg_quality_score:.1f}/100")
    
    # 统计主题匹配准确性
    topic_matches = sum(1 for r in results if r.get("topic_matches_expected", True))
    topic_accuracy = (topic_matches / len(test_cases)) * 100
    logger.info(f"主题匹配准确率: {topic_matches}/{len(test_cases)} ({topic_accuracy:.2f}%)")
    
    # 统计中医知识覆盖
    tcm_knowledge_count = sum(
        1 for r in results 
        if r.get("quality_evaluation", {}).get("contains_tcm_concepts", False)
    )
    tcm_coverage = (tcm_knowledge_count / len(test_cases)) * 100
    logger.info(f"中医知识覆盖率: {tcm_knowledge_count}/{len(test_cases)} ({tcm_coverage:.2f}%)")
    
    # 保存详细结果到JSON
    results_file = f"{log_dir}/conversation_chain_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"详细结果已保存到 {results_file}")
    
    return results, success_rate, avg_response_time, avg_quality_score, topic_accuracy, tcm_coverage


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试对话链节点')
    parser.add_argument('--filter', type=str, default=None, help='按描述筛选测试用例')
    parser.add_argument('--quality-threshold', type=float, default=60.0, help='质量评分阈值')
    parser.add_argument('--topic-threshold', type=float, default=80.0, help='主题匹配准确率阈值')
    parser.add_argument('--tcm-threshold', type=float, default=70.0, help='中医知识覆盖率阈值')
    args = parser.parse_args()
    
    # 设置日志
    logger, log_dir, timestamp = setup_logging()
    
    logger.info("=" * 60)
    logger.info("对话链节点测试")
    logger.info("=" * 60)
    
    # 运行测试
    results, success_rate, avg_response_time, avg_quality_score, topic_accuracy, tcm_coverage = test_conversation_chain_node(logger, log_dir, timestamp)
    
    # 创建测试报告
    report = {
        "timestamp": timestamp,
        "total_tests": len(results),
        "success_count": sum(1 for r in results if r.get("success", False)),
        "success_rate": success_rate,
        "avg_response_time": avg_response_time,
        "avg_quality_score": avg_quality_score,
        "topic_accuracy": topic_accuracy,
        "tcm_coverage": tcm_coverage,
        "quality_threshold": args.quality_threshold,
        "topic_threshold": args.topic_threshold,
        "tcm_threshold": args.tcm_threshold,
        "results": results
    }
    
    # 保存报告
    report_file = f"{log_dir}/conversation_chain_report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"测试报告已保存到 {report_file}")
    
    # 检查测试结果
    all_passed = all(r.get("success", False) for r in results)
    quality_threshold_met = avg_quality_score >= args.quality_threshold
    topic_threshold_met = topic_accuracy >= args.topic_threshold
    tcm_threshold_met = tcm_coverage >= args.tcm_threshold
    
    if all_passed and quality_threshold_met and topic_threshold_met and tcm_threshold_met:
        logger.info("所有测试均通过且指标达标！✅")
        sys.exit(0)
    else:
        issues = []
        
        if not all_passed:
            failed_tests = [r for r in results if not r.get("success", False)]
            issues.append(f"部分测试失败 ({len(failed_tests)}):")
            for i, test in enumerate(failed_tests):
                issues.append(f"  {i+1}. {test['test_case']['description']}")
                if test.get('error'):
                    issues.append(f"     错误: {test['error']}")
                if not test.get('route_matches_expected', True):
                    issues.append(f"     路由不匹配")
                if not test.get('topic_matches_expected', True):
                    issues.append(f"     主题不匹配")
        
        if not quality_threshold_met:
            issues.append(f"平均质量评分 ({avg_quality_score:.1f}) 未达到阈值 ({args.quality_threshold})")
        
        if not topic_threshold_met:
            issues.append(f"主题匹配准确率 ({topic_accuracy:.1f}%) 未达到阈值 ({args.topic_threshold}%)")
        
        if not tcm_threshold_met:
            issues.append(f"中医知识覆盖率 ({tcm_coverage:.1f}%) 未达到阈值 ({args.tcm_threshold}%)")
        
        for issue in issues:
            logger.info(issue + " ❌")
        
        sys.exit(1)


if __name__ == "__main__":
    main()