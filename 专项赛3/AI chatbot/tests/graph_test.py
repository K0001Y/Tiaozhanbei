#!/usr/bin/env python
import sys
import os
import logging
import json
import datetime
import argparse
from typing import Dict, List, Any, TypedDict, Optional
import time

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main graph functions - modify this import based on your project structure
from graph import run_tcm_graph, State


def setup_logging():
    """设置日志记录"""
    # 创建日志目录
    log_dir = "tests/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建时间戳用于日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/tcm_graph_test_{timestamp}.log"
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("TCMGraphTest")
    logger.info(f"日志将保存到: {log_file}")
    
    return logger, log_dir, timestamp


def validate_intent_recognition(intent: str, expected_intent: str) -> Dict[str, Any]:
    """验证意图识别的准确性"""
    validation = {
        "intent_correct": intent == expected_intent,
        "intent_valid": intent in ["diagnosis", "consultation", "general", "emergency"],
        "intent_detected": intent is not None and intent != ""
    }
    return validation


def validate_symptom_extraction(symptoms_list: List[Dict[str, Any]], expected_symptoms: List[str]) -> Dict[str, Any]:
    """验证症状提取的质量"""
    if not symptoms_list:
        return {
            "symptoms_extracted": False,
            "symptom_count": 0,
            "expected_symptoms_found": 0,
            "extraction_completeness": 0.0
        }
    
    extracted_symptom_names = [s.get("name", "").lower() for s in symptoms_list]
    expected_found = sum(
        1 for expected in expected_symptoms 
        if any(expected.lower() in extracted.lower() for extracted in extracted_symptom_names)
    )
    
    validation = {
        "symptoms_extracted": len(symptoms_list) > 0,
        "symptom_count": len(symptoms_list),
        "expected_symptoms_found": expected_found,
        "extraction_completeness": expected_found / len(expected_symptoms) if expected_symptoms else 1.0,
        "symptoms_have_details": all(
            s.get("description") and s.get("severity") and s.get("duration") 
            for s in symptoms_list
        )
    }
    return validation


def validate_diagnosis_quality(diagnosis_data: Dict[str, Any]) -> Dict[str, Any]:
    """验证诊断分析的质量"""
    if not diagnosis_data:
        return {"has_diagnosis": False, "diagnosis_score": 0}
    
    validation = {
        "has_diagnosis": True,
        "has_pattern_type": bool(diagnosis_data.get("pattern_type")),
        "has_pathogenesis": bool(diagnosis_data.get("pathogenesis")),
        "has_analysis": bool(diagnosis_data.get("analysis")),
        "confidence_valid": 0 <= diagnosis_data.get("confidence", -1) <= 1,
        "has_differential": bool(diagnosis_data.get("differential_diagnosis")),
        "analysis_detailed": len(diagnosis_data.get("analysis", "")) > 50,
        "pathogenesis_detailed": len(diagnosis_data.get("pathogenesis", "")) > 20
    }
    
    # 计算诊断质量评分
    score_items = [
        "has_pattern_type", "has_pathogenesis", "has_analysis", 
        "confidence_valid", "has_differential", "analysis_detailed", "pathogenesis_detailed"
    ]
    score = sum(validation.get(item, False) for item in score_items)
    validation["diagnosis_score"] = (score / len(score_items)) * 100
    
    return validation


def validate_prescription_quality(prescription_data: Dict[str, Any]) -> Dict[str, Any]:
    """验证处方推荐的质量"""
    if not prescription_data:
        return {"has_prescription": False, "prescription_score": 0}
    
    validation = {
        "has_prescription": True,
        "has_formula_name": bool(prescription_data.get("formula_name")),
        "has_herbs": bool(prescription_data.get("herbs")),
        "has_dosage": bool(prescription_data.get("dosage_instructions")),
        "has_usage": bool(prescription_data.get("usage_instructions")),
        "has_precautions": bool(prescription_data.get("precautions")),
        "herbs_detailed": len(prescription_data.get("herbs", [])) > 0
    }
    
    # 检查草药详情
    herbs = prescription_data.get("herbs", [])
    if herbs:
        validation["herbs_have_dosage"] = all(
            herb.get("dosage") for herb in herbs
        )
        validation["herbs_have_function"] = all(
            herb.get("function") for herb in herbs
        )
    else:
        validation["herbs_have_dosage"] = False
        validation["herbs_have_function"] = False
    
    # 计算处方质量评分
    score_items = [
        "has_formula_name", "has_herbs", "has_dosage", "has_usage", 
        "has_precautions", "herbs_detailed", "herbs_have_dosage", "herbs_have_function"
    ]
    score = sum(validation.get(item, False) for item in score_items)
    validation["prescription_score"] = (score / len(score_items)) * 100
    
    return validation


def validate_safety_checks(result: Dict[str, Any]) -> Dict[str, Any]:
    """验证安全检查的效果"""
    safety_check = result.get("safety_check", {})
    safety_violations = result.get("safety_violations", [])
    
    validation = {
        "safety_check_performed": bool(safety_check),
        "emergency_detected": safety_check.get("is_emergency", False),
        "safety_violations_checked": safety_violations is not None,
        "response_safe": len(safety_violations or []) == 0,
        "has_safety_score": "safety_score" in safety_check
    }
    
    return validation


def evaluate_graph_execution_quality(result: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
    """评估整个图执行的质量"""
    # 基础检查
    evaluation = {
        "execution_successful": not bool(result.get("error")),
        "has_response": bool(result.get("response")),
        "response_length_adequate": len(result.get("response", "")) > 20,
        "intent_accuracy": {},
        "symptom_extraction": {},
        "diagnosis_quality": {},
        "prescription_quality": {},
        "safety_validation": {}
    }
    
    # 意图识别验证
    expected_intent = test_case.get("expected_intent")
    if expected_intent:
        evaluation["intent_accuracy"] = validate_intent_recognition(
            result.get("intent"), expected_intent
        )
    
    # 症状提取验证
    expected_symptoms = test_case.get("expected_symptoms", [])
    evaluation["symptom_extraction"] = validate_symptom_extraction(
        result.get("symptoms_list", []), expected_symptoms
    )
    
    # 诊断质量验证
    evaluation["diagnosis_quality"] = validate_diagnosis_quality(
        result.get("diagnosis_data", {})
    )
    
    # 处方质量验证
    evaluation["prescription_quality"] = validate_prescription_quality(
        result.get("prescription_data", {})
    )
    
    # 安全检查验证
    evaluation["safety_validation"] = validate_safety_checks(result)
    
    # 计算总体质量评分
    quality_scores = [
        evaluation["intent_accuracy"].get("intent_correct", 0) * 100,
        evaluation["symptom_extraction"].get("extraction_completeness", 0) * 100,
        evaluation["diagnosis_quality"].get("diagnosis_score", 0),
        evaluation["prescription_quality"].get("prescription_score", 0),
        evaluation["safety_validation"].get("response_safe", 0) * 100
    ]
    
    # 只对有效的评分计算平均值
    valid_scores = [score for score in quality_scores if score > 0]
    evaluation["overall_quality_score"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    
    return evaluation


def run_test_case(test_case: Dict[str, Any], logger) -> Dict[str, Any]:
    """运行单个测试用例"""
    logger.info(f"测试用例: {test_case['description']}")
    logger.info(f"用户输入: {test_case['user_input']}")
    
    # 记录测试开始时间
    start_time = datetime.datetime.now()
    
    try:
        # 运行图
        result = run_tcm_graph(
            user_input=test_case["user_input"],
            messages=test_case.get("messages", []),
            memory=test_case.get("memory"),
            config=test_case.get("config", {"retriever_k": 4})
        )
        
        # 计算响应时间
        response_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # 记录基本结果
        logger.info(f"执行成功: {not bool(result.get('error'))}")
        logger.info(f"用户意图: {result.get('intent', 'N/A')}")
        logger.info(f"提取症状数: {len(result.get('symptoms_list', []))}")
        logger.info(f"对话状态: {result.get('conversation_state', 'N/A')}")
        logger.info(f"响应长度: {len(result.get('response', ''))}")
        logger.info(f"响应时间: {response_time:.2f}秒")
        
        # 诊断结果
        diagnosis_data = result.get("diagnosis_data")
        if diagnosis_data:
            logger.info(f"诊断证型: {diagnosis_data.get('pattern_type', 'N/A')}")
            logger.info(f"诊断置信度: {diagnosis_data.get('confidence', 'N/A')}")
        
        # 处方结果
        prescription_data = result.get("prescription_data")
        if prescription_data:
            logger.info(f"推荐方剂: {prescription_data.get('formula_name', 'N/A')}")
            logger.info(f"草药数量: {len(prescription_data.get('herbs', []))}")
        
        # 安全检查结果
        safety_check = result.get("safety_check", {})
        if safety_check:
            logger.info(f"紧急情况检测: {safety_check.get('is_emergency', False)}")
            logger.info(f"安全评分: {safety_check.get('safety_score', 'N/A')}")
        
        # 错误信息
        if result.get("error"):
            logger.warning(f"执行错误: {result.get('error')}")
        
        # 评估执行质量
        quality_evaluation = evaluate_graph_execution_quality(result, test_case)
        logger.info(f"整体质量评分: {quality_evaluation['overall_quality_score']:.1f}/100")
        
        # 检查是否符合预期
        expected_intent = test_case.get("expected_intent")
        intent_matches = (
            not expected_intent or 
            result.get("intent") == expected_intent
        )
        
        expected_conversation_state = test_case.get("expected_conversation_state")
        state_matches = (
            not expected_conversation_state or 
            result.get("conversation_state") == expected_conversation_state
        )
        
        # 构建测试结果
        test_result = {
            "test_case": test_case,
            "execution_result": result,
            "response_time": response_time,
            "quality_evaluation": quality_evaluation,
            "intent_matches_expected": intent_matches,
            "state_matches_expected": state_matches,
            "has_error": bool(result.get("error")),
            "success": (
                not bool(result.get("error")) and 
                intent_matches and 
                state_matches and 
                quality_evaluation["overall_quality_score"] >= 60
            )
        }
        
    except Exception as e:
        logger.error(f"测试用例执行失败: {str(e)}", exc_info=True)
        response_time = (datetime.datetime.now() - start_time).total_seconds()
        test_result = {
            "test_case": test_case,
            "error": str(e),
            "response_time": response_time,
            "success": False
        }
    
    logger.info("-" * 60)
    return test_result


def test_tcm_graph(logger, log_dir, timestamp):
    """测试中医智能对话系统图"""
    logger.info("开始中医智能对话系统图测试")
    
    # 测试用例
    test_cases = [
        {
            "description": "完整诊断流程 - 典型症状",
            "user_input": "医生，我最近总是感到头晕目眩，还有腰膝酸软，晚上失眠多梦，手心脚心发热，请帮我看看是什么问题？",
            "expected_intent": "diagnosis",
            "expected_symptoms": ["头晕目眩", "腰膝酸软", "失眠多梦", "五心烦热"],
            "expected_conversation_state": None,  # 应该直接进入诊断，不需要追问
            "category": "complete_diagnosis"
        },
        {
            "description": "需要追问的诊断流程 - 症状不足",
            "user_input": "我咳嗽",
            "expected_intent": "diagnosis", 
            "expected_symptoms": ["咳嗽"],
            "expected_conversation_state": "awaiting_follow_up",
            "category": "follow_up_needed"
        },
        {
            "description": "典型脾虚证诊断",
            "user_input": "我最近总是感到疲乏无力，不想吃饭，大便也不成形，容易腹胀，请医生帮我诊断一下。",
            "expected_intent": "diagnosis",
            "expected_symptoms": ["疲乏无力", "食欲不振", "大便溏薄", "腹胀"],
            "expected_conversation_state": None,
            "category": "spleen_deficiency"
        },
        {
            "description": "肝郁气滞证诊断", 
            "user_input": "医生，我胸胁部位经常胀痛，心情抑郁，容易生气，还经常嗳气，这是怎么回事？",
            "expected_intent": "diagnosis",
            "expected_symptoms": ["胸胁胀痛", "情志抑郁", "易怒", "嗳气"],
            "expected_conversation_state": None,
            "category": "liver_qi_stagnation"
        },
        {
            "description": "一般咨询 - 非诊断意图",
            "user_input": "请问中医的基本理论是什么？",
            "expected_intent": "consultation",
            "expected_symptoms": [],
            "expected_conversation_state": None,
            "category": "general_consultation"
        },
        {
            "description": "紧急情况 - 安全检查",
            "user_input": "我胸痛得很厉害，呼吸困难，感觉快要死了！",
            "expected_intent": "emergency",
            "expected_symptoms": ["胸痛", "呼吸困难"],
            "expected_conversation_state": None,
            "category": "emergency"
        },
        {
            "description": "心肾不交证诊断",
            "user_input": "我最近心慌心跳，失眠严重，腰酸腿软，小便频数，记忆力也不好了。",
            "expected_intent": "diagnosis",
            "expected_symptoms": ["心悸", "失眠", "腰酸", "小便频数", "健忘"],
            "expected_conversation_state": None,
            "category": "heart_kidney_disharmony"
        },
        {
            "description": "边界测试 - 空输入",
            "user_input": "",
            "expected_intent": None,
            "expected_symptoms": [],
            "expected_conversation_state": None,
            "category": "boundary_test"
        },
        {
            "description": "边界测试 - 模糊症状",
            "user_input": "我感觉不舒服",
            "expected_intent": "diagnosis",
            "expected_symptoms": [],
            "expected_conversation_state": "awaiting_follow_up",
            "category": "vague_symptoms"
        },
        {
            "description": "复杂症状组合 - 肝肾阴虚",
            "user_input": "医生，我头晕耳鸣，腰膝酸软，口干咽燥，手心脚心发热，夜间盗汗，失眠多梦，眼睛干涩，月经不调。",
            "expected_intent": "diagnosis", 
            "expected_symptoms": ["头晕", "耳鸣", "腰膝酸软", "口干咽燥", "五心烦热", "盗汗", "失眠多梦", "眼干", "月经不调"],
            "expected_conversation_state": None,
            "category": "complex_symptoms"
        },
        {
            "description": "性能测试 - 长文本输入",
            "user_input": "医生您好，我想详细描述一下我的症状，希望您能给我一个准确的诊断。" + "我最近几个月来一直感觉身体不适，" * 20 + "主要症状是头晕、乏力、失眠。",
            "expected_intent": "diagnosis",
            "expected_symptoms": ["头晕", "乏力", "失眠"],
            "expected_conversation_state": None,
            "category": "performance_test"
        },
        {
            "description": "药物咨询",
            "user_input": "请问六味地黄丸的功效和适应症是什么？",
            "expected_intent": "consultation",
            "expected_symptoms": [],
            "expected_conversation_state": None,
            "category": "medicine_consultation"
        }
    ]
    
    # 运行测试
    results = []
    category_stats = {}
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"执行测试 {i+1}/{len(test_cases)}")
        result = run_test_case(test_case, logger)
        results.append(result)
        
        # 统计分类结果
        category = test_case.get("category", "uncategorized")
        if category not in category_stats:
            category_stats[category] = {"total": 0, "success": 0}
        category_stats[category]["total"] += 1
        if result.get("success", False):
            category_stats[category]["success"] += 1
    
    # 计算总体统计
    total_tests = len(test_cases)
    success_count = sum(1 for r in results if r.get("success", False))
    success_rate = (success_count / total_tests) * 100
    
    # 计算平均响应时间
    response_times = [r.get("response_time", 0) for r in results if "response_time" in r]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # 计算平均质量评分
    quality_scores = [
        r.get("quality_evaluation", {}).get("overall_quality_score", 0) 
        for r in results 
        if r.get("quality_evaluation")
    ]
    avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    # 统计意图识别准确性
    intent_correct = sum(
        1 for r in results 
        if r.get("intent_matches_expected", True)
    )
    intent_accuracy = (intent_correct / total_tests) * 100
    
    # 输出结果摘要
    logger.info("=" * 60)
    logger.info("测试结果摘要")
    logger.info("=" * 60)
    logger.info(f"总测试数: {total_tests}")
    logger.info(f"成功测试数: {success_count}")
    logger.info(f"成功率: {success_rate:.2f}%")
    logger.info(f"平均响应时间: {avg_response_time:.2f}秒")
    logger.info(f"平均质量评分: {avg_quality_score:.1f}/100")
    logger.info(f"意图识别准确率: {intent_accuracy:.2f}%")
    
    # 分类统计
    logger.info("\n分类测试结果:")
    for category, stats in category_stats.items():
        success_rate_cat = (stats["success"] / stats["total"]) * 100
        logger.info(f"  {category}: {stats['success']}/{stats['total']} ({success_rate_cat:.1f}%)")
    
    # 保存详细结果
    results_file = f"{log_dir}/tcm_graph_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"\n详细结果已保存到: {results_file}")
    
    return results, success_rate, avg_response_time, avg_quality_score, intent_accuracy, category_stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试中医智能对话系统LangGraph')
    parser.add_argument('--filter', type=str, default=None, help='按描述或分类筛选测试用例')
    parser.add_argument('--category', type=str, default=None, help='只运行指定分类的测试用例')
    parser.add_argument('--quality-threshold', type=float, default=60.0, help='质量评分阈值')
    parser.add_argument('--success-threshold', type=float, default=80.0, help='成功率阈值')
    parser.add_argument('--response-time-limit', type=float, default=30.0, help='响应时间限制（秒）')
    args = parser.parse_args()
    
    # 设置日志
    logger, log_dir, timestamp = setup_logging()
    
    logger.info("=" * 80)
    logger.info("中医智能对话系统LangGraph测试")
    logger.info("=" * 80)
    
    # 运行测试
    results, success_rate, avg_response_time, avg_quality_score, intent_accuracy, category_stats = test_tcm_graph(
        logger, log_dir, timestamp
    )
    
    # 创建测试报告
    report = {
        "timestamp": timestamp,
        "test_config": {
            "quality_threshold": args.quality_threshold,
            "success_threshold": args.success_threshold,
            "response_time_limit": args.response_time_limit,
            "filter": args.filter,
            "category": args.category
        },
        "summary": {
            "total_tests": len(results),
            "success_count": sum(1 for r in results if r.get("success", False)),
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "avg_quality_score": avg_quality_score,
            "intent_accuracy": intent_accuracy
        },
        "category_stats": category_stats,
        "detailed_results": results
    }
    
    # 保存测试报告
    report_file = f"{log_dir}/tcm_graph_report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"测试报告已保存到: {report_file}")
    
    # 检查测试是否通过
    tests_passed = success_rate >= args.success_threshold
    quality_passed = avg_quality_score >= args.quality_threshold
    performance_passed = avg_response_time <= args.response_time_limit
    
    logger.info("\n" + "=" * 60)
    logger.info("最终测试结果")
    logger.info("=" * 60)
    
    if tests_passed and quality_passed and performance_passed:
        logger.info("✅ 所有测试指标均达标！")
        logger.info(f"✅ 成功率: {success_rate:.2f}% (>= {args.success_threshold}%)")
        logger.info(f"✅ 质量评分: {avg_quality_score:.1f} (>= {args.quality_threshold})")
        logger.info(f"✅ 响应时间: {avg_response_time:.2f}s (<= {args.response_time_limit}s)")
        sys.exit(0)
    else:
        logger.info("❌ 部分测试指标未达标：")
        
        if not tests_passed:
            logger.info(f"❌ 成功率: {success_rate:.2f}% (< {args.success_threshold}%)")
        else:
            logger.info(f"✅ 成功率: {success_rate:.2f}% (>= {args.success_threshold}%)")
            
        if not quality_passed:
            logger.info(f"❌ 质量评分: {avg_quality_score:.1f} (< {args.quality_threshold})")
        else:
            logger.info(f"✅ 质量评分: {avg_quality_score:.1f} (>= {args.quality_threshold})")
            
        if not performance_passed:
            logger.info(f"❌ 响应时间: {avg_response_time:.2f}s (> {args.response_time_limit}s)")
        else:
            logger.info(f"✅ 响应时间: {avg_response_time:.2f}s (<= {args.response_time_limit}s)")
        
        # 显示失败的测试用例
        failed_tests = [r for r in results if not r.get("success", False)]
        if failed_tests:
            logger.info(f"\n失败的测试用例 ({len(failed_tests)}):")
            for i, test in enumerate(failed_tests[:5]):  # 只显示前5个
                logger.info(f"  {i+1}. {test['test_case']['description']}")
                if test.get('error'):
                    logger.info(f"     错误: {test['error']}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()