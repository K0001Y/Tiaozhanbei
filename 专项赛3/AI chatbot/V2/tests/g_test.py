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
from nodes.g_diagnosis_node import diagnosis_node


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
    diagnosis_data: Optional[Dict[str, Any]]  # 辨证分析结果


def setup_logging():
    """设置日志记录"""
    # 创建日志目录
    log_dir = "tests/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建时间戳用于日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/diagnosis_node_test_{timestamp}.log"
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("DiagnosisNodeTest")
    logger.info(f"日志将保存到: {log_file}")
    
    return logger, log_dir, timestamp


def validate_tcm_pattern(pattern_type: str) -> Dict[str, Any]:
    """验证中医证型的合理性"""
    # 常见的中医证型关键词
    tcm_patterns = {
        "气": ["气虚", "气滞", "气陷", "气逆", "气不固"],
        "血": ["血虚", "血瘀", "血热", "血寒"],
        "阴阳": ["阴虚", "阳虚", "阴阳两虚", "阴虚火旺", "阳虚水泛"],
        "脏腑": ["肝", "心", "脾", "肺", "肾", "胆", "胃", "大肠", "小肠", "膀胱", "三焦"],
        "六淫": ["风", "寒", "暑", "湿", "燥", "火"],
        "经络": ["经络不通", "经脉阻滞"],
        "津液": ["津液不足", "痰湿", "水湿停滞"]
    }
    
    validation = {
        "is_valid_pattern": False,
        "contains_tcm_elements": False,
        "pattern_category": [],
        "confidence_reasonable": True
    }
    
    if pattern_type and pattern_type != "无法确定" and pattern_type != "未能确定证型":
        validation["is_valid_pattern"] = True
        
        # 检查是否包含中医元素
        pattern_lower = pattern_type.lower()
        for category, keywords in tcm_patterns.items():
            for keyword in keywords:
                if keyword in pattern_type:
                    validation["contains_tcm_elements"] = True
                    validation["pattern_category"].append(category)
                    break
    
    return validation


def evaluate_diagnosis_quality(diagnosis_data: Dict[str, Any], test_case: Dict) -> Dict[str, Any]:
    """评估辨证分析的质量"""
    if not diagnosis_data:
        return {"overall_score": 0, "has_data": False}
    
    evaluation = {
        "has_data": True,
        "has_pattern_type": bool(diagnosis_data.get("pattern_type")),
        "has_pathogenesis": bool(diagnosis_data.get("pathogenesis")),
        "has_analysis": bool(diagnosis_data.get("analysis")),
        "confidence_valid": 0 <= diagnosis_data.get("confidence", -1) <= 1,
        "has_differential": bool(diagnosis_data.get("differential_diagnosis")),
        "stage_correct": diagnosis_data.get("stage") == "diagnosis"
    }
    
    # 验证证型
    pattern_validation = validate_tcm_pattern(diagnosis_data.get("pattern_type", ""))
    evaluation.update(pattern_validation)
    
    # 检查分析内容质量
    analysis = diagnosis_data.get("analysis", "")
    evaluation["analysis_detailed"] = len(analysis) > 50
    evaluation["analysis_mentions_symptoms"] = any(
        symptom.get("name", "").lower() in analysis.lower()
        for symptom in test_case.get("symptoms_list", [])
        if symptom.get("name")
    )
    
    # 检查病机分析
    pathogenesis = diagnosis_data.get("pathogenesis", "")
    evaluation["pathogenesis_detailed"] = len(pathogenesis) > 20
    
    # 检查鉴别诊断
    diff_diagnosis = diagnosis_data.get("differential_diagnosis", [])
    evaluation["has_multiple_differentials"] = len(diff_diagnosis) > 1
    
    # 计算总分
    score_items = [
        "has_pattern_type", "has_pathogenesis", "has_analysis", 
        "confidence_valid", "has_differential", "stage_correct",
        "is_valid_pattern", "contains_tcm_elements", "analysis_detailed",
        "pathogenesis_detailed"
    ]
    
    score = sum(evaluation.get(item, False) for item in score_items)
    max_score = len(score_items)
    evaluation["overall_score"] = (score / max_score) * 100
    
    return evaluation


def run_test_case(test_case, logger):
    """运行单个测试用例"""
    logger.info(f"测试用例: {test_case['description']}")
    logger.info(f"症状列表: {[s.get('name') for s in test_case.get('symptoms_list', [])]}")
    
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
        "relevant_context": test_case.get("relevant_context", "无相关上下文"),
        "symptoms_list": test_case.get("symptoms_list", []),
        "missing_info_list": test_case.get("missing_info_list", []),
        "conversation_state": test_case.get("conversation_state"),
        "diagnosis_data": None
    }
    
    # 记录测试开始时间
    start_time = datetime.datetime.now()
    
    try:
        # 调用辨证分析节点
        updated_state, route = diagnosis_node(state)
        
        # 计算响应时间
        response_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # 获取结果
        diagnosis_data = updated_state.get("diagnosis_data", {})
        conversation_state = updated_state.get("conversation_state", "")
        error = updated_state.get("error")
        
        # 记录日志
        if diagnosis_data:
            logger.info(f"辨证结果:")
            logger.info(f"  证型: {diagnosis_data.get('pattern_type', 'N/A')}")
            logger.info(f"  置信度: {diagnosis_data.get('confidence', 'N/A')}")
            logger.info(f"  病机: {diagnosis_data.get('pathogenesis', 'N/A')[:100]}...")
            logger.info(f"  鉴别诊断数量: {len(diagnosis_data.get('differential_diagnosis', []))}")
        
        logger.info(f"对话状态: {conversation_state}")
        logger.info(f"路由: {route}")
        logger.info(f"响应时间: {response_time:.2f}秒")
        
        # 评估诊断质量
        quality_eval = evaluate_diagnosis_quality(diagnosis_data, test_case)
        logger.info(f"质量评分: {quality_eval['overall_score']:.1f}/100")
        
        # 检查是否符合预期路由
        expected_route = test_case.get("expected_route", "to_prescription")
        route_matches = route == expected_route
        logger.info(f"路由是否符合预期: {route_matches} (预期: {expected_route}, 实际: {route})")
        
        # 检查是否符合预期证型（如果有）
        expected_pattern = test_case.get("expected_pattern")
        pattern_matches = True
        if expected_pattern:
            actual_pattern = diagnosis_data.get("pattern_type", "")
            pattern_matches = expected_pattern.lower() in actual_pattern.lower()
            logger.info(f"证型是否符合预期: {pattern_matches} (预期包含: {expected_pattern}, 实际: {actual_pattern})")
        
        # 检查是否有错误
        has_error = error is not None
        if has_error:
            logger.warning(f"节点报告了错误: {error}")
        
        result = {
            "test_case": test_case,
            "diagnosis_data": diagnosis_data,
            "conversation_state": conversation_state,
            "route": route,
            "route_matches_expected": route_matches,
            "pattern_matches_expected": pattern_matches,
            "quality_evaluation": quality_eval,
            "has_error": has_error,
            "error": error,
            "response_time": response_time,
            "success": (not has_error and route_matches and pattern_matches and 
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


def test_diagnosis_node(logger, log_dir, timestamp):
    """测试辨证分析节点"""
    logger.info("开始辨证分析节点测试")
    
    # 测试用例
    test_cases = [
        {
            "description": "典型肝郁气滞证",
            "symptoms_list": [
                {
                    "name": "胸胁胀痛",
                    "description": "两侧胸胁部位胀满疼痛",
                    "severity": "中等",
                    "duration": "2周"
                },
                {
                    "name": "情志抑郁",
                    "description": "心情抑郁，易怒",
                    "severity": "明显",
                    "duration": "1个月"
                },
                {
                    "name": "嗳气",
                    "description": "经常嗳气",
                    "severity": "轻微",
                    "duration": "1周"
                }
            ],
            "relevant_context": "中医理论：肝主疏泄，调畅气机。肝郁气滞常见于情志不畅者。",
            "expected_route": "to_prescription",
            "expected_pattern": "肝郁气滞",
            "user_input": "我最近胸胁胀痛，心情不好"
        },
        {
            "description": "典型脾虚证",
            "symptoms_list": [
                {
                    "name": "乏力",
                    "description": "全身乏力，精神不振",
                    "severity": "严重",
                    "duration": "3个月"
                },
                {
                    "name": "食欲不振",
                    "description": "不想吃饭，食量减少",
                    "severity": "中等",
                    "duration": "2个月"
                },
                {
                    "name": "大便溏薄",
                    "description": "大便不成形",
                    "severity": "中等",
                    "duration": "1个月"
                }
            ],
            "relevant_context": "中医理论：脾主运化，脾虚则运化失司，水谷精微不能上输。",
            "expected_route": "to_prescription",
            "expected_pattern": "脾虚",
            "user_input": "我最近很累，不想吃饭，大便也不好"
        },
        {
            "description": "复杂证型 - 肝肾阴虚",
            "symptoms_list": [
                {
                    "name": "头晕目眩",
                    "description": "头晕眼花",
                    "severity": "中等",
                    "duration": "1个月"
                },
                {
                    "name": "腰膝酸软",
                    "description": "腰部和膝盖酸软无力",
                    "severity": "中等",
                    "duration": "2个月"
                },
                {
                    "name": "五心烦热",
                    "description": "手心脚心发热",
                    "severity": "轻微",
                    "duration": "3周"
                },
                {
                    "name": "失眠多梦",
                    "description": "夜间失眠，多梦",
                    "severity": "严重",
                    "duration": "1个月"
                }
            ],
            "relevant_context": "中医理论：肝肾同源，肝藏血，肾藏精。肝肾阴虚常见于久病或年老体弱者。",
            "expected_route": "to_prescription",
            "expected_pattern": "肝肾阴虚",
            "user_input": "我头晕腰酸，晚上睡不好"
        },
        {
            "description": "症状较少的情况",
            "symptoms_list": [
                {
                    "name": "咳嗽",
                    "description": "干咳",
                    "severity": "轻微",
                    "duration": "1周"
                }
            ],
            "relevant_context": "咳嗽可见于多种证型，需要更多症状才能准确辨证。",
            "expected_route": "to_prescription",
            "user_input": "我咳嗽"
        },
        {
            "description": "无症状情况",
            "symptoms_list": [],
            "relevant_context": "无症状信息，无法进行准确的中医辨证。",
            "expected_route": "to_prescription",
            "user_input": "我想看中医"
        },
        {
            "description": "包含丰富上下文的情况",
            "symptoms_list": [
                {
                    "name": "心悸",
                    "description": "心慌心跳",
                    "severity": "中等",
                    "duration": "2周"
                },
                {
                    "name": "失眠",
                    "description": "难以入睡",
                    "severity": "严重",
                    "duration": "1个月"
                }
            ],
            "relevant_context": """
            中医理论：心主神明，心血不足则神不守舍，出现心悸失眠。
            相关病机：心血虚、心阴虚、心肾不交等都可引起类似症状。
            鉴别要点：需要结合其他症状如面色、舌象、脉象等进行综合判断。
            """,
            "expected_route": "to_prescription",
            "expected_pattern": "心",
            "user_input": "我心慌失眠"
        },
        {
            "description": "边界测试 - 症状描述模糊",
            "symptoms_list": [
                {
                    "name": "不舒服",
                    "description": "身体不舒服",
                    "severity": "未提及",
                    "duration": "未提及"
                }
            ],
            "relevant_context": "症状描述过于模糊，需要更具体的信息进行辨证。",
            "expected_route": "to_prescription",
            "user_input": "我感觉不舒服"
        },
        {
            "description": "性能测试 - 大量症状",
            "symptoms_list": [
                {"name": f"症状{i}", "description": f"症状{i}的描述", "severity": "轻微", "duration": "1天"}
                for i in range(1, 16)  # 15个症状
            ],
            "relevant_context": "大量症状信息，测试系统处理复杂情况的能力。",
            "expected_route": "to_prescription",
            "user_input": "我有很多症状"
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
    
    # 统计证型准确性
    pattern_matches = sum(1 for r in results if r.get("pattern_matches_expected", True))
    pattern_accuracy = (pattern_matches / len(test_cases)) * 100
    logger.info(f"证型匹配准确率: {pattern_matches}/{len(test_cases)} ({pattern_accuracy:.2f}%)")
    
    # 保存详细结果到JSON
    results_file = f"{log_dir}/diagnosis_node_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"详细结果已保存到 {results_file}")
    
    return results, success_rate, avg_response_time, avg_quality_score, pattern_accuracy


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试辨证分析节点')
    parser.add_argument('--filter', type=str, default=None, help='按描述筛选测试用例')
    parser.add_argument('--quality-threshold', type=float, default=60.0, help='质量评分阈值')
    parser.add_argument('--pattern-threshold', type=float, default=70.0, help='证型匹配准确率阈值')
    args = parser.parse_args()
    
    # 设置日志
    logger, log_dir, timestamp = setup_logging()
    
    logger.info("=" * 60)
    logger.info("辨证分析节点测试")
    logger.info("=" * 60)
    
    # 运行测试
    results, success_rate, avg_response_time, avg_quality_score, pattern_accuracy = test_diagnosis_node(logger, log_dir, timestamp)
    
    # 创建测试报告
    report = {
        "timestamp": timestamp,
        "total_tests": len(results),
        "success_count": sum(1 for r in results if r.get("success", False)),
        "success_rate": success_rate,
        "avg_response_time": avg_response_time,
        "avg_quality_score": avg_quality_score,
        "pattern_accuracy": pattern_accuracy,
        "quality_threshold": args.quality_threshold,
        "pattern_threshold": args.pattern_threshold,
        "results": results
    }
    
    # 保存报告
    report_file = f"{log_dir}/diagnosis_node_report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"测试报告已保存到 {report_file}")
    
    # 检查测试结果
    all_passed = all(r.get("success", False) for r in results)
    quality_threshold_met = avg_quality_score >= args.quality_threshold
    pattern_threshold_met = pattern_accuracy >= args.pattern_threshold
    
    if all_passed and quality_threshold_met and pattern_threshold_met:
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
                if not test.get('pattern_matches_expected', True):
                    issues.append(f"     证型不匹配")
        
        if not quality_threshold_met:
            issues.append(f"平均质量评分 ({avg_quality_score:.1f}) 未达到阈值 ({args.quality_threshold})")
        
        if not pattern_threshold_met:
            issues.append(f"证型匹配准确率 ({pattern_accuracy:.1f}%) 未达到阈值 ({args.pattern_threshold}%)")
        
        for issue in issues:
            logger.info(issue + " ❌")
        
        sys.exit(1)


if __name__ == "__main__":
    main()