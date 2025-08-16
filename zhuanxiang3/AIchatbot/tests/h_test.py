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
from nodes.h_prescription_node import prescription_node


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
    prescription_data: Optional[Dict[str, Any]]  # 处方推荐数据


def setup_logging():
    """设置日志记录"""
    # 创建日志目录
    log_dir = "tests/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建时间戳用于日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/prescription_node_test_{timestamp}.log"
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("PrescriptionNodeTest")
    logger.info(f"日志将保存到: {log_file}")
    
    return logger, log_dir, timestamp


def validate_tcm_formula(formula_name: str) -> Dict[str, Any]:
    """验证中医方剂的合理性"""
    # 常见中医方剂名称关键词
    formula_keywords = [
        "汤", "散", "丸", "膏", "饮", "方", "丹", "露", "酒", "油",
        "逍遥", "四君子", "四物", "六君子", "补中益气", "玉屏风", 
        "甘麦大枣", "小柴胡", "大柴胡", "温胆", "理中", "当归", 
        "川芎", "白芍", "熟地", "人参", "黄芪", "茯苓", "甘草"
    ]
    
    validation = {
        "is_valid_formula": False,
        "contains_formula_keywords": False,
        "formula_type": []
    }
    
    if formula_name and formula_name != "无法确定" and formula_name.strip():
        # 检查是否包含方剂关键词
        for keyword in formula_keywords:
            if keyword in formula_name:
                validation["contains_formula_keywords"] = True
                validation["is_valid_formula"] = True
                break
        
        # 判断方剂类型
        if "汤" in formula_name:
            validation["formula_type"].append("汤剂")
        if any(word in formula_name for word in ["散", "丸", "膏"]):
            validation["formula_type"].append("其他剂型")
        if any(word in formula_name for word in ["补", "益"]):
            validation["formula_type"].append("补益类")
        if any(word in formula_name for word in ["理", "调", "和"]):
            validation["formula_type"].append("调理类")
    
    return validation


def validate_tcm_herbs(composition: List[str]) -> Dict[str, Any]:
    """验证中药组成的合理性"""
    # 常见中药材
    common_herbs = [
        "人参", "黄芪", "当归", "川芎", "白芍", "熟地", "生地", "茯苓", "白术", "甘草",
        "柴胡", "黄芩", "半夏", "陈皮", "枳壳", "枳实", "厚朴", "苍术", "泽泻", "猪苓",
        "桂枝", "白芍", "生姜", "大枣", "麻黄", "杏仁", "石膏", "知母", "栀子", "连翘",
        "金银花", "黄连", "黄柏", "龙胆草", "板蓝根", "蒲公英", "野菊花", "薄荷", "桔梗"
    ]
    
    validation = {
        "has_herbs": len(composition) > 0,
        "valid_herb_count": 0,
        "has_monarch_herbs": False,
        "has_dosage": False,
        "herb_categories": set()
    }
    
    if composition:
        for herb_entry in composition:
            herb_str = str(herb_entry)
            
            # 检查是否包含已知中药材
            for herb in common_herbs:
                if herb in herb_str:
                    validation["valid_herb_count"] += 1
                    break
            
            # 检查是否包含用量信息
            if re.search(r'\d+[克钱两]|\d+g', herb_str):
                validation["has_dosage"] = True
            
            # 分类常见药材
            if any(herb in herb_str for herb in ["人参", "黄芪", "当归", "熟地"]):
                validation["herb_categories"].add("补益类")
            if any(herb in herb_str for herb in ["柴胡", "黄芩", "半夏"]):
                validation["herb_categories"].add("疏肝理气类")
            if any(herb in herb_str for herb in ["茯苓", "白术", "陈皮"]):
                validation["herb_categories"].add("健脾化湿类")
        
        # 如果有效药材数量合理，认为有君药
        validation["has_monarch_herbs"] = validation["valid_herb_count"] >= 3
    
    validation["herb_categories"] = list(validation["herb_categories"])
    return validation


def evaluate_prescription_quality(prescription_data: Dict[str, Any], diagnosis_data: Dict[str, Any], test_case: Dict) -> Dict[str, Any]:
    """评估处方推荐的质量"""
    if not prescription_data:
        return {"overall_score": 0, "has_data": False}
    
    evaluation = {
        "has_data": True,
        "has_formula_name": bool(prescription_data.get("formula_name")),
        "has_composition": bool(prescription_data.get("composition")),
        "has_preparation": bool(prescription_data.get("preparation_method")),
        "has_usage": bool(prescription_data.get("usage")),
        "has_treatment_principle": bool(prescription_data.get("treatment_principle")),
        "has_contraindications": bool(prescription_data.get("contraindications")),
        "has_modifications": bool(prescription_data.get("modifications")),
        "has_evidence": bool(prescription_data.get("evidence")),
        "stage_correct": prescription_data.get("stage") == "prescription"
    }
    
    # 验证方剂名称
    formula_validation = validate_tcm_formula(prescription_data.get("formula_name", ""))
    evaluation.update(formula_validation)
    
    # 验证药物组成
    composition = prescription_data.get("composition", [])
    if isinstance(composition, str):
        composition = [composition]
    herb_validation = validate_tcm_herbs(composition)
    evaluation.update(herb_validation)
    
    # 检查与诊断的一致性
    pattern_type = diagnosis_data.get("pattern_type", "") if diagnosis_data else ""
    formula_name = prescription_data.get("formula_name", "")
    treatment_principle = prescription_data.get("treatment_principle", "")
    
    evaluation["matches_diagnosis"] = False
    if pattern_type and formula_name:
        # 简单的一致性检查
        if ("气滞" in pattern_type and any(word in formula_name for word in ["逍遥", "柴胡"])) or \
           ("脾虚" in pattern_type and any(word in formula_name for word in ["君子", "理中", "补中"])) or \
           ("阴虚" in pattern_type and any(word in formula_name for word in ["地黄", "当归", "滋阴"])):
            evaluation["matches_diagnosis"] = True
    
    # 检查用法用量的详细程度
    usage = prescription_data.get("usage", "")
    evaluation["usage_detailed"] = len(usage) > 20 and ("日" in usage or "次" in usage)
    
    # 检查禁忌事项的完整性
    contraindications = prescription_data.get("contraindications", [])
    if isinstance(contraindications, str):
        contraindications = [contraindications]
    evaluation["contraindications_comprehensive"] = len(contraindications) >= 2
    
    # 计算总分
    score_items = [
        "has_formula_name", "has_composition", "has_usage", "has_treatment_principle",
        "has_contraindications", "stage_correct", "is_valid_formula", "has_monarch_herbs",
        "has_dosage", "matches_diagnosis", "usage_detailed", "contraindications_comprehensive"
    ]
    
    score = sum(evaluation.get(item, False) for item in score_items)
    max_score = len(score_items)
    evaluation["overall_score"] = (score / max_score) * 100
    
    return evaluation


def evaluate_final_response_quality(response: str, test_case: Dict) -> Dict[str, Any]:
    """评估最终响应的质量"""
    evaluation = {
        "has_response": bool(response and response.strip()),
        "adequate_length": len(response) > 100,
        "has_disclaimer": "免责" in response or "仅供参考" in response or "专业医师" in response,
        "structured_content": False,
        "mentions_symptoms": False,
        "mentions_diagnosis": False,
        "mentions_prescription": False,
        "professional_tone": False
    }
    
    if response:
        # 检查结构化内容
        if any(marker in response for marker in ["1.", "2.", "3.", "】", "【", "一、", "二、"]):
            evaluation["structured_content"] = True
        
        # 检查是否提及症状
        symptoms = test_case.get("symptoms_list", [])
        for symptom in symptoms:
            if symptom.get("name", "") in response:
                evaluation["mentions_symptoms"] = True
                break
        
        # 检查是否提及诊断
        diagnosis_data = test_case.get("diagnosis_data", {})
        pattern_type = diagnosis_data.get("pattern_type", "")
        if pattern_type and pattern_type in response:
            evaluation["mentions_diagnosis"] = True
        
        # 检查是否提及处方
        if any(word in response for word in ["方剂", "汤", "用法", "服用", "药材"]):
            evaluation["mentions_prescription"] = True
        
        # 检查专业性
        if any(word in response for word in ["中医", "辨证", "病机", "治疗原则"]):
            evaluation["professional_tone"] = True
    
    # 计算总分
    score = sum(evaluation.values())
    max_score = len(evaluation)
    evaluation["overall_score"] = (score / max_score) * 100
    
    return evaluation


def run_test_case(test_case, logger):
    """运行单个测试用例"""
    logger.info(f"测试用例: {test_case['description']}")
    
    diagnosis_data = test_case.get("diagnosis_data", {})
    logger.info(f"输入证型: {diagnosis_data.get('pattern_type', 'N/A')}")
    
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
        "diagnosis_data": diagnosis_data,
        "prescription_data": None
    }
    
    # 记录测试开始时间
    start_time = datetime.datetime.now()
    
    try:
        # 调用处方推荐节点
        updated_state, route = prescription_node(state)
        
        # 计算响应时间
        response_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # 获取结果
        prescription_data = updated_state.get("prescription_data", {})
        final_response = updated_state.get("response", "")
        conversation_state = updated_state.get("conversation_state", "")
        error = updated_state.get("error")
        
        # 记录日志
        if prescription_data:
            logger.info(f"生成的处方:")
            logger.info(f"  方剂名称: {prescription_data.get('formula_name', 'N/A')}")
            logger.info(f"  药材数量: {len(prescription_data.get('composition', []))}")
            logger.info(f"  治疗原则: {prescription_data.get('treatment_principle', 'N/A')[:50]}...")
        
        logger.info(f"最终响应长度: {len(final_response)} 字符")
        logger.info(f"对话状态: {conversation_state}")
        logger.info(f"路由: {route}")
        logger.info(f"响应时间: {response_time:.2f}秒")
        
        # 评估处方质量
        prescription_quality = evaluate_prescription_quality(prescription_data, diagnosis_data, test_case)
        logger.info(f"处方质量评分: {prescription_quality['overall_score']:.1f}/100")
        
        # 评估响应质量
        response_quality = evaluate_final_response_quality(final_response, test_case)
        logger.info(f"响应质量评分: {response_quality['overall_score']:.1f}/100")
        
        # 检查是否符合预期路由
        expected_route = test_case.get("expected_route", "to_safety_check")
        route_matches = route == expected_route
        logger.info(f"路由是否符合预期: {route_matches} (预期: {expected_route}, 实际: {route})")
        
        # 检查是否符合预期方剂类型（如果有）
        expected_formula_type = test_case.get("expected_formula_type")
        formula_type_matches = True
        if expected_formula_type and prescription_data:
            formula_name = prescription_data.get("formula_name", "")
            formula_type_matches = expected_formula_type.lower() in formula_name.lower()
            logger.info(f"方剂类型是否符合预期: {formula_type_matches}")
        
        # 检查是否有错误
        has_error = error is not None
        if has_error:
            logger.warning(f"节点报告了错误: {error}")
        
        # 综合评分
        overall_score = (prescription_quality["overall_score"] + response_quality["overall_score"]) / 2
        
        result = {
            "test_case": test_case,
            "prescription_data": prescription_data,
            "final_response": final_response,
            "conversation_state": conversation_state,
            "route": route,
            "route_matches_expected": route_matches,
            "formula_type_matches_expected": formula_type_matches,
            "prescription_quality": prescription_quality,
            "response_quality": response_quality,
            "overall_score": overall_score,
            "has_error": has_error,
            "error": error,
            "response_time": response_time,
            "success": (not has_error and route_matches and formula_type_matches and 
                       overall_score >= 60)
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


def test_prescription_node(logger, log_dir, timestamp):
    """测试处方推荐节点"""
    logger.info("开始处方推荐节点测试")
    
    # 测试用例
    test_cases = [
        {
            "description": "典型肝郁气滞证 - 逍遥散类方",
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
                }
            ],
            "diagnosis_data": {
                "pattern_type": "肝郁气滞",
                "pathogenesis": "肝失疏泄，气机不畅",
                "analysis": "患者胸胁胀痛，情志抑郁，为典型的肝郁气滞证候",
                "confidence": 0.9,
                "stage": "diagnosis"
            },
            "relevant_context": "肝郁气滞常用逍遥散加减治疗",
            "expected_route": "to_safety_check",
            "expected_formula_type": "逍遥",
            "user_input": "我胸胁胀痛，心情不好"
        },
        {
            "description": "脾虚证 - 四君子汤类方",
            "symptoms_list": [
                {
                    "name": "乏力",
                    "description": "全身乏力，精神不振",
                    "severity": "严重",
                    "duration": "3个月"
                },
                {
                    "name": "食欲不振",
                    "description": "不想吃饭",
                    "severity": "中等",
                    "duration": "2个月"
                }
            ],
            "diagnosis_data": {
                "pattern_type": "脾气虚",
                "pathogenesis": "脾失健运，气血化源不足",
                "analysis": "患者乏力食少，为脾气虚证",
                "confidence": 0.85,
                "stage": "diagnosis"
            },
            "relevant_context": "脾气虚常用四君子汤或补中益气汤治疗",
            "expected_route": "to_safety_check",
            "expected_formula_type": "君子",
            "user_input": "我很累，不想吃饭"
        },
        {
            "description": "肾阴虚证 - 六味地黄丸类方",
            "symptoms_list": [
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
                }
            ],
            "diagnosis_data": {
                "pattern_type": "肾阴虚",
                "pathogenesis": "肾阴亏虚，虚火内扰",
                "analysis": "患者腰膝酸软，五心烦热，为肾阴虚证",
                "confidence": 0.8,
                "stage": "diagnosis"
            },
            "relevant_context": "肾阴虚常用六味地黄丸滋阴补肾",
            "expected_route": "to_safety_check",
            "expected_formula_type": "地黄",
            "user_input": "我腰酸手心发热"
        },
        {
            "description": "证型不明确的情况",
            "symptoms_list": [
                {
                    "name": "不舒服",
                    "description": "身体不舒服",
                    "severity": "未提及",
                    "duration": "未提及"
                }
            ],
            "diagnosis_data": {
                "pattern_type": "无法确定",
                "pathogenesis": "症状信息不足",
                "analysis": "无法进行准确辨证",
                "confidence": 0.1,
                "stage": "diagnosis"
            },
            "relevant_context": "症状不明确，无法辨证",
            "expected_route": "to_safety_check",
            "user_input": "我感觉不舒服"
        },
        {
            "description": "复杂证型 - 气血两虚",
            "symptoms_list": [
                {
                    "name": "乏力",
                    "description": "全身乏力",
                    "severity": "严重",
                    "duration": "3个月"
                },
                {
                    "name": "面色苍白",
                    "description": "面色无血色",
                    "severity": "明显",
                    "duration": "2个月"
                },
                {
                    "name": "心悸",
                    "description": "心慌心跳",
                    "severity": "中等",
                    "duration": "1个月"
                }
            ],
            "diagnosis_data": {
                "pattern_type": "气血两虚",
                "pathogenesis": "气虚血少，脏腑失养",
                "analysis": "患者乏力面白心悸，为气血两虚证",
                "confidence": 0.88,
                "stage": "diagnosis"
            },
            "relevant_context": "气血两虚常用八珍汤或十全大补汤",
            "expected_route": "to_safety_check",
            "expected_formula_type": "珍",
            "user_input": "我很累，脸色也不好，还心慌"
        },
        {
            "description": "边界测试 - 空的症状列表",
            "symptoms_list": [],
            "diagnosis_data": {
                "pattern_type": "肝郁脾虚",
                "pathogenesis": "肝郁克脾，脾失健运",
                "analysis": "基于其他信息的辨证结果",
                "confidence": 0.7,
                "stage": "diagnosis"
            },
            "relevant_context": "肝郁脾虚常用逍遥散",
            "expected_route": "to_safety_check",
            "user_input": "请给我开方"
        },
        {
            "description": "性能测试 - 复杂的丰富上下文",
            "symptoms_list": [
                {
                    "name": "失眠",
                    "description": "夜间失眠多梦",
                    "severity": "严重",
                    "duration": "1个月"
                }
            ],
            "diagnosis_data": {
                "pattern_type": "心肾不交",
                "pathogenesis": "心火亢盛，肾水不足，水火不济",
                "analysis": "详细的辨证分析过程...",
                "confidence": 0.85,
                "stage": "diagnosis"
            },
            "relevant_context": """
            心肾不交证是中医内科常见证型之一。心属火，肾属水，正常情况下心火下降温肾水，
            肾水上济制心火，形成水火既济的生理状态。当各种原因导致心火不能下降，
            肾水不能上济时，就会出现心肾不交证。常见症状包括失眠多梦、心烦、腰酸等。
            治疗多用交泰丸、黄连阿胶汤等方剂。
            """,
            "expected_route": "to_safety_check",
            "user_input": "我失眠很严重"
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
    overall_scores = [
        r.get("overall_score", 0) 
        for r in results 
        if "overall_score" in r
    ]
    avg_overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    logger.info(f"平均综合质量评分: {avg_overall_score:.1f}/100")
    
    # 统计方剂类型匹配准确性
    formula_matches = sum(1 for r in results if r.get("formula_type_matches_expected", True))
    formula_accuracy = (formula_matches / len(test_cases)) * 100
    logger.info(f"方剂类型匹配准确率: {formula_matches}/{len(test_cases)} ({formula_accuracy:.2f}%)")
    
    # 保存详细结果到JSON
    results_file = f"{log_dir}/prescription_node_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"详细结果已保存到 {results_file}")
    
    return results, success_rate, avg_response_time, avg_overall_score, formula_accuracy


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试处方推荐节点')
    parser.add_argument('--filter', type=str, default=None, help='按描述筛选测试用例')
    parser.add_argument('--quality-threshold', type=float, default=60.0, help='综合质量评分阈值')
    parser.add_argument('--formula-threshold', type=float, default=70.0, help='方剂类型匹配准确率阈值')
    args = parser.parse_args()
    
    # 设置日志
    logger, log_dir, timestamp = setup_logging()
    
    logger.info("=" * 60)
    logger.info("处方推荐节点测试")
    logger.info("=" * 60)
    
    # 运行测试
    results, success_rate, avg_response_time, avg_overall_score, formula_accuracy = test_prescription_node(logger, log_dir, timestamp)
    
    # 创建测试报告
    report = {
        "timestamp": timestamp,
        "total_tests": len(results),
        "success_count": sum(1 for r in results if r.get("success", False)),
        "success_rate": success_rate,
        "avg_response_time": avg_response_time,
        "avg_overall_score": avg_overall_score,
        "formula_accuracy": formula_accuracy,
        "quality_threshold": args.quality_threshold,
        "formula_threshold": args.formula_threshold,
        "results": results
    }
    
    # 保存报告
    report_file = f"{log_dir}/prescription_node_report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"测试报告已保存到 {report_file}")
    
    # 检查测试结果
    all_passed = all(r.get("success", False) for r in results)
    quality_threshold_met = avg_overall_score >= args.quality_threshold
    formula_threshold_met = formula_accuracy >= args.formula_threshold
    
    if all_passed and quality_threshold_met and formula_threshold_met:
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
                if not test.get('formula_type_matches_expected', True):
                    issues.append(f"     方剂类型不匹配")
        
        if not quality_threshold_met:
            issues.append(f"平均综合质量评分 ({avg_overall_score:.1f}) 未达到阈值 ({args.quality_threshold})")
        
        if not formula_threshold_met:
            issues.append(f"方剂类型匹配准确率 ({formula_accuracy:.1f}%) 未达到阈值 ({args.formula_threshold}%)")
        
        for issue in issues:
            logger.info(issue + " ❌")
        
        sys.exit(1)


if __name__ == "__main__":
    main()