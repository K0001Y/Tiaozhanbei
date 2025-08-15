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
from nodes.e_symptom_extraction_node import symptom_extraction_node

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

def setup_logging():
    """设置日志记录"""
    # 创建日志目录
    log_dir = "tests/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建时间戳用于日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/symptom_extraction_test_{timestamp}.log"
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("SymptomExtractionTest")
    logger.info(f"日志将保存到: {log_file}")
    
    return logger, log_dir, timestamp

def create_mock_message(role: str, content: str):
    """创建模拟消息对象"""
    class MockMessage:
        def __init__(self, type_val: str, content_val: str):
            self.type = type_val
            self.content = content_val
    
    return MockMessage(role, content)

def run_test_case(test_case, logger):
    """运行单个测试用例"""
    logger.info(f"测试用例: {test_case['description']}")
    logger.info(f"输入: {test_case['user_input']}")
    
    # 创建消息历史（如果有）
    messages = []
    if test_case.get("message_history"):
        try:
            for msg in test_case["message_history"]:
                messages.append(create_mock_message(msg["role"], msg["content"]))
            logger.info(f"创建了 {len(messages)} 条历史消息")
        except Exception as e:
            logger.error(f"创建消息历史时出错: {str(e)}", exc_info=True)
    
    # 创建状态
    state: State = {
        "user_input": test_case["user_input"],
        "messages": messages,
        "query": None,
        "memory": None,
        "documents": None,
        "response": None,
        "error": None,
        "config": {},
        "safety_check": None,
        "intent": None,
        "intent_details": None,
        "relevant_context": test_case.get("relevant_context", "相关医疗信息参考"),
        "symptoms_list": None,
        "missing_info_list": None
    }
    
    logger.info(f"创建的状态信息:")
    logger.info(f"  - user_input: {state['user_input']}")
    logger.info(f"  - messages数量: {len(state['messages'])}")
    logger.info(f"  - relevant_context: {state['relevant_context']}")
    
    # 记录测试开始时间
    start_time = datetime.datetime.now()
    
    try:
        logger.info("开始调用症状提取节点...")
        
        # 检查节点是否可调用
        if not hasattr(symptom_extraction_node, '__call__'):
            raise AttributeError("symptom_extraction_node 不是可调用对象")
        
        # 调用症状提取节点
        logger.info("正在执行节点调用...")
        updated_state, route = symptom_extraction_node(state)
        
        logger.info("节点调用完成")
        
        # 计算响应时间
        response_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # 详细检查返回值
        logger.info(f"返回值类型检查:")
        logger.info(f"  - updated_state类型: {type(updated_state)}")
        logger.info(f"  - route类型: {type(route)}")
        logger.info(f"  - route值: {route}")
        
        if not isinstance(updated_state, dict):
            raise TypeError(f"updated_state应该是字典类型，但得到: {type(updated_state)}")
        
        if not isinstance(route, str):
            raise TypeError(f"route应该是字符串类型，但得到: {type(route)}")
        
        # 获取结果
        symptoms_list = updated_state.get("symptoms_list", [])
        missing_info_list = updated_state.get("missing_info_list", [])
        error = updated_state.get("error")
        
        logger.info(f"状态更新检查:")
        logger.info(f"  - symptoms_list类型: {type(symptoms_list)}, 长度: {len(symptoms_list) if symptoms_list else 0}")
        logger.info(f"  - missing_info_list类型: {type(missing_info_list)}, 长度: {len(missing_info_list) if missing_info_list else 0}")
        logger.info(f"  - error: {error}")
        
        # 记录日志
        logger.info(f"提取到的症状数量: {len(symptoms_list) if symptoms_list else 0}")
        if symptoms_list:
            logger.info(f"症状详情: {json.dumps(symptoms_list, ensure_ascii=False, indent=2)}")
        else:
            logger.warning("未提取到任何症状")
            
        logger.info(f"缺失信息数量: {len(missing_info_list) if missing_info_list else 0}")
        if missing_info_list:
            logger.info(f"缺失信息: {json.dumps(missing_info_list, ensure_ascii=False, indent=2)}")
        else:
            logger.info("无缺失信息")
            
        logger.info(f"路由: {route}")
        logger.info(f"响应时间: {response_time:.2f}秒")
        
        if error:
            logger.warning(f"处理过程中的错误: {error}")
        
        # 检查是否符合预期
        expected_route = test_case["expected_route"]
        route_matches = route == expected_route
        
        # 检查症状数量是否符合预期范围
        expected_symptoms_min = test_case.get("expected_symptoms_min", 0)
        expected_symptoms_max = test_case.get("expected_symptoms_max", 999)
        actual_symptoms_count = len(symptoms_list) if symptoms_list else 0
        symptoms_count_valid = expected_symptoms_min <= actual_symptoms_count <= expected_symptoms_max
        
        # 检查是否识别了预期的症状关键词
        expected_symptoms = test_case.get("expected_symptoms", [])
        detected_expected_symptoms = []
        if expected_symptoms and symptoms_list:
            for symptom in symptoms_list:
                try:
                    if isinstance(symptom, dict):
                        # 尝试多个可能的键名
                        symptom_name = (symptom.get("症状名称") or 
                                      symptom.get("name") or 
                                      symptom.get("symptom_name") or 
                                      symptom.get("symptom") or 
                                      str(symptom))
                    else:
                        symptom_name = str(symptom)
                    
                    logger.debug(f"检查症状: {symptom_name}")
                    
                    for expected in expected_symptoms:
                        if expected.lower() in symptom_name.lower():
                            detected_expected_symptoms.append(expected)
                            logger.debug(f"匹配到预期症状: {expected}")
                            
                except Exception as symptom_check_error:
                    logger.warning(f"检查症状时出错: {symptom_check_error}, 症状对象: {symptom}")
        
        symptoms_match = len(detected_expected_symptoms) > 0 if expected_symptoms else True
        
        # 综合评估
        overall_success = route_matches and symptoms_count_valid and symptoms_match
        
        logger.info(f"测试结果评估:")
        logger.info(f"  - 路由是否符合预期: {route_matches} (预期: {expected_route}, 实际: {route})")
        logger.info(f"  - 症状数量是否合理: {symptoms_count_valid} (范围: {expected_symptoms_min}-{expected_symptoms_max}, 实际: {actual_symptoms_count})")
        logger.info(f"  - 是否识别预期症状: {symptoms_match} (识别到: {detected_expected_symptoms})")
        logger.info(f"  - 整体测试是否通过: {overall_success}")
        
        result = {
            "test_case": test_case,
            "symptoms_list": symptoms_list,
            "symptoms_count": actual_symptoms_count,
            "missing_info_list": missing_info_list,
            "missing_info_count": len(missing_info_list) if missing_info_list else 0,
            "detected_route": route,
            "route_matches": route_matches,
            "symptoms_count_valid": symptoms_count_valid,
            "symptoms_match": symptoms_match,
            "detected_expected_symptoms": detected_expected_symptoms,
            "overall_success": overall_success,
            "response_time": response_time,
            "error": error
        }
        
    except ImportError as e:
        error_msg = f"导入错误: {str(e)}"
        logger.error(error_msg)
        logger.error("请检查导入路径是否正确: from nodes.e_symptom_extraction_node import symptom_extraction_node")
        logger.error("确保文件路径存在且模块可以正常导入")
        result = {
            "test_case": test_case,
            "error": error_msg,
            "error_type": "ImportError",
            "overall_success": False,
            "response_time": (datetime.datetime.now() - start_time).total_seconds()
        }
        
    except AttributeError as e:
        error_msg = f"属性错误: {str(e)}"
        logger.error(error_msg)
        logger.error("可能的原因:")
        logger.error("1. symptom_extraction_node 对象不存在")
        logger.error("2. 对象没有 __call__ 方法")
        logger.error("3. 导入的对象不是预期的节点实例")
        result = {
            "test_case": test_case,
            "error": error_msg,
            "error_type": "AttributeError", 
            "overall_success": False,
            "response_time": (datetime.datetime.now() - start_time).total_seconds()
        }
        
    except TypeError as e:
        error_msg = f"类型错误: {str(e)}"
        logger.error(error_msg)
        logger.error("可能的原因:")
        logger.error("1. 节点函数参数类型不匹配")
        logger.error("2. 返回值类型不符合预期")
        logger.error("3. State对象结构不正确")
        result = {
            "test_case": test_case,
            "error": error_msg,
            "error_type": "TypeError",
            "overall_success": False,
            "response_time": (datetime.datetime.now() - start_time).total_seconds()
        }
        
    except KeyError as e:
        error_msg = f"键错误: {str(e)}"
        logger.error(error_msg)
        logger.error("可能的原因:")
        logger.error("1. State对象缺少必需的键")
        logger.error("2. 节点返回的状态对象结构不正确")
        result = {
            "test_case": test_case,
            "error": error_msg,
            "error_type": "KeyError",
            "overall_success": False,
            "response_time": (datetime.datetime.now() - start_time).total_seconds()
        }
        
    except Exception as e:
        error_msg = f"未知错误: {str(e)}"
        logger.error(error_msg, exc_info=True)
        logger.error("完整的错误信息:")
        logger.error(f"错误类型: {type(e).__name__}")
        logger.error(f"错误模块: {e.__class__.__module__}")
        if hasattr(e, 'args') and e.args:
            logger.error(f"错误参数: {e.args}")
        
        result = {
            "test_case": test_case,
            "error": error_msg,
            "error_type": type(e).__name__,
            "overall_success": False,
            "response_time": (datetime.datetime.now() - start_time).total_seconds()
        }
    
    logger.info("-" * 50)
    return result

def test_symptom_extraction(logger, log_dir, timestamp):
    """测试症状提取节点"""
    logger.info("开始症状提取测试")
    
    # 首先进行导入测试
    logger.info("检查节点导入状态...")
    try:
        logger.info(f"symptom_extraction_node类型: {type(symptom_extraction_node)}")
        logger.info(f"symptom_extraction_node是否可调用: {callable(symptom_extraction_node)}")
        
        if hasattr(symptom_extraction_node, '__class__'):
            logger.info(f"节点类名: {symptom_extraction_node.__class__.__name__}")
            logger.info(f"节点模块: {symptom_extraction_node.__class__.__module__}")
        
        if hasattr(symptom_extraction_node, '__call__'):
            logger.info("节点具有 __call__ 方法")
        else:
            logger.warning("节点缺少 __call__ 方法")
            
        # 检查节点的必要属性
        required_attrs = ['llm', 'output_parser', 'symptom_prompt', 'chain']
        for attr in required_attrs:
            if hasattr(symptom_extraction_node, attr):
                logger.info(f"节点具有属性: {attr}")
            else:
                logger.warning(f"节点缺少属性: {attr}")
                
    except Exception as import_check_error:
        logger.error(f"检查导入时出错: {str(import_check_error)}", exc_info=True)
        return [], 0, 0
    
    # 测试用例
    test_cases = [
        {
            "user_input": "我最近头疼得厉害，特别是太阳穴的位置，疼了大概一周了，而且还有点恶心。",
            "expected_route": "diagnosis",
            "expected_symptoms_min": 2,
            "expected_symptoms_max": 5,
            "expected_symptoms": ["头疼", "恶心"],
            "description": "明确症状描述 - 头疼和恶心，信息相对充足",
            "relevant_context": "头痛常见原因包括紧张性头痛、偏头痛等，常伴有恶心症状"
        },
        {
            "user_input": "我感冒了",
            "expected_route": "follow_up",
            "expected_symptoms_min": 0,
            "expected_symptoms_max": 2,
            "expected_symptoms": ["感冒"],
            "description": "模糊症状描述 - 需要更多具体信息",
            "relevant_context": "感冒症状通常包括流鼻涕、咳嗽、发热等"
        },
        {
            "user_input": "我昨天开始发烧，体温38.5度，还咳嗽，有黄痰，喉咙也很痛，浑身无力。",
            "expected_route": "diagnosis",
            "expected_symptoms_min": 3,
            "expected_symptoms_max": 8,
            "expected_symptoms": ["发烧", "咳嗽", "喉咙痛"],
            "description": "详细症状描述 - 发烧、咳嗽、喉咙痛等多个症状",
            "relevant_context": "发热伴咳嗽和咽痛常见于上呼吸道感染"
        },
        {
            "user_input": "我胃不舒服",
            "expected_route": "follow_up",
            "expected_symptoms_min": 0,
            "expected_symptoms_max": 2,
            "expected_symptoms": ["胃不舒服"],
            "description": "模糊胃部症状 - 需要询问具体表现",
            "relevant_context": "胃部不适可能包括胃痛、胃胀、恶心等症状"
        },
        {
            "user_input": "我右下腹疼痛，从昨晚开始的，疼痛程度大概7分（满分10分），按压的时候特别痛，还有点发热。",
            "expected_route": "diagnosis",
            "expected_symptoms_min": 2,
            "expected_symptoms_max": 5,
            "expected_symptoms": ["腹痛", "发热"],
            "description": "急性腹痛症状 - 位置、程度、时间都很明确",
            "relevant_context": "右下腹疼痛伴发热需要考虑急性阑尾炎等急性腹症"
        },
        {
            "user_input": "最近睡眠不好，经常失眠，白天没精神，情绪也不太好，食欲下降。",
            "expected_route": "diagnosis",
            "expected_symptoms_min": 3,
            "expected_symptoms_max": 8,
            "expected_symptoms": ["失眠", "食欲下降"],
            "description": "多系统症状 - 睡眠、情绪、食欲问题",
            "relevant_context": "失眠伴情绪和食欲改变可能与抑郁或焦虑相关"
        },
        {
            "user_input": "我腿疼",
            "expected_route": "follow_up",
            "expected_symptoms_min": 0,
            "expected_symptoms_max": 2,
            "expected_symptoms": ["腿疼"],
            "description": "极简症状描述 - 需要大量补充信息",
            "relevant_context": "腿部疼痛需要了解具体位置、性质、诱因等"
        },
        {
            "user_input": "我有糖尿病史，最近血糖控制不好，经常口渴，尿频，体重也在下降。",
            "expected_route": "diagnosis",
            "expected_symptoms_min": 3,
            "expected_symptoms_max": 6,
            "expected_symptoms": ["口渴", "尿频", "体重下降"],
            "description": "慢性病症状恶化 - 有明确病史背景",
            "relevant_context": "糖尿病患者出现三多一少症状提示血糖控制不佳",
            "message_history": [
                {"role": "user", "content": "我有糖尿病"}
            ]
        },
        {
            "user_input": "医生，我想问一下高血压的注意事项",
            "expected_route": "follow_up",
            "expected_symptoms_min": 0,
            "expected_symptoms_max": 1,
            "expected_symptoms": [],
            "description": "非症状咨询 - 健康教育类问题",
            "relevant_context": "高血压管理包括生活方式调整和药物治疗"
        },
        {
            "user_input": "我心跳很快，胸闷气短，特别是爬楼梯的时候，有时候还会胸痛，持续了两个月了。",
            "expected_route": "diagnosis",
            "expected_symptoms_min": 3,
            "expected_symptoms_max": 7,
            "expected_symptoms": ["心跳快", "胸闷", "气短", "胸痛"],
            "description": "心血管系统症状 - 多个相关症状组合",
            "relevant_context": "心悸、胸闷、气短可能提示心脏疾病"
        }
    ]
    
    # 运行测试
    results = []
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"测试 {i+1}/{len(test_cases)}")
        result = run_test_case(test_case, logger)
        results.append(result)
    
    # 计算成功率
    success_count = sum(1 for r in results if r.get("overall_success", False))
    success_rate = (success_count / len(test_cases)) * 100
    
    logger.info(f"测试摘要: {success_count}/{len(test_cases)} 测试通过 ({success_rate:.2f}%)")
    
    # 计算平均响应时间
    response_times = [r.get("response_time", 0) for r in results if "response_time" in r]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    logger.info(f"平均响应时间: {avg_response_time:.2f}秒")
    
    # 计算各项指标的通过率
    route_matches = sum(1 for r in results if r.get("route_matches", False))
    symptoms_count_valid = sum(1 for r in results if r.get("symptoms_count_valid", False))
    symptoms_match = sum(1 for r in results if r.get("symptoms_match", False))
    
    logger.info(f"路由准确率: {route_matches}/{len(test_cases)} ({route_matches/len(test_cases)*100:.1f}%)")
    logger.info(f"症状数量合理率: {symptoms_count_valid}/{len(test_cases)} ({symptoms_count_valid/len(test_cases)*100:.1f}%)")
    logger.info(f"症状识别准确率: {symptoms_match}/{len(test_cases)} ({symptoms_match/len(test_cases)*100:.1f}%)")
    
    # 保存详细结果到JSON
    results_file = f"{log_dir}/symptom_extraction_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"详细结果已保存到 {results_file}")
    
    return results, success_rate, avg_response_time

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试症状提取节点')
    parser.add_argument('--filter', type=str, default=None, help='按描述筛选测试用例')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    args = parser.parse_args()
    
    # 如果启用调试模式，设置更详细的日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 设置日志
    logger, log_dir, timestamp = setup_logging()
    
    logger.info("=" * 60)
    logger.info("症状提取节点测试")
    logger.info("=" * 60)
    
    # 环境信息检查
    logger.info("环境信息:")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"当前工作目录: {os.getcwd()}")
    logger.info(f"Python路径: {sys.path}")
    logger.info(f"脚本路径: {os.path.abspath(__file__)}")
    
    # 尝试导入检查
    logger.info("尝试导入节点...")
    try:
        # 重新导入以确保最新版本
        import importlib
        if 'nodes.e_symptom_extraction_node' in sys.modules:
            logger.info("重新加载模块...")
            importlib.reload(sys.modules['nodes.e_symptom_extraction_node'])
        
        from nodes.e_symptom_extraction_node import symptom_extraction_node
        logger.info("节点导入成功")
        
    except ImportError as e:
        logger.error(f"导入失败: {str(e)}")
        logger.error("请检查以下项目:")
        logger.error("1. 确保 nodes/e_symptom_extraction_node.py 文件存在")
        logger.error("2. 确保 nodes/ 目录包含 __init__.py 文件")
        logger.error("3. 检查文件路径和模块名是否正确")
        logger.error("4. 确保所有依赖项都已安装")
        sys.exit(1)
    except Exception as e:
        logger.error(f"导入时发生未知错误: {str(e)}", exc_info=True)
        sys.exit(1)
    
    # 运行测试
    try:
        results, success_rate, avg_response_time = test_symptom_extraction(logger, log_dir, timestamp)
    except Exception as test_error:
        logger.error(f"测试执行失败: {str(test_error)}", exc_info=True)
        sys.exit(1)
    
    # 创建测试报告
    report = {
        "timestamp": timestamp,
        "total_tests": len(results),
        "success_count": sum(1 for r in results if r.get("overall_success", False)),
        "success_rate": success_rate,
        "avg_response_time": avg_response_time,
        "route_accuracy": sum(1 for r in results if r.get("route_matches", False)) / len(results) * 100,
        "symptom_count_accuracy": sum(1 for r in results if r.get("symptoms_count_valid", False)) / len(results) * 100,
        "symptom_recognition_accuracy": sum(1 for r in results if r.get("symptoms_match", False)) / len(results) * 100,
        "results": results
    }
    
    # 保存报告
    report_file = f"{log_dir}/symptom_extraction_report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"测试报告已保存到 {report_file}")
    
    # 检查是否所有测试都通过
    all_passed = all(r.get("overall_success", False) for r in results)
    
    if all_passed:
        logger.info("所有测试均通过！✅")
        sys.exit(0)
    else:
        logger.info("部分测试失败。请查看日志了解详情。❌")
        # 打印失败的测试
        failed_tests = [r for r in results if not r.get("overall_success", False)]
        logger.info(f"失败的测试 ({len(failed_tests)}):")
        for i, test in enumerate(failed_tests):
            test_case = test['test_case']
            logger.info(f"  {i+1}. 输入: '{test_case['user_input']}'")
            logger.info(f"     预期路由: '{test_case['expected_route']}', 实际路由: '{test.get('detected_route', 'ERROR')}'")
            logger.info(f"     症状数量: {test.get('symptoms_count', 0)}, 预期范围: {test_case.get('expected_symptoms_min', 0)}-{test_case.get('expected_symptoms_max', 999)}")
            if test.get('error'):
                logger.info(f"     错误: {test['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()