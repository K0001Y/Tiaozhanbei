import logging
import re
from typing import Dict, List, Any, TypedDict, Optional, Union, Tuple, Set

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 状态类型定义
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
    safety_violations: Optional[List[Dict[str, Any]]]  # 安全违规记录

# =============== 中药安全知识库 ===============

# 十八反药配伍（相反相克）
EIGHTEEN_INCOMPATIBLES = [
    {"herbs": ["甘草", "甘草片", "炙甘草"], "contra": ["海藻", "海浮石", "海蛤壳", "海螵蛸"], "reason": "甘草与咸寒药材配伍可导致药性相反"},
    {"herbs": ["乌头", "川乌", "草乌", "附子", "天雄", "乌头炮制品"], "contra": ["贝母", "川贝母", "平贝母", "浙贝母"], "reason": "乌头类与贝母配伍有毒"},
    {"herbs": ["乌头", "川乌", "草乌", "附子", "天雄", "乌头炮制品"], "contra": ["瓜蒌", "瓜蒌皮", "瓜蒌子", "天花粉"], "reason": "乌头类与瓜蒌配伍有毒"},
    {"herbs": ["乌头", "川乌", "草乌", "附子", "天雄", "乌头炮制品"], "contra": ["半夏", "白半夏", "清半夏"], "reason": "乌头类与半夏配伍有毒"},
    {"herbs": ["乌头", "川乌", "草乌", "附子", "天雄", "乌头炮制品"], "contra": ["白及"], "reason": "乌头类与白及配伍有毒"},
    {"herbs": ["乌头", "川乌", "草乌", "附子", "天雄", "乌头炮制品"], "contra": ["白蔹"], "reason": "乌头类与白蔹配伍有毒"},
    {"herbs": ["甘遂", "芫花", "大戟"], "contra": ["人参", "党参", "太子参", "红参", "西洋参"], "reason": "峻下逐水药与补气药配伍损伤元气"},
    {"herbs": ["芫花"], "contra": ["郁金"], "reason": "芫花与郁金配伍有毒"},
]

# 十九畏药配伍（相畏）
NINETEEN_FEARS = [
    {"herb": "硫磺", "fears": ["朱砂"], "reason": "硫磺畏朱砂"},
    {"herb": "水银", "fears": ["砒霜"], "reason": "水银畏砒霜"},
    {"herb": "狼毒", "fears": ["密陀僧"], "reason": "狼毒畏密陀僧"},
    {"herb": "巴豆", "fears": ["牵牛子"], "reason": "巴豆畏牵牛子"},
    {"herb": "丁香", "fears": ["郁金"], "reason": "丁香畏郁金"},
    {"herb": "川乌", "fears": ["犀角"], "reason": "川乌畏犀角"},
    {"herb": "牙硝", "fears": ["三棱"], "reason": "牙硝畏三棱"},
    {"herb": "芫花", "fears": ["石膏"], "reason": "芫花畏石膏"},
    {"herb": "官桂", "fears": ["赤石脂"], "reason": "官桂畏赤石脂"},
    {"herb": "人参", "fears": ["五灵脂"], "reason": "人参畏五灵脂"},
    {"herb": "黄连", "fears": ["干漆"], "reason": "黄连畏干漆"},
    {"herb": "黄芩", "fears": ["干漆"], "reason": "黄芩畏干漆"},
    {"herb": "黄柏", "fears": ["干漆"], "reason": "黄柏畏干漆"},
]

# 高风险/毒性中药
TOXIC_HERBS = [
    {"name": "附子", "max_dose": "15克", "notes": "必须炮制后使用，生附子极毒", "keywords": ["附子", "制附片", "黑顺片"]},
    {"name": "川乌", "max_dose": "10克", "notes": "必须炮制后使用，生川乌极毒", "keywords": ["川乌", "乌头"]},
    {"name": "草乌", "max_dose": "10克", "notes": "必须炮制后使用，生草乌极毒", "keywords": ["草乌"]},
    {"name": "马钱子", "max_dose": "0.6克", "notes": "含有马钱子碱，过量致命", "keywords": ["马钱子", "马钱", "番木鳖"]},
    {"name": "雷公藤", "max_dose": "6克", "notes": "有严重肝肾毒性", "keywords": ["雷公藤", "千斤拔", "昆明山海棠"]},
    {"name": "水银", "max_dose": "0.1克", "notes": "有累积毒性", "keywords": ["水银", "轻粉"]},
    {"name": "砒霜", "max_dose": "0.03克", "notes": "含砷，剧毒", "keywords": ["砒霜", "砒石", "石砒"]},
    {"name": "斑蝥", "max_dose": "0.03克", "notes": "含斑蝥素，极毒", "keywords": ["斑蝥", "青娘子", "绿班蝥"]},
    {"name": "蟾酥", "max_dose": "0.1克", "notes": "含蟾蜍毒素，过量致命", "keywords": ["蟾酥", "干蟾"]},
    {"name": "罂粟壳", "max_dose": "10克", "notes": "含吗啡类生物碱，有成瘾性", "keywords": ["罂粟壳", "御米壳", "米壳"]},
    {"name": "天南星", "max_dose": "9克", "notes": "生品有毒，须炮制后使用", "keywords": ["天南星", "南星", "半夏曲"]},
    {"name": "巴豆", "max_dose": "0.1克", "notes": "含巴豆油，剧毒", "keywords": ["巴豆", "巴豆霜"]},
    {"name": "洋金花", "max_dose": "0.6克", "notes": "含莨菪碱，过量致命", "keywords": ["洋金花", "曼陀罗花", "曼陀罗"]},
    {"name": "芫花", "max_dose": "3克", "notes": "有强烈刺激性", "keywords": ["芫花", "芫花素"]},
    {"name": "甘遂", "max_dose": "1.5克", "notes": "泻下药，过量伤肾", "keywords": ["甘遂"]},
    {"name": "大戟", "max_dose": "3克", "notes": "泻下药，过量伤肾", "keywords": ["大戟"]},
]

# 特殊人群禁用药材
SPECIAL_POPULATION_RESTRICTIONS = [
    {"population": "孕妇", "herbs": ["柴胡", "牛膝", "泽兰", "蟹爪兰", "三棱", "莪术", "牡丹皮", "赤芍", "虻虫", "水蛭", "斑蝥", "牛黄", "麝香"], 
     "reason": "具有活血化瘀或刺激子宫收缩作用，可能导致流产"},
    {"population": "孕妇", "herbs": ["附子", "川乌", "草乌", "巴豆", "芫花", "甘遂", "大戟"], 
     "reason": "毒性较大，可能伤害胎儿"},
    {"population": "哺乳期妇女", "herbs": ["大黄", "芒硝", "甘遂", "商陆", "芫花", "甘遂", "大戟", "巴豆"], 
     "reason": "可能通过乳汁影响婴儿"},
    {"population": "儿童", "herbs": ["附子", "川乌", "草乌", "马钱子", "雷公藤", "水银", "砒霜", "斑蝥", "蟾酥", "罂粟壳"],
     "reason": "毒性较大，儿童用药安全范围窄"},
    {"population": "肝功能不全患者", "herbs": ["雷公藤", "苍耳子", "夹竹桃", "杏仁"],
     "reason": "可能加重肝损伤"},
    {"population": "肾功能不全患者", "herbs": ["关木通", "广防己", "青木香", "马兜铃", "天仙藤", "寻骨风"],
     "reason": "含马兜铃酸，可能加重肾损伤"}
]

# =============== 节点实现 ===============

class ResponseSafetyNode:
    """响应安全检查节点类"""
    
    def __init__(self):
        """初始化响应安全检查节点"""
        logger.info("响应安全检查节点初始化完成")
    
    def _check_incompatible_herbs(self, response: str) -> List[Dict[str, Any]]:
        """检查十八反药配伍禁忌"""
        violations = []
        
        for incompatible in EIGHTEEN_INCOMPATIBLES:
            # 检查是否同时提到主药和禁忌药
            main_herbs_mentioned = []
            contra_herbs_mentioned = []
            
            for herb in incompatible["herbs"]:
                if herb in response:
                    main_herbs_mentioned.append(herb)
            
            for herb in incompatible["contra"]:
                if herb in response:
                    contra_herbs_mentioned.append(herb)
            
            if main_herbs_mentioned and contra_herbs_mentioned:
                violations.append({
                    "type": "十八反配伍禁忌",
                    "herbs": main_herbs_mentioned + contra_herbs_mentioned,
                    "reason": incompatible["reason"],
                    "risk_level": "高风险"
                })
        
        return violations
    
    def _check_fearful_combinations(self, response: str) -> List[Dict[str, Any]]:
        """检查十九畏药配伍禁忌"""
        violations = []
        
        for fear in NINETEEN_FEARS:
            # 检查是否同时提到主药和畏药
            if fear["herb"] in response:
                fears_mentioned = []
                
                for feared_herb in fear["fears"]:
                    if feared_herb in response:
                        fears_mentioned.append(feared_herb)
                
                if fears_mentioned:
                    violations.append({
                        "type": "十九畏配伍禁忌",
                        "herbs": [fear["herb"]] + fears_mentioned,
                        "reason": fear["reason"],
                        "risk_level": "中风险"
                    })
        
        return violations
    
    def _check_toxic_herbs(self, response: str) -> List[Dict[str, Any]]:
        """检查高风险/毒性中药"""
        violations = []
        
        for herb in TOXIC_HERBS:
            # 检查是否提到了高风险药材
            mentioned = False
            for keyword in herb["keywords"]:
                if keyword in response:
                    mentioned = True
                    break
            
            if mentioned:
                # 检查是否提到了剂量
                dose_pattern = re.compile(r'(\d+\.?\d*)\s*(?:克|g|毫克|mg|斤|两|钱)[^。，,；;]*' + re.escape(herb["name"]))
                dose_match = dose_pattern.search(response)
                
                if dose_match:
                    dose_str = dose_match.group(1)
                    try:
                        dose = float(dose_str)
                        max_dose = float(re.search(r'(\d+\.?\d*)', herb["max_dose"]).group(1))
                        
                        # 检查剂量是否超过最大安全剂量
                        if dose > max_dose:
                            violations.append({
                                "type": "毒性药材剂量过大",
                                "herb": herb["name"],
                                "suggested_dose": herb["max_dose"],
                                "found_dose": f"{dose}克",
                                "reason": f"{herb['name']}为毒性药材，建议剂量不超过{herb['max_dose']}，{herb['notes']}",
                                "risk_level": "高风险"
                            })
                    except (ValueError, AttributeError):
                        # 无法解析剂量，添加一般警告
                        violations.append({
                            "type": "毒性药材使用警告",
                            "herb": herb["name"],
                            "reason": f"{herb['name']}为毒性药材，{herb['notes']}，请在专业中医师指导下谨慎使用",
                            "risk_level": "中风险"
                        })
                else:
                    # 没有提到剂量，添加一般警告
                    violations.append({
                        "type": "毒性药材使用警告",
                        "herb": herb["name"],
                        "reason": f"{herb['name']}为毒性药材，{herb['notes']}，请在专业中医师指导下谨慎使用",
                        "risk_level": "中风险"
                    })
        
        return violations
    
    def _check_special_population_restrictions(self, response: str, user_input: str) -> List[Dict[str, Any]]:
        """检查特殊人群禁用药材"""
        violations = []
        
        # 合并用户输入和响应以检查特殊人群信息
        combined_text = user_input + " " + response
        
        for restriction in SPECIAL_POPULATION_RESTRICTIONS:
            population = restriction["population"]
            
            # 检查是否提到了特殊人群
            population_mentioned = False
            population_keywords = [population]
            
            if population == "孕妇":
                population_keywords.extend(["怀孕", "妊娠", "有孕", "备孕", "孕期"])
            elif population == "哺乳期妇女":
                population_keywords.extend(["哺乳", "喂奶", "产后", "授乳", "乳母"])
            elif population == "儿童":
                population_keywords.extend(["小孩", "儿科", "幼儿", "婴儿", "宝宝", "小朋友"])
            elif "肝功能" in population:
                population_keywords.extend(["肝病", "肝炎", "肝硬化", "肝功能异常", "转氨酶高"])
            elif "肾功能" in population:
                population_keywords.extend(["肾病", "肾炎", "肾功能异常", "肾功能不全", "尿毒症"])
            
            for keyword in population_keywords:
                if keyword in combined_text:
                    population_mentioned = True
                    break
            
            if population_mentioned:
                herbs_mentioned = []
                
                for herb in restriction["herbs"]:
                    if herb in response:
                        herbs_mentioned.append(herb)
                
                if herbs_mentioned:
                    violations.append({
                        "type": f"{population}用药禁忌",
                        "herbs": herbs_mentioned,
                        "reason": restriction["reason"],
                        "risk_level": "高风险"
                    })
        
        return violations
    
    def _check_disclaimer(self, response: str) -> List[Dict[str, Any]]:
        """检查是否包含免责声明"""
        violations = []
        
        disclaimer_keywords = ["免责", "仅供参考", "不构成医疗建议", "咨询医师", "在医师指导下"]
        has_disclaimer = any(keyword in response for keyword in disclaimer_keywords)
        
        if not has_disclaimer:
            violations.append({
                "type": "缺少免责声明",
                "reason": "响应中应包含免责声明，提醒用户这仅是参考信息，不能替代专业医疗建议",
                "risk_level": "低风险"
            })
        
        return violations
    
    def _create_safety_warning(self, violations: List[Dict[str, Any]]) -> str:
        """根据违规创建安全警告响应"""
        warning = "⚠️ 安全警告：检测到以下中药安全风险 ⚠️\n\n"
        
        # 按风险等级分组
        high_risk = [v for v in violations if v.get("risk_level") == "高风险"]
        medium_risk = [v for v in violations if v.get("risk_level") == "中风险"]
        low_risk = [v for v in violations if v.get("risk_level") == "低风险"]
        
        # 添加高风险警告
        if high_risk:
            warning += "【高风险警告】\n"
            for i, violation in enumerate(high_risk, 1):
                if violation["type"] == "毒性药材剂量过大":
                    warning += f"{i}. {violation['herb']}剂量过大: 建议不超过{violation['suggested_dose']}，检测到{violation['found_dose']}\n"
                    warning += f"   原因: {violation['reason']}\n\n"
                elif "配伍禁忌" in violation["type"]:
                    warning += f"{i}. {violation['type']}: {', '.join(violation['herbs'])}\n"
                    warning += f"   原因: {violation['reason']}\n\n"
                elif "用药禁忌" in violation["type"]:
                    warning += f"{i}. {violation['type']}: {', '.join(violation['herbs'])}\n"
                    warning += f"   原因: {violation['reason']}\n\n"
                else:
                    warning += f"{i}. {violation['type']}: {violation.get('reason', '未提供原因')}\n\n"
        
        # 添加中风险警告
        if medium_risk:
            warning += "【中风险警告】\n"
            for i, violation in enumerate(medium_risk, 1):
                if "毒性药材" in violation["type"]:
                    warning += f"{i}. {violation['herb']}: {violation['reason']}\n\n"
                elif "配伍禁忌" in violation["type"]:
                    warning += f"{i}. {violation['type']}: {', '.join(violation['herbs'])}\n"
                    warning += f"   原因: {violation['reason']}\n\n"
                else:
                    warning += f"{i}. {violation['type']}: {violation.get('reason', '未提供原因')}\n\n"
        
        # 添加低风险建议
        if low_risk:
            warning += "【建议】\n"
            for violation in low_risk:
                warning += f"- {violation['reason']}\n"
        
        # 添加安全提示
        warning += "\n请注意：\n"
        warning += "1. 以上信息仅用于安全提示，不构成医疗建议\n"
        warning += "2. 中药应在专业中医师指导下使用\n"
        warning += "3. 某些中药具有毒性，使用不当可能危及健康\n"
        warning += "4. 特殊人群（孕妇、儿童、老人、肝肾功能不全患者）用药需格外谨慎\n"
        
        return warning
    
    def _add_disclaimer_if_needed(self, response: str) -> str:
        """如果需要，添加免责声明"""
        disclaimer_keywords = ["免责", "仅供参考", "不构成医疗建议", "咨询医师", "在医师指导下"]
        has_disclaimer = any(keyword in response for keyword in disclaimer_keywords)
        
        if not has_disclaimer:
            disclaimer = "\n\n【免责声明】：以上内容仅供参考，不构成医疗建议。请在专业中医师指导下使用中药，切勿自行配药或更改剂量。"
            return response + disclaimer
        
        return response
    
    def __call__(self, state: State) -> Tuple[State, str]:
        """节点主函数"""
        try:
            # 获取输入
            response = state.get("response", "")
            user_input = state.get("user_input", "")
            
            if not response:
                logger.warning("响应为空，无法进行安全检查")
                return state, "to_output"
            
            # 执行安全检查
            violations = []
            
            # 十八反药配伍检查
            incompatible_violations = self._check_incompatible_herbs(response)
            violations.extend(incompatible_violations)
            
            # 十九畏药配伍检查
            fearful_violations = self._check_fearful_combinations(response)
            violations.extend(fearful_violations)
            
            # 毒性药材检查
            toxic_violations = self._check_toxic_herbs(response)
            violations.extend(toxic_violations)
            
            # 特殊人群禁忌检查
            special_violations = self._check_special_population_restrictions(response, user_input)
            violations.extend(special_violations)
            
            # 免责声明检查
            disclaimer_violations = self._check_disclaimer(response)
            violations.extend(disclaimer_violations)
            
            # 记录安全检查结果
            safety_check_result = {
                "has_violations": bool(violations),
                "violations_count": len(violations),
                "violations": violations
            }
            
            # 更新状态
            updated_state = {
                **state,
                "safety_violations": violations,
                "safety_check": safety_check_result
            }
            
            # 处理违规
            if violations:
                # 分类违规
                high_med_violations = [v for v in violations if v.get("risk_level") in ["高风险", "中风险"]]
                only_low_violations = all(v.get("risk_level") == "低风险" for v in violations)
                
                if high_med_violations:
                    # 存在高风险或中风险违规，替换响应为警告
                    logger.warning(f"检测到{len(high_med_violations)}个高/中风险安全违规，替换响应为警告")
                    safety_warning = self._create_safety_warning(violations)
                    updated_state["response"] = safety_warning
                elif only_low_violations:
                    # 仅存在低风险违规，添加免责声明
                    logger.info("仅检测到低风险违规，添加免责声明")
                    updated_state["response"] = self._add_disclaimer_if_needed(response)
            else:
                # 无违规，检查是否需要添加免责声明
                logger.info("未检测到安全违规")
                updated_state["response"] = self._add_disclaimer_if_needed(response)
            
            return updated_state, "to_output"
        
        except Exception as e:
            error_msg = f"响应安全检查过程中出错: {str(e)}"
            logger.error(error_msg)
            
            # 发生错误时保留原响应并添加安全提示
            safety_reminder = (
                "\n\n【安全提示】：中药应在专业中医师指导下使用，某些药材具有毒性或配伍禁忌，"
                "特殊人群（孕妇、儿童、老人、肝肾功能不全患者）用药需格外谨慎。"
            )
            
            return {
                **state,
                "error": error_msg,
                "response": (state.get("response", "") + safety_reminder)
            }, "to_output"

# 导出节点实例以便在图中使用
response_safety_node = ResponseSafetyNode()