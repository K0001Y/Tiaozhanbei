# 中医智能对话系统 - LangGraph节点使用文档

## 1. 系统概述

本系统使用LangGraph框架构建了一个中医智能对话系统，通过多个专用节点处理用户输入、安全检查、RAG检索、意图识别、症状提取、诊断分析、处方生成和响应安全过滤等功能。系统能够理解用户意图，区分普通咨询和诊断需求，并提供相应的专业回复。

## 2. 节点详解

### 2.1 输入处理节点 (input_node)

**功能**：校验用户输入，将输入转换为HumanMessage并添加到状态的messages列表中。

**主要函数**：
- `input_node(state)` - 处理输入并返回更新后的状态和路由指示

**输入字段**：
- `user_input`: 用户输入的文本

**输出字段**：
- `messages`: 更新后的消息历史
- `memory`: 更新后的对话内存
- `query`: 初始查询（与用户输入相同）
- `error`: 如果输入无效，设置错误信息

**路由**：
- 如果输入有效：`"next"` → 安全检查节点
- 如果输入无效：`"end"` → 结束

### 2.2 安全检查节点 - 输入阶段 (safety_guard_node)

**功能**：对用户输入执行安全协议检查，检测紧急关键词（如"胸痛"、"急救"等）。

**主要函数**：
- `safety_guard_node(state)` - 执行安全检查并返回更新后的状态和路由指示

**输入字段**：
- `user_input`: 用户输入

**输出字段**：
- `safety_check`: 安全检查结果，包含is_emergency、detected_keywords和risk_level
- `response`: 如果检测到紧急情况，设置为紧急提醒消息

**路由**：
- 如果检测到紧急情况：`"emergency"` → 输出节点
- 如果安全检查通过：`"continue"` → RAG检索节点

### 2.3 RAG检索节点 (RAGRetrievalNode)

**功能**：执行向量存储检索，获取与用户查询相关的文档。

**主要函数**：
- `__call__(state)` - 执行检索并返回更新后的状态
- `get_info()` - 获取RAG检索节点信息

**输入字段**：
- `query`: 用户查询
- `config`: 配置信息，包括检索数量k

**输出字段**：
- `documents`: 检索到的相关文档
- `error`: 如果检索失败，设置错误信息
- `relevant_context`: 从文档提取的相关上下文

**路由**：
- 线性到意图识别节点

### 2.4 意图识别节点 (recognize_intent_node)

**功能**：基于大模型分析用户输入，识别用户意图（诊断需求或一般问题）。

**主要函数**：
- `__init__()` - 初始化LLM和提示模板
- `_create_llm()` - 创建LLM实例
- `__call__(state)` - 识别意图并返回更新后的状态和路由指示

**输入字段**：
- `user_input`: 用户输入

**输出字段**：
- `intent`: 识别的意图（"diagnosis"、"question"或"unclear"）
- `intent_details`: 意图详细信息，包括置信度、关键词和推理过程

**路由**：
- 如果意图是"diagnosis"：`"diagnosis"` → 症状提取节点
- 如果意图不是"diagnosis"：`"conversation"` → 对话链处理节点

### 2.5 症状提取节点 (symptom_extraction_node)

**功能**：从用户输入中提取症状信息，并识别缺失的重要信息。

**主要函数**：
- `__init__()` - 初始化LLM、解析器和链
- `_create_llm()` - 创建LLM实例
- `_create_output_parser()` - 创建结构化输出解析器
- `_create_symptom_prompt()` - 创建症状提取提示
- `__call__(state)` - 提取症状并返回更新后的状态和路由指示

**输入字段**：
- `user_input`: 用户输入
- `messages`: 消息历史
- `relevant_context`: RAG检索相关上下文

**输出字段**：
- `symptoms_list`: 提取的症状列表，每个症状包含名称、描述、严重程度和持续时间
- `missing_info_list`: 缺失的信息列表

**路由**：
- 如果缺失信息或症状不足：`"follow_up"` → 追问节点
- 如果信息充足：`"diagnosis"` → 辨证分析节点

### 2.6 追问节点 (follow_up_question_node)

**功能**：基于缺失信息列表生成追问响应，确保专业和礼貌。

**主要函数**：
- `__init__()` - 初始化LLM和提示
- `_create_llm()` - 创建LLM实例
- `_create_question_prompt()` - 创建追问提示模板
- `_extract_previous_questions()` - 从历史消息提取之前的问题
- `_filter_duplicate_missing_info()` - 过滤重复的缺失信息
- `__call__(state)` - 生成追问并返回更新后的状态和路由指示

**输入字段**：
- `missing_info_list`: 缺失的信息列表
- `symptoms_list`: 已提取的症状列表
- `messages`: 消息历史

**输出字段**：
- `response`: 设置为追问文本
- `conversation_state`: 设置为"awaiting_follow_up"或"ready_for_diagnosis"

**路由**：
- 如果需要追问：`"to_output"` → 输出节点（等待用户回答）
- 如果没有新问题且已有足够信息：`"to_diagnosis"` → 辨证分析节点

### 2.7 辨证分析节点 (diagnosis_node)

**功能**：基于症状和相关上下文进行中医辨证分析，输出结构化诊断结果。

**主要函数**：
- `__init__()` - 初始化LLM、解析器和链
- `_create_llm()` - 创建LLM实例
- `_create_output_parser()` - 创建结构化输出解析器
- `_create_diagnosis_prompt()` - 创建辨证分析提示
- `_format_symptoms()` - 格式化症状列表
- `__call__(state)` - 执行辨证分析并返回更新后的状态和路由指示

**输入字段**：
- `symptoms_list`: 提取的症状列表
- `relevant_context`: RAG检索相关上下文

**输出字段**：
- `diagnosis_data`: 辨证分析结果，包含证型、病机、分析依据、置信度和鉴别诊断
- `conversation_state`: 设置为"diagnosis_complete"

**路由**：
- 线性到处方推荐节点：`"to_prescription"` → 处方推荐节点

### 2.8 处方推荐节点 (prescription_node)

**功能**：基于辨证分析结果生成方剂推荐，整合所有阶段结果到响应。

**主要函数**：
- `__init__()` - 初始化LLM、解析器和链
- `_create_llm()` - 创建LLM实例
- `_create_output_parser()` - 创建结构化输出解析器
- `_create_prescription_prompt()` - 创建处方推荐提示
- `_create_response_prompt()` - 创建响应整合提示
- `_create_fallback_response()` - 创建备用响应
- `__call__(state)` - 生成处方并返回更新后的状态和路由指示

**输入字段**：
- `diagnosis_data`: 辨证分析结果
- `symptoms_list`: 提取的症状列表
- `relevant_context`: RAG检索相关上下文

**输出字段**：
- `prescription_data`: 处方推荐数据，包含方剂名称、组成、用法等
- `response`: 整合了症状、诊断和处方的完整响应文本
- `conversation_state`: 设置为"prescription_complete"

**路由**：
- 线性到响应安全检查节点：`"to_safety_check"` → 响应安全检查节点

### 2.9 对话链处理节点 (conversation_chain_node)

**功能**：处理一般中医问题，使用ConversationChain生成响应。

**主要函数**：
- `__init__()` - 初始化LLM、提示和链
- `_create_llm()` - 创建LLM实例
- `_create_conversation_prompt()` - 创建对话提示模板
- `_format_history()` - 格式化消息历史
- `__call__(state)` - 生成对话回复并返回更新后的状态和路由指示

**输入字段**：
- `user_input`: 用户输入
- `messages`: 消息历史
- `relevant_context`: RAG检索相关上下文

**输出字段**：
- `response`: 生成的对话回复

**路由**：
- 线性到响应安全检查节点：`"to_safety_check"` → 响应安全检查节点

### 2.10 响应安全检查节点 (response_safety_node)

**功能**：对生成的响应执行安全检查，检测中药配伍禁忌、毒性药材等风险。

**主要函数**：
- `__init__()` - 初始化节点
- `_check_incompatible_herbs()` - 检查十八反药配伍禁忌
- `_check_fearful_combinations()` - 检查十九畏药配伍禁忌
- `_check_toxic_herbs()` - 检查高风险/毒性中药
- `_create_safety_warning()` - 创建安全警告响应
- `__call__(state)` - 执行安全检查并返回更新后的状态和路由指示

**输入字段**：
- `response`: 生成的响应
- `user_input`: 用户输入

**输出字段**：
- `safety_violations`: 检测到的安全违规列表
- `safety_check`: 安全检查结果
- `response`: 如果检测到高风险违规，更新为安全警告响应

**路由**：
- 线性到输出节点：`"to_output"` → 输出节点

### 2.11 输出处理节点 (output_node)

**功能**：将响应转换为AIMessage添加到messages，保存到内存和数据库。

**主要函数**：
- `__init__()` - 初始化节点和数据库管理器
- `_create_default_error_message()` - 创建默认错误响应
- `_collect_metadata()` - 从状态中收集元数据
- `__call__(state)` - 处理输出并返回最终状态

**输入字段**：
- `response`: 最终响应
- `messages`: 消息历史
- `memory`: 对话内存

**输出字段**：
- `messages`: 更新后的消息历史（添加了AIMessage）
- 其他字段保持不变

**路由**：
- 终止节点，返回最终状态

## 3. 状态字段说明

LangGraph中的状态对象在节点间传递，包含以下关键字段：

| 字段名 | 类型 | 说明 |
|-------|------|------|
| user_input | str | 用户输入的原始文本 |
| query | str | 处理后的查询文本 |
| messages | List[Any] | 消息历史，包含HumanMessage和AIMessage |
| memory | Any | 对话内存，用于保存历史 |
| documents | List[Dict] | RAG检索到的相关文档 |
| response | str | 生成的响应文本 |
| error | str | 错误信息 |
| config | Dict | 配置信息 |
| safety_check | Dict | 安全检查结果 |
| intent | str | 用户意图 |
| intent_details | Dict | 意图详细信息 |
| relevant_context | str | RAG检索相关上下文 |
| symptoms_list | List[Dict] | 提取的症状列表 |
| missing_info_list | List[str] | 缺失的信息列表 |
| conversation_state | str | 对话状态标记 |
| diagnosis_data | Dict | 辨证分析结果 |
| prescription_data | Dict | 处方推荐数据 |
| safety_violations | List[Dict] | 安全违规记录 |

## 4. 图结构

```
                                                  ┌─────────────────┐
                                                  │  follow_up_     │
                                                  │  question_node  │◄────────┐
                                                  └────────┬────────┘         │
                                                           │                  │
                                                           ▼                  │
┌─────────┐     ┌─────────────┐     ┌─────────┐     ┌────────────┐     ┌─────┴──────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────┐
│ input_  │     │ safety_     │     │ RAG_    │     │ recognize_ │     │ symptom_   │     │ diagnosis_   │     │ prescription_ │     │ response_  │     ┌─────────┐
│ node    ├────►│ guard_node  ├────►│ node    ├────►│ intent_node├────►│ extraction_├────►│ node         ├────►│ node          ├────►│ safety_node├────►│ output_ │
│         │     │             │     │         │     │            │     │ node       │     │              │     │               │     │            │     │ node    │
└─────────┘     └─────────────┘     └─────────┘     └──────┬─────┘     └────────────┘     └──────────────┘     └───────────────┘     └────────────┘     └─────────┘
                       │                                    │                                                                                                 ▲
                       │                                    ▼                                                                                                 │
                       │                            ┌──────────────┐                                                                                          │
                       │                            │ conversation_│                                                                                          │
                       │                            │ chain_node   ├──────────────────────────────────────────────────────────────────────────────────────────┘
                       │                            └──────────────┘
                       │
                       └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────►
```

### 路由逻辑

1. **input_node**:
   - 输入有效 → safety_guard_node
   - 输入无效 → END

2. **safety_guard_node**:
   - 检测到紧急关键词 → output_node
   - 安全检查通过 → RAG_node

3. **RAG_node** → recognize_intent_node

4. **recognize_intent_node**:
   - 意图是"diagnosis" → symptom_extraction_node
   - 意图不是"diagnosis" → conversation_chain_node

5. **symptom_extraction_node**:
   - 缺失信息或症状不足 → follow_up_question_node
   - 信息充足 → diagnosis_node

6. **follow_up_question_node**:
   - 需要追问 → output_node（等待用户回答）
   - 没有新问题且已有足够信息 → diagnosis_node

7. **diagnosis_node** → prescription_node

8. **prescription_node** → response_safety_node

9. **conversation_chain_node** → response_safety_node

10. **response_safety_node** → output_node

11. **output_node** → END

## 5. 使用示例

### 5.1 创建和编译图

```python
from langgraph.graph import StateGraph
from typing import Dict, List, Any, TypedDict, Optional

# 导入所有节点
from input_node import input_node
from safety_guard_node import safety_guard_node
from rag_node import RAGRetrievalNode
from recognize_intent_node import recognize_intent_node
from symptom_extraction_node import symptom_extraction_node
from follow_up_question_node import follow_up_question_node
from diagnosis_node import diagnosis_node
from prescription_node import prescription_node
from conversation_chain_node import conversation_chain_node
from response_safety_node import response_safety_node
from output_node import output_node

# 定义状态类型
class State(TypedDict):
    """LangGraph状态类型"""
    user_input: str
    query: Optional[str]
    messages: List[Any]
    # ... 其他字段 ...

# 创建RAG检索节点实例
rag_node = RAGRetrievalNode(
    model_path="path/to/models",
    vector_store_path="path/to/vector_store",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# 创建图实例
graph = StateGraph(State)

# 添加所有节点
graph.add_node("input", input_node)
graph.add_node("safety_guard", safety_guard_node)
graph.add_node("rag", rag_node)
graph.add_node("recognize_intent", recognize_intent_node)
graph.add_node("symptom_extraction", symptom_extraction_node)
graph.add_node("follow_up_question", follow_up_question_node)
graph.add_node("diagnosis", diagnosis_node)
graph.add_node("prescription", prescription_node)
graph.add_node("conversation_chain", conversation_chain_node)
graph.add_node("response_safety", response_safety_node)
graph.add_node("output", output_node)

# 添加边和条件路由
# 输入节点到安全检查节点
graph.add_edge("input", "safety_guard", condition=lambda x: not x.get("error"))
graph.add_edge("input", "output", condition=lambda x: bool(x.get("error")))

# 安全检查节点路由
graph.add_edge("safety_guard", "rag", 
              condition=lambda x: x.get("safety_check", {}).get("is_emergency") == False)
graph.add_edge("safety_guard", "output", 
              condition=lambda x: x.get("safety_check", {}).get("is_emergency") == True)

# RAG节点到意图识别节点
graph.add_edge("rag", "recognize_intent")

# 意图识别节点路由
graph.add_edge("recognize_intent", "symptom_extraction", 
              condition=lambda x: x.get("intent") == "diagnosis")
graph.add_edge("recognize_intent", "conversation_chain", 
              condition=lambda x: x.get("intent") != "diagnosis")

# 症状提取节点路由
graph.add_edge("symptom_extraction", "follow_up_question", 
              condition=lambda x: len(x.get("missing_info_list", [])) > 0 or 
                                 len(x.get("symptoms_list", [])) < 2)
graph.add_edge("symptom_extraction", "diagnosis", 
              condition=lambda x: len(x.get("missing_info_list", [])) == 0 and 
                                 len(x.get("symptoms_list", [])) >= 2)

# 追问节点路由
graph.add_edge("follow_up_question", "output", 
              condition=lambda x: x.get("conversation_state") == "awaiting_follow_up")
graph.add_edge("follow_up_question", "diagnosis", 
              condition=lambda x: x.get("conversation_state") == "ready_for_diagnosis")

# 诊断节点到处方节点
graph.add_edge("diagnosis", "prescription")

# 处方节点到安全检查节点
graph.add_edge("prescription", "response_safety")

# 对话链节点到安全检查节点
graph.add_edge("conversation_chain", "response_safety")

# 响应安全检查节点到输出节点
graph.add_edge("response_safety", "output")

# 设置输出节点为终止节点
graph.set_finish_node("output")

# 编译图
app = graph.compile()
```

### 5.2 运行图

```python
# 创建初始状态
initial_state = {
    "user_input": "我最近总是感到头晕，还伴有口干舌燥，请问是什么原因？",
    "messages": [],
    "config": {"k": 3}  # 配置RAG检索返回3个结果
}

# 运行图
result = app.invoke(initial_state)

# 获取响应
response = result.get("response", "")
print(response)
```

## 6. 注意事项

1. **LLM配置**: 每个使用LLM的节点都有显式的配置选项，可以根据需要调整模型、温度等参数。

2. **安全检查**: 系统包含两个安全检查节点，分别在输入和输出阶段，确保系统的安全性。

3. **状态管理**: 所有节点都遵循不可变状态更新模式，返回更新后的新状态而非修改原状态。

4. **错误处理**: 每个节点都包含异常处理，确保即使在错误情况下也能返回合理的响应。

5. **数据库集成**: 输出节点支持可选的数据库集成，可以通过提供连接字符串来启用。

6. **向量存储**: RAG节点需要预先准备好的向量存储，需要确保路径正确且文件存在。

7. **多轮对话**: 系统支持多轮对话，特别是在症状收集阶段，可能需要多轮交互才能收集足够信息。

8. **模块化设计**: 系统设计为高度模块化，各节点可以独立修改和替换，只要保持接口一致。

通过这种模块化的LangGraph设计，系统能够灵活处理各种中医咨询场景，提供专业、安全的回复。