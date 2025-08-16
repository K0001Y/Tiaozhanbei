### LangGraph 重构 ChatBot 对话链的具体节点规划

以下是基于原 ChatBot 代码的核心逻辑，使用 LangGraph 构建的节点规划。整个图采用 StateGraph 结构，状态（State）使用 TypedDict 定义，包含共享字段如 messages（消息历史）、user_input（用户输入）、intent（意图）、safety_check（安全结果）、relevant_context（RAG 上下文）、response（最终响应）等。节点是函数形式，每个节点接收 State 并返回更新后的 State。边连接为线性 + 条件路由（e.g., 基于 intent 或 safety_check）。

#### 节点列表与详细规划
1. **input_node（输入处理节点）**  
   - **作用**：校验用户输入（非空、非空白），将输入转换为 HumanMessage 添加到状态的 messages 列表中。同时保存到内存（ConversationBufferMemory）。如果输入无效，设置错误响应并路由到结束。  
   - **输入**：State 中的 user_input。  
   - **输出**：更新 messages 和潜在错误 response。  
   - **路由**：线性到下一个节点（safety_guard_node）。  

2. **safety_guard_node（安全检查节点 - 输入阶段）**  
   - **作用**：对用户输入执行安全协议检查，包括检测紧急关键词（e.g., "胸痛"、"急救"）。更新状态的 safety_check 字段。如果检测到紧急，立即设置 response 为紧急提醒消息，并路由到结束节点（跳过后续处理）。  
   - **输入**：State 中的 user_input。  
   - **输出**：更新 safety_check 和潜在 response（如果紧急）。  
   - **路由**：如果紧急，条件路由到 output_node 或 END；否则线性到 rag_retrieve_node。  

3. **rag_retrieve_node（RAG 检索节点）**  
   - **作用**：如果 RAGManager 就绪，基于 user_input 执行检索（k=3 或 4，根据链类型），提取相关文档上下文，添加到状态的 relevant_context。处理检索失败时设置为空字符串。  
   - **输入**：State 中的 user_input。  
   - **输出**：更新 relevant_context。  
   - **路由**：线性到 recognize_intent_node（可并行执行，如果 LangGraph 配置异步）。  

4. **recognize_intent_node（意图识别节点）**  
   - **作用**：基于关键词匹配分析 user_input，设置状态的 intent（"diagnosis"、"question" 或 "unclear"）。整合文档关键词（e.g., "病历"）到诊断意图。  
   - **输入**：State 中的 user_input。  
   - **输出**：更新 intent。  
   - **路由**：条件路由：如果 "diagnosis"，到 llm_chain_node；否则到 conversation_chain_node。  

5. **symptom_extraction_node（症状提取节点 - 新拆分，原 llm_chain_node 第一阶段）**  
   - **作用**：使用 LLMChain + symptom_prompt 提取症状，输出 JSON（如 symptoms、missing_info、stage="symptom_collection"）。整合 RAG 上下文和历史消息。解析 JSON，结果存入状态（如 symptoms_list 和 missing_info_list）。  
   - **输入**：State 中的 user_input、relevant_context、messages。  
   - **输出**：更新 symptoms_list 和 missing_info_list。  
   - **路由**：条件路由：如果 missing_info_list 不为空或 symptoms_list 不足（e.g., 长度 < 阈值，如 2），到 follow_up_question_node；否则线性到 diagnosis_node。

6. **follow_up_question_node（追问节点 - 新增）**  
   - **作用**：基于 missing_info_list 生成追问响应，例如 "为了更准确诊断，请补充以下信息：1. 头痛位置 2. 持续时间"。可以可选使用 LLM 生成自然语言追问提示（e.g., PromptTemplate 如 "基于{missing_info}，生成礼貌追问"），确保专业和礼貌。设置 response 为追问消息，并标记状态为 "awaiting_follow_up"（可选，用于多轮跟踪）。如果有历史上下文（如前轮追问），避免重复。  
   - **输入**：State 中的 missing_info_list、user_input、messages（历史）。  
   - **输出**：更新 response 为追问文本。  
   - **路由**：线性到 response_safety_node（或直接到 output_node，如果追问无需二次安全检查）。

7. **diagnosis_node（辨证分析节点 - 新拆分，原 llm_chain_node 第二阶段）**  
   - **作用**：使用 diagnosis_chain 基于 symptoms_list 和 relevant_context 进行辨证，输出 JSON（如 证型、病机、依据、stage="diagnosis"）。  
   - **输入**：State 中的 symptoms_list、relevant_context。  
   - **输出**：更新 diagnosis_data（如 证型、病机）。  
   - **路由**：线性到 prescription_node。

8. **prescription_node（处方推荐节点 - 新拆分，原 llm_chain_node 第三阶段）**  
   - **作用**：使用 prescription_chain 基于 diagnosis_data 生成方剂推荐，输出 JSON（如 方剂名、组成、禁忌检查、stage="prescription"）。整合所有阶段结果到 response（e.g., 症状 + 诊断 + 方剂的文本总结）。  
   - **输入**：State 中的 diagnosis_data、relevant_context。  
   - **输出**：更新 response 为完整诊断响应。  
   - **路由**：线性到 response_safety_node。

6. **conversation_chain_node（ConversationChain 处理节点 - 问答链）**  
   - **作用**：针对问答或不明意图，使用 ConversationChain 生成响应，融入 messages 历史、relevant_context 和 document_content。构造增强输入模板，确保专业中医建议。  
   - **输入**：State 中的 user_input、relevant_context、messages。  
   - **输出**：更新 response。  
   - **路由**：线性到 response_safety_node。  

7. **response_safety_node（安全检查节点 - 响应阶段）**  
   - **作用**：对生成的 response 执行二次安全检查，包括用药剂量红线、十八反十九畏禁忌等。如果检测到风险，覆盖 response 为阻塞消息（e.g., "检测到用药安全风险" + 警告列表）。  
   - **输入**：State 中的 response、user_input。  
   - **输出**：更新 response（如果阻塞）。  
   - **路由**：线性到 output_node。  

8. **output_node（输出处理节点）**  
   - **作用**：将 response 转换为 AIMessage 添加到 messages，保存到内存和数据库（如果 DatabaseManager 就绪，包括 metadata 如 safety_check、intent）。处理最终异常，返回默认错误消息。  
   - **输入**：State 中的 response、messages。  
   - **输出**：最终 State（用于返回 AIMessage）。  
   - **路由**：到 END。  

#### 需要注意的功能
- **状态管理**：使用 Annotated[Sequence[BaseMessage], "add_messages"] 确保消息历史自动累积。添加自定义字段如 document_content（从上传文档导入），以支持第一轮对话或病历文件上下文。
- **条件路由**：在图构建时，使用 conditional_edge（如基于 intent == "diagnosis" 路由到 llm_chain_node）。紧急安全检查需优先路由到 END，避免不必要计算。
- **RAG 集成**：确保节点检查 _rag_ready 和 _vector_store_loaded 标志。如果未就绪，跳过检索或设置空上下文。支持 k 值根据链类型调整（诊断 k=3，问答 k=4）。
- **内存与历史**：所有节点共享 ConversationBufferMemory，确保 save_context 在 input/output 节点调用。支持 _configure_memory 切换内存类型（buffer/summary/window），但在图中固定为 buffer 以简化。
- **安全协议全面性**：双阶段检查（输入 + 响应），覆盖紧急、剂量、禁忌、地域等。节点需处理 false positive（如误判关键词），并记录 warnings 到状态。
- **阶段化与 JSON 输出**：在 llm_chain_node 中，使用多个子 LLMChain，确保每个阶段输出 JSON 格式，便于解析和整合。添加错误处理（如 JSON 解析失败时 fallback）。
- **异常与日志**：每个节点内 try-except 捕获错误，设置 _last_error 并记录日志（使用原 logger）。避免图崩溃，返回友好错误响应。
- **可扩展性**：图设计支持添加节点，如 upload_document_node（上传病历，更新 document_content）。启用异步（async def）以并行 RAG 和意图识别，提高效率。
- **性能与限制**：限制 RAG 检索 k 值避免过载；意图识别可增强为 LLM-based 以提高准确性（但保持简单关键词以匹配原代码）。测试时，确保图编译（compile()）后可运行 invoke({"user_input": "test"})。
- **兼容原代码**：保留 PromptTemplate 和链创建逻辑，直接从原方法移植（e.g., _create_llm_chain）。如果有第一轮内存（first_round_memory），整合到状态的 messages。
- **条件路由与循环支持**：在 symptom_extraction_node 后，使用 conditional_edge 基于 missing_info_list 是否为空路由。如果追问，图结束当前调用，但状态持久化（LangGraph 支持 checkpoint），下轮用户输入可继续（e.g., 意图识别时检查历史 missing_info）。这实现多轮互动，而非单轮诊断。
- **症状不足判断**：不止检查 missing_info 非空，还可添加阈值（如 symptoms < 3 或 LLM 输出 "insufficient" 标志）。在节点内处理 JSON 解析失败（fallback 到默认追问，如 "请详细描述症状"）。
- **响应生成**：追问节点确保响应礼貌、专业（参考原代码回答要求：礼貌、清晰、诚实、使用用户语言）。整合 RAG（如 "根据知识库，需补充舌象"）。
- **状态扩展**：在 AgentState 添加 symptoms_list、missing_info_list、diagnosis_data 等字段，便于子节点共享。使用 Annotated[List[str], "add_to_list"] 等操作符自动累积。
- **内存集成**：在 follow_up_question_node 保存追问到内存，作为 AI 消息；下轮输入时，历史 messages 可用于避免重复追问。
- **安全与边缘ケース**：追问响应无需用药检查，但仍路由到 response_safety_node 以统一。如果连续多轮追问，添加上限（e.g., 状态计数器 > 3 时建议上传文档或就医）。
- **可扩展性**：节点设计支持未来添加如 "remind_user_node"（如果无历史，提醒上传病历），放在 input_node 后。测试时，确保路由循环不死锁（e.g., invoke 图多次模拟多轮）。
- **性能注意**：拆分节点增加 LLM 调用（每个子链一次），但提升模块化；可并行 RAG 和症状提取如果异步。保持原代码的 JSON 输出格式，确保解析可靠。 