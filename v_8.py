import os
import random

from typing import TypedDict, List, Dict, Literal, Optional

import numpy as np
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_deepseek import ChatDeepSeek

# pip install -U duckduckgo-search -i https://pypi.tuna.tsinghua.edu.cn/simple
'''
重构版
增加了MongoDB
'''
from langchain_community.document_loaders.csv_loader import CSVLoader
from typing import List
from langchain_community.tools import DuckDuckGoSearchResults
from tenacity import retry, stop_after_attempt, wait_random_exponential

from datetime import datetime
from v_8_dialogues_memory import DialogueManager

# 初始化对话管理器
CONN_STR = "mongodb://mongodb:DTNxf5TYcZWFKDYY@116.62.149.204:27017/"
dialogue_manager = DialogueManager(CONN_STR)

# 引入音频生成
from v_8_TTS import TTSHandler

tts = TTSHandler()


class RAGManager:
    """统一的RAG管理类，支持不同的embedding模型和向量存储"""

    def __init__(
            self,
            embedding_model_path: str = r"D:\Embedding\xrunda\m3e-base",
            vector_store_path: str = "vector_store/faiss_index",
            device: str = "cpu",
            normalize_embeddings: bool = False
    ):
        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings}
        )
        self.vector_store_path = vector_store_path
        self.vector_store = None

    def init_vectorstore(self, documents: List[Document] = None) -> FAISS:
        """初始化或加载向量存储"""
        if os.path.exists(self.vector_store_path):
            print("加载已有向量存储")
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embedding,
                allow_dangerous_deserialization=True  # debug 显式声明信任源 防止报错
            )
        elif documents:
            print("创建新向量存储")
            self.vector_store = FAISS.from_documents(documents, self.embedding)
            self.vector_store.save_local(self.vector_store_path)
        else:
            raise ValueError("需要提供文档来初始化向量存储")
        return self.vector_store

    def retrieve(
            self,
            query: str,
            search_type: str = "mmr",
            k: int = 5,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[Dict] = None
    ) -> List[Document]:
        """统一检索接口"""
        if not self.vector_store:
            self.init_vectorstore()

        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
                "filter": filter
            }
        )
        return retriever.invoke(query)


# 初始化RAG管理实例
rag_manager = RAGManager()


def load_shop_products_csv(file_path: str) -> List[Document]:
    """加载CSV商品数据并初始化向量存储"""
    loader = CSVLoader(
        file_path=file_path,
        source_column="product_id",
        metadata_columns=["product_id", "product_name", "product_price", "category", "position", "stock", "sales"],
        encoding="utf-8",
        csv_args={
            "fieldnames": [
                "id", "order_id", "product_id", "product_name",
                "product_url", "product_price", "product_count",
                "create_time", "update_time", "dt",
                "category", "position", "stock", "sales"  # 为业务外的店铺自定义字段，可以随csv等元数据继续添加
            ],
        }
    )
    raw_docs = loader.load()
    processed_docs = [
        Document(
            page_content=f"品名:{doc.metadata['product_name']};类别:{doc.metadata['category']};价格:{doc.metadata['product_price']};",
            metadata=doc.metadata
        ) for doc in raw_docs
    ]

    # 使用RAG管理类初始化向量存储
    rag_manager.init_vectorstore(processed_docs)
    return processed_docs

    # 店铺内可以对具体商品添加个性化文档 目前略


# llm = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=200,
#     timeout=60,
#     max_retries=2,
#     api_key="sk-228111842ebc45789d19c30dba1714e5",
# )
llm = ChatOpenAI(
    base_url="https://api.siliconflow.cn/v1",
    model="Qwen/Qwen2.5-7B-Instruct",  # Qwen/Qwen3-8B(免费)  Qwen/Qwen3-14B   Qwen/Qwen2.5-7B-Instruct(免费)
    temperature=0.7,
    max_tokens=200,
    timeout=30,
    max_retries=2,
    api_key="sk-tdvgqeujlplwxkczbzoyicgadzzdkdgulgdxzzbkcaybyhit",
)


class State(TypedDict):
    session_id: str
    purchased_items: List[str]
    current_user: dict
    shop_info: dict
    messages: List[dict]
    current_intent: Optional[str]
    last_response: Optional[str]
    payment_status: Literal["pending", "success"]
    exit_flag: bool


# ===== 工具定义 =====

@tool("handle_web_search")
def handle_web_search(query: str, dialect: str) -> str:
    """获取实时信息（天气、新闻、本地事件等）"""
    try:
        # 创建定制化搜索API包装器（设置中国大陆参数）
        api_wrapper = DuckDuckGoSearchAPIWrapper(
            region="cn-zh",  # "cn-zh"中国大陆地区   "wt-wt"：全球范围（默认）    "us-en"：美国英语
            time="w",  # 最近一周  可分别设置天d、周w、月m、年y
            max_results=5,  # 限制结果数量
            safesearch="Moderate",  # 适度安全过滤
            backend="auto",  # backend='api' is deprecated
            # timeout=10  # 超时时间
        )

        # 创建搜索工具实例
        search = DuckDuckGoSearchResults(
            api_wrapper=api_wrapper,
            source="news,web"  # 综合新闻和网页结果
        )

        # 带重试机制的搜索（最多1次，指数退避）
        @retry(stop=stop_after_attempt(1), wait=wait_random_exponential(multiplier=0.5, max=3))
        def safe_search():
            print("[调试信息] 检索尝试")
            return search.invoke(query)

        results = safe_search()

        # 调试输出原始结果结构
        print("[调试信息] 原始搜索结果:", results)
        print("━ ━ ━")
        # 信息整合提示模板
        prompt = ChatPromptTemplate.from_template("""
        问题：{query}
        搜索结果：
        {context}
        用{dialect}方言回复，并
        1. 去除语义无关和推广信息
        2. 合并重复内容
        3. 保留数字和关键点
        4. 简短
        """)

        chain = prompt | llm
        msg = chain.invoke({
            "context": "\n\n".join(results),  # 取前两个结果
            "query": query,
            "dialect": dialect,
        }).content
        print("[调试信息] 节点输出：" + msg)
        return msg
    except Exception as e:
        print(f"联网搜索异常 优先考虑Proxy问题：{str(e)}")
        # 降级处理
        msg = random.choice([
            "当前网络不太稳定，您可以问问店内商品信息~",
            "暂时无法获取实时数据，推荐查看今日促销活动",
            "连接超时啦，要不先看看推荐商品？"
        ])
        print(f"[调试信息] 节点输出：{msg}")
        return msg


@tool("handle_chitchat")
def handle_chitchat(region: str, name: str, dialect: str, session_id: str, history: List[dict], input: str) -> str:
    """日常对话处理，使用方言"""
    print(f"[调试] 开始处理闲聊，session_id: {session_id}")

    # 从数据库获取完整对话历史
    # dialogue = dialogue_manager.get_dialogue(session_id)
    # if not dialogue:
    #     print(f"[警告] 未找到对话记录，session_id: {session_id}")
    #     return "当前会话异常，请重新开始"
    #
    # # 获取最近3条消息（含系统消息）
    # db_history = dialogue.get("messages", [])[-5:]
    # print(f"[调试信息] 数据库查询结果条数: {len(db_history)}")
    #
    # # 转换历史记录格式
    # history_str = "\n".join([
    #     f"{msg['type']}：{msg['content']}"
    #     for msg in db_history if msg['type'] in ['human', 'system']
    # ])
    history_str = dialogue_manager.get_latest_messages(session_id, 6)

    print(f"[历史记录] {history_str}")

    # web_info = handle_web_search(f"{input}")
    web_info = handle_web_search.invoke({"query": input, "dialect": dialect})
    print("[调试信息] web输出：" + web_info)

    prompt = ChatPromptTemplate.from_template("""
    你是位于{region}，名为{name}的无人店铺智能助手，用{dialect}方言友好交流。
    对话历史：{history}
    最新消息：顾客：{input}
    注意可能参考的实时信息{web_info}
    简短 自然亲切 口语化 并用表情符号
    """)

    chain = prompt | llm
    msg = chain.invoke({
        "region": region,
        "name": name,
        "dialect": dialect,
        "history": history_str,
        "input": input,
        "web_info": web_info,
    }).content
    print("[调试信息] 节点输出：" + msg)

    # TTS接入测试
    try:
        tts.synthesize_and_play(msg)  # 同步播放语音
    except Exception as e:
        print(f"[TTS相关] 语音播报失败: {str(e)}")

    return msg


@tool("handle_shopping_guide")
def handle_shopping_guide(query: str, remark: str, dialect: str) -> str:
    """提供商品导购和位置指引，支持自然语言查询"""
    docs = rag_manager.retrieve(
        query=query,
        search_type="mmr",
        k=20,
        fetch_k=100,
        lambda_mult=0.7
    )
    print(f"[调试信息] 语义检索docs结果：{[d.metadata['product_name'] for d in docs]}")
    product_info = [
        f"· {doc.metadata['product_name']}（{doc.metadata['category']}）"
        f" 位置：{doc.metadata['position']} 价格：{doc.metadata['product_price']}元"
        for doc in docs
    ]
    prompt = ChatPromptTemplate.from_template("""
    用{dialect}方言回答位置问题：
    商品信息：{product_info}
    用户问题：{input}
    含位置和价格，并附支付指引
    注意与{remark}内容冲突部分需指出
    """)
    msg = (prompt | llm).invoke({
        "product_info": "\n".join(product_info),
        "remark": remark,
        "input": query,
        "dialect": dialect
    }).content

    print("[调试信息] 节点输出：" + msg)
    # TTS接入测试
    try:
        tts.synthesize_and_play(msg)  # 同步播放语音
    except Exception as e:
        print(f"[TTS相关] 语音播报失败: {str(e)}")

    return msg


@tool("handle_recommendation")
def handle_recommendation(query: str, remark: str, preferences: List[str], promotions: List[str], dialect: str) -> str:
    """根据用户偏好和在售商品提供推荐"""
    # 增强语义查询
    semantic_query = f"推荐条件：{query}，用户偏好：{', '.join(preferences)}"

    docs = rag_manager.retrieve(
        query=semantic_query,
        search_type="mmr",
        k=10,
        fetch_k=100,
        lambda_mult=0.8,
        filter={'stock': {'$gt': "0"}}
    )
    print("first docs:", docs)
    # 偏好二次过滤
    pref_embeddings = rag_manager.embedding.embed_documents(preferences)
    doc_embeddings = [rag_manager.embedding.embed_query(doc.page_content) for doc in docs]

    # 计算偏好相似度
    similarity_scores = []
    for doc_emb in doc_embeddings:
        max_score = max(np.dot(pref_emb, doc_emb)
                        for pref_emb in pref_embeddings)
        similarity_scores.append(max_score)

    # 综合排序
    sorted_docs = sorted(
        zip(docs, similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    print("sorted_docs:", [d[0].metadata['product_name'] for d in sorted_docs])
    recommendations = []
    # 在推荐描述中突出匹配特征
    for doc, score in sorted_docs:
        metadata = doc.metadata
        highlight = ""
        if any(pref in metadata['category'] for pref in preferences):
            highlight += "[偏好分类]"
        if any(pref in metadata['product_name'] for pref in preferences):
            highlight += "[偏好商品]"

        rec = f"{highlight} {metadata['product_name']}（{metadata['category']}）\
                ¥{metadata['product_price']} ←{metadata['position']}"

        recommendations.append(rec)

    # for doc in docs[:5]:
    #     name = doc.metadata.get('product_name')
    #     price = doc.metadata.get('product_price', '价格未标')
    #     pos = doc.metadata.get('position', '位置未标')
    #     print(doc.metadata)
    #     rec = f"- {name}（{price}元）{pos}"

    prompt = ChatPromptTemplate.from_template("""
    用{dialect}方言推荐商品：
    推荐列表：{recommendations}
    促销活动：{promotions}
    注意与{remark}内容冲突部分需指出
    """)
    # 说明推荐理由并包含促销信息
    msg = (prompt | llm).invoke({
        "recommendations": "\n".join(recommendations),
        "remark": remark,
        "promotions": "、".join(promotions),
        "dialect": dialect
    }).content

    print("[调试信息] 节点输出：" + msg)
    # TTS接入测试
    try:
        tts.synthesize_and_play(msg)  # 同步播放语音
    except Exception as e:
        print(f"[TTS相关] 语音播报失败: {str(e)}")
    return msg


@tool("handle_human_transfer")
def handle_human_transfer(contact_info: str) -> str:
    """处理转人工服务"""
    msg = f"""已为您转接值班经理（工作时间9:00-21:00）
    等待时间约{random.randint(1, 3)}分钟
    {contact_info}"""

    print("[调试信息] 节点输出：" + msg)
    try:
        tts.synthesize_and_play(msg)  # 同步播放语音
    except Exception as e:
        print(f"[TTS相关] 语音播报失败: {str(e)}")

    return msg


@tool("handle_report")
def handle_report(sales: int, profit: int, hot_products: List[str]) -> str:
    """处理报表"""
    msg = f"""昨日经营简报：
    销售额：¥{sales}
    净利润：¥{profit}
    热销商品：{'、'.join(hot_products)}
    可视化建议：柱状图（销售额趋势）\n饼图（商品占比）"""
    print("[调试信息] 节点输出：" + msg)

    try:
        tts.synthesize_and_play(msg)  # 同步播放语音
    except Exception as e:
        print(f"[TTS相关] 语音播报失败: {str(e)}")

    return msg


@tool("handle_payment")
def handle_payment() -> str:
    """处理支付"""
    msg = "支付成功！请取走商品" if random.random() > 0.2 else "支付失败，请重试"
    print("[调试信息] 节点输出：" + msg)
    return msg


# ===== 状态处理节点 =====


def entry_node(state: State):
    # 生成唯一session_id（示例格式）
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    session_id = f"{state['shop_info']['shop_id']}_{state['current_user'].get('user_id', 'anonymous')}_{timestamp}"

    # [DEBUG] 控制是否使用历史消息
    if state.get("session_id"):
        session_id = state["session_id"]

    # 检查是否已存在该 session_id 的对话
    existing_dialogue = dialogue_manager.get_dialogue(session_id)
    if not existing_dialogue:
        # 仅当不存在时才创建初始记录
        initial_doc = {
            "session_id": session_id,
            "shop_info": {
                "shop_id": state["shop_info"].get("shop_id", "default_shop"),
                "shop_name": state["shop_info"]["name"],
                "region": state["shop_info"]["region"],
                "contact": state["shop_info"].get("contact", "")
            },
            "user_info": {
                "user_id": state["current_user"].get("user_id", "anonymous"),
                "username": state["current_user"].get("username", "Guest"),
                "preferences": state["current_user"].get("preferences", []),
                "dialect": state["current_user"].get("dialect", "普通话")
            },
            "messages": [],
            "status": {
                "payment_status": "pending",
                "exit_flag": False
            },
            "start_time": datetime.utcnow()
        }
        dialogue_manager.create_dialogue(initial_doc)
        print(f"[调试信息] 新建对话记录: {session_id}")
    else:
        print(f"[调试信息] 复用现有对话记录: {session_id}")

    return {
        "session_id": session_id,
        "messages": [],
        "purchased_items": []
    }


def personalized_welcome(state: State):
    semantic_query = f"""
       {', '.join(state["current_user"]["preferences"])}"""

    # 使用统一RAG接口检索
    docs = rag_manager.retrieve(
        query=semantic_query,
        search_type="mmr",
        k=10,
        fetch_k=100,
        lambda_mult=0.2,  # 侧重多样性
        filter={  # 保留原有元数据过滤
            '$or': [
                {'product_name': {'$in': state["current_user"]["preferences"]}},
                {'category': {'$in': state["current_user"]["preferences"]}},
            ]
        }
    )
    print("[调试信息] docs: ", docs)
    recommends = [doc.metadata.get('product_name', doc.page_content.split(' ')[0]) for doc in docs]
    print("[调试信息] recommends: ", recommends)
    prompt = ChatPromptTemplate.from_template("""
    欢迎来到{region}的名为{name}的无人值守店！
    精选推荐：{recommends}
    今日活动：{promotions}
    使用{dialect}方言带表情符号的欢迎词，力求简短""")

    response = (prompt | llm).invoke({
        "region": state["shop_info"]["region"],
        "name": state["shop_info"]["name"],
        "recommends": "、".join(recommends),
        "promotions": "、".join(state["shop_info"]["promotions"]),
        "dialect": state["current_user"]["dialect"]
    })
    print("[调试信息] 欢迎节点输出：" + response.content)

    # 生成欢迎词后立即持久化
    welcome_msg = {
        "message_id": "msg_1",
        "type": "system",
        "content": response.content,
        "timestamp": datetime.utcnow(),
        "metadata": {
            "intent": "welcome",
            "tool_used": "personalized_welcome"
        }
    }

    dialogue_manager.append_message(state["session_id"], welcome_msg)
    print('[调试信息] 欢迎词已持久化')

    # TTS接入测试
    try:
        tts.synthesize_and_play(response.content)  # 同步播放语音
    except Exception as e:
        print(f"[TTS相关] 语音播报失败: {str(e)}")

    return update_state(state, response.content)


def intent_recognition(state: State):
    history = "\n".join([
        f"{'顾客' if isinstance(msg, HumanMessage) else '助手'}：{msg.content}"
        for msg in state["messages"][-5:]
    ])

    prompt = ChatPromptTemplate.from_template("""
    分析消息类型（返回工具名称）：
    历史：{history}
    新消息：{message}

    可选工具：
    - handle_web_search：查询实时联网信息
    - handle_chitchat：日常对话/问候
    - handle_shopping_guide：商品位置/支付问题
    - handle_recommendation：明确要求推荐商品
    - handle_human_transfer：转人工请求
    - handle_report：销售数据查询
    - handle_payment：支付相关
    只需返回工具名称""")

    msg = (prompt | llm).invoke({
        "message": state["messages"][-1].content,
        "history": history
    }).content
    print("[调试信息] 意图节点输出：" + msg)

    # DEBUG 由于很多模型不一定会按照格式输出  要截取
    msg = msg.replace(' ', '')
    msg = msg.replace('\n', '')
    # 如果为思考模型 则去掉两个<think>和中间的文本块
    start = '<think>'
    end = '</think>'
    while start in msg and end in msg:
        start_index = msg.find(start)
        end_index = msg.find(end, start_index) + len(end)
        msg = msg[:start_index] + msg[end_index:]
    print("[调试信息] 意图节点输出：" + msg)

    return {
        "current_intent": msg
    }


# ===== 新增用户输入处理节点 =====
def collect_human_input(state: State):
    if state.get("exit_flag"):
        return {"exit_flag": True}

    try:
        user_input = input("用户：")
    except EOFError:
        return {"exit_flag": True}

    if user_input.lower() in ["退出", "exit", "quit"]:
        return {"exit_flag": True}

    # 保存用户消息到数据库
    user_msg = {
        "message_id": f"msg_{len(state['messages']) + 1}",
        "type": "human",
        "content": user_input,
        "timestamp": datetime.utcnow(),
        "metadata": {}
    }
    dialogue_manager.append_message(state["session_id"], user_msg)
    print('[调试信息] 用户消息已持久化')

    new_message = HumanMessage(content=user_input)
    return {
        "messages": state["messages"] + [new_message],
        "last_response": None
    }


def update_state(state: dict, response: str) -> dict:
    # 更新支付状态
    updates = {}
    if "支付成功" in response:
        updates["payment_status"] = "success"

    # 更新数据库状态
    dialogue_manager.update_status(
        state["session_id"],
        {
            **updates,
            "exit_flag": state.get("exit_flag", False),
            "purchased_items": state.get("purchased_items", [])
        }
    )

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "last_response": response
    }


# ===== 工具调用节点 =====
def create_tool_node(tool_name: str):
    def tool_node(state: State):
        last_msg = state["messages"][-1].content
        # 转换消息为字典格式
        converted_history = [
            {
                "type": "human" if isinstance(msg, HumanMessage) else "system",
                "content": msg.content
            }
            for msg in state["messages"][-5:]
        ]

        params = {
            "session_id": state["session_id"],
            "region": state["shop_info"]["region"],
            "name": state["shop_info"]["name"],
            "dialect": state["current_user"]["dialect"],
            "history": converted_history,  # 使用转换后的历史记录
            "input": last_msg,
            "remark": state["shop_info"]["remark"],  # 引入店铺备注
            "products_docs": state["shop_info"]["products_docs"],
            "preferences": state["current_user"]["preferences"],
            "promotions": state["shop_info"]["promotions"],
            "contact_info": state["shop_info"].get("contact", "400-1234-5678")
        }

        if tool_name == "handle_chitchat":
            response = handle_chitchat.invoke(params)
        elif tool_name == "handle_web_search":
            response = handle_web_search.invoke({
                "query": last_msg,  # 传递用户输入作为查询参数
                "dialect": params["dialect"],
            })
        elif tool_name == "handle_shopping_guide":
            response = handle_shopping_guide.invoke({
                "query": last_msg,
                "remark": params["remark"],
                "products_docs": params["products_docs"],
                "dialect": params["dialect"]
            })
        elif tool_name == "handle_recommendation":
            response = handle_recommendation.invoke({
                "query": last_msg,
                "remark": params["remark"],
                "preferences": params["preferences"],
                "products_docs": params["products_docs"],
                "promotions": params["promotions"],
                "dialect": params["dialect"]
            })
        elif tool_name == "handle_human_transfer":
            response = handle_human_transfer.invoke({
                "contact_info": params["contact_info"]
            })
        elif tool_name == "handle_report":
            response = handle_report.invoke({
                "sales": random.randint(5000, 10000),
                "profit": random.randint(1500, 3000),
                "hot_products": ["椰树椰汁", "农夫山泉"][:random.randint(1, 2)]
            })
        elif tool_name == "handle_payment":
            # TODO 最后告别词也可以持久化
            response = handle_payment.invoke({})
            new_state = update_state(state, response)
            if "成功" in response:
                return {
                    **new_state,
                    "payment_status": "success",
                    "exit_flag": True
                }
            else:
                # 支付失败时添加重试提示
                new_state["messages"].append(AIMessage(
                    content="支付失败，请检查余额或更换支付方式"
                ))
                return new_state

        ai_msg = {
            "message_id": f"msg_{len(state['messages']) + 1}",
            "type": "system",
            "content": response,
            "timestamp": datetime.utcnow(),
            "metadata": {
                "tool_used": tool_name,
                "intent": state.get("current_intent", "")
            }
        }
        dialogue_manager.append_message(state["session_id"], ai_msg)
        print('[调试信息] AI输出已持久化')

        return update_state(state, response)

    return tool_node


# ===== 构建流程图 =====
builder = StateGraph(State)

# ===== 调整流程图结构 =====
# 添加新节点
# 定义节点
builder.add_node("entry", entry_node)
builder.add_node("welcome", personalized_welcome)
builder.add_node("collect_input", collect_human_input)
builder.add_node("intent_recognition", intent_recognition)

# 设置流程
builder.set_entry_point("entry")

# 定义边
# 调整流程边
builder.add_edge("entry", "welcome")
builder.add_edge("welcome", "collect_input")
builder.add_edge("collect_input", "intent_recognition")

# 添加工具节点
tools = ["handle_web_search", "handle_chitchat", "handle_shopping_guide", "handle_recommendation",
         "handle_human_transfer", "handle_report", "handle_payment"]
for tool_name in tools:
    builder.add_node(tool_name, create_tool_node(tool_name))

# 意图路由
builder.add_conditional_edges(
    "intent_recognition",
    lambda state: state["current_intent"],
    {tool_name: tool_name for tool_name in tools}
)

# 支付结果路由
builder.add_conditional_edges(
    "handle_payment",
    lambda state: "goodbye" if state.get("payment_status") == "success" else "handle_payment",
    {"goodbye": "goodbye", "handle_payment": "handle_payment"}
)

# 循环处理
for tool_name in tools:
    if tool_name != "handle_payment":
        builder.add_conditional_edges(
            tool_name,
            lambda state: "collect_input" if not state.get("exit_flag") else END,
            {"collect_input": "collect_input", END: END}
        )

# 结束流程
builder.add_node("goodbye", lambda state: update_state(state, "感谢光临！"))
builder.add_edge("goodbye", END)

# ===== 测试执行 =====
if __name__ == "__main__":
    sample_user = {
        "user_id": "user-123",
        "dialect": "上海",
        "preferences": ["茅台", "中华", ]
    }
    # 商品信息Document修改为读取CSV
    products_docs_list = load_shop_products_csv(
        "data/store_cloud_duty_store_order_goods_day.csv")  # 替换为实际路径
    sample_shop = {
        "shop_id": "shop-123",
        "region": "杭州",
        "name": "西湖123便利店",
        "remark": "门密码888；购物袋免费；成条烟不拆开；店内不堂食；可以接热水；",
        "promotions": ["满30减5", "买条烟送一个打火机", "糖果免费"],
        "products_docs": products_docs_list,  # 使用加载后的增强数据
        "contact": "X老板, 123-1234-5678",
    }

    flow = builder.compile()

    # 绘图功能测试 暂略
    # flow.get_graph()

    initial_state = {
        "current_user": sample_user,
        "shop_info": sample_shop,
        "messages": [],
        "payment_status": "pending",
        "exit_flag": False,
        "session_id": "shop-123_user-123_20250519103421"  # DEBUG 可选的指定session_id 表示在历史记录上追加
    }

    # 运行对话循环
    while True:
        # 执行流程
        events = flow.stream(initial_state)

        # 处理事件流
        final_state = None
        for event in events:
            if "__end__" in event:
                final_state = event["__end__"]
                break

            # 实时显示AI回复
            if "messages" in event and event["messages"]:
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    pass
                    # print(f"\nAI：{last_msg.content}")  因为与调试信息的输出重合

        # 更新初始状态
        if final_state and final_state.get("exit_flag"):
            print("对话结束")
            break
        initial_state = final_state or initial_state

#   TODO 自定义的持久化记忆功能  包括自定义sessionID和记录  OKK
#   TODO 考虑TTS 语音合成 阿里的CosyVoice  OKK  若要排除 直接搜索TTS相关即可
#   TODO 周边商品推荐功能

#   TODO 考虑ASR语音识别  目前考虑阿里paraformer-realtime-v2支持的语种：中文（含普通话和各种方言）、英文、日语、韩语、德语、法语、俄语
#   支持的中文方言：上海话、吴语、闽南语、东北话、甘肃话、贵州话、河南话、湖北话、湖南话、江西话、宁夏话、山西话、陕西话、山东话、四川话、天津话、云南话、粤语
#   核心能力：
#   支持标点符号预测
#   支持逆文本正则化（ITN）
#   指定语种：通过language_hints参数能够指定待识别语种，提升识别效果
#   支持定制热词
