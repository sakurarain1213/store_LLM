import os
import random

import asyncio
# 在主流程中重用现有事件循环
# loop = asyncio.get_event_loop()
# 涉及BUG: web_search相关tool

from typing import List, Dict, Literal, Optional
from typing_extensions import TypedDict

import numpy as np
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
import graphviz  # 添加graphviz导入  仅用于调试绘图 可以删除

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_deepseek import ChatDeepSeek

from v_12_web import fetch_search_results

# from v_12_ASR import ASRHandler  # 添加ASR处理器导入

from v_12_dialogues_memory import DialogueManager

# from v_12_TTS import TTSHandler

# pip install -U duckduckgo-search -i https://pypi.tuna.tsinghua.edu.cn/simple

"""
v11: 新增ASR语音识别支持
    - 支持中文(含方言)、英文等多语种识别
    - 集成实时语音输入功能
    - 优化语音交互流程
"""

from langchain_community.document_loaders.csv_loader import CSVLoader
from typing import List

from datetime import datetime


# TODO UserInfo似乎没用上 以及State类的使用问题 严格定义一下使用类来传递参数
class UserInfo(TypedDict):
    user_id: str
    username: str
    dialect: str
    gender: Literal["男", "女", "其他"]  # 新增性别字段
    preferences: List[str]
    # 后续新增属性在此添加


class ShopInfo(TypedDict):
    shop_id: str
    name: str
    region: str
    contact: str
    remark: str
    promotions: List[str]
    products_docs: List[Document]
    # 后续新增属性在此添加


llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=200,
    timeout=60,
    max_retries=2,
    api_key="sk-228111842ebc45789d19c30dba1714e5",
)

# llm = ChatOpenAI(
#     base_url="https://api.siliconflow.cn/v1",
#     model="Qwen/Qwen2.5-7B-Instruct",  # Qwen/Qwen3-8B(免费)  Qwen/Qwen3-14B   Qwen/Qwen2.5-7B-Instruct(免费)  THUDM/GLM-4-9B-0414( 智谱免费，但幻觉强) 
#     temperature=0.7,
#     max_tokens=200,
#     timeout=30,
#     max_retries=2,
#     api_key="sk-tdvgqeujlplwxkczbzoyicgadzzdkdgulgdxzzbkcaybyhit",
# )

# ===== 初始化对话管理器 =====
CONN_STR = "mongodb://mongodb:DTNxf5TYcZWFKDYY@116.62.149.204:27017/"
dialogue_manager = DialogueManager(CONN_STR)


# ===== 引入音频生成 =====
# tts = TTSHandler()


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


class State(TypedDict):
    session_id: str
    purchased_items: List[str]
    current_user: UserInfo  # 使用强类型定义
    shop_info: ShopInfo  # 修改为强类型
    messages: List[dict]
    current_intent: Optional[str]
    last_response: Optional[str]
    payment_status: Literal["pending", "success"]
    exit_flag: bool


class NodeManager:
    """集中管理所有节点和工具的定义与注册"""

    def __init__(self):
        self.tools = {}
        self.nodes = {}
        self.edges = []
        self.conditional_edges = []

    def register_tool(self, tool_func):
        """注册工具函数并创建对应的节点"""
        tool_name = tool_func.__name__
        self.tools[tool_name] = tool_func

        # 创建工具节点
        def tool_node(state: State):
            last_msg = state["messages"][-1].content
            converted_history = [
                {
                    "type": "human" if isinstance(msg, HumanMessage) else "system",
                    "content": msg.content
                }
                for msg in state["messages"][-5:]
            ]

            # 使用inspect获取函数参数
            import inspect
            sig = inspect.signature(tool_func)
            params = {}

            # 根据参数名从state中获取对应的值
            for param_name in sig.parameters:
                if param_name == "query":
                    params[param_name] = last_msg
                elif param_name == "history":
                    params[param_name] = converted_history
                elif param_name == "session_id":
                    params[param_name] = state["session_id"]
                elif param_name == "input":
                    params[param_name] = last_msg
                elif param_name in state["current_user"]:
                    params[param_name] = state["current_user"][param_name]
                elif param_name in state["shop_info"]:
                    params[param_name] = state["shop_info"][param_name]
                elif param_name == "sales":
                    params[param_name] = random.randint(5000, 10000)
                elif param_name == "profit":
                    params[param_name] = random.randint(1500, 3000)
                elif param_name == "hot_products":
                    params[param_name] = ["椰树椰汁", "农夫山泉"][:random.randint(1, 2)]

            try:
                # 直接调用工具函数
                response = tool_func(**params)

                # 处理支付特殊逻辑
                if tool_name == "handle_payment":
                    new_state = update_state(state, response)
                    if "成功" in response:
                        return {
                            **new_state,
                            "payment_status": "success",
                            "exit_flag": True
                        }
                    else:
                        new_state["messages"].append(AIMessage(
                            content="支付失败，请检查余额或更换支付方式"
                        ))
                        return new_state

            except Exception as e:
                error_msg = f"工具调用异常: {str(e)}"
                print(f"[错误] {error_msg}")
                response = f"抱歉，处理您的请求时出现了问题，请稍后再试. "

            # 持久化消息
            ai_msg = {
                "message_id": f"msg_{len(state['messages']) + 1}",
                "type": "system",
                "content": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {
                    "tool_used": tool_name,
                    "intent": state.get("current_intent", "")
                }
            }
            dialogue_manager.append_message(state["session_id"], ai_msg)
            print('[调试信息] AI输出已持久化')

            return update_state(state, response)

        self.nodes[tool_name] = tool_node
        return tool_func

    def register_node(self, name, node_func):
        """注册普通节点"""
        self.nodes[name] = node_func

    def add_edge(self, from_node, to_node, condition=None):
        """添加边或条件边"""
        if condition:
            self.conditional_edges.append((from_node, to_node, condition))
        else:
            self.edges.append((from_node, to_node))

    def add_conditional_edges(self, from_node, condition, edge_map):
        """添加条件边映射"""
        # 确保edge_map是一个字典
        if isinstance(edge_map, dict):
            self.conditional_edges.append((from_node, condition, edge_map))
        else:
            # 如果是字符串或其他类型，转换为字典
            edge_dict = {str(k): k for k in edge_map} if isinstance(edge_map, (list, tuple)) else {
                str(edge_map): edge_map}
            self.conditional_edges.append((from_node, condition, edge_dict))

    def build_graph(self, builder: StateGraph):
        """构建完整的图结构"""
        # 添加所有节点
        for name, node_func in self.nodes.items():
            builder.add_node(name, node_func)

        # 添加普通边
        for from_node, to_node in self.edges:
            builder.add_edge(from_node, to_node)

        # 添加条件边
        for from_node, condition, edge_map in self.conditional_edges:
            builder.add_conditional_edges(from_node, condition, edge_map)


# 创建节点管理器实例
node_manager = NodeManager()


# 重新定义工具装饰器
def tool(name):
    def decorator(func):
        return node_manager.register_tool(func)

    return decorator


# 重新定义工具函数
@tool("handle_web_search")
def handle_web_search(query: str, dialect: str, gender: str, username: str) -> str:
    """获取实时信息( 天气、新闻、本地事件等) """
    # 配置服务 URL
    service_url = "http://appbuilder.baidu.com/v2/ai_search/mcp/sse?" + \
                  "api_key=Bearer+bce-v3/ALTAK-lp91La4lRwuifo4dSNURU/70cb3ab0e2e87e267f6840f76e9fd052adfca877"

    # 执行异步搜索并返回结果，使用新的事件循环避免冲突
    try:
        # 创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        search_results = loop.run_until_complete(fetch_search_results(query, service_url))
        loop.close()
    except Exception as e:
        print(f"[错误] 搜索异常: {str(e)}")
        search_results = ["无法获取实时信息"]

    # 调试信息
    print(f"[调试信息] 搜索结果: {search_results}")

    # 信息整合提示模板
    prompt = ChatPromptTemplate.from_template("""
        问题: {query}
        搜索结果: 
        {context}
        用{dialect}方言回复，涵盖{username}和{gender}并
        1. 去除语义无关和推广信息
        2. 合并重复内容
        3. 保留数字和关键点
        4. 简短
    """)

    # 链式调用
    chain = prompt | llm
    msg = chain.invoke({
        "context": "\n\n".join(search_results) if isinstance(search_results, list) else search_results,
        "query": query,
        "dialect": dialect,
        "username": username,
        "gender": gender,
    }).content

    # 调试信息
    print("[调试信息] 节点输出: " + msg)
    return msg


@tool("handle_chitchat")
def handle_chitchat(region: str, name: str, dialect: str, session_id: str, history: List[dict], input: str, gender: str,
                    username: str) -> str:
    """日常对话处理，使用方言"""
    print(f"[调试] 开始处理闲聊，session_id: {session_id}")

    # 获取历史消息
    history_str = dialogue_manager.get_latest_messages(session_id, 6)
    # print(f"[调试信息] 历史记录{history_str}")

    # 调用web搜索并获取结果，但使用新的事件循环
    try:
        # 避免使用其他函数的工具调用方式，直接实现web搜索逻辑
        service_url = "http://appbuilder.baidu.com/v2/ai_search/mcp/sse?" + \
                      "api_key=Bearer+bce-v3/ALTAK-lp91La4lRwuifo4dSNURU/70cb3ab0e2e87e267f6840f76e9fd052adfca877"

        # 为这次调用创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        web_info = loop.run_until_complete(fetch_search_results(input, service_url))
        loop.close()

        print("[调试信息] web信息\n", web_info)
    except Exception as e:
        print(f"[错误] web搜索异常: {str(e)}")
        web_info = "无法获取相关信息"

    # 获取系统时间
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 创建提示模板，明确列出所有变量，确保没有'type'字段冲突
    prompt_template = """
    你是位于{region},名为{name}的无人店铺智能助手，使用{dialect}方言. 
    系统时间为{now}.
    用户信息: {username}, 性别{gender}
    对话历史: {history}
    最新消息: {input}
    可考的网络信息: {web_info}
    简短 自然亲切 口语化 使用表情符号
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | llm
    msg = chain.invoke({
        "region": region,
        "name": name,
        "dialect": dialect,
        "username": username,
        "gender": gender,
        "history": history_str,
        "input": input,
        "web_info": web_info,
        "now": now
    }).content
    print("[调试信息] 节点输出: " + msg)

    # TTS接入测试已注释
    # try:
    #     tts.synthesize_and_play(msg)  # 同步播放语音
    # except Exception as e:
    #     print(f"[TTS相关] 语音播报失败: {str(e)}")

    return msg


@tool("handle_shopping_guide")
def handle_shopping_guide(query: str, remark: str, dialect: str, gender: str, username: str, ) -> str:
    """提供商品导购和位置指引，支持自然语言查询"""

    # 固定话术映射
    fixed_responses = {
        "扫码": "您好，请在前往收银台自助扫码买单. 使用桌面的扫码枪对准商品条形码，发出'滴'的一声即可，收银屏上会出现对应商品结算信息，并使用微信或者支付宝扫码付款. ",
        "买单": "您好，请在前往收银台自助扫码买单. 使用桌面的扫码枪对准商品条形码，发出'滴'的一声即可，收银屏上会出现对应商品结算信息，并使用微信或者支付宝扫码付款. ",
        "付钱": "您好，请打开微信或者支付宝二维码，将手机屏幕放置于扫码枪前，完成扫码支付操作. ",
        "支付": "您好，请打开微信或者支付宝二维码，将手机屏幕放置于扫码枪前，完成扫码支付操作. ",
        "退款": "您好，请将不购买的商品放回货架原位. 若需退款请次日联系门店老板描述情况，经老板同意可进行退款. ",
        "离店": "您好，请扫描门上的二维码出门. 欢迎下次光临，再见. ",
        "出门": "您好，请扫描门上的二维码出门. 欢迎下次光临，再见. ",
        "开门": "您好，请扫描门上的二维码出门. 欢迎下次光临，再见. ",
        "转人工": "您好，每个门店后台都有专属的值守人员实时看管，值守人员将通过实时监控画面了解到门店情况. 已为您转接人工. "
    }

    # 检查是否匹配固定话术
    for key, response in fixed_responses.items():
        if key in query:
            # 使用方言和用户信息个性化固定话术
            prompt = ChatPromptTemplate.from_template("""
            用{dialect}方言回答: 
            {response}
            注意要符合{remark}内容，使用表情
            用户信息: {username},性别{gender}
            """)

            msg = (prompt | llm).invoke({
                "dialect": dialect,
                "username": username,
                "gender": gender,
                "remark": remark
            }).content

            print("[调试信息] 节点输出: " + msg)
            return msg

    # 如果不是固定话术，先通过LLM标准化用户描述
    standardization_prompt = ChatPromptTemplate.from_template("""
    你是一个商品描述标准化助手。请将用户的商品描述转换为标准化的商品名称和特征描述。
    用户描述: {query}
    请按以下格式输出:
    1. 标准商品名称: [将代称、口语化、网络流行语转换为标准商品名称，例如：华子=中华香烟，快乐水=可乐]
    2. 商品特征: [提取用户提到的商品特征，如品牌、规格、口味等]
    3. 位置相关: [提取用户提到的位置相关描述]
    4. 价格相关: [提取用户提到的价格相关描述]
    注意:
    - 如果用户没有明确提到商品，保持描述的原意
    - 如果用户使用了代称，转换为可能的商品名称
    - 输出格式必须有商品名称，其它字段可以为空
    """)

    standardized_query = (standardization_prompt | llm).invoke({
        "query": query
    }).content

    print(f"[调试信息] 标准化后的查询: {standardized_query}")

    # 使用标准化后的查询进行语义检索
    docs = rag_manager.retrieve(
        query=standardized_query,
        search_type="mmr",
        k=20,
        fetch_k=100,
        lambda_mult=0.7
    )
    print(f"[调试信息] 语义检索docs结果: {[d.metadata['product_name'] for d in docs]}")
    product_info = [
        f"· {doc.metadata['product_name']}( {doc.metadata['category']}) "
        f" 位置: {doc.metadata['position']} 价格: {doc.metadata['product_price']}元"
        for doc in docs
    ]
    prompt = ChatPromptTemplate.from_template("""
    用{dialect}方言回答用户问题: {input}。
    注意商品信息回答包含商品名称，位置和价格。
    可能的商品信息: {product_info},如果涉及则详细描述
    可能需要附{remark}内容,使用表情。
    用户信息: {username}, 性别{gender}。
    """)
    msg = (prompt | llm).invoke({
        "product_info": "\n".join(product_info),
        "remark": remark,
        "input": query,
        "dialect": dialect,
        "username": username,
        "gender": gender,
    }).content

    print("[调试信息] 节点输出: " + msg)
    # TTS接入测试已注释
    # try:
    #     tts.synthesize_and_play(msg)  # 同步播放语音
    # except Exception as e:
    #     print(f"[TTS相关] 语音播报失败: {str(e)}")

    return msg


@tool("handle_recommendation")
def handle_recommendation(query: str, remark: str, preferences: List[str], promotions: List[str], dialect: str,
                          gender: str, username: str, ) -> str:
    """根据用户偏好和在售商品提供推荐"""
    # 增强语义查询
    semantic_query = f"推荐条件: {query},用户偏好: {', '.join(preferences)}"

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

        rec = f"{highlight} {metadata['product_name']} ( {metadata['category']} ) \
                ¥{metadata['product_price']} ←{metadata['position']}"

        recommendations.append(rec)

    # for doc in docs[:5]:
    #     name = doc.metadata.get('product_name')
    #     price = doc.metadata.get('product_price', '价格未标')
    #     pos = doc.metadata.get('position', '位置未标')
    #     print(doc.metadata)
    #     rec = f"- {name}( {price}元) {pos}"

    prompt = ChatPromptTemplate.from_template("""
    用{dialect}方言推荐商品: 
    用户信息: {username},性别{gender}.
    推荐列表: {recommendations}.
    促销活动: {promotions}.
    使用表情，注意与{remark}内容冲突部分需指出. 
    """)
    # 说明推荐理由并包含促销信息
    msg = (prompt | llm).invoke({
        "recommendations": "\n".join(recommendations),
        "remark": remark,
        "promotions": ",".join(promotions),
        "dialect": dialect,
        "username": username,
        "gender": gender,
    }).content

    print("[调试信息] 节点输出: " + msg)
    # TTS接入测试已注释
    # try:
    #     tts.synthesize_and_play(msg)  # 同步播放语音
    # except Exception as e:
    #     print(f"[TTS相关] 语音播报失败: {str(e)}")
    return msg


@tool("handle_human_transfer")
def handle_human_transfer(contact: str) -> str:
    """处理转人工服务"""
    msg = f"""已为您转接值班经理( 工作时间9:00-21:00) 
    等待时间约{random.randint(1, 3)}分钟
    {contact}"""

    print("[调试信息] 节点输出: " + msg)
    # TTS接入测试已注释
    # try:
    #     tts.synthesize_and_play(msg)  # 同步播放语音
    # except Exception as e:
    #     print(f"[TTS相关] 语音播报失败: {str(e)}")

    return msg


@tool("handle_report")
def handle_report(sales: int, profit: int, hot_products: List[str]) -> str:
    """处理报表"""
    msg = f"""昨日经营简报: 
    销售额: ¥{sales}
    净利润: ¥{profit}
    热销商品: {'、'.join(hot_products)}
    可视化建议: 柱状图( 销售额趋势) \n饼图( 商品占比) """
    print("[调试信息] 节点输出: " + msg)

    # TTS接入测试已注释
    # try:
    #     tts.synthesize_and_play(msg)  # 同步播放语音
    # except Exception as e:
    #     print(f"[TTS相关] 语音播报失败: {str(e)}")

    return msg


@tool("handle_payment")
def handle_payment() -> str:
    """处理支付"""
    msg = "支付成功！请取走商品" if random.random() > 0.2 else "支付失败，请重试"
    print("[调试信息] 节点输出: " + msg)
    return msg


# ===== 状态处理节点 =====


def update_user_info(existing_user_info: dict, new_user_info: dict) -> dict:
    """更新用户信息，合并新旧偏好"""
    updated_info = existing_user_info.copy()

    # 更新基本信息( 包括所有可能的字段) 
    for key in ['username', 'gender', 'dialect', 'user_id']:
        if key in new_user_info and new_user_info[key]:
            updated_info[key] = new_user_info[key]

    # 合并偏好列表，去重
    existing_prefs = set(existing_user_info.get('preferences', []))
    new_prefs = set(new_user_info.get('preferences', []))
    updated_info['preferences'] = list(existing_prefs.union(new_prefs))

    return updated_info


def entry_node(state: State):
    # 生成唯一session_id( 示例格式) 
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
                "gender": state["current_user"]["gender"],
                "preferences": state["current_user"].get("preferences", []),
                "dialect": state["current_user"].get("dialect", "普通话")
            },
            "messages": [],
            "status": {
                "payment_status": "pending",
                "exit_flag": False
            },
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        dialogue_manager.create_dialogue(initial_doc)
        print(f"[调试信息] 新建对话记录: {session_id}")
    else:
        print(f"[调试信息] 复用现有对话记录: {session_id}")
        # 更新用户信息
        existing_user_info = existing_dialogue.get("user_info", {})
        updated_user_info = update_user_info(existing_user_info, state["current_user"])

        # 更新数据库中的用户信息
        dialogue_manager.update_user_info(session_id, updated_user_info)
        print(f"[调试信息] 已更新用户信息: {updated_user_info}")

        # 更新当前状态中的用户信息
        state["current_user"] = updated_user_info

    # 返回完整的状态更新，包括更新后的用户信息
    return {
        "session_id": session_id,
        "messages": [],
        "purchased_items": [],
        "current_user": state["current_user"],  # 确保返回更新后的用户信息
        "shop_info": state["shop_info"],  # 保持店铺信息不变
        "payment_status": "pending",
        "exit_flag": False
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

    # 获取系统时间
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt = ChatPromptTemplate.from_template("""
    用户姓名{username},性别{gender}.
    系统时间为{now}.
    精选推荐: {recommends}
    今日活动: {promotions}
    使用{dialect}方言带表情符号的欢迎词，
    热情欢迎用户来到{region}的名为{name}的无人值守店。           
    """)

    response = (prompt | llm).invoke({
        "now": now,
        "region": state["shop_info"]["region"],
        "name": state["shop_info"]["name"],
        "gender": state["current_user"]["gender"],
        "username": state["current_user"]["username"],
        "recommends": ",".join(recommends),
        "promotions": ",".join(state["shop_info"]["promotions"]),
        "dialect": state["current_user"]["dialect"]
    })
    print("[调试信息] 欢迎节点输出: " + response.content)

    # 生成欢迎词后立即持久化
    welcome_msg = {
        "message_id": "msg_1",
        "type": "system",
        "content": response.content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": {
            "intent": "welcome",
            "tool_used": "personalized_welcome"
        }
    }

    dialogue_manager.append_message(state["session_id"], welcome_msg)
    print('[调试信息] 欢迎词已持久化')

    # TTS接入测试已注释
    # try:
    #     tts.synthesize_and_play(response.content)  # 同步播放语音
    # except Exception as e:
    #     print(f"[TTS相关] 语音播报失败: {str(e)}")

    return update_state(state, response.content)


def intent_recognition(state: State):
    history = "\n".join([
        f"{'顾客' if isinstance(msg, HumanMessage) else '助手'}: {msg.content}"
        for msg in state["messages"][-5:]
    ])

    prompt = ChatPromptTemplate.from_template("""
    分析消息类型( 返回工具名称) : 
    历史: {history}
    新消息: {message}
    注意非必要不使用handle_web_search，且只返回工具名称
    可选工具: 
    - handle_web_search: 查询实时联网信息
    - handle_chitchat: 日常对话/问候
    - handle_shopping_guide: 商品位置/支付问题
    - handle_recommendation: 明确要求推荐商品
    - handle_human_transfer: 转人工请求
    - handle_report: 销售数据查询
    - handle_payment: 支付相关
    - goodbye: 告别/退出/离开/结束请求
    只需返回工具名称""")

    msg = (prompt | llm).invoke({
        "message": state["messages"][-1].content,
        "history": history
    }).content
    # print("[调试信息] 意图节点输出: " + msg)

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

    print("[调试信息] 意图节点输出: " + msg)

    # 如果识别到告别意图，设置退出标志
    if msg == "goodbye":
        return {
            "current_intent": msg,
            "exit_flag": True
        }

    return {
        "current_intent": msg
    }


# ===== 新增用户输入处理节点 v11新增ASR支持=====
def collect_human_input(state: State):
    if state.get("exit_flag"):
        return {"exit_flag": True}

    try:
        user_input = input("用户(voice启动语音 暂时禁用): ")

        # 检查是否进入语音输入模式
        # if user_input.lower() == "voice":
        #     print("已进入语音输入模式")
        #     # 初始化ASR处理器
        #     asr = ASRHandler(
        #         input_source='mic',
        #         language_hints=["zh", "en"],
        #         disfluency_removal_enabled=True
        #     )

        #     try:
        #         # 启动ASR并等待用户按回车结束
        #         asr.start()

        #         # 停止ASR并获取结果
        #         asr.stop()
        #         user_input = asr.get_final_result()
        #         if user_input.strip():  # 只在有识别结果时显示
        #             print(f"识别结果: {user_input}")
        #         else:
        #             print("未检测到语音输入")
        #             return {"messages": state["messages"]}

        #     except Exception as e:
        #         print(f"语音识别错误: {str(e)}")
        #         if asr:
        #             asr.stop()
        #         return {"messages": state["messages"]}

        # 检查退出命令
        if user_input.lower() in ["退出", "exit", "quit"]:
            return {"exit_flag": True}

        # 保存用户消息到数据库
        user_msg = {
            "message_id": f"msg_{len(state['messages']) + 1}",
            "type": "human",
            "content": user_input,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": {
                "input_mode": "voice" if user_input.lower() == "voice" else "console"
            }
        }
        dialogue_manager.append_message(state["session_id"], user_msg)
        print('[调试信息] 用户消息已持久化')

        new_message = HumanMessage(content=user_input)
        return {
            "messages": state["messages"] + [new_message],
            "last_response": None
        }
    except EOFError:
        return {"exit_flag": True}


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


# 注册基础节点
node_manager.register_node("entry", entry_node)
node_manager.register_node("welcome", personalized_welcome)
node_manager.register_node("collect_input", collect_human_input)
node_manager.register_node("intent_recognition", intent_recognition)


def personalized_goodbye(state: State):
    # 硬编码购买商品 实际使用时接入支付接口即可
    state["purchased_items"] = [
        "农夫山泉矿泉水",
        "可口可乐",
        "中华香烟",
    ]

    """生成个性化的告别词"""
    prompt = ChatPromptTemplate.from_template("""
    用户信息: {username},性别{gender},使用{dialect}方言
    店铺信息: {region}的{name}店
    购买商品: {purchased_items}
    使用{dialect}方言生成告别词，要求: 
    1. 简短自然
    2. 提及用户购买的商品
    3. 使用表情符号
    4. 欢迎再次光临
    """)
    print("[调试信息] 购买记录", state["purchased_items"])
    response = (prompt | llm).invoke({
        "username": state["current_user"]["username"],
        "gender": state["current_user"]["gender"],
        "dialect": state["current_user"]["dialect"],
        "region": state["shop_info"]["region"],
        "name": state["shop_info"]["name"],
        "purchased_items": ",".join(state["purchased_items"]) if state["purchased_items"] else "未购买商品"
    })

    print("[调试信息] 告别节点输出: " + response.content)

    # 持久化告别消息
    goodbye_msg = {
        "message_id": f"msg_{len(state['messages']) + 1}",
        "type": "system",
        "content": response.content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": {
            "intent": "goodbye",
            "tool_used": "personalized_goodbye"
        }
    }
    dialogue_manager.append_message(state["session_id"], goodbye_msg)
    print('[调试信息] 告别词已持久化')

    # TTS接入测试已注释
    # try:
    #     tts.synthesize_and_play(response.content)  # 同步播放语音
    # except Exception as e:
    #     print(f"[TTS相关] 语音播报失败: {str(e)}")

    return update_state(state, response.content)


node_manager.register_node("goodbye", personalized_goodbye)

# 设置入口点
node_manager.add_edge("entry", "welcome")
node_manager.add_edge("welcome", "collect_input")
node_manager.add_edge("collect_input", "intent_recognition")

# 添加工具节点路由
tools = ["handle_web_search", "handle_chitchat", "handle_shopping_guide", "handle_recommendation",
         "handle_human_transfer", "handle_report", "handle_payment", "goodbye"]

# 意图路由
node_manager.add_conditional_edges(
    "intent_recognition",
    lambda state: state["current_intent"],
    {tool_name: tool_name for tool_name in tools}
)

# 支付结果路由
node_manager.add_conditional_edges(
    "handle_payment",
    lambda state: "goodbye" if state.get("payment_status") == "success" else "handle_payment",
    {"goodbye": "goodbye", "handle_payment": "handle_payment"}
)

# 循环处理
for tool_name in tools:
    if tool_name not in ["handle_payment", "goodbye"]:
        node_manager.add_conditional_edges(
            tool_name,
            lambda state: "collect_input" if not state.get("exit_flag") else END,
            {"collect_input": "collect_input", END: END}
        )

# 结束流程
node_manager.add_edge("goodbye", END)

# ===== 测试执行 =====
if __name__ == "__main__":
    sample_user = {
        "user_id": "user-123",
        "dialect": "粤语",
        "gender": "男",
        "preferences": ["可乐", "香烟"],
        "username": "张三"
    }

    products_docs_list = load_shop_products_csv(
        "data/store_cloud_duty_store_order_goods_day.csv")
    sample_shop = ShopInfo(
        shop_id="shop-123",
        name="西湖123便利店",
        region="杭州",
        contact="X老板, 123-1234-5678",
        remark="购物袋免费；成条烟不拆开；店内不堂食；可以接热水；",
        promotions=["满30减5", "送口香糖"],
        products_docs=products_docs_list
    )

    # 使用新的图构建方式
    builder = StateGraph(State)
    builder.set_entry_point("entry")
    node_manager.build_graph(builder)
    flow = builder.compile()

    # 绘图功能测试  调试完毕后删除  显示图结构
    graph = flow.get_graph()
    print("\n=== 绘制langgraph图结构 ===")
    # 生成图
    dot = graphviz.Digraph(comment='对话流程图', format='png')
    dot.attr(rankdir='LR')  # 从左到右的布局
    # 添加节点
    for node in graph.nodes.values():
        # 设置节点样式
        if node.id == '__start__':
            dot.node(node.id, '开始', shape='circle', style='filled', fillcolor='green')
        elif node.id == '__end__':
            dot.node(node.id, '结束', shape='circle', style='filled', fillcolor='red')
        else:
            dot.node(node.id, node.id, shape='box')
    # 添加边
    for edge in graph.edges:
        # 设置边的样式
        if edge.conditional:
            dot.edge(edge.source, edge.target, style='dashed')
        else:
            dot.edge(edge.source, edge.target)
    # 保存为PNG文件到当前目录
    output_path = os.path.join(os.getcwd(), 'conversation_flow')
    dot.render(output_path, view=False, cleanup=True)
    print(f"流程图已保存为 {output_path}.png")
    # 结束绘图  调试后可删除

    initial_state = {
        "current_user": sample_user,
        "shop_info": sample_shop,
        "messages": [],
        "payment_status": "pending",
        "exit_flag": False,
        "session_id": "shop-123_user-123_20250529114143"  # DEBUG 可选的指定session_id 表示在历史记录上追加
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

            # 实时显示AI回复和节点状态
            if "messages" in event and event["messages"]:
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    pass
                    # print(f"\nAI: {last_msg.content}")  因为与调试信息的输出重合

        # 更新初始状态
        if final_state and final_state.get("exit_flag"):
            print("对话结束")
            break
        initial_state = final_state or initial_state

#   TODO 周边商品推荐功能

#   ASR语音识别  目前考虑阿里paraformer-realtime-v2支持的语种: 中文( 含普通话和各种方言) 、英文、日语、韩语、德语、法语、俄语
#   支持的中文方言: 上海话、吴语、闽南语、东北话、甘肃话、贵州话、河南话、湖北话、湖南话、江西话、宁夏话、山西话、陕西话、山东话、四川话、天津话、云南话、粤语
#   核心能力: 
#   支持标点符号预测
#   支持逆文本正则化( ITN) 
#   指定语种: 通过language_hints参数能够指定待识别语种，提升识别效果
#   支持定制热词
