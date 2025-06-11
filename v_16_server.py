
# 全局调试控制
DEBUG_MODE = True
# 全局混合流控制 开启则启用音频  否则仅文本
USE_MIXED_STREAM = False  # 控制是否使用混合流输出

'''
TODO 虽然不影响结果 但是 修一下联网功能的异步报错 尽量不动本身 

v_16  文本流混合音频流的输出  OKK 
首文响应时间: 0.39s | 首音响应时间: 1.89s | 混合流总传输: 47.13s

A    ASR+打断 在前端实现即可 
B    后端接入预备音频问题 本地化生成+语义树模型+优先输出+格式适配即可
C    暂不做 各种文档 配置项整理


新建对话（流式响应）
POST http://localhost:8000/chat/new/stream
Content-Type: application/json
入参
{
    "user_info": {
      "user_id": "user-123",
      "dialect": "上海",
      "gender": "男",
      "preferences": ["可乐", "香烟"],
      "username": "张三"
    },
    "shop_info": {
      "shop_id":"shop-123",
      "name": "西湖123便利店",
      "region": "杭州",
      "contact": "X老板, 123-1234-5678",
      "remark": "购物袋免费;成条烟不拆开;店内不堂食;可以接热水;",
      "promotions": ["满30减5", "送口香糖"]
    }
  }
出参（SSE格式混合流）
data: {"type": "text", "session_id": "shop-123_user-123_20250611160645", "status": "success"}

data: {"type": "text", "response": "\n", "status": "success"}

data: {"type": "text", "response": "\u54ce", "status": "success"}

data: {"type": "text", "response": "\u5440", "status": "success"}

data: {"type": "text", "response": "\uff0c", "status": "success"}

{
    "type": "audio_chunk",
    "sentence_id": 11,
    "text": "快来挑挑看看，有什么需要帮忙的尽管说哦！",
    "data": "base64格式的字节码 较长 10KB级 交给前端处理即可",
}

继续对话（流式响应）
POST http://localhost:8000/chat/continue/stream
Content-Type: application/json
入参
{
    "session_id": "shop-123_user-123_20250603111110",
    "message": "结账"
}
出参（SSE格式混合流）
同上

1. 接口使用 Server-Sent Events (SSE) 格式返回响应
2. 每个响应都以 "data: " 开头，以 "\n\n" 结尾


'''

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import asyncio
from datetime import datetime
import time
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from v_16 import *
from v_16_web import fetch_search_results
from appbuilder.mcp_server.client import MCPClient
import random # 仅仅用于转人工和报表节点的测试  若后续接入接口则不需要随机数


from v_16_TTS import StreamingTTS  # 导入TTS模块
from v_16_mixed_stream import (  # 导入混合流模块
    init_audio_system, 
    create_improved_mixed_stream,
    cancel_audio_generation,
)
# 初始化TTS客户端
tts_client = StreamingTTS()
# 初始化音频系统
init_audio_system(tts_client)



def debug_print(node_name: str, content: str):
    """统一的调试打印函数"""
    if DEBUG_MODE:
        print(f"[{node_name}] {content}")

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
llm = ChatOpenAI(
    base_url="https://api.siliconflow.cn/v1",
    model="THUDM/glm-4-9b-chat",
    temperature=0.4,
    max_tokens=1000, # 核心  可以更大 RAG时
    timeout=10,
    max_retries=1,
    api_key="sk-tdvgqeujlplwxkczbzoyicgadzzdkdgulgdxzzbkcaybyhit",
    streaming=True,
)

CONN_STR = "mongodb://mongodb:DTNxf5TYcZWFKDYY@116.62.149.204:27017/"
dialogue_manager = DialogueManager(CONN_STR)
node_manager = NodeManager()
rag_manager = RAGManager()  # 初始化 RAG 管理器

# Request models
class UserInfo(BaseModel):
    user_id: str
    dialect: str
    gender: str
    preferences: List[str]
    username: str

class ShopInfo(BaseModel):
    shop_id: str
    name: str
    region: str
    contact: str
    remark: str
    promotions: List[str]

class NewChatRequest(BaseModel):
    user_info: UserInfo
    shop_info: ShopInfo

class ContinueChatRequest(BaseModel):
    session_id: str
    message: str

# 混合流只改了这一个功能 10:42
async def personalized_welcome_stream(state: State):
    """Stream personalized welcome message with audio"""
    start_time = time.time()
    semantic_query = f"""
       {', '.join(state["current_user"]["preferences"])}"""

    # Get system time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt = ChatPromptTemplate.from_template("""
    用户姓名{username},性别{gender}.
    系统时间为{now}.
    精选推荐: {recommends}
    今日活动: {promotions}
    使用{dialect}方言带表情符号的欢迎词，
    热情欢迎用户来到{region}的名为{name}的无人值守店。
    注意要符合{remark}内容，使用表情。
    """)

    response = (prompt | llm).stream({
        "now": now,
        "region": state["shop_info"]["region"],
        "name": state["shop_info"]["name"],
        "gender": state["current_user"]["gender"],
        "username": state["current_user"]["username"],
        "recommends": ",".join(state["current_user"]["preferences"]),
        "promotions": ",".join(state["shop_info"]["promotions"]),
        "dialect": state["current_user"]["dialect"],
        "remark": state["shop_info"]["remark"]
    })

    # 创建原始文本流生成器
    async def original_text_stream():
        full_content = ""
        for chunk in response:
            if hasattr(chunk, 'content'):
                full_content += chunk.content
                # Yield each character individually
                for char in chunk.content:
                    yield f"data: {json.dumps({'type': 'text', 'response': char, 'status': 'success'})}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
        
        # 持久化欢迎消息
        welcome_msg = {
            "message_id": "msg_1",
            "type": "system",
            "content": full_content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": {
                "intent": "welcome",
                "tool_used": "personalized_welcome"
            }
        }
        dialogue_manager.append_message(state["session_id"], welcome_msg)
        print('[调试信息] 欢迎词已持久化')
        
        response_time = time.time() - start_time
        print(f"\n[AI响应时间] 生成耗时: {response_time:.2f} 秒")

    # 根据配置决定是否使用混合流
    if USE_MIXED_STREAM:
        async for chunk in create_improved_mixed_stream(
            original_text_stream(), 
            state["session_id"]
        ):
            yield chunk
    else:
        async for chunk in original_text_stream():
            yield chunk


def intent_recognition(state: State):
    """Intent recognition node"""
    history = "\n".join([
        f"{'顾客' if isinstance(msg, HumanMessage) else '助手'}：{msg.content}"
        for msg in state["messages"][-5:]
    ])

    prompt = ChatPromptTemplate.from_template("""
    分析消息类型(返回工具名称): 
    历史对话:
    {history}
    新消息: {message}
    注意:
    1. 非必要不使用handle_web_search
    2. 如果用户询问之前提到的商品价格，使用handle_shopping_guide
    3. 如果用户询问之前提到的商品详情，使用handle_recommendation
    4. 只返回工具名称，不要包含其他文字
    
    可选工具: 
    - handle_web_search: 查询实时联网信息
    - handle_chitchat: 日常对话/问候
    - handle_shopping_guide: 商品位置/价格问题
    - handle_recommendation: 明确要求推荐商品
    - handle_human_transfer: 转人工请求
    - handle_report: 销售数据查询
    - handle_payment: 支付相关
    - handle_goodbye: 告别/退出/离开/结束请求
    只需返回工具名称，不要包含其他任何文字""")

    msg = (prompt | llm).invoke({
        "message": state["messages"][-1].content,
        "history": history
    }).content

    # Clean up output
    msg = msg.replace(' ', '').replace('\n', '')
    if '<think>' in msg and '</think>' in msg:
        start_index = msg.find('<think>')
        end_index = msg.find('</think>', start_index) + len('</think>')
        msg = msg[:start_index] + msg[end_index:]
    
    # Extract tool name
    valid_tools = ["handle_web_search", "handle_chitchat", "handle_shopping_guide", 
                  "handle_recommendation", "handle_human_transfer", "handle_report", 
                  "handle_payment", "handle_goodbye"]
    
    for tool in valid_tools:
        if tool in msg:
            msg = tool
            break
    else:
        msg = "handle_chitchat"
    
    print("[调试信息] 清理后的意图: " + msg)

    if msg == "handle_goodbye" or state.get("exit_flag"):
        return {
            "current_intent": "handle_goodbye",
            "exit_flag": True
        }

    return {
        "current_intent": msg
    }

async def handle_chitchat(state: State):
    """Handle casual conversation"""
    history = "\n".join([
        f"{'顾客' if isinstance(msg, HumanMessage) else '助手'}：{msg.content}"
        for msg in state["messages"][-5:]
    ])

    prompt = ChatPromptTemplate.from_template("""
    历史对话:
    {history}
    新消息: {message}
    请用{dialect}方言回复，语气要自然友好。
    注意要符合{remark}内容，使用表情
    用户信息: {username},性别{gender}
    """)

    response = (prompt | llm).stream({
        "message": state["messages"][-1].content,
        "history": history,
        "dialect": state["current_user"]["dialect"],
        "remark": state["shop_info"]["remark"],
        "username": state["current_user"]["username"],
        "gender": state["current_user"]["gender"]
    })

    # 创建原始文本流生成器
    async def original_text_stream():
        full_content = ""
        for chunk in response:
            if hasattr(chunk, 'content'):
                full_content += chunk.content
                # Yield each character individually
                for char in chunk.content:
                    yield f"data: {json.dumps({'type': 'text', 'response': char, 'status': 'success'})}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
        
        debug_print("handle_chitchat", f"回复内容: {full_content}")

    # 根据配置决定是否使用混合流
    if USE_MIXED_STREAM:
        async for chunk in create_improved_mixed_stream(
            original_text_stream(), 
            state["session_id"]
        ):
            yield chunk
    else:
        async for chunk in original_text_stream():
            yield chunk

async def handle_shopping_guide(state: State):
    """Handle shopping related queries"""
    history = "\n".join([
        f"{'顾客' if isinstance(msg, HumanMessage) else '助手'}：{msg.content}"
        for msg in state["messages"][-5:]
    ])

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
    query = state["messages"][-1].content
    for key, response in fixed_responses.items():
        if key in query:
            prompt = ChatPromptTemplate.from_template("""
            用{dialect}方言回答: 
            {response}
            注意要符合{remark}内容，使用表情
            用户信息: {username},性别{gender}
            历史对话: {history}
            """)

            response = (prompt | llm).stream({
                "dialect": state["current_user"]["dialect"],
                "username": state["current_user"]["username"],
                "gender": state["current_user"]["gender"],
                "remark": state["shop_info"]["remark"],
                "history": history,
                "response": response
            })

            # 创建原始文本流生成器
            async def original_text_stream():
                full_content = ""
                for chunk in response:
                    if hasattr(chunk, 'content'):
                        full_content += chunk.content
                        # Yield each character individually
                        for char in chunk.content:
                            yield f"data: {json.dumps({'type': 'text', 'response': char, 'status': 'success'})}\n\n"
                            await asyncio.sleep(0)  # Allow other tasks to run
            
            # 根据配置决定是否使用混合流
            if USE_MIXED_STREAM:
                async for chunk in create_improved_mixed_stream(
                    original_text_stream(), 
                    state["session_id"]
                ):
                    yield chunk
            else:
                async for chunk in original_text_stream():
                    yield chunk
            return

    # 如果不是固定话术，使用 RAG 检索相关信息
    debug_print("handle_shopping_guide", "开始 RAG 检索")
    docs = rag_manager.retrieve(
        query=query + "\n历史对话: " + history,
        search_type="mmr",
        k=10,
        fetch_k=50,
        lambda_mult=0.5
    )
    
    # 提取检索到的商品信息，包含完整信息
    product_info = [
        f"· {doc.metadata['product_name']}（{doc.metadata['category']}）"
        f" 位置：{doc.metadata['position']}"
        f" 价格：{doc.metadata['product_price']}元"
        # f" 库存：{doc.metadata['stock']}件"
        # f" 销量：{doc.metadata['sales']}件"
        for doc in docs
    ]
    debug_print("handle_shopping_guide", f"检索到的商品信息: {product_info}")

    # 使用通用导购处理
    prompt = ChatPromptTemplate.from_template("""
    历史对话:
    {history}
    新消息: {message}
    商品信息:
    {product_info}
    请用{dialect}方言回复，语气要专业友好。
    注意：
    1. 如果询问价格，请说明具体价格
    2. 如果询问位置，请说明具体位置
    3. 如果询问库存，请说明库存状态
    4. 注意要符合{remark}内容，使用表情
    用户信息: {username},性别{gender}
    """)

    response = (prompt | llm).stream({
        "message": query,
        "history": history,
        "product_info": "\n".join(product_info),
        "dialect": state["current_user"]["dialect"],
        "remark": state["shop_info"]["remark"],
        "username": state["current_user"]["username"],
        "gender": state["current_user"]["gender"]
    })

    # 创建原始文本流生成器
    async def original_text_stream():
        full_content = ""
        for chunk in response:
            if hasattr(chunk, 'content'):
                full_content += chunk.content
                # Yield each character individually
                for char in chunk.content:
                    yield f"data: {json.dumps({'type': 'text', 'response': char, 'status': 'success'})}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
        
        debug_print("handle_shopping_guide", f"通用导购回复: {full_content}")

    # 根据配置决定是否使用混合流
    if USE_MIXED_STREAM:
        async for chunk in create_improved_mixed_stream(
            original_text_stream(), 
            state["session_id"]
        ):
            yield chunk
    else:
        async for chunk in original_text_stream():
            yield chunk

async def handle_recommendation(state: State):
    """Handle product recommendations"""
    history = "\n".join([
        f"{'顾客' if isinstance(msg, HumanMessage) else '助手'}：{msg.content}"
        for msg in state["messages"][-5:]
    ])

    # 使用 RAG 检索相关信息
    debug_print("handle_recommendation", "开始 RAG 检索")
    docs = rag_manager.retrieve(
        query=state["messages"][-1].content + "\n历史对话: " + history,
        search_type="mmr",
        k=10,
        fetch_k=50,
        lambda_mult=0.5
    )
    
    # 提取检索到的商品信息，包含完整信息
    product_info = [
        f"· {doc.metadata['product_name']}（{doc.metadata['category']}）"
        f" 位置：{doc.metadata['position']}"
        f" 价格：{doc.metadata['product_price']}元"
        # f" 库存：{doc.metadata['stock']}件"
        # f" 销量：{doc.metadata['sales']}件"
        for doc in docs
    ]
    debug_print("handle_recommendation", f"检索到的商品信息: {product_info}")

    prompt = ChatPromptTemplate.from_template("""
    历史对话:
    {history}
    新消息: {message}
    用户偏好: {preferences}
    今日活动: {promotions}
    商品信息:
    {product_info}
    请用{dialect}方言推荐商品，语气要热情友好。
    注意：
    1. 根据用户偏好推荐相关商品
    2. 说明推荐理由
    3. 可以推荐多个商品
    4. 注意要符合{remark}内容，使用表情
    用户信息: {username},性别{gender}
    """)

    response = (prompt | llm).stream({
        "message": state["messages"][-1].content,
        "history": history,
        "preferences": ", ".join(state["current_user"]["preferences"]),
        "promotions": ", ".join(state["shop_info"]["promotions"]),
        "product_info": "\n".join(product_info),
        "dialect": state["current_user"]["dialect"],
        "remark": state["shop_info"]["remark"],
        "username": state["current_user"]["username"],
        "gender": state["current_user"]["gender"]
    })

    # 创建原始文本流生成器
    async def original_text_stream():
        full_content = ""
        for chunk in response:
            if hasattr(chunk, 'content'):
                full_content += chunk.content
                # Yield each character individually
                for char in chunk.content:
                    yield f"data: {json.dumps({'type': 'text', 'response': char, 'status': 'success'})}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
        
        debug_print("handle_recommendation", f"推荐回复: {full_content}")

    # 根据配置决定是否使用混合流
    if USE_MIXED_STREAM:
        async for chunk in create_improved_mixed_stream(
            original_text_stream(), 
            state["session_id"]
        ):
            yield chunk
    else:
        async for chunk in original_text_stream():
            yield chunk

async def handle_payment(state: State):
    """Handle payment related queries"""
    history = "\n".join([
        f"{'顾客' if isinstance(msg, HumanMessage) else '助手'}：{msg.content}"
        for msg in state["messages"][-5:]
    ])

    prompt = ChatPromptTemplate.from_template("""
    历史对话:
    {history}
    新消息: {message}
    请用{dialect}方言回复，语气要专业友好。
    注意：
    1. 说明支付方式
    2. 说明支付流程
    3. 说明支付注意事项
    4. 注意要符合{remark}内容，使用表情
    用户信息: {username},性别{gender}
    """)

    response = (prompt | llm).stream({
        "message": state["messages"][-1].content,
        "history": history,
        "dialect": state["current_user"]["dialect"],
        "remark": state["shop_info"]["remark"],
        "username": state["current_user"]["username"],
        "gender": state["current_user"]["gender"]
    })

    # 创建原始文本流生成器
    async def original_text_stream():
        full_content = ""
        for chunk in response:
            if hasattr(chunk, 'content'):
                full_content += chunk.content
                # Yield each character individually
                for char in chunk.content:
                    yield f"data: {json.dumps({'type': 'text', 'response': char, 'status': 'success'})}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
        
        debug_print("handle_payment", f"支付回复: {full_content}")

    # 根据配置决定是否使用混合流
    if USE_MIXED_STREAM:
        async for chunk in create_improved_mixed_stream(
            original_text_stream(), 
            state["session_id"]
        ):
            yield chunk
    else:
        async for chunk in original_text_stream():
            yield chunk

async def handle_goodbye(state: State):
    """Handle goodbye messages"""
    history = "\n".join([
        f"{'顾客' if isinstance(msg, HumanMessage) else '助手'}：{msg.content}"
        for msg in state["messages"][-5:]
    ])

    prompt = ChatPromptTemplate.from_template("""
    历史对话:
    {history}
    新消息: {message}
    请用{dialect}方言回复，语气要热情友好。
    注意：
    1. 表达感谢
    2. 欢迎再次光临
    3. 可以适当使用表情符号
    4. 注意要符合{remark}内容
    用户信息: {username},性别{gender}
    """)

    response = (prompt | llm).stream({
        "message": state["messages"][-1].content,
        "history": history,
        "dialect": state["current_user"]["dialect"],
        "remark": state["shop_info"]["remark"],
        "username": state["current_user"]["username"],
        "gender": state["current_user"]["gender"]
    })

    # 创建原始文本流生成器
    async def original_text_stream():
        full_content = ""
        for chunk in response:
            if hasattr(chunk, 'content'):
                full_content += chunk.content
                # Yield each character individually
                for char in chunk.content:
                    yield f"data: {json.dumps({'type': 'text', 'response': char, 'status': 'success'})}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
        
        debug_print("handle_goodbye", f"告别回复: {full_content}")

    # 根据配置决定是否使用混合流
    if USE_MIXED_STREAM:
        async for chunk in create_improved_mixed_stream(
            original_text_stream(), 
            state["session_id"]
        ):
            yield chunk
    else:
        async for chunk in original_text_stream():
            yield chunk

async def handle_human_transfer(state: State):
    """Handle human transfer requests"""
    history = "\n".join([
        f"{'顾客' if isinstance(msg, HumanMessage) else '助手'}：{msg.content}"
        for msg in state["messages"][-5:]
    ])

    prompt = ChatPromptTemplate.from_template("""
    历史对话:
    {history}
    新消息: {message}
    请用{dialect}方言回复，语气要专业友好。
    注意：
    1. 说明已转接值班经理
    2. 说明工作时间（9:00-21:00）
    3. 说明预计等待时间
    4. 注意要符合{remark}内容，使用表情
    用户信息: {username},性别{gender}
    """)

    response = (prompt | llm).stream({
        "message": state["messages"][-1].content,
        "history": history,
        "dialect": state["current_user"]["dialect"],
        "remark": state["shop_info"]["remark"],
        "username": state["current_user"]["username"],
        "gender": state["current_user"]["gender"]
    })

    # 创建原始文本流生成器
    async def original_text_stream():
        full_content = ""
        for chunk in response:
            if hasattr(chunk, 'content'):
                full_content += chunk.content
                # Yield each character individually
                for char in chunk.content:
                    yield f"data: {json.dumps({'type': 'text', 'response': char, 'status': 'success'})}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
        
        debug_print("handle_human_transfer", f"转人工回复: {full_content}")

    # 根据配置决定是否使用混合流
    if USE_MIXED_STREAM:
        async for chunk in create_improved_mixed_stream(
            original_text_stream(), 
            state["session_id"]
        ):
            yield chunk
    else:
        async for chunk in original_text_stream():
            yield chunk

# 确保 handle_report 等函数也正确处理流结束
async def handle_report(state: State):
    """Handle report requests"""
    history = "\n".join([
        f"{'顾客' if isinstance(msg, HumanMessage) else '助手'}：{msg.content}"
        for msg in state["messages"][-5:]
    ])

    # 生成随机销售数据
    sales = random.randint(5000, 10000)
    profit = random.randint(1500, 3000)
    hot_products = ["椰树椰汁", "农夫山泉"][:random.randint(1, 2)]

    prompt = ChatPromptTemplate.from_template("""
    历史对话:
    {history}
    新消息: {message}
    销售数据:
    销售额: ¥{sales}
    净利润: ¥{profit}
    热销商品: {hot_products}
    请用{dialect}方言回复，语气要专业友好。
    注意：
    1. 展示销售数据
    2. 说明热销商品
    3. 提供可视化建议
    4. 注意要符合{remark}内容，使用表情
    用户信息: {username},性别{gender}
    """)

    response = (prompt | llm).stream({
        "message": state["messages"][-1].content,
        "history": history,
        "sales": sales,
        "profit": profit,
        "hot_products": "、".join(hot_products),
        "dialect": state["current_user"]["dialect"],
        "remark": state["shop_info"]["remark"],
        "username": state["current_user"]["username"],
        "gender": state["current_user"]["gender"]
    })

    # 创建原始文本流生成器
    async def original_text_stream():
        full_content = ""
        for chunk in response:
            if hasattr(chunk, 'content'):
                full_content += chunk.content
                # Yield each character individually
                for char in chunk.content:
                    yield f"data: {json.dumps({'type': 'text', 'response': char, 'status': 'success'})}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
                    
        debug_print("handle_report", f"报表回复: {full_content}")

    # 根据配置决定是否使用混合流
    if USE_MIXED_STREAM:
        async for chunk in create_improved_mixed_stream(
            original_text_stream(), 
            state["session_id"]
        ):
            yield chunk
    else:
        async for chunk in original_text_stream():
            yield chunk


async def handle_web_search(state: State):
    """Handle web search queries"""
    history = "\n".join([
        f"{'顾客' if isinstance(msg, HumanMessage) else '助手'}：{msg.content}"
        for msg in state["messages"][-5:]
    ])

    # 配置服务 URL
    service_url = "http://appbuilder.baidu.com/v2/ai_search/mcp/sse?" + \
                  "api_key=Bearer+bce-v3/ALTAK-lp91La4lRwuifo4dSNURU/70cb3ab0e2e87e267f6840f76e9fd052adfca877"

    # 执行异步搜索并返回结果
    try:
        # 初始化MCPClient
        client = MCPClient()
        await client.connect_to_server(service_url=service_url)

        # 调用MCP搜索工具
        result = await client.call_tool(
            "AIsearch",
            {
                "query": state["messages"][-1].content,
                "search_top_k": 1,
                "instruction": "过滤广告,只要摘要",
                "search_recency_filter": "week",
                "search_resource_type": [{"type": "web", "top_k": 10}]
            }
        )

        # 提取搜索结果文本内容
        if result and hasattr(result, 'content') and result.content:
            search_results = result.content[0].text
        else:
            search_results = "未找到相关信息"
        debug_print("handle_web_search", f"搜索结果: {search_results}")
    except Exception as e:
        print(f"[错误] 搜索异常: {str(e)}")
        search_results = "无法获取实时信息"
        debug_print("handle_web_search", "搜索失败，使用默认回复")

    # 信息整合提示模板
    prompt = ChatPromptTemplate.from_template("""
        问题: {query}
        搜索结果: 
        {context}
        切记用{dialect}方言回复，涵盖{username}和{gender}并
        1. 去除语义无关和推广信息
        2. 合并重复内容
        3. 保留数字和关键点
        4. 简短
    """)

    # 链式调用
    response = (prompt | llm).stream({
        "context": search_results,
        "query": state["messages"][-1].content,
        "dialect": state["current_user"]["dialect"],
        "username": state["current_user"]["username"],
        "gender": state["current_user"]["gender"]
    })

    # 创建原始文本流生成器
    async def original_text_stream():
        full_content = ""
        for chunk in response:
            if hasattr(chunk, 'content'):
                full_content += chunk.content
                # Yield each character individually
                for char in chunk.content:
                    yield f"data: {json.dumps({'type': 'text', 'response': char, 'status': 'success'})}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
        
        debug_print("handle_web_search", f"搜索回复: {full_content}")

    # 根据配置决定是否使用混合流
    if USE_MIXED_STREAM:
        async for chunk in create_improved_mixed_stream(
            original_text_stream(), 
            state["session_id"]
        ):
            yield chunk
    else:
        async for chunk in original_text_stream():
            yield chunk

# Register all nodes
node_manager.register_node("intent_recognition", intent_recognition)
node_manager.register_node("handle_chitchat", handle_chitchat)
node_manager.register_node("handle_shopping_guide", handle_shopping_guide)
node_manager.register_node("handle_recommendation", handle_recommendation)
node_manager.register_node("handle_payment", handle_payment)
node_manager.register_node("handle_goodbye", handle_goodbye)
node_manager.register_node("handle_web_search", handle_web_search)
node_manager.register_node("handle_human_transfer", handle_human_transfer)
node_manager.register_node("handle_report", handle_report)

# 涉及continue接口的核心修改
async def process_message(state: State):
    """Process message and generate response"""
    try:
        # Recognize intent
        intent_state = node_manager.nodes["intent_recognition"](state)
        state.update(intent_state)
        debug_print("intent_recognition", f"识别意图: {state['current_intent']}")

        # Get tool node based on intent
        tool_name = state["current_intent"]
        if tool_name not in node_manager.nodes:
            error_msg = f'抱歉，我无法处理这个请求。当前意图 "{tool_name}" 无效。'
            debug_print("process_message", f"无效意图: {tool_name}")
            yield f"data: {json.dumps({'type': 'text', 'response': error_msg, 'status': 'error'})}\n\n"
            # 添加流结束标志
            yield f"data: {json.dumps({'type': 'stream_end', 'status': 'error'})}\n\n"
            return

        # Get response from tool - 这里各个工具函数内部已经处理了混合流
        async for chunk in node_manager.nodes[tool_name](state):
            yield chunk

    except Exception as e:
        error_msg = f"抱歉，处理您的消息时出现了错误，先看看店内商品吧。"
        debug_print("process_message", f"处理错误: {str(e)}")
        yield f"data: {json.dumps({'type': 'text', 'response': error_msg, 'status': 'error'})}\n\n"
        # 添加流结束标志
        yield f"data: {json.dumps({'type': 'stream_end', 'status': 'error'})}\n\n"

# 修改后的接口
@app.post("/chat/new/stream")
async def new_chat(request: NewChatRequest):
    """Create new chat session with streaming response including audio"""
    try:
        # Generate session ID
        session_id = f"{request.shop_info.shop_id}_{request.user_info.user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 原有的初始化逻辑...
        initial_doc = {
            "session_id": session_id,
            "shop_info": {
                "shop_id": request.shop_info.shop_id,
                "shop_name": request.shop_info.name,
                "region": request.shop_info.region,
                "contact": request.shop_info.contact,
                "remark": request.shop_info.remark,
                "promotions": request.shop_info.promotions
            },
            "user_info": {
                "user_id": request.user_info.user_id,
                "username": request.user_info.username,
                "gender": request.user_info.gender,
                "preferences": request.user_info.preferences,
                "dialect": request.user_info.dialect
            },
            "messages": [],
            "status": {
                "payment_status": "pending",
                "exit_flag": False
            },
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        dialogue_manager.create_dialogue(initial_doc)
        debug_print("new_chat", f"新建对话记录: {session_id}")
        
        # Initialize state
        state = {
            "session_id": session_id,
            "messages": [],
            "current_user": {
                "user_id": request.user_info.user_id,
                "username": request.user_info.username,
                "gender": request.user_info.gender,
                "preferences": request.user_info.preferences,
                "dialect": request.user_info.dialect
            },
            "shop_info": {
                "shop_id": request.shop_info.shop_id,
                "name": request.shop_info.name,
                "region": request.shop_info.region,
                "contact": request.shop_info.contact,
                "remark": request.shop_info.remark,
                "promotions": request.shop_info.promotions
            },
            "current_intent": None
        }

        # Stream welcome message with audio
        async def generate():
            # First yield the session info
            yield f"data: {json.dumps({'type': 'text', 'session_id': session_id, 'status': 'success'})}\n\n"
            
            # Then stream the welcome message with audio
            async for chunk in personalized_welcome_stream(state):
                if isinstance(chunk, str):
                    yield chunk
                    await asyncio.sleep(0)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        print(f"[错误] 新建对话失败: {str(e)}")
        error_message = str(e)
        async def error_generate():
            yield f"data: {json.dumps({'type': 'text', 'error': error_message})}\n\n"
        return StreamingResponse(error_generate(), media_type="text/event-stream")


@app.post("/chat/continue/stream")
async def continue_chat(request: ContinueChatRequest):
    """Continue existing chat session with streaming response including audio"""
    try:
        # 确保取消之前的音频生成
        await cancel_audio_generation(request.session_id)
        
        # 获取会话
        session = dialogue_manager.get_dialogue(request.session_id)
        if not session:
            async def error_generate():
                yield f"data: {json.dumps({'type': 'text', 'error': 'Session not found'})}\n\n"
                yield f"data: {json.dumps({'type': 'stream_end', 'status': 'error'})}\n\n"
            return StreamingResponse(error_generate(), media_type="text/event-stream")

        # 初始化状态
        state = {
            "session_id": request.session_id,
            "current_user": {
                "user_id": session["user_info"].get("user_id", ""),
                "username": session["user_info"].get("username", ""),
                "gender": session["user_info"].get("gender", "未知"),
                "preferences": session["user_info"].get("preferences", []),
                "dialect": session["user_info"].get("dialect", "普通话")
            },
            "shop_info": {
                "shop_id": session["shop_info"].get("shop_id", ""),
                "name": session["shop_info"].get("shop_name", ""),
                "region": session["shop_info"].get("region", ""),
                "contact": session["shop_info"].get("contact", ""),
                "remark": session["shop_info"].get("remark", ""),
                "promotions": session["shop_info"].get("promotions", [])
            },
            "messages": [],
            "last_response": None
        }

        # 添加用户消息
        state["messages"].append(HumanMessage(content=request.message))

        # 持久化用户消息
        user_msg = {
            "message_id": f"msg_{len(session.get('messages', [])) + 1}",
            "type": "human",
            "content": request.message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        dialogue_manager.append_message(request.session_id, user_msg)

        async def generate():
            content = ""
            try:
                async for chunk in process_message(state):
                    if isinstance(chunk, str):
                        try:
                            if chunk.startswith("data: "):
                                data = json.loads(chunk[6:])
                                if data.get("type") == "text" and "response" in data:
                                    content += data["response"]
                        except:
                            pass
                        yield chunk
                        await asyncio.sleep(0)
                
                # 确保发送流结束标志
                yield f"data: {json.dumps({'type': 'stream_end', 'status': 'complete'})}\n\n"
                
            except Exception as e:
                print(f"[错误] 流处理异常: {str(e)}")
                yield f"data: {json.dumps({'type': 'text', 'error': '处理消息时出现错误', 'status': 'error'})}\n\n"
                yield f"data: {json.dumps({'type': 'stream_end', 'status': 'error'})}\n\n"
            finally:
                # 确保取消音频生成
                await cancel_audio_generation(request.session_id)
                
                # 持久化AI回复
                if content.strip():
                    ai_msg = {
                        "message_id": f"msg_{len(session.get('messages', [])) + 2}",
                        "type": "system",
                        "content": content,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "metadata": {
                            "intent": "response",
                            "confidence": 1.0,
                            "tools_used": []
                        }
                    }
                    dialogue_manager.append_message(request.session_id, ai_msg)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        print(f"[错误] 继续对话失败: {str(e)}")
        # 确保异常时也释放资源
        await cancel_audio_generation(request.session_id)
        
        async def error_generate():
            yield f"data: {json.dumps({'type': 'text', 'error': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'stream_end', 'status': 'error'})}\n\n"
        return StreamingResponse(error_generate(), media_type="text/event-stream")


# 新增取消音频接口
@app.post("/chat/cancel_audio")
async def cancel_audio(request: dict):
    """Cancel audio generation for a session"""
    try:
        session_id = request.get("session_id")
        if not session_id:
            return {"error": "Session ID is required"}
        
        await cancel_audio_generation(session_id)
        return {"status": "success", "message": "Audio generation cancelled"}
    
    except Exception as e:
        return {"error": str(e)}

# 清理接口，在会话结束时调用
@app.post("/chat/cleanup")
async def cleanup_session(request: dict):
    """Clean up session resources"""
    try:
        session_id = request.get("session_id")
        if session_id:
            await cancel_audio_generation(session_id)
        
        return {"status": "success"}
    
    except Exception as e:
        return {"error": str(e)}





if __name__ == "__main__":
    import uvicorn
    
    # Start the FastAPI application with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,       # Run on port 8000
        log_level="info" # Set log level to info
    )



'''

要求5 队列应该可由前端的请求进行阻断 可以简单地断开连接，但更好的方式是可以任意在队列中执行暂停、继续、清空、插入操作
简洁又完整地实现一下

'''