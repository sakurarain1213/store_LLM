'''
TODO 虽然不影响结果 但是 修一下联网功能的异步报错 尽量不动v_12本身 
接口文档
    TODO 语音的识别和生成的接口还要研究一下 核心的语音识别一定要有 可以付费但是要快

新建对话
POST http://localhost:8000/chat/new
Content-Type: application/json
入参
{
    "user_id": "user-123",
    "dialect": "上海",
    "gender": "男",
    "preferences": ["可乐", "香烟"],
    "username": "张三"
}

出参
{
    "session_id": "shop-123_user-123_20250603111110",
    "response": "\n哎哟，张三哥啊，欢迎光临西湖123便利店🌟🍀！今朝个天气真好，适合来我这家无人值守店逛逛哦。看这边，精选推荐给你看，可乐、香烟、烟酒都有，还有各种好烟供你挑选，比如黄鹤楼、好猫、N王冠、兰州、白沙、七星、娇子、中南海、贵烟，保证让你挑花眼😍🍻。\n\n今天还有活动哦，满30块减5块，买够30块还能送你一包口香糖🤤，快来试试手气吧！😉🎉\n\n杭州是个好地方，希望你能在这里找到心仪的商品，也祝你在杭州过得开心哦！🌸🌳",
    "status": "success",
    "error": null
}



继续对话
http://localhost:8000/chat/continue
Content-Type: application/json
入参
{
    "session_id": "shop-123_user-123_20250603111110",
    "message": "结账"
}
出参
{
    "session_id": "shop-123_user-123_20250603111110",
    "response": "支付成功！请取走商品",
    "status": "success",
    "error": null
}


'''


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal
import uvicorn
from fastapi.middleware.cors import CORSMiddleware # DEBUG 跨域支持 本地前端展示
from v_12 import (
    State, UserInfo, ShopInfo, StateGraph, NodeManager,
    load_shop_products_csv, dialogue_manager, entry_node,
    personalized_welcome, collect_human_input, intent_recognition,
    personalized_goodbye, handle_chitchat,
    handle_shopping_guide, handle_recommendation, handle_human_transfer,
    handle_report, handle_payment, llm
)
from v_12_web import fetch_search_results
from langgraph.graph import END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import random
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
import time
from langchain.tools import tool
from v_13 import handle_web_search as v13_handle_web_search
import asyncio

app = FastAPI(title="无人店值守对话大模型API")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 初始化节点管理器
node_manager = NodeManager()

# 注册基础节点
node_manager.register_node("entry", entry_node)
node_manager.register_node("welcome", personalized_welcome)
node_manager.register_node("collect_input", collect_human_input)
node_manager.register_node("intent_recognition", intent_recognition)
node_manager.register_node("handle_goodbye", personalized_goodbye)

# 注册工具节点
node_manager.register_node("handle_chitchat", handle_chitchat)
node_manager.register_node("handle_shopping_guide", handle_shopping_guide)
node_manager.register_node("handle_recommendation", handle_recommendation)
node_manager.register_node("handle_human_transfer", handle_human_transfer)
node_manager.register_node("handle_report", handle_report)
node_manager.register_node("handle_payment", handle_payment)
node_manager.register_node("handle_web_search", v13_handle_web_search)

# 设置入口点
node_manager.add_edge("entry", "welcome")
node_manager.add_edge("welcome", "collect_input")
node_manager.add_edge("collect_input", "intent_recognition")

# 添加工具节点路由
tools = ["handle_chitchat", "handle_shopping_guide", "handle_recommendation",
         "handle_human_transfer", "handle_report", "handle_payment", "handle_goodbye", "handle_web_search"]

# 意图路由
node_manager.add_conditional_edges(
    "intent_recognition",
    lambda state: state["current_intent"],
    {tool_name: tool_name for tool_name in tools}
)

# 支付结果路由
node_manager.add_conditional_edges(
    "handle_payment",
    lambda state: "handle_goodbye" if state.get("payment_status") == "success" else "handle_payment",
    {"handle_goodbye": "handle_goodbye", "handle_payment": "handle_payment"}
)

# 循环处理
for tool_name in tools:
    if tool_name not in ["handle_payment", "handle_goodbye"]:
        node_manager.add_conditional_edges(
            tool_name,
            lambda state: "collect_input" if not state.get("exit_flag") else "handle_goodbye",
            {"collect_input": "collect_input", "handle_goodbye": "handle_goodbye"}
        )

# 结束流程
node_manager.add_edge("handle_goodbye", END)

# 初始化图结构
builder = StateGraph(State)
builder.set_entry_point("entry")
node_manager.build_graph(builder)
flow = builder.compile()

# 加载示例店铺数据
products_docs_list = load_shop_products_csv("data/store_cloud_duty_store_order_goods_day.csv")
sample_shop = ShopInfo(
    shop_id="shop-123",
    name="西湖123便利店",
    region="杭州",
    contact="X老板, 123-1234-5678",
    remark="购物袋免费；成条烟不拆开；店内不堂食；可以接热水；",
    promotions=["满30减5", "送口香糖"],
    products_docs=products_docs_list
)

# 请求模型
class ShopInfoRequest(BaseModel):
    name: str
    region: str
    contact: str
    remark: str
    promotions: List[str]

class NewChatRequest(BaseModel):
    user_id: str
    dialect: str
    gender: Literal["男", "女", "其他"]
    preferences: List[str]
    username: str
    shop_info: ShopInfoRequest

class ChatRequest(BaseModel):
    session_id: str
    message: str

# 响应模型
class ChatResponse(BaseModel):
    session_id: str
    response: str
    status: str
    error: Optional[str] = None

@app.post("/chat/new", response_model=ChatResponse)
async def new_chat(request: NewChatRequest):
    """创建新的对话会话"""
    try:
        # 构建店铺信息
        shop = ShopInfo(
            shop_id=f"shop-{random.randint(100, 999)}",
            name=request.shop_info.name,
            region=request.shop_info.region,
            contact=request.shop_info.contact,
            remark=request.shop_info.remark,
            promotions=request.shop_info.promotions,
            products_docs=products_docs_list  # 使用示例商品数据
        )

        # 构建初始状态
        initial_state = {
            "current_user": {
                "user_id": request.user_id,
                "dialect": request.dialect,
                "gender": request.gender,
                "preferences": request.preferences,
                "username": request.username
            },
            "shop_info": shop,
            "messages": [],
            "payment_status": "pending",
            "exit_flag": False
        }

        # 执行流程直到welcome节点
        events = flow.stream(initial_state, {"recursion_limit": 2})
        
        # 获取welcome节点的响应
        welcome_response = None
        session_id = None
        for event in events:
            if "welcome" in event:
                welcome_response = event["welcome"]["last_response"]
                session_id = event["welcome"]["session_id"]
                break

        if not welcome_response or not session_id:
            raise HTTPException(status_code=500, detail="Failed to generate welcome message")

        # 保存对话到数据库
        dialogue_manager.create_dialogue({
            "session_id": session_id,
            "user_info": {
                "user_id": request.user_id,
                "dialect": request.dialect,
                "gender": request.gender,
                "preferences": request.preferences,
                "username": request.username
            },
            "shop_info": {
                "shop_id": shop["shop_id"],
                "shop_name": shop["name"],
                "region": shop["region"],
                "contact": shop["contact"],
                "remark": shop["remark"],
                "promotions": shop["promotions"]
            },
            "messages": [{
                "message_id": "msg_1",
                "type": "system",
                "content": welcome_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {
                    "intent": "welcome"
                }
            }],
            "status": {
                "payment_status": "pending",
                "exit_flag": False
            },
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        return ChatResponse(
            session_id=session_id,
            response=welcome_response,
            status="success"
        )

    except Exception as e:
        print(f"Error in new_chat: {str(e)}")  # 添加错误日志
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/continue", response_model=ChatResponse)
async def continue_chat(request: ChatRequest):
    """继续现有对话"""
    try:
        # 获取现有对话状态
        existing_dialogue = dialogue_manager.get_dialogue(request.session_id)
        if not existing_dialogue:
            raise HTTPException(status_code=404, detail="Session not found")

        print("[调试信息] 开始处理对话请求")
        print(f"[调试信息] 用户消息: {request.message}")

        # 获取历史消息
        history_messages = []
        for msg in existing_dialogue.get("messages", []):
            if msg["type"] == "human":
                history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "system":
                history_messages.append(AIMessage(content=msg["content"]))

        # 添加当前消息
        history_messages.append(HumanMessage(content=request.message))

        # 保存用户消息到数据库
        human_msg = {
            "message_id": f"msg_{len(existing_dialogue.get('messages', [])) + 1}",
            "type": "human",
            "content": request.message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": {
                "intent": "user_input"
            }
        }
        dialogue_manager.append_message(request.session_id, human_msg)

        # 构建状态
        state = {
            "session_id": request.session_id,
            "current_user": existing_dialogue.get("user_info", {}),
            "shop_info": sample_shop,
            "messages": history_messages,
            "payment_status": existing_dialogue.get("status", {}).get("payment_status", "pending"),
            "exit_flag": False,
            "purchased_items": existing_dialogue.get("status", {}).get("purchased_items", [])
        }

        print("[调试信息] 状态构建完成")
        print(f"[调试信息] 当前用户: {state['current_user']}")
        print(f"[调试信息] 消息历史长度: {len(state['messages'])}")

        # 直接调用意图识别节点
        intent_result = intent_recognition(state)
        print("[调试信息] 意图识别结果:", intent_result)
        state.update(intent_result)

        # 检查是否是告别意图
        if state.get("current_intent") == "handle_goodbye" or state.get("exit_flag"):
            print("[调试信息] 检测到告别意图，准备生成告别词")
            print("[调试信息] 已购商品:", state.get("purchased_items", []))
            
            # 调用告别节点生成告别消息
            goodbye_result = personalized_goodbye(state)
            response = goodbye_result["last_response"]
            
            # 保存告别消息到数据库
            ai_msg = {
                "message_id": f"msg_{len(existing_dialogue.get('messages', [])) + 2}",
                "type": "system",
                "content": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {
                    "tool_used": "handle_goodbye",
                    "intent": "handle_goodbye",
                    "exit_flag": True,
                    "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            dialogue_manager.append_message(request.session_id, ai_msg)
            
            return ChatResponse(
                session_id=request.session_id,
                response=response,
                status="success"
            )

        # 根据意图调用相应的工具节点
        if state.get("current_intent") in tools:
            tool_name = state["current_intent"]
            print(f"[调试信息] 调用工具节点: {tool_name}")

            # 准备工具函数参数
            tool_params = {
                "query": request.message,
                "history": [
                    {
                        "type": "human" if isinstance(msg, HumanMessage) else "system",
                        "content": msg.content
                    }
                    for msg in state["messages"][-5:]
                ],
                "session_id": state["session_id"],
                "input": request.message,
                "dialect": state["current_user"].get("dialect", "普通话"),
                "gender": state["current_user"].get("gender", "男"),
                "username": state["current_user"].get("username", "用户"),
                "region": state["shop_info"]["region"],
                "name": state["shop_info"]["name"],
                "remark": state["shop_info"]["remark"],
                "preferences": state["current_user"].get("preferences", []),
                "promotions": state["shop_info"]["promotions"],
                "contact": state["shop_info"]["contact"]
            }

            try:
                if tool_name == "handle_payment":
                    print("[调试信息] 开始处理支付")
                    # 调用支付处理函数
                    response = handle_payment.invoke({})
                    print(f"[调试信息] 支付处理结果: {response}")
                    
                    # 更新状态
                    state["last_response"] = response
                    state["messages"].append(AIMessage(content=response))
                    
                    # 如果支付成功，设置退出标志并触发告别
                    if "成功" in response:
                        print("[调试信息] 支付成功，准备生成告别消息")
                        state["exit_flag"] = True
                        state["current_intent"] = "handle_goodbye"
                        
                        # 调用告别节点生成告别消息
                        goodbye_result = personalized_goodbye(state)
                        response = goodbye_result["last_response"]
                        print(f"[调试信息] 生成的告别消息: {response}")
                        
                        # 更新对话状态为已结束
                        dialogue_manager.append_message(request.session_id, {
                            "message_id": f"msg_{len(existing_dialogue.get('messages', [])) + 2}",
                            "type": "system",
                            "content": response,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "metadata": {
                                "tool_used": "handle_goodbye",
                                "intent": "handle_goodbye",
                                "exit_flag": True,
                                "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        })
                        
                        print("[调试信息] 准备返回告别响应")
                        return ChatResponse(
                            session_id=request.session_id,
                            response=response,
                            status="success"
                        )
                    else:
                        # 支付失败时添加重试提示
                        response = "支付失败，请检查余额或更换支付方式"
                        print(f"[调试信息] 支付失败，返回消息: {response}")
                elif tool_name == "handle_web_search":
                    print(f"[调试信息] 调用联网节点: {tool_name}")
                    try:
                        # 设置服务URL
                        service_url = ("http://appbuilder.baidu.com/v2/ai_search/mcp/sse?"
                                     "api_key=Bearer+bce-v3/ALTAK-lp91La4lRwuifo4dSNURU/70cb3ab0e2e87e267f6840f76e9fd052adfca877")
                        
                        # 直接使用fetch_search_results
                        search_results = await fetch_search_results(
                            query=tool_params["query"],
                            service_url=service_url
                        )
                        
                        # 构建提示模板
                        prompt = ChatPromptTemplate.from_template("""
                        用户信息: {username},性别{gender},使用{dialect}方言
                        搜索结果: {search_results}
                        使用{dialect}方言生成回答，要求:
                        1. 简短自然
                        2. 使用表情符号
                        3. 如果搜索结果为空或错误，说明无法获取实时信息
                        """)
                        
                        # 生成回答
                        response = (prompt | llm).invoke({
                            "username": tool_params["username"],
                            "gender": tool_params["gender"],
                            "dialect": tool_params["dialect"],
                            "search_results": search_results
                        })
                        
                        response = response.content
                    except Exception as e:
                        print(f"[错误] 搜索异常: {str(e)}")
                        response = f"抱歉，{tool_params['username']}，我暂时无法获取实时信息，请稍后再试。"
                else:
                    print(f"[调试信息] 调用其他工具节点: {tool_name}")
                    # 根据不同的工具函数传递相应的参数
                    if tool_name == "handle_shopping_guide":
                        response = node_manager.nodes[tool_name](
                            query=request.message,
                            remark=tool_params["remark"],
                            dialect=tool_params["dialect"],
                            gender=tool_params["gender"],
                            username=tool_params["username"]
                        )
                    elif tool_name == "handle_chitchat":
                        response = node_manager.nodes[tool_name](
                            region=tool_params["region"],
                            name=tool_params["name"],
                            dialect=tool_params["dialect"],
                            session_id=tool_params["session_id"],
                            history=tool_params["history"],
                            input=tool_params["input"],
                            gender=tool_params["gender"],
                            username=tool_params["username"]
                        )
                    elif tool_name == "handle_recommendation":
                        response = node_manager.nodes[tool_name](
                            query=request.message,
                            remark=tool_params["remark"],
                            preferences=tool_params["preferences"],
                            promotions=tool_params["promotions"],
                            dialect=tool_params["dialect"],
                            gender=tool_params["gender"],
                            username=tool_params["username"]
                        )
                    elif tool_name == "handle_human_transfer":
                        response = node_manager.nodes[tool_name](
                            contact=tool_params["contact"],
                            dialect=tool_params["dialect"],
                            gender=tool_params["gender"],
                            username=tool_params["username"]
                        )
                    elif tool_name == "handle_report":
                        response = node_manager.nodes[tool_name](
                            sales=random.randint(5000, 10000),
                            profit=random.randint(1500, 3000),
                            hot_products=["椰树椰汁", "农夫山泉"][:random.randint(1, 2)],
                            dialect=tool_params["dialect"],
                            gender=tool_params["gender"],
                            username=tool_params["username"]
                        )
                    elif tool_name == "handle_goodbye":
                        response = personalized_goodbye(state)["last_response"]
                    else:
                        raise HTTPException(status_code=500, detail=f"Unknown tool: {tool_name}")
            except Exception as e:
                print(f"[错误] 工具节点执行失败: {str(e)}")
                print(f"[错误] 工具名称: {tool_name}")
                print(f"[错误] 参数: {tool_params}")
                raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

        # 更新状态
        state["last_response"] = response
        state["messages"].append(AIMessage(content=response))

        if not response:
            print("[错误] 响应为空")
            raise HTTPException(status_code=500, detail="Failed to generate response")

        # 保存新的AI消息到数据库
        ai_msg = {
            "message_id": f"msg_{len(existing_dialogue.get('messages', [])) + 2}",
            "type": "system",
            "content": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": {
                "tool_used": state.get("current_intent", ""),
                "intent": state.get("current_intent", "")
            }
        }
        dialogue_manager.append_message(request.session_id, ai_msg)

        print("[调试信息] 准备返回最终响应")
        return ChatResponse(
            session_id=request.session_id,
            response=response,
            status="success"
        )

    except Exception as e:
        print(f"[错误] 处理请求时发生异常: {str(e)}")
        print(f"[错误] 异常类型: {type(e)}")
        import traceback
        print(f"[错误] 异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

def update_state(state: State, response: str) -> State:
    """更新对话状态"""
    # 添加新的AI消息到历史记录
    state["messages"].append(AIMessage(content=response))
    # 更新最后响应
    state["last_response"] = response
    return state

def personalized_goodbye(state: State):
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
    
    response = (prompt | llm).invoke({
        "username": state["current_user"]["username"],
        "gender": state["current_user"]["gender"],
        "dialect": state["current_user"]["dialect"],
        "region": state["shop_info"]["region"],
        "name": state["shop_info"]["name"],
        "purchased_items": ",".join(state.get("purchased_items", ["未购买商品"]))
    })
    
    print("[调试信息] 告别节点输出: " + response.content)
    
    # 持久化告别消息
    goodbye_msg = {
        "message_id": f"msg_{len(state['messages']) + 1}",
        "type": "system",
        "content": response.content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": {
            "intent": "handle_goodbye",
            "tool_used": "personalized_goodbye"
        }
    }
    dialogue_manager.append_message(state["session_id"], goodbye_msg)
    print('[调试信息] 告别词已持久化')
    
    return update_state(state, response.content)

def intent_recognition(state: State):
    # 获取最新消息
    latest_message = state["messages"][-1].content if state["messages"] else ""
    
    # 定义告别关键词列表
    goodbye_keywords = ["再见", "拜拜", "拜拜了", "再见啦", "再见喽", "拜拜喽", "拜拜啦", "结束", "退出", "quit", "exit", "bye", "goodbye", "走了"]
    
    # 定义支付关键词列表
    payment_keywords = ["结账", "付款", "支付", "买单", "收银", "扫码", "刷卡", "现金", "付钱", "给钱"]
    
    # 检查是否包含告别关键词
    if any(keyword in latest_message for keyword in goodbye_keywords):
        print("[调试信息] 检测到告别关键词，直接返回告别意图")
        return {
            "current_intent": "handle_goodbye",
            "exit_flag": True
        }
    
    # 检查是否包含支付关键词
    if any(keyword in latest_message for keyword in payment_keywords):
        print("[调试信息] 检测到支付关键词，直接返回支付意图")
        return {
            "current_intent": "handle_payment"
        }

    history = "\n".join([
        f"{'顾客' if isinstance(msg, HumanMessage) else '助手'}: {msg.content}"
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
    4. 只返回工具名称
    
    可选工具: 
    - handle_web_search: 查询实时联网信息
    - handle_chitchat: 日常对话/问候
    - handle_shopping_guide: 商品位置/价格问题
    - handle_recommendation: 明确要求推荐商品
    - handle_human_transfer: 转人工请求
    - handle_report: 销售数据查询
    - handle_payment: 支付相关
    - handle_goodbye: 告别/退出/离开/结束请求，包括但不限于：再见、拜拜、结束、退出、quit、exit等告别词
    只需返回工具名称""")

    msg = (prompt | llm).invoke({
        "message": state["messages"][-1].content,
        "history": history
    }).content
    print("[调试信息] 意图节点输出: " + msg)

    # 清理输出
    msg = msg.replace(' ', '').replace('\n', '')
    if '<think>' in msg and '</think>' in msg:
        start_index = msg.find('<think>')
        end_index = msg.find('</think>', start_index) + len('</think>')
        msg = msg[:start_index] + msg[end_index:]
    
    print("[调试信息] 清理后的意图: " + msg)

    # 如果识别到告别意图，设置退出标志
    if msg == "handle_goodbye" or state.get("exit_flag"):
        return {
            "current_intent": "handle_goodbye",
            "exit_flag": True
        }

    return {
        "current_intent": msg
    }

@tool("handle_payment")
def handle_payment() -> str:
    """处理支付"""
    start_time = time.time()  # 添加时间记录
    msg = "支付成功！请取走商品" if random.random() > 0.2 else "支付失败，请重试"

    response_time = time.time() - start_time  # 添加时间记录
    print(f"\n[AI响应时间] 生成耗时: {response_time:.2f} 秒")  # 添加时间记录
    print("[调试信息] 节点输出: " + msg)
    return msg

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 