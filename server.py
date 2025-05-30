'''
TODO 虽然不影响结果 但是 修一下联网功能的异步报错 尽量不动v_12本身 
接口文档


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
    "session_id": "shop-123_user-123_20250530152916",
    "response": "\n哎哟，张三哥啊，欢迎光临西湖123便利店🌟🍀！今朝个天气真好，适合来我这家无人值守店逛逛哦。看这边，精选推荐给你看，可乐、香烟、烟酒都有，还有各种好烟供你挑选，比如黄鹤楼、好猫、N王冠、兰州、白沙、七星、娇子、中南海、贵烟，保证让你挑花眼😍🍻。\n\n今天还有活动哦，满30块减5块，买够30块还能送你一包口香糖🤤，快来试试手气吧！😉🎉\n\n杭州是个好地方，希望你能在这里找到心仪的商品，也祝你在杭州过得开心哦！🌸🌳",
    "status": "success",
    "error": null
}



继续对话
http://localhost:8000/chat/continue
Content-Type: application/json
入参
{
    "session_id": "shop-123_user-123_20250530152916",
    "message": "结账"
}
出参
{
    "session_id": "shop-123_user-123_20250530152916",
    "response": "支付成功！请取走商品",
    "status": "success",
    "error": null
}


'''


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal
import uvicorn
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

app = FastAPI(title="智能对话系统API")

# 初始化节点管理器
node_manager = NodeManager()

# 注册基础节点
node_manager.register_node("entry", entry_node)
node_manager.register_node("welcome", personalized_welcome)
node_manager.register_node("collect_input", collect_human_input)
node_manager.register_node("intent_recognition", intent_recognition)
node_manager.register_node("goodbye", personalized_goodbye)

# 注册工具节点
node_manager.register_node("handle_chitchat", handle_chitchat)
node_manager.register_node("handle_shopping_guide", handle_shopping_guide)
node_manager.register_node("handle_recommendation", handle_recommendation)
node_manager.register_node("handle_human_transfer", handle_human_transfer)
node_manager.register_node("handle_report", handle_report)
node_manager.register_node("handle_payment", handle_payment)

# 设置入口点
node_manager.add_edge("entry", "welcome")
node_manager.add_edge("welcome", "collect_input")
node_manager.add_edge("collect_input", "intent_recognition")

# 添加工具节点路由
tools = ["handle_chitchat", "handle_shopping_guide", "handle_recommendation",
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
            lambda state: "collect_input" if not state.get("exit_flag") else "goodbye",
            {"collect_input": "collect_input", "goodbye": "goodbye"}
        )

# 结束流程
node_manager.add_edge("goodbye", END)

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
class NewChatRequest(BaseModel):
    user_id: str
    dialect: str
    gender: Literal["男", "女", "其他"]
    preferences: List[str]
    username: str

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
        # 构建初始状态
        initial_state = {
            "current_user": {
                "user_id": request.user_id,
                "dialect": request.dialect,
                "gender": request.gender,
                "preferences": request.preferences,
                "username": request.username
            },
            "shop_info": sample_shop,
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

        return ChatResponse(
            session_id=session_id,
            response=welcome_response,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/continue", response_model=ChatResponse)
async def continue_chat(request: ChatRequest):
    """继续现有对话"""
    try:
        # 获取现有对话状态
        existing_dialogue = dialogue_manager.get_dialogue(request.session_id)
        if not existing_dialogue:
            raise HTTPException(status_code=404, detail="Session not found")

        # 获取历史消息
        history_messages = []
        for msg in existing_dialogue.get("messages", []):
            if msg["type"] == "human":
                history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "system":
                history_messages.append(AIMessage(content=msg["content"]))

        # 添加当前消息
        history_messages.append(HumanMessage(content=request.message))

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

        # 直接调用意图识别节点
        intent_result = intent_recognition(state)
        state.update(intent_result)

        # 根据意图调用相应的工具节点
        if state.get("current_intent") == "handle_web_search":
            # 使用异步方式调用web搜索
            try:
                service_url = "http://appbuilder.baidu.com/v2/ai_search/mcp/sse?" + \
                            "api_key=Bearer+bce-v3/ALTAK-lp91La4lRwuifo4dSNURU/70cb3ab0e2e87e267f6840f76e9fd052adfca877"
                
                # 直接使用当前事件循环
                search_results = await fetch_search_results(request.message, service_url)
                
                # 使用搜索结果生成响应
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
                
                chain = prompt | llm
                response = chain.invoke({
                    "context": "\n\n".join(search_results) if isinstance(search_results, list) else search_results,
                    "query": request.message,
                    "dialect": state["current_user"].get("dialect", "普通话"),
                    "username": state["current_user"].get("username", "用户"),
                    "gender": state["current_user"].get("gender", "男"),
                }).content
            except Exception as e:
                print(f"[错误] 搜索异常: {str(e)}")
                response = "抱歉，暂时无法获取实时信息，请稍后再试。"
        elif state.get("current_intent") in tools:
            tool_name = state["current_intent"]
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
                    contact=tool_params["contact"]
                )
            elif tool_name == "handle_report":
                response = node_manager.nodes[tool_name](
                    sales=random.randint(5000, 10000),
                    profit=random.randint(1500, 3000),
                    hot_products=["椰树椰汁", "农夫山泉"][:random.randint(1, 2)]
                )
            elif tool_name == "handle_payment":
                response = node_manager.nodes[tool_name]()
            else:
                raise HTTPException(status_code=500, detail=f"Unknown tool: {tool_name}")

        # 更新状态
        state["last_response"] = response
        state["messages"].append(AIMessage(content=response))

        if not response:
            raise HTTPException(status_code=500, detail="Failed to generate response")

        # 保存新的AI消息到数据库
        ai_msg = {
            "message_id": f"msg_{len(existing_dialogue.get('messages', [])) + 1}",
            "type": "system",
            "content": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": {
                "tool_used": state.get("current_intent", ""),
                "intent": state.get("current_intent", "")
            }
        }
        dialogue_manager.append_message(request.session_id, ai_msg)

        return ChatResponse(
            session_id=request.session_id,
            response=response,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 