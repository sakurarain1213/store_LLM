import asyncio
import httpx
import json
from appbuilder.mcp_server.client import MCPClient
from typing import List, Union, Optional


async def fetch_search_results(query: str, service_url: str) -> List[str]:
    """获取搜索结果，添加错误处理和降级机制"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 构建请求参数
            params = {
                "query": query,
                "top_k": 3,  # 限制返回结果数量
                "stream": False  # 使用非流式响应
            }
            
            # 发送请求
            response = await client.post(
                service_url,
                json=params,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict) and "results" in data:
                        return [item.get("content", "") for item in data["results"] if item.get("content")]
                    return ["无法解析搜索结果"]
                except json.JSONDecodeError:
                    return ["搜索结果格式错误"]
            else:
                return [f"搜索服务返回错误: {response.status_code}"]
                
    except httpx.TimeoutException:
        return ["搜索请求超时，请稍后重试"]
    except httpx.RequestError as e:
        return [f"网络请求错误: {str(e)}"]
    except Exception as e:
        return [f"搜索过程发生错误: {str(e)}"]
    finally:
        # 确保客户端被正确关闭
        await client.aclose()


# 示例使用
if __name__ == "__main__":
    service_url = ("http://appbuilder.baidu.com/v2/ai_search/mcp/sse?"
                   "api_key=Bearer+bce-v3/ALTAK-lp91La4lRwuifo4dSNURU/70cb3ab0e2e87e267f6840f76e9fd052adfca877")
    query = "杭州天气"

    # 执行异步函数
    loop = asyncio.get_event_loop()
    ans = loop.run_until_complete(fetch_search_results(query, service_url))
    print(ans)

# 降级处理函数
def get_fallback_response(query: str) -> List[str]:
    """当联网搜索失败时提供降级响应"""
    fallback_responses = [
        "抱歉，当前网络连接不稳定，无法获取实时信息。",
        "您可以尝试询问店内商品信息或促销活动。",
        "或者稍后再试。"
    ]
    return fallback_responses
