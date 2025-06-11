import asyncio
from appbuilder.mcp_server.client import MCPClient
from typing import List, Union, AsyncGenerator
import json


async def fetch_search_results(query: str, service_url: str, search_top_k: int = 1,
                               search_recency_filter: str = "week") -> AsyncGenerator[str, None]:
    """执行百度MCP搜索并返回流式结果"""
    client = None
    try:
        # 初始化MCPClient
        client = MCPClient()
        await client.connect_to_server(service_url=service_url)

        # 调用MCP搜索工具
        async for chunk in client.call_tool_stream(
            "AIsearch",
            {
                "query": query,
                "search_top_k": search_top_k,
                "instruction": "过滤广告,只要摘要",  # 过滤广告，返回摘要
                "search_recency_filter": search_recency_filter,  # 最近一周
                "search_resource_type": [{"type": "web", "top_k": 10}]
            }
        ):
            if chunk and hasattr(chunk, 'content') and chunk.content:
                # 将每个chunk转换为SSE格式
                yield f"data: {json.dumps({'response': chunk.content[0].text, 'status': 'success'})}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run

    except Exception as e:
        print(f"[fetch_search_results] 错误发生: {str(e)}")
        yield f"data: {json.dumps({'response': f'搜索发生错误: {str(e)}', 'status': 'error'})}\n\n"
    finally:
        # 移除对close方法的调用，因为MCPClient没有这个方法
        pass


# 示例使用
if __name__ == "__main__":
    service_url = ("http://appbuilder.baidu.com/v2/ai_search/mcp/sse?"
                   "api_key=Bearer+bce-v3/ALTAK-lp91La4lRwuifo4dSNURU/70cb3ab0e2e87e267f6840f76e9fd052adfca877")
    query = "杭州天气"

    # 执行异步函数
    async def main():
        async for result in fetch_search_results(query, service_url):
            print(result)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
