import asyncio
from appbuilder.mcp_server.client import MCPClient
from typing import List, Union


async def fetch_search_results(query: str, service_url: str, search_top_k: int = 1,
                               search_recency_filter: str = "week") -> Union[List[str], str]:
    """执行百度MCP搜索并返回结果列表"""
    client = None
    try:
        # 初始化MCPClient
        client = MCPClient()
        await client.connect_to_server(service_url=service_url)

        # 调用MCP搜索工具
        result = await client.call_tool(
            "AIsearch",
            {
                "query": query,
                "search_top_k": search_top_k,
                "instruction": "过滤广告,只要摘要",  # 过滤广告，返回摘要
                "search_recency_filter": search_recency_filter,  # 最近一周
                "search_resource_type": [{"type": "web", "top_k": 10}]
            }
        )

        # 提取搜索结果文本内容
        # raw_texts = [result.content[i].text for i in range(len(result.content))]
        if result and hasattr(result, 'content') and result.content:
            raw_texts = result.content[0].text
            return raw_texts
        else:
            return "未找到相关信息"

    except Exception as e:
        print(f"[fetch_search_results] 错误发生: {str(e)}")
        return f"搜索发生错误: {str(e)}"
    finally:
        # 移除对close方法的调用，因为MCPClient没有这个方法
        pass


# 示例使用
if __name__ == "__main__":
    service_url = ("http://appbuilder.baidu.com/v2/ai_search/mcp/sse?"
                   "api_key=Bearer+")
    query = "杭州天气"

    # 执行异步函数
    loop = asyncio.get_event_loop()
    ans = loop.run_until_complete(fetch_search_results(query, service_url))
    print(ans)
