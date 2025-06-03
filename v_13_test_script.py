import requests
import json
import time

def test_chat():
    base_url = "http://localhost:8000"
    session_id = "shop-123_user-123_20250603111110"
    
    # 多轮测试用例，每个元素为一组连续对话
    test_cases = [
        # 历史记忆A：连续问同一商品
        ["我想买可乐", "刚说的多少钱"],
        # 历史记忆B：推荐后直接下单
        ["推荐饮料", "就要你推荐的那个"],
        # 联网搜索E：商品价格对比
        ["可乐现在网上多少钱"],
        # 复杂混合F
        [
            "你好",
            "我想买可乐",
            "多少钱",
            "推荐点零食",
            "今天天气怎么样",
            "结账",
            "拜拜"
        ]
    ]
    
    for idx, case in enumerate(test_cases, 1):
        print(f"\n===== 测试用例{idx} 开始 =====")
        for message in case:
            print(f"\n测试消息: {message}")
            response = requests.post(
                f"{base_url}/chat/continue",
                json={
                    "session_id": session_id,
                    "message": message
                }
            )
            
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
            else:
                print(f"错误: {response.text}")
            
            time.sleep(1)  # 等待1秒再发送下一条消息
        print(f"===== 测试用例{idx} 结束 =====\n")

if __name__ == "__main__":
    test_chat() 