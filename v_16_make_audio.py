#coding=utf-8

import asyncio
import base64
import aiofiles
from v_16_TTS import StreamingTTS

async def generate_audio(text: str, output_file: str = "output.mp3"):
    """
    生成单个音频文件
    
    Args:
        text: 要转换为语音的文本
        output_file: 输出文件名，默认为 output.mp3
    """
    print(f"开始生成音频: {text}")
    
    # 初始化TTS客户端
    tts_client = StreamingTTS()
    
    # 收集音频数据
    audio_chunks = []
    
    try:
        # 生成语音
        async for result in tts_client.synthesize_streaming(text):
            if result['type'] == 'audio_chunk':
                # 解码base64音频数据
                audio_data = base64.b64decode(result['data'])
                audio_chunks.append(audio_data)
            elif result['type'] == 'audio_complete':
                # 保存完整音频文件
                complete_audio = base64.b64decode(result['data'])
                async with aiofiles.open(output_file, "wb") as f:
                    await f.write(complete_audio)
                print(f"音频已保存到 {output_file}, 大小: {len(complete_audio)} 字节")
                return True
            elif result['type'] == 'error':
                print(f"生成音频时出错: {result['message']}")
                return False
                
    except Exception as e:
        print(f"生成音频时发生异常: {str(e)}")
        return False

async def main():
    # 定义所有需要生成的音频场景
    scenarios = {
        "chitchat": {
            "text": "小助手很高兴和您聊天哦~",
            "audioFile": "chitchat.mp3"
        },
        "shopping_guide": {
            "text": "让我为您介绍一下店内的商品吧~",
            "audioFile": "shopping_guide.mp3"
        },
        "recommendation": {
            "text": "让我为您推荐一些不错的商品~",
            "audioFile": "recommendation.mp3"
        },
        "payment": {
            "text": "好的，让我帮您处理支付问题~",
            "audioFile": "payment.mp3"
        },
        "goodbye": {
            "text": "小助手在这里感谢您的光临~",
            "audioFile": "goodbye.mp3"
        },
        "web_search": {
            "text": "让我在网络上查询一下相关信息~",
            "audioFile": "web_search.mp3"
        },
        "human_transfer": {
            "text": "我这就帮您转接人工客服~",
            "audioFile": "human_transfer.mp3"
        },
        "report": {
            "text": "好的，我来汇报一下店内的状况哦~",
            "audioFile": "report.mp3"
        },
        "default": {
            "text": "小助手不是很明白呢，我可以导购，推荐，或者随便聊聊天哦。",
            "audioFile": "default.mp3"
        }
    }
    
    # 为每个场景生成音频文件
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n开始生成 {scenario_name} 场景的音频...")
        success = await generate_audio(scenario_data["text"], scenario_data["audioFile"])
        if success:
            print(f"{scenario_name} 场景音频生成成功！")
        else:
            print(f"{scenario_name} 场景音频生成失败！")

if __name__ == '__main__':
    # 运行主函数
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main()) 