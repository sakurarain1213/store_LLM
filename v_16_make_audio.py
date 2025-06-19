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
    "default_2": {
        "text": "抱歉，小助手没听清楚呢，您可以：描述商品，随便聊聊，或呼叫真人客服。",
        "audioFile": "default_2.mp3"
    },
    "default_3": {
        "text": "已为您切换至舒缓模式，可以慢慢说哦~",
        "audioFile": "default_3.mp3"
    },
    "default_4": {
        "text": "您的语速较快，正在加速响应哦，请描述需求~",
        "audioFile": "default_4.mp3"
    },
    "default_5": {
        "text": "您可以告诉我需要什么，无人店小助手虽然不能吃零食，但超会找零食哦！",
        "audioFile": "default_5.mp3"
    },
    "default_6": {
        "text": "这个问题我可能不清楚，可以帮您介绍下店内商品和服务哦。",
        "audioFile": "default_6.mp3"
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