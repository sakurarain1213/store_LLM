#coding=utf-8

'''
流式TTS实现
火山引擎 websocket接口文档
https://www.volcengine.com/docs/6561/79821
用量信息
https://console.volcengine.com/speech/service/8?AppID=1124713874
全部费用信息
https://console.volcengine.com/finance/account-overview/

'''

import asyncio
import websockets
import uuid
import json
import gzip
import copy
import base64
import aiofiles
from typing import AsyncGenerator, Dict, Any
import io
import logging

# 消息类型定义
MESSAGE_TYPES = {11: "仅音频服务器响应", 12: "前端服务器响应", 15: "服务器错误消息"}
MESSAGE_TYPE_SPECIFIC_FLAGS = {0: "无序列号", 1: "序列号 > 0",
                               2: "服务器最后消息 (seq < 0)", 3: "序列号 < 0"}
MESSAGE_SERIALIZATION_METHODS = {0: "无序列化", 1: "JSON", 15: "自定义类型"}
MESSAGE_COMPRESSIONS = {0: "无压缩", 1: "gzip", 15: "自定义压缩方法"}

# 配置信息
appid = "9263344469"  # 请替换为您的实际appid
token = "3_WKsoT2pq_HlbO80fwzCoSToyzqqwpF"  # 请替换为您的实际token
cluster = "volcano_tts"  # 通常使用volcano_tts

# 可用的音色类型及其配置
VOICE_TYPES = {
    "BV700_V2_streaming": {
        "name": "灿灿 2.0",
        "emotions": ["通用", "愉悦", "抱歉", "嗔怪", "开心", "愤怒", "惊讶", "厌恶", "悲伤", "害怕", 
                    "哭腔", "客服", "专业", "严肃", "傲娇", "安慰鼓励", "绿茶", "娇媚", "情感电台", 
                    "撒娇", "瑜伽", "讲故事"],
        "languages": ["中文", "英语", "日语", "葡语", "西语", "印尼"]
    },
    "BV705_streaming": {
        "name": "炀炀",
        "emotions": ["通用", "自然对话", "愉悦", "抱歉", "嗔怪", "安慰鼓励", "讲故事"]
    },
    "BV701_V2_streaming": {
        "name": "擎苍 2.0",
        "emotions": ["旁白-舒缓", "旁白-沉浸", "平和", "开心", "悲伤", "生气", "害怕", "厌恶", "惊讶", "哭腔"]
    },
    "BV001_V2_streaming": {
        "name": "通用女声 2.0"
    },
    "BV700_streaming": {
        "name": "灿灿",
        "emotions": ["通用", "愉悦", "抱歉", "嗔怪", "开心", "愤怒", "惊讶", "厌恶", "悲伤", "害怕", 
                    "哭腔", "客服", "专业", "严肃", "傲娇", "安慰鼓励", "绿茶", "娇媚", "情感电台", 
                    "撒娇", "瑜伽", "讲故事"],
        "languages": ["中文", "英语", "日语", "葡语", "西语", "印尼"]
    },
    "BV406_V2_streaming": {
        "name": "超自然音色-梓梓2.0"
    },
    "BV406_streaming": {
        "name": "超自然音色-梓梓",
        "emotions": ["通用", "开心", "悲伤", "生气", "害怕", "厌恶", "惊讶"]
    },
    "BV407_V2_streaming": {
        "name": "超自然音色-燃燃2.0"
    },
    "BV407_streaming": {
        "name": "超自然音色-燃燃"
    },
    "BV001_streaming": {
        "name": "通用女声",
        "emotions": ["通用", "开心", "悲伤", "生气", "害怕", "厌恶", "惊讶", "助手", "客服", 
                    "安慰鼓励", "广告", "讲故事"]
    },
    "BV002_streaming": {
        "name": "通用男声"
    }
}

# 默认音色
voice_type = "zh_female_shuangkuaisisi_moon_bigtts"  # 默认使用灿灿 2.0   "BV700_V2_streaming"   一般是"zh_female_shuangkuaisisi_moon_bigtts"  # 示例语音类型
host = "openspeech.bytedance.com"
api_url = f"wss://{host}/api/v1/tts/ws_binary"

# 添加日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认头部信息
# 版本: b0001 (4 位)
# 头部大小: b0001 (4 位)
# 消息类型: b0001 (完整客户端请求) (4位)
# 消息类型特定标志: b0000 (无) (4位)
# 消息序列化方法: b0001 (JSON) (4 位)
# 消息压缩: b0001 (gzip) (4位)
# 保留数据: 0x00 (1 字节)
default_header = bytearray(b'\x11\x10\x11\x00')

# 请求JSON模板
request_json = {
    "app": {
        "appid": appid,
        "token": "access_token",
        "cluster": cluster
    },
    "user": {
        "uid": "388808087185088"
    },
    "audio": {
        "voice_type": voice_type,
        "encoding": "mp3",          # 音频编码格式：wav/pcm/ogg_opus/mp3
        "rate": 24000,              # 音频采样率：8000/16000/24000
        "compression_rate": 1,       # opus格式时编码压缩比：[1, 20]
        "speed_ratio": 1.0,         # 语速：[0.2, 3]
        "volume_ratio": 1.0,        # 音量：[0.1, 3]
        "pitch_ratio": 1.0,         # 音高：[0.1, 3]
        "emotion": "通用",          # 情感/风格
        "language": "cn"            # 语言类型
    },
    "request": {
        "reqid": "xxx",
        "text": "字节跳动语音合成。",
        "text_type": "plain",       # plain/ssml
        "silence_duration": 125,    # 句尾静音时长(ms)
        "operation": "submit",      # query/submit
        "with_timestamp": 1,        # 是否启用时间戳
        "pure_english_opt": 1       # 英文前端优化
    }
}


class StreamingTTS:
    """流式TTS类，用于处理文本到语音的转换"""
    
    def __init__(self, appid=appid, token=token, cluster=cluster, voice_type=voice_type):
        self.appid = appid
        self.token = token
        self.cluster = cluster
        self.voice_type = voice_type
        self.api_url = api_url
        logger.info(f"初始化TTS客户端: appid={appid}, cluster={cluster}, voice_type={voice_type}")
        
    async def synthesize_streaming(self, text: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式合成语音
        
        Args:
            text: 要合成的文本
            
        Yields:
            Dict: 包含音频数据和状态信息的字典
        """
        logger.info(f"开始合成文本: {text[:50]}...")
        
        # 准备请求JSON
        submit_request_json = copy.deepcopy(request_json)
        submit_request_json["app"]["appid"] = self.appid
        submit_request_json["app"]["token"] = self.token  # 确保使用正确的token
        submit_request_json["app"]["cluster"] = self.cluster
        submit_request_json["audio"]["voice_type"] = self.voice_type
        submit_request_json["request"]["reqid"] = str(uuid.uuid4())
        submit_request_json["request"]["text"] = text
        submit_request_json["request"]["operation"] = "submit"
        
        # 构建请求字节
        payload_bytes = str.encode(json.dumps(submit_request_json))
        payload_bytes = gzip.compress(payload_bytes)  # 压缩载荷
        full_client_request = bytearray(default_header)
        full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # 载荷大小(4字节)
        full_client_request.extend(payload_bytes)  # 载荷
        
        # 建立WebSocket连接
        header = {"Authorization": f"Bearer; {self.token}"}  # 使用分号分隔Bearer和token
        
        try:
            logger.info("正在连接TTS服务器...")
            async with websockets.connect(
                self.api_url,
                subprotocols=None,
                ping_interval=None,
                extra_headers=header
            ) as ws:
                logger.info("已连接到TTS服务器，发送请求...")
                await ws.send(full_client_request)
                
                audio_chunks = []
                while True:
                    res = await ws.recv()
                    audio_chunk, done = self._parse_response(res)
                    
                    if audio_chunk:
                        audio_chunks.append(audio_chunk)
                        # 实时返回音频片段
                        yield {
                            "type": "audio_chunk",
                            "data": base64.b64encode(audio_chunk).decode(),
                            "status": "streaming"
                        }
                    
                    if done:
                        # 合成完成，返回完整音频
                        complete_audio = b''.join(audio_chunks)
                        logger.info(f"语音合成完成，总大小: {len(complete_audio)} 字节")
                        yield {
                            "type": "audio_complete",
                            "data": base64.b64encode(complete_audio).decode(),
                            "status": "complete",
                            "size": len(complete_audio)
                        }
                        break
                        
        except Exception as e:
            logger.error(f"TTS合成失败: {str(e)}")
            yield {
                "type": "error",
                "message": str(e),
                "status": "error"
            }
    
    def _parse_response(self, res):
        """
        解析服务器响应
        
        Args:
            res: 服务器响应的字节数据
            
        Returns:
            tuple: (音频数据, 是否完成)
        """
        # 解析头部信息
        protocol_version = res[0] >> 4
        header_size = res[0] & 0x0f
        message_type = res[1] >> 4
        message_type_specific_flags = res[1] & 0x0f
        serialization_method = res[2] >> 4
        message_compression = res[2] & 0x0f
        reserved = res[3]
        header_extensions = res[4:header_size*4]
        payload = res[header_size*4:]
        
        if message_type == 0xb:  # 仅音频服务器响应
            if message_type_specific_flags == 0:  # 无序列号作为ACK
                return None, False
            else:
                sequence_number = int.from_bytes(payload[:4], "big", signed=True)
                payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                audio_data = payload[8:]
                
                # 如果序列号小于0，表示这是最后一个音频片段
                return audio_data, sequence_number < 0
                
        elif message_type == 0xf:  # 错误消息
            code = int.from_bytes(payload[:4], "big", signed=False)
            msg_size = int.from_bytes(payload[4:8], "big", signed=False)
            error_msg = payload[8:]
            if message_compression == 1:
                error_msg = gzip.decompress(error_msg)
            error_msg = str(error_msg, "utf-8")
            raise Exception(f"TTS服务器错误 若是3003则额度用完了 需要充钱 (代码: {code}): {error_msg}")
            
        elif message_type == 0xc:  # 前端服务器响应
            msg_size = int.from_bytes(payload[:4], "big", signed=False)
            payload = payload[4:]
            if message_compression == 1:
                payload = gzip.decompress(payload)
            print(f"前端消息: {payload}")
            return None, False
            
        else:
            raise Exception(f"未定义的消息类型: {message_type}")


class TextToSpeechAdapter:
    """
    文本流到语音流的适配器
    用于将现有的文本流接口转换为包含语音的流式响应
    """
    
    def __init__(self, tts_client: StreamingTTS):
        self.tts_client = tts_client
        self.text_buffer = ""
        self.sentence_endings = ['.', '!', '?', '。', '！', '？', '\n']
    
    async def process_text_stream(self, text_stream: AsyncGenerator[Dict, None]) -> AsyncGenerator[Dict, None]:
        """
        处理文本流并生成包含语音的混合流
        
        Args:
            text_stream: 原始文本流
            
        Yields:
            Dict: 文本或音频数据
        """
        async for item in text_stream:
            # 首先返回原始文本
            yield item
            
            if item.get("type") == "text" and item.get("status") == "success":
                text_chunk = item.get("response", "")
                self.text_buffer += text_chunk
                
                # 检查是否到达句子结尾
                if any(ending in text_chunk for ending in self.sentence_endings):
                    # 找到完整句子
                    sentences = self._extract_complete_sentences(self.text_buffer)
                    for sentence in sentences:
                        if sentence.strip():
                            # 为每个完整句子生成语音
                            async for audio_item in self.tts_client.synthesize_streaming(sentence.strip()):
                                yield audio_item
        
        # 处理剩余的文本缓冲区
        if self.text_buffer.strip():
            async for audio_item in self.tts_client.synthesize_streaming(self.text_buffer.strip()):
                yield audio_item
    
    def _extract_complete_sentences(self, text: str) -> list:
        """提取完整的句子"""
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in self.sentence_endings:
                sentences.append(current_sentence)
                current_sentence = ""
        
        # 更新缓冲区为未完成的句子
        self.text_buffer = current_sentence
        return sentences


# 示例使用函数
async def example_text_stream():
    """模拟现有的文本流接口"""
    text_responses = [
        {"type": "text", "response": "你好", "status": "success"},
        {"type": "text", "response": "，", "status": "success"},
        {"type": "text", "response": "欢迎", "status": "success"},
        {"type": "text", "response": "使用", "status": "success"},
        {"type": "text", "response": "流式", "status": "success"},
        {"type": "text", "response": "TTS", "status": "success"},
        {"type": "text", "response": "服务", "status": "success"},
        {"type": "text", "response": "。", "status": "success"},
        {"type": "text", "response": "这是", "status": "success"},
        {"type": "text", "response": "第二", "status": "success"},
        {"type": "text", "response": "句话", "status": "success"},
        {"type": "text", "response": "！", "status": "success"}
    ]
    
    for response in text_responses:
        yield response
        await asyncio.sleep(0.1)  # 模拟流式延迟


# async def test_streaming_tts():
#     """测试流式TTS功能"""
#     print("=== 测试单独的TTS功能 ===")
#     tts_client = StreamingTTS()
    
#     async for result in tts_client.synthesize_streaming("你好，这是一个测试语音合成。"):
#         print(f"TTS结果: {result['type']}, 状态: {result['status']}")
#         if result['type'] == 'audio_complete':
#             print(f"音频大小: {result['size']} 字节")


async def test_text_to_speech_adapter():
    """测试文本流到语音流的适配器"""
    print("\n=== 测试文本流适配器 ===")
    tts_client = StreamingTTS()
    adapter = TextToSpeechAdapter(tts_client)
    
    # 模拟文本流
    text_stream = example_text_stream()
    
    async for result in adapter.process_text_stream(text_stream):
        if result['type'] == 'text':
            print(f"文本: {result['response']}")
        elif result['type'] == 'audio_chunk':
            print(f"音频片段: {len(result['data'])} 字符 (base64)")
        elif result['type'] == 'audio_complete':
            print(f"完整音频: {result['size']} 字节")


# async def save_audio_example():
#     """保存音频文件的示例"""
#     print("\n=== 保存音频文件示例 ===")
#     tts_client = StreamingTTS()
    
#     audio_chunks = []
#     async for result in tts_client.synthesize_streaming("这是要保存到文件的语音内容。"):
#         if result['type'] == 'audio_chunk':
#             # 解码base64音频数据
#             audio_data = base64.b64decode(result['data'])
#             audio_chunks.append(audio_data)
#         elif result['type'] == 'audio_complete':
#             # 保存完整音频文件
#             complete_audio = base64.b64decode(result['data'])
#             async with aiofiles.open("output.mp3", "wb") as f:
#                 await f.write(complete_audio)
#             print(f"音频已保存到 output.mp3, 大小: {len(complete_audio)} 字节")


if __name__ == '__main__':
    # 运行测试
    loop = asyncio.get_event_loop()
    
    # 测试单独的TTS功能
    # loop.run_until_complete(test_streaming_tts())
    
    # 测试文本流适配器
    loop.run_until_complete(test_text_to_speech_adapter())
    
    # 保存音频文件示例
    # loop.run_until_complete(save_audio_example())