# coding=utf-8
# pip install pyaudio

# 科大讯飞 TTS websocket 流式接口


import websocket
import json
import base64
import hmac
import hashlib
import time
import threading
from datetime import datetime
from urllib.parse import urlencode, urlparse
import pyaudio
import wave
import io
import tempfile
import os


class TTSHandler:
    def __init__(self, app_id="c5f02e97", api_key="9b5cbc5e7d7b48e3f91ff29f4b32e599", 
                 api_secret="ZjY2ZTEzNmNhODM0NGNkZTI0ZDlhODBi"):
        """
        初始化TTS处理器
        
        Args:
            app_id: 应用ID
            api_key: API密钥
            api_secret: API密钥
        """
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "wss://tts-api.xfyun.cn/v2/tts"
        
        # 音频数据存储
        self.audio_data = []
        self.synthesis_complete = False
        self.synthesis_error = None
        
        # 音频播放配置
        self.sample_rate = 16000
        self.channels = 1
        self.sample_width = 2  # 16位 = 2字节
        
    def _generate_auth_url(self):
        """
        生成认证URL
        """
        # 解析URL
        parsed_url = urlparse(self.base_url)
        host = parsed_url.netloc
        path = parsed_url.path
        
        # 生成时间戳 (UTC时间，RFC1123格式)
        now = datetime.utcnow()
        date = now.strftime('%a, %d %b %Y %H:%M:%S GMT')
        
        # 构造签名字符串
        signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
        
        # 使用HMAC-SHA256签名
        signature_sha = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        # Base64编码签名
        signature = base64.b64encode(signature_sha).decode('utf-8')
        
        # 构造authorization原始字符串
        authorization_origin = (
            f'api_key="{self.api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature}"'
        )
        
        # Base64编码authorization
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
        
        # 构造查询参数
        params = {
            'authorization': authorization,
            'date': date,
            'host': host
        }
        
        return f"{self.base_url}?{urlencode(params)}"
    
    def _on_message(self, ws, message):
        """
        处理WebSocket消息
        """
        try:
            data = json.loads(message)
            code = data.get('code', -1)
            
            if code != 0:
                self.synthesis_error = f"错误码: {code}, 信息: {data.get('message', '未知错误')}"
                return
            
            # 获取音频数据
            audio_data = data.get('data', {})
            if audio_data:
                audio_base64 = audio_data.get('audio')
                if audio_base64:
                    # 解码音频数据
                    audio_bytes = base64.b64decode(audio_base64)
                    self.audio_data.append(audio_bytes)
                
                # 检查是否合成完成
                status = audio_data.get('status', 0)
                if status == 2:
                    self.synthesis_complete = True
                    
        except Exception as e:
            self.synthesis_error = f"解析消息失败: {str(e)}"
    
    def _on_error(self, ws, error):
        """
        处理WebSocket错误
        """
        self.synthesis_error = f"WebSocket错误: {str(error)}"
    
    def _on_close(self, ws, close_status_code, close_msg):
        """
        处理WebSocket关闭
        """
        pass
    
    def _on_open(self, ws):
        """
        WebSocket连接打开后发送请求数据
        """
        def run():
            # 构造请求参数
            params = {
                "common": {
                    "app_id": self.app_id
                },
                "business": {
                    "aue": "raw",  # 原始PCM格式
                    "vcn": "x4_xiaoyan",  # 发音人xiaoyan
                    "speed": 50,  # 语速
                    "volume": 50,  # 音量
                    "pitch": 50,  # 音高
                    "auf": "audio/L16;rate=16000",  # 16k采样率
                    "tte": "UTF8"  # 文本编码格式
                },
                "data": {
                    "status": 2,
                    "text": base64.b64encode(self.text_to_synthesize.encode('utf-8')).decode('utf-8')
                }
            }
            
            '''   
            讯飞小燕 普通话 x4_xiaoyan
            讯飞小露 普通话 x4_yezi
            讯飞许久 普通话 aisjiuxu
            讯飞小婧 普通话 aisjinger
            讯飞许小宝 普通话 aisbabyxu
            '''
            
            # 发送数据
            ws.send(json.dumps(params))
        
        # 启动发送线程
        threading.Thread(target=run).start()
    
    def synthesize_text(self, text, timeout=30):
        """
        合成文本为音频数据
        
        Args:
            text: 要合成的文本
            timeout: 超时时间（秒）
            
        Returns:
            bytes: 音频数据，失败返回None
        """
        # 重置状态
        self.audio_data = []
        self.synthesis_complete = False
        self.synthesis_error = None
        self.text_to_synthesize = text
        
        # 生成认证URL
        auth_url = self._generate_auth_url()
        
        # 创建WebSocket连接
        ws = websocket.WebSocketApp(
            auth_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        # 启动WebSocket连接
        def run_ws():
            ws.run_forever()
        
        ws_thread = threading.Thread(target=run_ws)
        ws_thread.daemon = True
        ws_thread.start()
        
        # 等待合成完成或超时
        start_time = time.time()
        while not self.synthesis_complete and not self.synthesis_error:
            if time.time() - start_time > timeout:
                ws.close()
                raise TimeoutError(f"语音合成超时 ({timeout}秒)")
            time.sleep(0.1)
        
        # 关闭连接
        ws.close()
        
        # 检查错误
        if self.synthesis_error:
            raise Exception(self.synthesis_error)
        
        # 合并音频数据
        if self.audio_data:
            return b''.join(self.audio_data)
        else:
            return None
    
    def play_audio(self, audio_data):
        """
        播放音频数据
        
        Args:
            audio_data: PCM音频数据
        """
        try:
            # 初始化pyaudio
            p = pyaudio.PyAudio()
            
            # 打开音频流
            stream = p.open(
                format=pyaudio.paInt16,  # 16位
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )
            
            # 播放音频
            chunk_size = 1024
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                stream.write(chunk)
            
            # 清理资源
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            print(f"播放音频失败: {str(e)}")
    
    def save_audio(self, audio_data, filename):
        """
        保存音频到文件
        
        Args:
            audio_data: PCM音频数据
            filename: 文件名
        """
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
            print(f"音频已保存到: {filename}")
        except Exception as e:
            print(f"保存音频失败: {str(e)}")
    
    def synthesize_and_play(self, text):
        """
        合成文本并播放音频
        
        Args:
            text: 要合成的文本
        """
        try:
            print(f"正在合成:")
            # print(text)
            audio_data = self.synthesize_text(text)
            
            if audio_data:
                print("合成完成，正在播放...")
                self.play_audio(audio_data)
                print("播放完成")
            else:
                print("合成失败：未获取到音频数据")
                
        except Exception as e:
            print(f"语音合成失败: {str(e)}")
    
    def synthesize_and_save(self, text, filename):
        """
        合成文本并保存音频文件
        
        Args:
            text: 要合成的文本
            filename: 保存的文件名
        """
        try:
            print(f"正在合成: {text}")
            audio_data = self.synthesize_text(text)
            
            if audio_data:
                self.save_audio(audio_data, filename)
            else:
                print("合成失败：未获取到音频数据")
                
        except Exception as e:
            print(f"语音合成失败: {str(e)}")


# 使用示例
if __name__ == "__main__":
    print("程序开始执行...")
    try:
        # 创建TTS处理器实例
        print("正在初始化TTS处理器...")
        tts = TTSHandler()
        print("TTS处理器初始化完成")
        
        print("test")
        # 示例1: 合成并播放
        print("开始合成语音...")
        tts.synthesize_and_play("现在时间是下午三点，当前室外温度28度，空气质量优")
        print("语音合成完成")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())