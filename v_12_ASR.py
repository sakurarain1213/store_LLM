import os
import sys
import time
import signal
import pyaudio
from datetime import datetime
from dashscope.audio.asr import *
import dashscope
import threading


# 待实际接入
class ASRHandler:
    """
    实时语音识别处理器（支持文件输入和麦克风实时输入）
    """

    def __init__(self,
                 input_source='mic',
                 model='paraformer-realtime-v2',
                 format='pcm',
                 sample_rate=16000,
                 file_path=None,
                 language_hints=None,
                 block_size=12800,
                 disfluency_removal_enabled=False):
        """
        实时语音识别类

        参数:
            input_source: 输入源类型 ('mic' 或 'file')
            model: 语音识别模型  paraformer-realtime-v2
            format: 音频格式 (pcm, wav等) 支持的音频格式：pcm、wav、mp3、opus、speex、aac、amr。重要 对于opus和speex格式的音频，需要ogg封装；对于wav格式的音频，需要pcm编码。
            sample_rate: 采样率 (Hz) 16000
            file_path: 文件路径（当输入源为文件时必需）
            language_hints: 语言提示（仅支持paraformer-realtime-v2模型） None
                zh: 中文  en: 英文  ja: 日语  yue: 粤语  ko: 韩语  de：德语  fr：法语  ru：俄语  传入方式: ["zh", "en"]
            block_size: 每次发送的音频块大小（字节） 3200
            disfluency_removal_enabled:是否过滤语气词 False
            自动断句等 更多参数详见文档 https://help.aliyun.com/zh/model-studio/paraformer-real-time-speech-recognition-python-api
        """

        # 配置基础参数
        self.input_source = input_source
        self.model = model
        self.format = format
        self.sample_rate = sample_rate
        self.file_path = file_path
        self.block_size = block_size
        self.language_hints = language_hints or []
        self.disfluency_removal_enabled = disfluency_removal_enabled

        # 初始化音频设备相关
        self.mic = None
        self.stream = None
        self.recognition = None
        self.is_running = False

        # 配置DashScope API
        if 'DASHSCOPE_API_KEY' in os.environ:
            dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
        else:
            dashscope.api_key = "sk-"
            # raise ValueError("请设置环境变量DASHSCOPE_API_KEY")

    def get_final_result(self):
        """获取最终识别文本"""
        if self.recognition and hasattr(self.recognition, '_callback'):
            callback = self.recognition._callback
            # 确保未完成的句子也被添加到结果中
            callback.get_full_text()
            # 返回连接后的字符串
            return ' '.join(callback.full_text)
        return ""

    class RecognitionCallback(RecognitionCallback):
        """
        自定义回调处理类  核心获取文本
        """

        def __init__(self, outer):
            self.outer = outer
            self.last_text = ""
            self.full_text = []  # 新增：存储完整文本的列表
            self.current_sentence = ""  # 新增：临时存储当前句子的中间结果

        def on_open(self):
            """麦克风输入初始化"""
            if self.outer.input_source == 'mic':
                self.outer.mic = pyaudio.PyAudio()
                self.outer.stream = self.outer.mic.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.outer.sample_rate,
                    input=True
                )
                print(f"[{self._get_timestamp()}] 麦克风已启动")

        def on_event(self, result: RecognitionResult) -> None:
            sentence = result.get_sentence()
            if 'text' in sentence:
                current_text = sentence['text']

                # 实时更新当前句子（无论是否结束）
                self.current_sentence = current_text  # 覆盖而非累积

                # 当检测到句子结束时提交到完整结果
                if RecognitionResult.is_sentence_end(sentence):
                    if current_text.strip():
                        self.full_text.append(current_text.strip())
                    self.current_sentence = ""  # 重置

                # 实时打印逻辑
                if current_text != self.last_text:
                    print(f"[{self._get_timestamp()}] 识别结果: {current_text}")
                    self.last_text = current_text

        def get_full_text(self):
            """将未完成的句子强制提交到完整结果"""
            if self.current_sentence.strip():
                self.full_text.append(self.current_sentence.strip())
                self.current_sentence = ""
            """获取完整文本"""
            return ' '.join(self.full_text)

        def on_error(self, result):
            """错误处理"""
            print(f"[ERROR] 请求ID: {result.request_id} | 错误信息: {result.message}")
            self.outer.stop()

        def _get_timestamp(self):
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def start(self):
        """启动语音识别流程"""
        # 初始化识别引擎
        self.recognition = Recognition(
            model=self.model,
            format=self.format,
            sample_rate=self.sample_rate,
            language_hints=self.language_hints if self.model == 'paraformer-realtime-v2' else None,
            semantic_punctuation_enabled=self.disfluency_removal_enabled,
            callback=self.RecognitionCallback(self)
        )

        # 信号处理（Ctrl+C终止）
        signal.signal(signal.SIGINT, self._signal_handler)

        # 启动识别线程
        self.recognition.start()
        self.is_running = True

        # 根据输入源选择数据读取方式
        if self.input_source == 'file':
            self._process_file()
        else:
            self._process_mic()

    def stop(self):
        """停止识别并释放资源"""
        try:
            if self.recognition and self.is_running:
                self.recognition.stop()
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 识别已停止")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 停止识别时出现异常: {str(e)}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.mic:
                self.mic.terminate()
            self.is_running = False

    def _process_file(self):
        """处理文件输入流"""
        try:
            with open(self.file_path, 'rb') as f:
                while self.is_running:
                    data = f.read(self.block_size)
                    if not data:
                        break
                    self.recognition.send_audio_frame(data)
                    time.sleep(0.1)  # 控制发送频率
        except Exception as e:
            print(f"文件处理错误: {str(e)}")
            self.stop()

    def _process_mic(self):
        """处理麦克风实时流"""
        print("请开始说话（按回车键结束）...")
        try:
            # 创建一个线程来监听回车键
            def check_enter():
                input("正在聆听...")  # 提供更清晰的提示
                self.is_running = False
            
            # 启动监听线程
            enter_thread = threading.Thread(target=check_enter)
            enter_thread.daemon = True  # 设置为守护线程，这样主程序退出时线程会自动结束
            enter_thread.start()
            
            # 主循环处理音频
            while self.is_running and self.stream and self.recognition:
                try:
                    data = self.stream.read(self.block_size,
                                          exception_on_overflow=False)
                    if self.is_running:  # 再次检查，避免在读取后发送前停止
                        self.recognition.send_audio_frame(data)
                except Exception as e:
                    if self.is_running:  # 只在仍然运行时打印错误
                        print(f"音频处理错误: {str(e)}")
                    break
        except Exception as e:
            print(f"麦克风处理错误: {str(e)}")
        finally:
            self.stop()  # 确保资源被释放

    def _signal_handler(self, sig, frame):
        """信号处理函数（保留作为备用）"""
        print("\n检测到终止信号...")
        self.stop()
        # sys.exit(0)  不要终止整个程序


if __name__ == "__main__":
    # 使用示例

    # asr = ASRHandler(
    #     input_source='file',
    #     file_path=r"C:\Users\w1625\Desktop\test.wav",
    #     model='paraformer-realtime-v2',
    #     format='wav',
    #     language_hints=["zh"],
    # )
    #
    # try:
    #     asr.start()
    # except KeyboardInterrupt:
    #     asr.stop()
    # final_text = asr.get_final_result()
    # print("\n完整结果：\n", final_text)

    # # 麦克风实时识别示例
    asr = ASRHandler(
        input_source='mic',
        language_hints=["zh", "en"],
        disfluency_removal_enabled=True
    )
    try:
        asr.start()
    except KeyboardInterrupt:
        asr.stop()
    final_text = asr.get_final_result()
    print("\n完整结果：\n", final_text)
