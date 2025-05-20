# 使用alibaba的百炼平台接口

# coding=utf-8

import os
import subprocess

import dashscope
from dashscope.audio.tts_v2 import *


class TTSHandler:
    def __init__(self, api_key="sk-effb15582c0a45a38120d65b1a2ad05a",
                 model="cosyvoice-v2",
                 voice="longhua_v2",
                 format=AudioFormat.WAV_22050HZ_MONO_16BIT):
        """
        初始化语音合成器
        :param api_key: 阿里云API密钥
        :param model: 语音模型，默认cosyvoice-v2
        :param voice: 音色选择，默认longxiaochun_v2
        voice = "longxiaochun_v2"  # 中文普通话+英文
        # longcheng_v2  中文普通话 男
        # longhua_v2 中文普通话  儿童
        # longwan_v2 中文普通话  女
        # longshu_v2 中文普通话  男解说
        # loongbella_v2 中文普通话  女解说

        # longxiaoxia_v2 中文普通话
        :param format: 音频格式，默认WAV_22050HZ_MONO_16BIT
        """
        dashscope.api_key = api_key
        self.model = model  # 保存模型参数
        self.voice = voice  # 保存音色参数
        self.format = format  # 保存音频格式参数
        self.temp_file = "temp_output." + ("wav" if format == AudioFormat.WAV_22050HZ_MONO_16BIT else "mp3")

    def synthesize_and_play(self, text):
        """
        文本转语音并自动播放（每次调用新建SpeechSynthesizer实例）
        :param text: 待转换文本
        """
        try:
            # 每次调用创建新的SpeechSynthesizer实例
            synthesizer = SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                format=self.format
            )

            # 生成语音数据
            audio = synthesizer.call(text)

            # 检查返回的音频数据是否有效
            if not audio:
                raise ValueError("合成空数据")

            # 保存临时文件
            with open(self.temp_file, 'wb') as f:
                f.write(audio)

            # 调用系统播放器（阻塞直到播放完成）
            subprocess.run(
                ["cmd", "/c", "start", "/wait", "", self.temp_file],
                shell=True,
                check=True
            )
            print("已播放")

        except Exception as e:
            print(f"语音合成或播放失败：{str(e)}")
        finally:
            # 确保清理临时文件
            if os.path.exists(self.temp_file):
                try:
                    os.remove(self.temp_file)
                    print(f"临时文件 {self.temp_file} 已清理")
                except Exception as e:
                    print(f"清理临时文件失败：{str(e)}")

# 将 SpeechSynthesizer 的实例化从 __init__ 移动到 synthesize_and_play 方法内部，确保每次调用都使用全新的实例，避免阿里云 SDK 可能的内部状态残留问题。
#  用法
# tts = TTSHandler(api_key="sk-effb15582c0a45a38120d65b1a2ad05a")
# tts.synthesize_and_play("现在时间是下午三点，当前室外温度28度，空气质量优")
