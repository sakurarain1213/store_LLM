# 混合文本音频流实现
import asyncio
import json
import base64
from typing import AsyncGenerator, Dict, Any, Optional
from dataclasses import dataclass
from queue import Queue
import threading
import time

# 音频队列管理器
class AudioQueueManager:
    """管理音频生成队列，支持取消功能"""
    
    def __init__(self, tts_client):
        self.tts_client = tts_client
        self.queues = {}  # session_id -> AudioQueue
        
    def get_or_create_queue(self, session_id: str) -> 'AudioQueue':
        """获取或创建音频队列"""
        if session_id not in self.queues:
            self.queues[session_id] = AudioQueue(self.tts_client, session_id)
        return self.queues[session_id]
    
    def cancel_queue(self, session_id: str):
        """取消指定会话的音频队列"""
        if session_id in self.queues:
            self.queues[session_id].cancel()
            del self.queues[session_id]

class AudioQueue:
    """音频生成队列 - 优化版本"""
    
    def __init__(self, tts_client, session_id: str):
        self.tts_client = tts_client
        self.session_id = session_id
        self.cancelled = False
        self.sentence_counter = 0
        self.active_tasks = set()  # 跟踪活跃任务
        self.next_sentence_id = 1  # 下一个要输出的句子ID
        self.pending_audio = {}  # 存储待输出的音频 {sentence_id: audio_chunk}
        
    async def add_sentence(self, text: str) -> int:
        """添加句子到队列 - 立即并行处理"""
        if self.cancelled:
            return -1
            
        self.sentence_counter += 1
        sentence_id = self.sentence_counter
        
        # 立即创建并行任务，不等待队列
        task = asyncio.create_task(
            self._generate_audio_direct(sentence_id, text.strip())
        )
        self.active_tasks.add(task)
        
        # 清理完成的任务
        task.add_done_callback(lambda t: self.active_tasks.discard(t))
            
        return sentence_id
    
    def cancel(self):
        """取消所有音频任务"""
        self.cancelled = True
        for task in list(self.active_tasks):
            if not task.done():
                task.cancel()
        self.active_tasks.clear()
    
    async def _generate_audio_direct(self, sentence_id: int, text: str):
        """直接生成音频 - 移除队列等待"""
        try:
            if self.cancelled:
                return
                
            # 直接调用TTS，移除中间环节
            audio_data = await self._synthesize_audio(text)
            
            if not self.cancelled and audio_data:
                # 存储到待输出队列
                audio_chunk = {
                    "type": "audio_chunk",
                    "sentence_id": sentence_id,
                    "text": text,
                    "data": audio_data
                }
                self.pending_audio[sentence_id] = audio_chunk
                
        except Exception as e:
            print(f"[音频生成错误] {e}")
    
    async def get_next_audio(self) -> Optional[dict]:
        """获取下一个要输出的音频"""
        # 检查是否有下一个要输出的音频
        if self.next_sentence_id in self.pending_audio:
            audio_chunk = self.pending_audio.pop(self.next_sentence_id)
            self.next_sentence_id += 1
            return audio_chunk
        return None
    
    async def _synthesize_audio(self, text: str) -> Optional[str]:
        """优化的TTS合成 - 减少延迟"""
        try:
            # 移除chunk合并逻辑，直接等待完整音频
            async for chunk in self.tts_client.synthesize_streaming(text):
                if chunk.get("type") == "audio_complete":
                    return chunk.get("data")
            return None
        except Exception as e:
            print(f"[TTS合成错误] {e}")
            return None


# 全局音频结果队列
audio_result_queue = asyncio.Queue()

# 全局音频队列管理器
audio_queue_manager = None

def init_audio_system(tts_client):
    """初始化音频系统"""
    global audio_queue_manager
    audio_queue_manager = AudioQueueManager(tts_client)









# 文本处理器
class TextProcessor:
    """优化的文本处理器"""
    
    def __init__(self):
        # 简化句子结束符
        self.sentence_endings = {'.', '!', '?', '。', '！', '？'}
        self.text_buffer = ""
        self.min_sentence_length = 5  # 最小句子长度
    
    def process_char(self, char: str) -> Optional[str]:
        """优化的字符处理"""
        self.text_buffer += char
        
        if char in self.sentence_endings and len(self.text_buffer.strip()) >= self.min_sentence_length:
            sentence = self.text_buffer.strip()
            self.text_buffer = ""
            return sentence
        
        # 对于较长文本，也可以在逗号处分割以提前生成音频
        if len(self.text_buffer) > 50 and char in {',', '，', ';', '；'}:
            sentence = self.text_buffer.strip()
            self.text_buffer = ""
            return sentence
            
        return None
        
    def get_remaining_text(self) -> Optional[str]:
        """获取缓冲区中剩余的文本"""
        if self.text_buffer.strip():
            remaining = self.text_buffer.strip()
            self.text_buffer = ""
            return remaining
        return None



# 混合流生成器
# async def create_mixed_stream(
#     original_stream: AsyncGenerator[str, None],
#     session_id: str
# ) -> AsyncGenerator[str, None]:
#     """
#     创建混合文本+音频流
    
#     Args:
#         original_stream: 原始文本流
#         session_id: 会话ID
#     """
#     if not audio_queue_manager:
#         # 如果音频系统未初始化，只返回原始文本流
#         async for chunk in original_stream:
#             yield chunk
#         return
    
#     text_processor = TextProcessor()
#     audio_queue = audio_queue_manager.get_or_create_queue(session_id)
    
#     # 创建音频结果监听任务
#     audio_task = asyncio.create_task(
#         _handle_audio_results(session_id)
#     )
    
#     try:
#         # 处理原始文本流
#         async for chunk in original_stream:
#             # 立即返回文本chunk
#             yield chunk
            
#             # 提取文本内容并检查句子边界
#             if chunk.startswith("data: "):
#                 try:
#                     data = json.loads(chunk[6:])
#                     if data.get("type") == "text" and "response" in data:
#                         char = data["response"]
#                         sentence = text_processor.process_char(char)
                        
#                         if sentence:
#                             # 异步添加到音频队列
#                             await audio_queue.add_sentence(sentence)
                            
#                 except json.JSONDecodeError:
#                     pass
        
#         # 处理剩余文本
#         remaining_text = text_processor.get_remaining_text()
#         if remaining_text:
#             await audio_queue.add_sentence(remaining_text)
        
#         # 等待一段时间让音频处理完成
#         await asyncio.sleep(2.0)
        
#     finally:
#         # 清理
#         audio_task.cancel()
#         try:
#             await audio_task
#         except asyncio.CancelledError:
#             pass

# async def _handle_audio_results(session_id: str):
#     """处理音频结果并发送到客户端"""
#     while True:
#         try:
#             # 等待音频结果
#             audio_result = await asyncio.wait_for(
#                 audio_result_queue.get(), timeout=1.0
#             )
            
#             # 检查是否属于当前会话
#             if audio_result.get("session_id") == session_id:
#                 # 发送音频结果
#                 audio_chunk = {
#                     "type": "audio_chunk",
#                     "sentence_id": audio_result["sentence_id"],
#                     "text": audio_result["text"],
#                     "data": audio_result["data"]
#                 }
                
#                 # 这里需要通过某种方式发送到客户端
#                 # 由于无法直接yield，我们将结果放到会话特定的队列中
#                 await _send_audio_to_session(session_id, audio_chunk)
                
#         except asyncio.TimeoutError:
#             continue
#         except asyncio.CancelledError:
#             break
#         except Exception as e:
#             print(f"[音频结果处理错误] {e}")

# 会话音频队列
session_audio_queues = {}

# async def _send_audio_to_session(session_id: str, audio_data: dict):
#     """发送音频数据到会话"""
#     if session_id not in session_audio_queues:
#         session_audio_queues[session_id] = asyncio.Queue()
    
#     await session_audio_queues[session_id].put(audio_data)

async def get_session_audio_stream(session_id: str) -> AsyncGenerator[str, None]:
    """获取会话的音频流"""
    if session_id not in session_audio_queues:
        return
    
    queue = session_audio_queues[session_id]
    
    while True:
        try:
            audio_data = await asyncio.wait_for(queue.get(), timeout=0.1)
            yield f"data: {json.dumps(audio_data)}\n\n"
        except asyncio.TimeoutError:
            break
        except Exception as e:
            print(f"[音频流错误] {e}")
            break

# 核心中的核心修改 改进的混合流生成器  注意不要求文本快  要求音频快 文本比音频早一点即可
async def create_improved_mixed_stream(
    original_stream: AsyncGenerator[str, None],
    session_id: str
) -> AsyncGenerator[str, None]:
    """优化的混合流生成器 - 真正的文本音频交替输出"""
    if not audio_queue_manager:
        async for chunk in original_stream:
            yield chunk
        return
    
    text_processor = TextProcessor()
    audio_queue = audio_queue_manager.get_or_create_queue(session_id)
    
    try:
        # 将原始流转换为异步迭代器
        text_iterator = original_stream.__aiter__()
        text_done = False
        
        while True:
            # 首先检查是否有准备好的音频
            audio_chunk = await audio_queue.get_next_audio()
            if audio_chunk:
                yield f"data: {json.dumps(audio_chunk)}\n\n"
                continue
            
            # 然后尝试获取下一个文本chunk（非阻塞）
            if not text_done:
                try:
                    text_chunk = await asyncio.wait_for(
                        text_iterator.__anext__(), timeout=0.001  # 极短超时，避免阻塞音频
                    )
                    
                    # 立即输出文本
                    yield text_chunk
                    
                    # 处理文本生成音频（异步，不等待）
                    if text_chunk.startswith("data: "):
                        try:
                            data = json.loads(text_chunk[6:])
                            if data.get("type") == "text" and "response" in data:
                                char = data["response"]
                                sentence = text_processor.process_char(char)
                                
                                if sentence:
                                    # 立即触发音频生成，不等待
                                    asyncio.create_task(audio_queue.add_sentence(sentence))
                        except json.JSONDecodeError:
                            pass
                    
                    continue
                    
                except StopAsyncIteration:
                    text_done = True
                    # 处理剩余文本
                    remaining_text = text_processor.get_remaining_text()
                    if remaining_text:
                        asyncio.create_task(audio_queue.add_sentence(remaining_text))
                        
                except asyncio.TimeoutError:
                    pass  # 继续循环，优先检查音频
            
            # 如果文本已结束，检查是否还有音频任务在处理
            if text_done:
                # 如果没有活跃任务且没有待输出音频，结束
                if not audio_queue.active_tasks and not audio_queue.pending_audio:
                    break
            
            # 极短暂休眠，避免CPU占用过高
            await asyncio.sleep(0.001)
        
    finally:
        # 清理
        pass

# async def _process_text_stream(
#     original_stream: AsyncGenerator[str, None],
#     text_processor: TextProcessor,
#     audio_queue: AudioQueue
# ) -> AsyncGenerator[str, None]:
#     """处理文本流"""
#     async for chunk in original_stream:
#         yield chunk
        
#         if chunk.startswith("data: "):
#             try:
#                 data = json.loads(chunk[6:])
#                 if data.get("type") == "text" and "response" in data:
#                     char = data["response"]
#                     sentence = text_processor.process_char(char)
                    
#                     if sentence:
#                         await audio_queue.add_sentence(sentence)
                        
#             except json.JSONDecodeError:
#                 pass
    
#     # 处理剩余文本
#     remaining_text = text_processor.get_remaining_text()
#     if remaining_text:
#         await audio_queue.add_sentence(remaining_text)

# async def _process_audio_stream(session_id: str, audio_queue: asyncio.Queue):
#     """处理音频流"""
#     while True:
#         try:
#             # 从全局音频结果队列获取结果
#             audio_result = await asyncio.wait_for(
#                 audio_result_queue.get(), timeout=1.0
#             )
            
#             if audio_result.get("session_id") == session_id:
#                 audio_chunk = {
#                     "type": "audio_chunk",
#                     "sentence_id": audio_result["sentence_id"],
#                     "text": audio_result["text"],
#                     "data": audio_result["data"]
#                 }
#                 await audio_queue.put(audio_chunk)
                
#         except asyncio.TimeoutError:
#             continue
#         except asyncio.CancelledError:
#             break
#         except Exception as e:
#             print(f"[音频流处理错误] {e}")

# 取消音频API
async def cancel_audio_generation(session_id: str):
    """取消指定会话的音频生成"""
    if audio_queue_manager:
        audio_queue_manager.cancel_queue(session_id)
    
    # 清理会话音频队列
    if session_id in session_audio_queues:
        del session_audio_queues[session_id]