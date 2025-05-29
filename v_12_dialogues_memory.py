from datetime import datetime
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from bson import ObjectId
from typing import List, Dict, Optional


class DialogueManager:
    """MongoDB对话记录管理类（支持连接池和事务）"""

    #  固定数据库和表（collection）名
    def __init__(self, conn_str: str, db_name: str = "chat_system", collection: str = "dialogues"):
        """
        初始化MongoDB连接
        :param conn_str: 连接字符串 mongodb://username:password@host:port/
        :param db_name: 数据库名称（默认chat_system）
        :param collection: 集合名称（默认dialogues）
        """
        try:
            self.client = MongoClient(
                conn_str,
                maxPoolSize=100,  # 连接池大小
                minPoolSize=10,  # 最小保持连接数
                socketTimeoutMS=30000,  # 30秒超时
                connectTimeoutMS=2000  # 2秒连接超时
            )
            self.db = self.client[db_name]
            self.collection = self.db[collection]
            self._create_indexes()  # 自动创建索引
            print("MongoDB连接") #✅
        except ConnectionFailure as e:
            raise RuntimeError(f"连接失败: {str(e)}")

    def _create_indexes(self):
        """创建优化索引（幂等操作）"""
        indexes = [
            {"key": [("session_id", 1)], "unique": True},  # 唯一索引
            {"key": [("shop_info.shop_id", 1)]},  # 普通索引
            {"key": [("user_info.user_id", 1)]},
            {"key": [("start_time", -1)]},
            {"key": [("status.payment_status", 1)]}
        ]
        for idx in indexes:
            # 提取索引参数
            keys = idx["key"]
            options = {k: v for k, v in idx.items() if k != "key"}
            self.collection.create_index(keys, **options)
        print("索引创建完成")# 🔄

    def create_dialogue(self, doc: Dict) -> str:
        """
        创建新对话记录（支持幂等性，重复创建返回现有ID）
        :param doc: 符合结构的文档
        :return: 文档ID（新增或现有）
        """
        try:
            # 尝试插入新文档
            doc.setdefault("start_time", datetime.utcnow())
            result = self.collection.insert_one(doc)
            return str(result.inserted_id)
        except DuplicateKeyError:
            # 查询现有文档
            existing = self.collection.find_one(
                {"session_id": doc["session_id"]},
                {"_id": 1}
            )
            if existing:
                print(f"⚠️ session_id 已存在，复用: {doc['session_id']}")
                return str(existing["_id"])
            else:
                raise ValueError("session_id 冲突但未找到文档")
        except Exception as e:
            raise RuntimeError(f"插入失败: {str(e)}")
        # try:
        #     # 自动生成时间戳和持续时间
        #     doc.setdefault("start_time", datetime.utcnow())
        #     if "end_time" in doc:
        #         doc["duration_seconds"] = int(
        #             (doc["end_time"] - doc["start_time"]).total_seconds()
        #         )
        #
        #     result = self.collection.insert_one(doc)
        #     return str(result.inserted_id)
        # except DuplicateKeyError:
        #     raise ValueError("session_id 必须唯一")
        # except Exception as e:
        #     raise RuntimeError(f"插入失败: {str(e)}")

    def get_dialogue(self, session_id: str) -> Optional[Dict]:
        """通过session_id获取完整对话记录"""
        return self.collection.find_one(
            {"session_id": session_id},
            {"_id": 0}  # 排除MongoDB的_id字段
        )

    def append_message(self, session_id: str, message: Dict) -> int:
        """
        追加新消息到对话记录
        :return: 更新后的消息总数
        """
        result = self.collection.find_one_and_update(
            {"session_id": session_id},
            {"$push": {"messages": message}},
            return_document=ReturnDocument.AFTER
        )
        return len(result["messages"]) if result else 0

    def update_status(self, session_id: str, updates: Dict) -> bool:
        """更新会话状态字段"""
        result = self.collection.update_one(
            {"session_id": session_id},
            {"$set": {"status": updates}}
        )
        return result.modified_count > 0

    def delete_dialogue(self, session_id: str) -> bool:
        """删除指定对话记录"""
        result = self.collection.delete_one({"session_id": session_id})
        return result.deleted_count > 0

    # 高级查询方法
    def query_by_shop(self, shop_id: str, limit: int = 100) -> List[Dict]:
        """按店铺分页查询最新对话"""
        return list(self.collection.find(
            {"shop_info.shop_id": shop_id},
            {"_id": 0, "messages": {"$slice": -5}},  # 返回最后5条消息
            sort=[("start_time", -1)],
            limit=limit
        ))

    def query_by_time_range(self, start: datetime, end: datetime) -> List[Dict]:
        """时间范围查询（含分析指标）"""
        return list(self.collection.find({
            "start_time": {"$gte": start, "$lte": end},
            "analytics.sentiment_score": {"$exists": True}
        }, {
            "session_id": 1,
            "start_time": 1,
            "analytics.sentiment_score": 1,
            "status.payment_status": 1
        }))

    def get_latest_messages(self, session_id: str, n: int = 3) -> Optional[str]:
        """
        获取指定对话的最新消息并格式化成多行字符串
        :param session_id: 会话唯一标识
        :param n: 需要获取的最近对话轮数（默认3轮）
        :return: 格式化后的多行字符串（None表示会话不存在）
        """
        try:
            # 使用投影操作符优化查询性能
            projection = {
                "messages": {
                    "$slice": -n  # 直接取最后n条消息
                },
                "_id": 0
            }

            # 带超时设置的查询
            result = self.collection.find_one(
                {"session_id": session_id},
                projection,
                max_time_ms=1000  # 1秒超时
            )

            if not result or not result.get('messages'):
                return None

            # 消息拼接处理
            formatted_lines = []
            for idx, message in enumerate(result['messages'], 1):
                try:
                    # 添加消息顺序标识
                    line = f"[第{idx}轮] \"{message['type']}\": \"{message['content']}\""
                    formatted_lines.append(line)
                except KeyError as e:
                    print(f"消息格式错误，缺失字段: {str(e)}")
                    continue

            return '\n'.join(formatted_lines)

        except Exception as e:
            print(f"数据库查询失败: {str(e)}")
            return None

    def update_user_info(self, session_id: str, user_info: dict) -> bool:
        """
        更新对话中的用户信息
        :param session_id: 会话ID
        :param user_info: 新的用户信息
        :return: 是否更新成功
        """
        try:
            result = self.collection.update_one(
                {"session_id": session_id},
                {"$set": {"user_info": user_info}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"[错误] 更新用户信息失败: {str(e)}")
            return False

if __name__ == "__main__":
    # 配置连接信息（示例）
    CONN_STR = "mongodb://mongodb:"
    mgr = DialogueManager(CONN_STR)
    print(mgr.get_latest_messages("shop-123_user-123_20250519103421",3))
    # # 初始化管理器
    # try:
    #     mgr = DialogueManager(CONN_STR)
    #
    #     # 测试用例1: 创建新对话
    #     sample_doc = {
    #         "session_id": "shop_123_user_789_20250512142600",
    #         "shop_info": {
    #             "shop_id": "shop_123",
    #             "shop_name": "测试店铺",
    #             "region": "上海",
    #             "contact": "李经理 138-0000-0000"
    #         },
    #         "user_info": {
    #             "user_id": "user_789",
    #             "username": "测试用户",
    #             "preferences": ["日用品"],
    #             "dialect": "上海话"
    #         },
    #         "messages": [{
    #             "message_id": "msg_001",
    #             "type": "system",
    #             "content": "欢迎光临测试店铺！",
    #             "timestamp": datetime.utcnow(),
    #             "metadata": {
    #                 "intent": "welcome",
    #                 "confidence": 0.95,
    #                 "tools_used": []
    #             }
    #         }],
    #         "status": {
    #             "payment_status": "pending",
    #             "exit_flag": False,
    #             "current_intent": "greeting"
    #         }
    #     }
    #     doc_id = mgr.create_dialogue(sample_doc)
    #     print(f"📝 创建文档ID: {doc_id}")
    #
    #     # 测试用例2: 追加消息
    #     new_msg = {
    #         "message_id": "msg_002",
    #         "type": "human",
    #         "content": "有洗发水吗？",
    #         "timestamp": datetime.utcnow()
    #     }
    #     msg_count = mgr.append_message(sample_doc["session_id"], new_msg)
    #     print(f"🔢 当前消息数: {msg_count}")
    #
    #     # 测试用例22: 追加消息
    #     new_msg = {
    #         "message_id": "msg_002",
    #         "type": "system",
    #         "content": "有的，在货架A",
    #         "timestamp": datetime.utcnow()
    #     }
    #     msg_count = mgr.append_message(sample_doc["session_id"], new_msg)
    #     print(f"🔢 当前消息数: {msg_count}")
    #
    #     # 测试用例3: 更新支付状态
    #     update_result = mgr.update_status(
    #         sample_doc["session_id"],
    #         {"payment_status": "success", "purchased_items": ["洗发水"]}
    #     )
    #     print(f"🔄 状态更新: {'成功' if update_result else '失败'}")
    #
    #     # 测试用例4: 复杂查询
    #     recent_chats = mgr.query_by_shop("shop_123", limit=2)
    #     print(f"🏪 最近店铺对话: {len(recent_chats)}条")
    #
    # except Exception as e:
    #     print(f"❌ 操作异常: {str(e)}")