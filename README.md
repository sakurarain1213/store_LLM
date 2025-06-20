# 无人店铺值守对话大模型DEMO 原理与演示

后端入口程序：v_xx_server.py
网页端入口程序：LANG_clint/v_xx_clint.html

## 界面
<img src="https://i.postimg.cc/L86V3s5D/2.png" style="width: 100%; height: auto;" />


## 用户体验
ASR实时文本显示延迟<0.5s
从用户说话结束到显示首token <1s
从用户说话结束到播放首音频片 <1.5s
支持打断AI说话




## 特性
代码极简 便于部署 模块解耦 可定制可拓展

## 技术栈

- 后端服务框架: Uvicorn + FastAPI
- 应用层框架: LangGraph
- 大模型: Qwen2.5-7B-Instruct + DeepSeekV3
- 数据存储: MongoDB
- RAG检索增强: 
  - 向量引擎: FAISS 
  - 嵌入模型: HuggingFace Embeddings
- 搜索服务: DuckDuckGo + 百度搜索
- 语音能力:
  - ASR: 科大讯飞 (支持中英文)
  - TTS: 火山引擎 (多音色中英文)
- 客户端(Web) : 传统三件套

## 功能模块

<img src="https://i.postimg.cc/Dfsp8tkf/1.png" style="width: 100%; height: auto;" />

Tools：

Welcome、chitchat 、shopping guide（万级商品表检索）、recommendation、payment、goodbye等

[![1.png](https://i.postimg.cc/hGhD9C8J/1.png)](https://postimg.cc/CZTp0sQY)

## 过程中状态管理
```python
class State(TypedDict):
    session_id: str           # 会话ID
    purchased_items: List[str] # 购物车清单
    current_user: UserInfo     # 用户档案
    shop_info: ShopInfo       # 店铺配置
    messages: List[dict]      # 对话历史
    current_intent: Optional[str] # 当前意图
    last_response: Optional[str] # 最后响应
    payment_status: Literal["pending","success"] # 支付状态
    exit_flag: bool           # 退出标志
```
## 前置输入
```python
商品表
products_docs_list = [
    "id",                # 商品ID
    "product_id",        # 商品69码
    "product_name",      # 商品名称
    "product_price",     # 商品价格
    "product_count",     # 商品数量
    "create_time",       # 创建时间
    "update_time",       # 更新时间
    "dt",               # 日期
    "category",         # 商品类别
    "position",         # 商品位置
    "stock",            # 库存
    "sales"             # 销量
]

店铺信息
sample_shop = ShopInfo(
    shop_id="shop-123",
    name="西湖123便利店",
    region="杭州",
    contact="X老板, 123-1234-5678",
    remark="购物袋免费；成条烟不拆开；店内不堂食；可以接热水；",
    promotions=["满30减5", "送口香糖"],
    products_docs=products_docs_list
)

进店用户
 sample_user = {
    "user_id": "user-123",
    "dialect": "粤语",
    "gender": "男",
    "preferences": ["可乐", "香烟"],
    "username": "张三"
}

```

## 效果展示（以粤语用户为例）
```python
[调试] 新建对话记录: shop-123_user-123_20250529114143
[调试] docs:  [
Document(id='34febf44-f29d-4eba-b922-0faa2a204502', metadata={'source': '6.90E+12', 'row': 7598, 'product_id': '6.90E+12', 'product_name': '可乐', 'product_price': '7.0', 'category': '饮料', 'position': 'A942', 'stock': '46', 'sales': '36'}, page_content='品名:可乐;类别:饮料;价格:7.0;'),
Document(id='d1544ca8-de6b-4435-935b-2198b3619cbd', metadata={'source': '6.90E+12', 'row': 16525, 'product_id': '6.90E+12', 'product_name': '黄鹤楼（硬1916如意）', 'product_price': '90.0', 'category': '香烟', 'position': 'A297', 'stock': '57', 'sales': '100'}, page_content='品名:黄鹤楼（硬1916如意）;类别:香烟;价格:90.0;'),
Document(id='183f99d1-8b8e-4830-b2a3-bd28eb6b449a', metadata={'source': '6.90E+12', 'row': 18159, 'product_id': '6.90E+12', 'product_name': '好猫(长乐九美)', 'product_price': '20.0', 'category': '香烟', 'position': 'A706', 'stock': '50', 'sales': '87'}, page_content='品名:好猫(长乐九美);类别:香烟;价格:20.0;'),
Document(id='f761c02b-a588-4236-800c-ffddb1403072', metadata={'source': '6.90E+12', 'row': 10882, 'product_id': '6.90E+12', 'product_name': 'N王冠(加勒比)香烟(包)', 'product_price': '16.0', 'category': '香烟', 'position': 'B805', 'stock': '60', 'sales': '2'}, page_content='品名:N王冠(加勒比)香烟(包);类别:香烟;价格:16.0;'),
Document(id='3cd08440-f793-4bee-ac3b-d2a72ad44d33', metadata={'source': '6.90E+12', 'row': 19837, 'product_id': '6.90E+12', 'product_name': '兰州(黑中支)', 'product_price': '30.0', 'category': '香烟', 'position': 'B167', 'stock': '16', 'sales': '58'}, page_content='品名:兰州(黑中支);类别:香烟;价格:30.0;'),
Document(id='36779b7d-4543-4593-855c-e0cb48da051d', metadata={'source': '6.90E+12', 'row': 14943, 'product_id': '6.90E+12', 'product_name': '白沙（硬精品三代）', 'product_price': '16.5', 'category': '香烟', 'position': 'B861', 'stock': '53', 'sales': '28'}, page_content='品名:白沙（硬精品三代）;类别:香烟;价格:16.5;'),
Document(id='031aaa9d-720e-4212-9045-7904803a7100', metadata={'source': '4.90E+12', 'row': 10037, 'product_id': '4.90E+12', 'product_name': '七星（风蓝）', 'product_price': '26.0', 'category': '香烟', 'position': 'A593', 'stock': '11', 'sales': '8'}, page_content='品名:七星（风蓝）;类别:香烟;价格:26.0;'),
Document(id='d5b111b4-9e77-4c7e-9888-b3003d89e385', metadata={'source': '6.90E+12', 'row': 3418, 'product_id': '6.90E+12', 'product_name': '娇子（蓝)*10包', 'product_price': '140.0', 'category': '香烟', 'position': 'B856', 'stock': '88', 'sales': '78'}, page_content='品名:娇子（蓝)*10包;类别:香烟;价格:140.0;'),
Document(id='5321a4e2-e810-40af-90a9-79e9f6a6f431', metadata={'source': '6.90E+12', 'row': 15723, 'product_id': '6.90E+12', 'product_name': '中南海（金装8mg）', 'product_price': '10.0', 'category': '香烟', 'position': 'A97', 'stock': '30', 'sales': '36'}, page_content='品名:中南海（金装8mg）;类别:香烟;价格:10.0;'),
Document(id='ceaee417-1375-4d83-8c0c-3db8b03ba58c', metadata={'source': '6.90E+12', 'row': 14112, 'product_id': '6.90E+12', 'product_name': '贵烟(软高遵)', 'product_price': '35.0', 'category': '香烟', 'position': 'A653', 'stock': '87', 'sales': '45'}, page_content='品名:贵烟(软高遵);类别:香烟;价格:35.0;')]
[调试] recommends:  ['可乐', '黄鹤楼（硬1916如意）', '好猫(长乐九美)', 'N王冠(加勒比)香烟(包)', '兰州(黑 中支)', '白沙（硬精品三代）', '七星（风蓝）', '娇子（蓝)*10包', '中南海（金装8mg）', '贵烟(软高遵)']
```

欢迎节点输出: （≧▽≦）哈佬啊张三哥！欢迎嚟到杭州西湖123便利店嘅无人店呀～🎉  

今日系2025年5月29号，下昼2点，阳光正好啱好扫货喔！🍀

我哋精选推荐可乐解渴💦，仲有黄鹤楼、好猫、七星等靓烟🚬，满30蚊即减5蚊再送粒香口胶，抵到笑啦～😝

需要帮手就随时㩒屏幕叫我，祝你扫货开心，西湖风景都冇你咁靓仔啊！✨
```python
[调试] 欢迎词已持久化
```
用户: 今天杭州天气怎么样
```python
[调试] 用户消息已持久化
INFO:httpx:HTTP Request: POST https://api.siliconflow.cn/v1/chat/completions "HTTP/1.1 200 OK"
[调试] 意图节点输出: handle_chitchat
[调试] 开始处理闲聊，session_id: shop-123_user-123_20250529114143
[调试] web信息
 Title: 【杭州天气预报15天_杭州天气预报15天查询】-中国天气网
Content: 未来15天,平均高温26℃,平均低温17℃ 8次降水 今天 05/21 小雨 33/23℃ 周四 05/22 多云转小雨 31/22℃ 周五 05/23 大雨转中雨 24/18℃ 周六 05/24 多云 25/17℃ 周日 05/25 多云 27/18℃ 周一 05/
URL: https://wap.weather.com.cn/mweather15d/101210101.shtml

Title: 紧急提醒:明日抵达!今年以来最强降雨登场!接下来天气...
Content: 6月1-2日,杭州全市将出现明显的阵雨或雷雨天气过程,恰逢端午小长假,降水极有可能给市民出行、游玩计划带来 
不便。 由于初夏天气变化大,也请大家关注临近预报和预警信息。
URL: https://finance.sina.cn/cj/2025-05-27/detail-inexyruc7014761.d.html

Title: 杭州气象
Content: 杭州市气象台5月27日19时50分发布的市区天气预报: 今天夜里多云到阴;明天多云到阴,有时有阵雨;后天阴转多
云。今天夜里到明天偏...
URL: https://www.hzqx.com/

Title: 中国天气网杭州今日详情-中国天气网
Content: 实况天气 当前气温 27.1° 东风3级 降水量 0mm 能见度 21km 气压1000 hPa 相对湿度 69% 20时 25° 东南风 <3 
级 21时 25° 东南风 <3级 22时 25° 东风 <3级 23时 ...
URL: http://sougou.weather.com.cn/mweather1d/index.shtml?aid=101210101

Title: 15天天气预报
Content: 晴 26/6℃ 周六 03/22 晴 25/10℃ 周日 03/23 阴转多云 26/11℃ 周一 03/24 晴 26/11℃ 周二 03/25 多云转小到 
中雨 28/16℃ 周三 03/26 雨 27/9℃ 周四 03/27...
URL: https://e.weather.com.cn/d/town/future?lat=30.057007&lon=119.99028&areaid=101210108

Title: 【杭州天气预报】杭州今天天气预报_杭州天气预报7天...
Content: 杭州天气预报,及时准确发布中央气象台天气信息,便捷查询北京今日天气,杭州周末天气,杭州一周天气预报,杭州15日天气预报,杭州40日天气预报,杭州天气预报还提供杭州各区县的生活指数、健康...
URL: http://yidian.weather.com.cn/mweather1d/101210101.shtml
INFO:httpx:HTTP Request: POST https://api.siliconflow.cn/v1/chat/completions "HTTP/1.1 200 OK"
```
LLM节点输出: ZANG桑你好呀！👋 今日杭州owa天气呢，最晒都系27度左右，仲有轻嘅东南风，户外空气清新啲。不过
呢日喺23时开始，天气会转阵云呢，可能有啲阵雨，咁记得带紧伞呀！☔
同埋，你知咩？明早到后日ewe有阵雨或者雷阵雨，尤其是端午节前后，可能会影响出游计划，好紧记关注临近预报啦！🌈  
今日推介继续推介🥤可樂同là翠螺，满30减5，送你一包口香糖！🎁 点解唔晒买紧几件好悭钱又有食又有饮呢！😉
```python
[调试] AI输出已持久化
```
用户: 可乐在哪
```python
[调试] 用户消息已持久化
INFO:httpx:HTTP Request: POST https://api.siliconflow.cn/v1/chat/completions "HTTP/1.1 200 OK"
[调试] 意图节点输出: handle_shopping_guide
INFO:httpx:HTTP Request: POST https://api.siliconflow.cn/v1/chat/completions "HTTP/1.1 200 OK"
[调试] 标准化后的查询: 1. 标准商品名称: 可乐
2. 商品特征:
3. 位置相关: 在哪
4. 价格相关:
[调试] 语义检索docs结果: ['可口可乐·可口可乐', '颜雯丽可乐丝', '可乐汽水12*300毫升', '可乐', '听装可乐', 
'百事可乐', '可口可乐', '可乐糖', '680ml可口可乐(经典原味)', '崂山可乐', '零度可口可乐500ml', '汽水可乐', '果
然多可乐味', '罐装可口可乐柠檬味330ml', '百事可乐生可乐', '双支可乐', '百事可乐可乐型汽水1L', '塑料瓶可口可乐
300克', '可口可乐纤维+500ml', '迷你可乐']
INFO:httpx:HTTP Request: POST https://api.siliconflow.cn/v1/chat/completions "HTTP/1.1 200 OK"
```
LLM节点输出: 阿哥，你可乐喺边买得呀？我帮佢哋找下喇。仲有啲信息要你留意下喇：

可口可乐（饮咗）喺B104，一条四點五块；

颜雯丽可乐丝（可乐）喺B35，一条几两块，仲cheap；

可乐汽水12支300毫升（饮咗）喺A103，一条十九块九；

可口可乐（饮咗）喺A362，一条三块；

听装可乐（饮咗）喺B867，一条两块五；

百事可乐（饮咗）喺B826，一条三块五；

可口可乐（饮咗）喺A317，一条四块；

果然多可乐味（可乐）喺B129，一条一·五块；

680毫升可口可乐（经典原味）（瓶装）喺A823，一条四块；

崂山可乐（饮品）喺B60，一条三块五；

零度可口可乐500毫升（饮咗）喺B186，一条四块；

汽水可乐（饮咗）喺B668，一条三块；

罐装可口可乐柠檬味330毫升（饮咗）喺A311，一条六块五；

百事可乐生可乐（碳酸）喺A317，一条四块；

双支可乐（饮咗）喺A193，一条一·块；

百事可乐可乐型汽水1L（饮咗）喺A393，一条四块五；

塑料瓶可口可乐300克（饮咗）喺B588，一条一十九块；

可口可乐纤维+500毫升（饮咗）喺B29，一条五块；

迷你可乐（迷你）喺A122，一条两块。

记得拎包俾佢哋免费，烟成条卖唔拆开，店内唔得食嘅，可以接热水。有需要就使使手啦！😊
```python
[调试] AI输出已持久化
```
用户: 靓仔推荐点香烟嘞  
```python
[调试] 用户消息已持久化
INFO:httpx:HTTP Request: POST https://api.siliconflow.cn/v1/chat/completions "HTTP/1.1 200 OK"
[调试] 意图节点输出: handle_recommendation
first docs: [
Document(id='9b42e7a2-4f4f-4c8b-a6d8-3427f46f27e2', metadata={'source': '6.90E+12', 'row': 10048, 'product_id': '6.90E+12', 'product_name': 'N软经典香烟(条)', 'product_price': '125.0', 'category': '香烟', 'position': 'B596', 'stock': '88', 'sales': '52'}, page_content='品名:N软经典香烟(条);类别:香烟;价格:125.0;'),
Document(id='701dd2c6-6592-406e-b558-75173862e4fd', metadata={'source': '6.94E+12', 'row': 6323, 'product_id': '6.94E+12', 'product_name': '可咯', 'product_price': '5.0', 'category': '可咯', 'position': 'B624', 'stock': '42', 'sales': '44'}, page_content='品名:可咯;类别:可咯;价格:5.0;'),
Document(id='183f99d1-8b8e-4830-b2a3-bd28eb6b449a', metadata={'source': '6.90E+12', 'row': 18159, 'product_id': '6.90E+12', 'product_name': '好猫(长乐九美)', 'product_price': '20.0', 'category': '香烟', 'position': 'A706', 'stock': '50', 'sales': '87'}, page_content='品名:好猫(长乐九美);类别:香烟;价格:20.0;'),
Document(id='b69104f7-af24-444b-baa0-70e086dbd7de', metadata={'source': '6.97E+12', 'row': 3258, 'product_id': '6.97E+12', 'product_name': '无意柑橘可可多用空气香氛 400ml', 'product_price': '9.9', 'category': '香氛', 'position': 'B816', 'stock': '45', 'sales': '98'}, page_content='品名:无意柑橘可可多用空气香氛 400ml;类别:香氛;价格:9.9;'),
Document(id='5cf965a4-fa59-41ad-9f4d-4683f7c7eafb', metadata={'source': '6.97E+12', 'row': 8139, 'product_id': '6.97E+12', 'product_name': '防风打火机蓝莓叔叔（24小时营业）', 'product_price': '2.0', 'category': '打火', 'position': 'A118', 'stock': '68', 'sales': '30'}, page_content='品名:防风打火机蓝莓叔叔（24小时营业）;类别:打火;价格:2.0;'),
Document(id='80a54af3-8782-471e-abed-53cb41100ac2', metadata={'source': '6.90E+12', 'row': 5097, 'product_id': '6.90E+12', 'product_name': '黄鹤楼(迷你醇味)', 'product_price': '20.0', 'category': '香烟', 'position': 'B317', 'stock': '35', 'sales': '16'}, page_content='品名:黄鹤楼(迷你醇味);类别:香烟;价格:20.0;'),
Document(id='a3aae68f-941f-49bd-a22e-deb47e651edf', metadata={'source': '6.90E+12', 'row': 6737, 'product_id': '6.90E+12', 'product_name': 'N双喜好日子精品硬盒香烟(包)()', 'product_price': '15.0', 'category': '烟盒', 'position': 'B727', 'stock': '42', 'sales': '54'}, page_content='品名:N双喜好日子精品硬盒香烟(包)();类别:烟盒;价格:15.0;'),
Document(id='d5b111b4-9e77-4c7e-9888-b3003d89e385', metadata={'source': '6.90E+12', 'row': 3418, 'product_id': '6.90E+12', 'product_name': '娇子（蓝)*10包', 'product_price': '140.0', 'category': '香烟', 'position': 'B856', 'stock': '88', 'sales': '78'}, page_content='品名:娇子（蓝)*10包;类别:香烟;价格:140.0;'),
Document(id='a5220ea1-0aaa-40b7-b52a-a9315d5f450b', metadata={'source': '6.90E+12', 'row': 48, 'product_id': '6.90E+12', 'product_name': '长城(迷你咖啡)香烟(包)', 'product_price': '25.0', 'category': '烟草', 'position': 'B14', 'stock': '12', 'sales': '75'}, page_content='品名:长城(迷你咖啡)香烟(包);类别:烟草;价格:25.0;'),
Document(id='a98923a0-037b-4053-859d-75dda025b5ef', metadata={'source': '6.95E+12', 'row': 3050, 'product_id': '6.95E+12', 'product_name': '舒 可乐纸抽', 'product_price': '3.0', 'category': '纸抽', 'position': 'B764', 'stock': '15', 'sales': '78'}, page_content='品名:舒可乐纸抽;类别:纸抽;价格:3.0;')]
sorted_docs: ['N软经典香烟(条)', '长城(迷你咖啡)香烟(包)', '娇子（蓝)*10包', '好猫(长乐九美)', 'N双喜好日子精品硬盒香烟(包)()']
INFO:httpx:HTTP Request: POST https://api.siliconflow.cn/v1/chat/completions "HTTP/1.1 200 OK"
```
LLM节点输出: 张三哥，你好呀！仲未吸烟啦？我这里帮佢推介咁几款烟，睇下仲有哪一款符合你嘅口味。

首先推介你嘅是 **N软经典香烟**，一条十四枝，真系够靓，价格公道，仲有仲大嘅优惠，满三百就返三十抵用，买得仲多 
返得仲多。睇紧啦，仲有连包口香糖送，吸完烟口即刻清新。

或者你仲可以用优惠去拣 **长城（迷你咖啡）香烟** ，包包装，啲咖啡味真系好适合喜欢特别口味嘅朋友，价格廉宜，又 
啲优惠，满三十减五，好划算！

如果仲有其他嘅选择，可以看看 **娇子（蓝）**，十包装，是你可以自己抽晒完嘅，价格唔算太贵，适合平日慢慢享受。  

还有 **好猫（长乐九美）**，味道唔错，价钱更加亲民，几适合嘴悭仔。

最后有款 **N双喜好日子精品硬盒香烟**，包裝靓，适合自用晒，价格唔算贵。

记得买烟要依咗规定，买成条烟唔可以拆开，店内唔可以吸烟，可以接热水，仲有口香糖送，收好咗！

希望张三哥可以拣到你满意嘅，食得开心，抽得健康！
```python
[调试] AI输出已持久化
```

用户: 店里销售额怎么样

```python
[调试] 用户消息已持久化
INFO:httpx:HTTP Request: POST https://api.siliconflow.cn/v1/chat/completions "HTTP/1.1 200 OK"
意图节点输出: handle_report
```

LLM节点输出: 昨日经营简报:
    销售额: ¥8937
    净利润: ¥1513
    热销商品: 椰树椰汁、农夫山泉
可视化: 柱状图( 销售额趋势) 饼图(商品占比)

```python
[调试] AI输出已持久化
```
用户: 你没懂我的意思 我要转人工 
```python
[调试] 用户消息已持久化
INFO:httpx:HTTP Request: POST https://api.siliconflow.cn/v1/chat/completions "HTTP/1.1 200 OK"
[调试] 意图节点输出: handle_human_transfer
```

LLM节点输出: 已为您转接值班经理( 工作时间9:00-21:00)

等待时间约2分钟

X老板, 123-1234-5678

```python
[调试] AI输出已持久化
```
用户: 结账吧小助手
```python
[调试] 用户消息已持久化
[调试] 意图节点输出: handle_payment
```
LLM节点输出: 支付失败，请重新支付

LLM节点输出: 支付成功！请取走商品

告别节点输出: 张生，多谢帮衬呀！农夫山泉、可乐同中华烟都齐喇～下次再嚟饮茶啦！😊👍

## 其它

成本控制：LLM API免费  网络API免费  ASR+TTS少量收费


