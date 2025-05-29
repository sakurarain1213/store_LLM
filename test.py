#  控制台激活/退出虚拟环境  使用   .venv\Scripts\activate
#     退出虚拟环境  使用   deactivate
import export
import requests
import os

# 中文个人系列教程 https://blog.csdn.net/jeffray1991/category_12591752.html
# langchain API  https://python.langchain.com/api_reference/langchain/index.html


#  注意 公司SSL限制  使用--trusted-host来安装
# pip install langchain -i https://pypi.tuna.tsinghua.edu.cn/simple   --trusted-host pypi.tuna.tsinghua.edu.cn
# pip install langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install langchain-community -i https://pypi.tuna.tsinghua.edu.cn/simple


# 设置环境变量
# 以下是第三方中转的API key
# os.environ['OPENAI_API_KEY'] = "sk-"
# os.environ['OPENAI_BASE_URL'] = "https://one.glbai.com"

# 以下是DEEPSEEK的API key
# https://platform.deepseek.com/sign_in
# https://api.deepseek.com
# sk-

# 以下是腾讯混元的API key
# api_key=os.environ.get("HUNYUAN_API_KEY"),  # 混元 APIKey
# base_url="https://api.hunyuan.cloud.tencent.com/v1",  # 混元 endpoint
# sk-


# # 构建请求体
# messages = [
#     {"role": "system", "content": "把下面的语句翻译为英文。"},
#     {"role": "user", "content": "今天天气怎么样？"}
# ]
#
# # 设置请求头
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
# }
#
# # 构建API请求URL
# url = f"{os.environ['OPENAI_BASE_URL']}/v1/chat/completions"
#
# # 发送POST请求
# response = requests.post(
#     url,
#     headers=headers,
#     json={
#         "model": "gpt-3.5-turbo",
#         "messages": messages
#     }
# )
#
# # 处理响应
# if response.status_code == 200:
#     print("Response content:", response.json()["choices"][0]["message"]["content"])
# else:
#     print("Error:", response.status_code, response.text)


#  以下  下载embedding模型m3e到本地
# pip install modelscope
# from modelscope import snapshot_download
# import os
#
# # 设置下载目标路径
# target_dir = "D:\\Embedding"  # Windows 路径需双反斜杠转义
# # 核心下载代码
# model_dir = snapshot_download(
#     'xrunda/m3e-base',
#     cache_dir=target_dir,  # 关键参数：指定下载路径
#     revision='master'       # 默认下载最新版本
# )
# print(f"模型已下载到：{model_dir}")


# import pandas as pd
# import os
# 
# # 输入和输出路径
# input_folder = os.path.join(os.getcwd(), 'data')  # 输入目录：当前目录下的data文件夹
# output_folder = os.path.join(input_folder, 'csv')  # 输出目录：data/csv
# 
# # 如果输出目录不存在则创建
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# 
# # 遍历data文件夹中的xlsx文件
# for filename in os.listdir(input_folder):
#     if filename.endswith(('.xlsx', '.xls')):
#         file_path = os.path.join(input_folder, filename)
# 
#         # 读取Excel文件（包含所有Sheet页）
#         excel_file = pd.ExcelFile(file_path)
#         all_sheets = pd.read_excel(excel_file, sheet_name=None)  # sheet_name=None读取所有Sheet
# 
#         # 方式1：合并所有Sheet页到一个CSV文件
#         # merged_df = pd.concat(all_sheets.values(), axis=0)
#         # csv_filename = os.path.splitext(filename)[0] + '.csv'
#         # merged_df.to_csv(os.path.join(output_folder, csv_filename), index=False, encoding='utf-8-sig')
# 
#         # 方式2：每个Sheet页保存为单独的CSV文件（按需选择）
#         for sheet_name, df in all_sheets.items():
#             csv_filename = f"{os.path.splitext(filename)[0]}_{sheet_name}.csv"
#             df.to_csv(os.path.join(output_folder, csv_filename), index=False, encoding='utf-8-sig')
# 
# print("转换完成！")

import pandas as pd
import random
import string
import chardet  # 新增编码检测库
# 定义分类映射规则（可根据实际情况扩展关键词）
category_keywords = {
    '家电': ['电热水壶', '烧水壶', '电磁炉', '电饭煲', '电风扇', '洗衣机', '冰箱'],
    '生鲜': ['小白菜', '青菜', '菠菜', '黄瓜', '西红柿', '苹果', '耙'],
    '服饰': ['T恤', '外套', '连衣裙', '裤子', '衬衫'],
    '美妆': ['口红', '面膜', '粉底', '睫毛膏', '香水'],
    '零食': ['薯片', '巧克力', '坚果', '饼干', '糖果'],
    '饮料': ['矿泉水', '果汁', '碳酸饮料', '茶饮', '咖啡'],
    '烟酒': ['香烟', '白酒', '红酒', '啤酒', '雪茄'],
    '副食': ['酱油', '醋', '食盐', '味精', '食用油'],
    '药品': ['感冒药', '创可贴', '维生素', '消炎药', '口罩', '避孕'],
    '数码': ['手机', '笔记本', '相机', '耳机', '平板'],
    '日用': ['纸巾', '牙刷', '毛巾', '洗发水', '肥皂'],
    '其它': []
}


def determine_category(name):
    for cat, keywords in category_keywords.items():
        if any(keyword in name for keyword in keywords):
            return cat
    return '其它'


def generate_position():
    return random.choice(string.ascii_uppercase) + random.choice(string.digits)


# --- 核心改进部分 ---
# 1. 检测文件原始编码
def detect_file_encoding(path):
    with open(path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']  # 返回检测到的编码(如'gb2312','utf-8')

# 2. 动态读取文件
file_path = './data/store_cloud_duty_store_order_goods_day.csv'
original_encoding = detect_file_encoding(file_path)  # 调用编码检测函数[4,11](@ref)

try:
    df = pd.read_csv(file_path, encoding=original_encoding)
except UnicodeDecodeError:
    # 备选方案：尝试常见中文编码[5,8](@ref)
    try:
        df = pd.read_csv(file_path, encoding='gbk')
        original_encoding = 'gbk'
    except:
        df = pd.read_csv(file_path, encoding='utf-8')
        original_encoding = 'utf-8'


df['category'] = df['product_name'].apply(determine_category)


df['position'] = df['position'].fillna(pd.Series([generate_position() for _ in range(len(df))]))


df['sales'] = [random.randint(0, 100) for _ in range(len(df))]
df['stock'] = [random.randint(0, 100) for _ in range(len(df))]

# 4. 按原始编码保存[3,5](@ref)
df.to_csv(
    'processed_data.csv',
    index=False,
    encoding=original_encoding  # 关键修改：使用检测到的编码
)

print(f"文件已保存为 {original_encoding} 编码格式")
# 核心debug问题  兼容中英文的格式UTF-8-SIG  并非utf8或gbk或gb2312
