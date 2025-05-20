import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm  # 进度条显示

# 初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-tdvgqeujlplwxkczbzoyicgadzzdkdgulgdxzzbkcaybyhit"
)


def get_correct_category(product_name, current_category):
    """调用大模型获取正确分类"""
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",  # "Qwen/Qwen3-8B",    THUDM/GLM-Z1-9B-0414
            messages=[{
                "role": "user",
                "content": f"商品名称：{product_name}\n对商品名称进行归类。最好两个字。禁止出现【正确】和【错误】字眼。"
            }],
            temperature=0.7,
            max_tokens=2,
            timeout=30
        )
        content = response.choices[0].message.content.strip()  # 先去除首尾空白字符
        last_two_chars = content[-3:]  # 核心操作
        print(f"{product_name},{current_category}====>{last_two_chars}")
        new_category = last_two_chars
        return new_category # if len(new_category) == 2 else current_category
    except Exception as e:
        print(f"API调用失败：{str(e)}")
        return current_category


# 读取CSV文件
file_path = r"C:\Users\w1625\Desktop\LANG\data\store_cloud_duty_store_order_goods_day.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig')



# 新增控制参数
SAVE_INTERVAL = 10  # 每处理10条保存一次[1](@ref)
SLEEP_SECONDS = 0.2   # 每次请求间隔s

# 断点续传初始化
start_row = 13250  # 指定开始行号
counter = 0  # 新增计数器

for index in tqdm(df.index, desc="处理进度"):
    if index < start_row:
        continue

    # 原有数据处理逻辑
    product = df.loc[index, 'product_name']
    original_cat = df.loc[index, 'category']
    corrected_cat = get_correct_category(product, original_cat)

    if corrected_cat != original_cat:
        df.at[index, 'category'] = corrected_cat

    # 新增延时控制
    time.sleep(SLEEP_SECONDS)  # 每次循环暂停1秒[6,8](@ref)

    # 分批次保存
    counter += 1
    if counter % SAVE_INTERVAL == 0:
        df.to_csv(file_path, index=False, encoding='utf-8-sig')  # 定期保存[1](@ref)
        print("saved")

# 最终保存确保数据持久化
df.to_csv(file_path, index=False, encoding='utf-8-sig')
print("处理完成，文件已保存！")