import os
from openai import OpenAI
import pandas as pd
import glob, json, re
import numpy as np
from tqdm import tqdm
from datetime import datetime



client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-2f73f0e455d54bb9a756757df0c7ea2d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen2.5-1.5b-instruct", # 这个模型免费
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
    # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
    # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
    extra_body={"enable_thinking": False},
)


def get_chunk_list(lst, chunk_size=200):
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def exteract_module_raw_text(module_name, lines):
    completion = client.chat.completions.create(
        model="qwen2.5-1.5b-instruct",
        messages=[
            {"role": "system",
             "content": f"""你是一个专业的保险行业的信息处理专家，请对下面的文本中抽取出与{module_name}相关的文本，只需要输出原文，不要有其他输出。如果相关的本文本不存在，则输出一个空格。"""},
            {"role": "user", "content": "".join(lines)},
        ],
        extra_body={"enable_thinking": False},
    )
    return completion.choices[0].message.content


def analysis_conflict(module_name, content1, content2):
    completion = client.chat.completions.create(
        model="qwen2.5-1.5b-instruct",
        messages=[
            {"role": "system", "content": f"""你是一个专业的保险行业的信息处理专家，请对下面文本进行严谨的一致性进行分析，判断是否相同条件的表达存在不一致的情况，只需要回答一致或不一致，不要有其他任何输出。
            - 基础产品销售信息：该保险产品的基础配置信息，包括产品名、附加的条款信息、销售限制等；
            - 投保条款：投保过程中的缴费约定、投被保人条件限制等；
            - 保障责任：约定该产品的保险责任细节，如保障范围、保险金额、增值服务等；
            - 保障相关时间：约定该产品的各类时间信息，包括但不限于犹豫期、等待期、宽限期等；
            - 赔付 & 领取规则：约定该产品的保险责任的赔付、给付、领取及免赔细节，如赔付年龄/比例/次数等；
            - 责任免除：约定该产品不承担保险责任的情形；
            - 续保条款：约定续保相关信息，包括但不限于续保条件、保证续保等；
            - 退保条款：约定退保相关信息，包括但不限于退保条件、退保手续费等；
            - 出险条款：约定出险相关信息，包括但不限于出险地点、出险方式等；
            - 附加条款：约定该产品的附加条款，如特别约定等；
            - 术语解释：约定该产品的术语解释，如名词定义等；"""},
            {"role": "user",
             "content": f"""片段1\n{content1}\n\n片段2\n{content2}\n\n分析上述两个片段的{module_name}是否存在相同的条件但表达不一致的情况。"""},
        ],
        extra_body={"enable_thinking": False},
    )
    return completion.choices[0].message.content


for row in tqdm(pd.read_json("../data/测试 A 集/data.jsonl", lines=True).iloc[:].iterrows()):
    module_name = row[1].rule.replace("该产品的", "").replace("在各材料中的定义没有冲突", "")
    now = datetime.now()
    print(now.strftime("2025.%m.%d %H:%M:%S"),row[1].material_id)

    module_content_list = []
    for path in glob.glob(f"../data/测试 A 集/materials/{row[1].material_id}/*/*"):
        now = datetime.now()
        print(now.strftime("2025.%m.%d %H:%M:%S"),path)
        lines = open(path,encoding='utf-8').readlines()
        module_lines = ""
        for chunk_lines in get_chunk_list(lines):
            res = exteract_module_raw_text(module_name, chunk_lines)
            if res:
                module_lines = module_lines + res

        module_content_list.append(module_lines)

    result = []
    for i in range(len(module_content_list)):
        for j in range(i, len(module_content_list)):
            if "不一致" in analysis_conflict(module_name, module_content_list[i], module_content_list[j]):
                result.append(1)
            else:
                result.append(0)

    with open("submit.jsonl", "a") as up:
        up.write(json.dumps({
            "material_id": row[1].material_id,
            "rule_id": row[1].rule_id,
            "result": bool(np.mean(result) < 0.3)
        }) + "\n")
    now = datetime.now()
    print(now.strftime("2025.%m.%d %H:%M:%S"),np.meam(result))