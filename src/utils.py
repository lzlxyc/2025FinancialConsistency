import markdown
import glob, os, re
from llms import AiBox
from tqdm import tqdm
import pandas as pd
import json

# from prompts import PT_Extract_RuleInfo

rule_clauses = {
    "基础产品销售信息": "该保险产品的基础配置信息，包括产品名、附加的条款信息、销售限制等",
    "投保条款": "投保过程中的缴费约定、投被保人条件限制等",
    "保障责任": "约定该产品的保险责任细节，如保障范围、保险金额、增值服务等",
    "保障相关时间": "约定该产品的各类时间信息，包括但不限于犹豫期、等待期、宽限期等",
    "赔付 & 领取规则": "约定该产品的保险责任的赔付、给付、领取及免赔细节，如赔付年龄/比例/次数等",
    "责任免除": "约定该产品不承担保险责任的情形,险人不承担给付保险金的责任",
    "续保条款": "约定续保相关信息，包括但不限于续保条件、保证续保等",
    "退保条款": "约定退保相关信息，包括但不限于退保条件、退保手续费等",
    "出险条款": "约定出险相关信息，包括但不限于出险地点、出险方式等",
    "附加条款": "约定该产品的附加条款，如特别约定等",
    "术语解释": "约定该产品的术语解释，如名词定义等"
}


def get_rule(rule):
    rule = re.sub('该产品的|在各材料中的定义没有冲突|与', '', rule)
    return re.sub('的时间', '时间', rule)


def get_mid2rule(data_name='验证集'):
    df = pd.read_json(f'../DATA/{data_name}/data.jsonl', lines=True)
    df['rule'] = df['rule'].apply(get_rule)
    df = df.groupby('material_id').agg(tuple).reset_index()
    mid2rule = dict(df[['material_id','rule']].values)
    with open(f'../DATA/{data_name}/mid2rule.json', 'w', encoding='utf-8') as f:
        json.dump(mid2rule, f, ensure_ascii=False, indent=4)

    return mid2rule


def read_markdown(path:str):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        html = markdown.markdown(text)

    return html


def save_sample(save_path:str, material_id:str, rule_id:str, result:bool) -> None:
    with open(save_path, "a") as up:
        up.write(json.dumps({
            "material_id": material_id,
            "rule_id": rule_id,
            "result": result
        }) + "\n")






    

    
