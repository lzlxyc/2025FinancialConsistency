import pandas as pd
import re
import markdown
import glob, os, re
from tqdm import tqdm
import pandas as pd
import json
from typing import Dict

import sys
sys.path.append('/data2/cwli16/2025FinancialConsistency')

from src.llms import AiBox

def read_markdown(path:str):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        html = markdown.markdown(text)

    return html


def get_rule(rule):
    rule = re.sub('该产品的|在各材料中的定义没有冲突|与', '', rule)
    return re.sub('的时间', '时间', rule)


def get_mid2rule(data_name='测试A集_clean'):
    #df = pd.read_json(f'../DATA/{data_name}/data.jsonl', lines=True)
    file_path = '/data2/cwli16/2025FinancialConsistency/DATA/测试A集_clean/data.jsonl'
    df = pd.read_json(file_path, lines=True)
    df['rule'] = df['rule'].apply(get_rule)
    df = df.groupby('material_id').agg(tuple).reset_index()
    mid2rule = dict(df[['material_id','rule']].values)
    output_path = f'/data2/cwli16/2025FinancialConsistency/DATA/{data_name}/mid2rule.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mid2rule, f, ensure_ascii=False, indent=4)

    return mid2rule



def rule_info_extract_form_md(rules: list, md_path: str, aibox: AiBox) -> str:
    '''提取出单个md文件中特定规则的完整数据'''
    System_Extract_RuleInfo = '''**任务指令**：
你是一名专业的保险条款处理专家，需要将原始保险条款文件进行分块，方便后续进行向量嵌入。请严格遵循以下规则：
1，不能修改和删除任何文本内容
2，输出为JSON格式，不要包含任何Markdown标记（如```json或```）
3，分块形式如下：
 - `block`：分块序号（从1开始的连续数字，如1, 2, 3...）
 - `title`：把条款标题或者开头标题写在这里，如果没有可根据文件内容提取
 - `content`：数组形式存放完整条款原文（保留原始编号和换行符）


**你必须返回严格符合以下JSON格式的内容，不要包含任何额外文本：
[
    {
        "block": 1,
        "title": "条款标题",
        "content": ["条款具体内容..."]
    }
]
'''

    prompt = read_markdown(md_path)

    print(System_Extract_RuleInfo)
    print(len(prompt))

    folder_name = os.path.basename(os.path.dirname(md_path))
    response =aibox.chat(prompt=prompt, system=System_Extract_RuleInfo)

    return response 



def rule_info_extract_from_file(rules: list, md_dir: str, aibox: AiBox, label: str) -> None:
    '''提取出一份素材（几个md文件）中特定规则的完整数据，分别保存每个文件的结果'''
    
    for path in glob.glob(os.path.join(md_dir, '*.md')):  # 获取所有 .md 文件
        print(f'Processing file: {path}...')
        sample_response = rule_info_extract_form_md(rules, path, aibox)
        print('*' * 200)

        if sample_response:
            # 使用文件名作为保存的文件名，添加后缀以区分
            file_name = os.path.splitext(os.path.basename(path))[0]  # 去掉文件扩展名
            save_path = os.path.join(md_dir, f"{file_name}_final_block.txt")
            
            # 检查文件是否已存在
            if os.path.exists(save_path):
                print(f'File {save_path} exists, skipping...')
                continue
            
            # 保存结果到单独的文件
            with open(save_path, 'w', encoding='utf-8') as fp:
                fp.write(sample_response + '\n\n')  # 只写入当前文件的结果




# ****************提取文件方式：通过规则名称提取文件,例如“保障责任”*****************
# def rule_info_extract(rules: list, doc_dir: str, aibox: AiBox, label: str) -> None:
#     '''
#     rule: 规则
#     doc_dir： materials子文件夹
#     '''
#     for doc in tqdm(os.listdir(doc_dir)):
#         md_dir = os.path.join(doc_dir, doc)
#         print(f'processing docs:{doc}...')
#         rule_info_extract_from_file(rules, md_dir, aibox,label)


# ****************提取文件方式：通过规则名称提取文件,例如“保障责任”*****************
# def rule_process(data_name='测试A集_clean', retain_rules:list=["保障责任"]):
#     aibox = AiBox(mode='api',model='qw72')
#     M_DIR = '/data2/cwli16/2025FinancialConsistency/DATA/测试A集_clean/materials'

#     mid2rule_map = get_mid2rule(data_name)

#     for material in os.listdir(M_DIR):
#         material_path = f'{M_DIR}/{material}'

#         rules = mid2rule_map[material]

#         #if mid2rule_map[material] not in retain_rules:
#         if not any(rule in retain_rules for rule in rules):
#             print(f"Skip file {material} processing.")
#             continue
#         RULES = mid2rule_map[material]
#         label = os.path.splitext(material)[0] 
#         print(f'processing material_path:{material_path}<<====>>{RULES}...')
#         rule_info_extract(RULES,material_path, aibox, label=label)


#*****************提取文件方式：通过文件名提取,如m_00014a*************************
def rule_process(folder_names:list=['m_00008a'],rules: list = []) -> None:
    '''对给定文件夹列表中的每个文件调用规则提取函数'''
    aibox = AiBox(mode='api',model='qw72')
    
    for folder_name in folder_names:
        M_DIR = f'/data2/cwli16/2025FinancialConsistency/DATA/测试A集_clean/materials/{folder_name}'
        
        for material in os.listdir(M_DIR):
            material_path = os.path.join(M_DIR, material)
            print(f'Processing material: {material_path}...')
            label = os.path.splitext(material)[0] 
            # 调用规则提取函数
            rule_info_extract_from_file(rules=rules, md_dir=material_path, aibox=aibox,label=label)


if __name__ == '__main__':
   # data_name='测试A集_clean'
  #  retain_rules=["该产品的保障责任在各材料中的定义没有冲突"]

    rule_process()

