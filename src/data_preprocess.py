import os, glob, re
import pandas as pd
from tqdm import tqdm

from mylogger import setup_logger

logger = setup_logger('../logs/data_preprocess.log')


from llms import AiBox
from utils import rule_clauses, read_markdown, get_mid2rule


def rule_info_extract_form_md(rule: str, md_path: str, aibox: AiBox) -> str:
    '''提取出单个md文件中特定规则的完整数据'''
    rule = f"{rule}（{rule_clauses[rule]}）"

    System_Extract_RuleInfo = f"提取出下面文本的关于“{rule}”的信息，要完整的、不要有遗漏的信息，更不要修改数据内容。如果相关的本文本不存在，则输出一个空字符串。"
    prompt = read_markdown(md_path)

    print(System_Extract_RuleInfo)
    print(len(prompt))
    return aibox.chat(prompt=prompt, system=System_Extract_RuleInfo)


def rule_info_extract_from_file(rule: str, md_dir: str, aibox: AiBox) -> None:
    '''提取出一份素材（几个md文件）中特定规则的完整数据'''
    save_path = f'{md_dir}/{rule}.txt'
    if os.path.exists(save_path):
        print(f'file {save_path} exists......')
        return

    all_infos = []
    for path in glob.glob(f"{md_dir}/*.md"):
        print(f'processing file:{path}...')
        sample_response = rule_info_extract_form_md(rule, path, aibox)
        print('*' * 200)
        # print(sample_response)

        if sample_response:
            all_infos.append(sample_response)

    with open(save_path, 'w', encoding='utf-8') as fp:
        for sample in all_infos:
            fp.write(sample + '\n\n')

    # return all_infos


def rule_info_extract(rule: str, doc_dir: str, aibox: AiBox) -> None:
    '''
    rule: 规则
    doc_dir： materials子文件夹
    '''
    for doc in tqdm(os.listdir(doc_dir)):
        md_dir = doc_dir + f'/{doc}'
        print(f'processing docs:{doc}...')
        rule_info_extract_from_file(rule, md_dir, aibox)


def rule_preprocess(data_name='验证集'):
    aibox = AiBox(mode='api',model='qw72')
    M_DIR = f'../DATA/{data_name}/materials'
    mid2rule_map = get_mid2rule(data_name)

    for material in os.listdir(M_DIR):
        material_path = f'{M_DIR}/{material}'

        # if material in ['m_00007a','m_00038a','m_00060a','m_00108a']: continue

        if material not in ['m_00002s', 'm_00005s', 'm_000014s', 'm_000020s', 'm_00001a', 'm_00003a','m_00004a','m_00007a','m_00139a','m_00061a']:
            continue

        RULES = mid2rule_map[material]
        for rule in RULES:
            print(f'processing material_path:{material_path}<<====>>{rule}...')
            rule_info_extract(rule,material_path, aibox)


if __name__ == '__main__':
    rule_preprocess()