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

        filter_materials = ['m_00007a','m_00038a','m_00060a','m_00108a','m_00128a']

        if material not in mid2rule_map or material in filter_materials:
            print(f"Skip file {material} processing.")
            continue
        RULES = mid2rule_map[material]
        for rule in RULES:
            print(f'processing material_path:{material_path}<<====>>{rule}...')
            rule_info_extract(rule,material_path, aibox)



def rule_txt_merge(data_name:str='', retain_rules:list=[]) -> None:
    '''合并同rule的txt文件'''
    from utils import rule_clauses, get_rule, save_sample

    M_DIR = f'../data/{data_name}/materials'
    df = pd.read_json(f"../data/{data_name}/data.jsonl", lines=True)

    df['rule'] = df['rule'].apply(get_rule)

    cnt = 0

    from src.data_splits import back_data_split
    module_content_list = []
    for row in df.iloc[:].iterrows():
        cnt += 1
        rule, rule_id, material_id = row[1].rule, row[1].rule_id, row[1].material_id
        label = row[1].result if 'result' in df.columns else None

        if rule not in retain_rules: continue

        material_path = f'{M_DIR}/{material_id}'

        logger.info(f"==============={cnt=} || {material_id=} || {rule=} || {label=}===============")


        for file in os.listdir(material_path):
            path = f'{material_path}/{file}/{rule}.txt'
            logger.info(f"load data {path}...")
            if not os.path.exists(path): continue

            sample = open(path, 'r', encoding='utf-8').read()
            if len(re.sub(r'[^\u4e00-\u9fff]', '', sample)) <=4: continue

            module_content_list.append('START@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            module_content_list.append(f"mid::{material_id} >> {file}")
            module_content_list.append(sample)
            module_content_list.append('END@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')

    save_path = f"../logs/{'#'.join(retain_rules)}_merge.txt"
    print(save_path)
    with open(save_path, 'w', encoding='utf-8') as fp:
        fp.write('\n******************************************************\n'.join(module_content_list))

        # datas = back_data_split(module_content_list)
        #
        # for data in datas:
        #     print(data)
        #     print("*"*150+'\n')
            # with open(f'../logs/{retain_rule}_merge.txt', 'w', encoding='utf-8') as fp:
            #     fp.write('\n******************************************************\n'.join(datas))





if __name__ == '__main__':
    rule_txt_merge("测试A集_clean", ["保障责任"])