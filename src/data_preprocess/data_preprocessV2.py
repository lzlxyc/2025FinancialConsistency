import os, glob, re
import pandas as pd
from tqdm import tqdm
from time import time
from bs4 import BeautifulSoup

from mylogger import setup_logger

logger = setup_logger('../logs/data_preprocess.log')


from llms import AiBox
from utils import rule_clauses, read_markdown, get_mid2rule

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from data_splits.disclaimer_split_block import __disclaimer_data_presplit, disclaimer_data_split
import modelscope.pipelines.nlp.document_segmentation_pipeline



from tools.word_map_tools import word_map


def keep_only_chinese_strict(text:str) ->str:
    '''只保留中文字符串'''
    return re.sub(r'[^\u4e00-\u9fff]', '', text)


model_path = 'D:/LZL/workspace/ModelHub/nlp_bert_document-segmentation_chinese-base'
p = pipeline(
    task=Tasks.document_segmentation,
    model=model_path)

split_sys = '\n' + '*' * 100 + '\n'
# 3. 文本内容是直接从pdf原文件中提取出的保险条款原文，字体排版存在一些空间结构，直接转换成字符串后会导致条款原文非常难以阅读。请把{rule}相关的内容重新组织成清晰可读的格式。
def rule_info_extract_form_md(rule: str, block: str, aibox: AiBox) -> str:
    '''提取出单个md文件中特定规则的完整数据'''
    rule_info = f"{rule}（{rule_clauses[rule]}）"

    System_Extract_RuleInfo = f'''你是一名专业的保险条款处理专家，需要从将原始保险条款文件进行特定的{rule}相关的条款的信息提取。'''
    prompt = f'''** 任务指令 **：
    ** 从将原始保险条款文件进行特定的{rule}相关的条款的信息提取
    ** 请严格遵循以下规则：
    1. 只提取跟{rule}有关的信息，其中：{rule_info}；
    2. 不能修改、增加任何文本内容,一定要保证数据的完整，不要有遗漏的信息；
    3. 如果跟{rule} 相关的文本不存在，就直接输出'None'；
    4. 不要输出任何多余的信息，比如解释、分析等，否则我将惩罚你。
    ** 待处理文本如下：\n
    {block}\n
    ** 请按要求输出：
    '''
    if len(prompt)>15000:
        print(len(prompt))
    response = aibox.chat(prompt=prompt, system=System_Extract_RuleInfo)
    return response


def rule_block_info_extract_form_md(md_path: str) -> list:
    '''加上?=断言:return:
    '''
    data = read_markdown(md_path).replace('.', '．')
    data = re.sub('\n\s', '\n', data)
    if len(data) <=6: return []

    for pattern in [
        r'第[一二三四五六七八九十]+部分\s',
        r'\n\b\d+	',
        r'\n\b\d+\．\s'
    ]:
        if len(re.findall(pattern, data))>=3:
            break

    # print(pattern)
    all_blocks = []

    if '部分' in pattern:
        big_split_datas = re.split(f'(?={pattern})|(?=\n.+（互联网.+条款)', data)
        # if len(max(big_split_datas, key=len)) > 10000:
        for split_data in big_split_datas:
            if split_data:
                all_blocks += re.split(r'(?=第[一二三四五六七八九十]+条[\n\t\s])', split_data)
        all_blocks = [d for d in all_blocks if d is not None]

    else:
        all_patterns = r'(?=\n.+（互联网专属）条款)|(?=\n[一二三四五六七八九十]+、)|(?=\n【\d+\．[\n\t\s])|(?=第[一二三四五六七八九十]+条\n)|(?=\n第[一二三四五六七八九十]+条[\n\t\s])'
        big_split_datas = re.split(all_patterns, data)
        for split_data in big_split_datas:
            if split_data:
                all_blocks += re.split(f'(?={pattern})', split_data)

    all_split_blocks = []
    for split_data in all_blocks:
        all_split_blocks += re.split(r'(?=【.*】[\n\t\s])|\n\n\n', split_data)


    if len(max(all_split_blocks, key=len)) > 100000:
        end_all_split_blocks = []
        for split_data in all_split_blocks:
            if len(split_data)<=10000:
                end_all_split_blocks.append(split_data)
            else:
                tmp = re.split(r'(?=\n\d+．\d+\s)|(?=\n\d+\s)|(?=\n（\d+）)', split_data)
                end_all_split_blocks += tmp

        all_split_blocks = [data for data in end_all_split_blocks if len(data) >=1]
    # split_data_str = f'{split_sys}'.join(all_split_blocks)

    # all_blocks = [d for d in all_blocks if len(d) >=15000]

    # if all_blocks:
    #     print(len(max(all_blocks, key=len)))
        # with open('../logs/data_split.log', 'a', encoding='utf-8') as f:
        #     f.write('\n'+ pattern+ md_path + '\n')
        #     f.write(f'{split_sys}'.join(all_split_blocks))

    return [a for d in all_split_blocks if len(a:=re.sub(r'\n+', '\n', d)) >=4]



def rule_info_extract_from_file(rule: str, md_dir: str, aibox: AiBox) -> None:
    '''提取出一份素材（几个md文件）中特定规则的完整数据'''
    save_block_path = f'{md_dir}/all_info.block'
    save_rule_path = f'{md_dir}/{rule}.txt'

    # if os.path.exists(save_block_path):
    #     print(f'file {save_block_path} exists......')
    #     return

    if os.path.exists(save_rule_path):
        print(f'file {save_rule_path} exists......')
        return

    # 全量数据分块
    all_blocks= []

    for path in glob.glob(f"{md_dir}/*.md"):
        print(f'processing file:{path}...')

        all_blocks += rule_block_info_extract_form_md(path)

    all_blocks = list(dict.fromkeys(all_blocks))

    print("召回数据量：", len(all_blocks))
    with open(save_block_path, 'w', encoding='utf-8') as fp:
        for sample in all_blocks:
            fp.write(sample + split_sys)

    # 大模型召回特定rule的分块
    rule_blocks = []
    for i, sample in enumerate(all_blocks):
        print(i, len(sample))
        rule_block = rule_info_extract_form_md(rule, sample, aibox).strip()
        if (len(keep_only_chinese_strict(rule_block))<=4
                or rule_blocks in [None,'None','null','NAN','nan']
                or "输出 'None'" in rule_block):
            continue

        if '请提供待处理的保险条款文本内容' in rule_block:
            continue

        rule_blocks.append(rule_block)

    with open(save_rule_path, 'w', encoding='utf-8') as fp:
        for sample in rule_blocks:
            fp.write(sample + split_sys)




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
    M_DIR = f'../data/{data_name}/materials'
    mid2rule_map = get_mid2rule(data_name)

    cnt = 0

    for material in os.listdir(M_DIR):
        material_path = f'{M_DIR}/{material}'

        filter_materials = []

        if material not in mid2rule_map or material in filter_materials:
            print(f"Skip file {material} processing.")
            continue
        RULES = mid2rule_map[material]
        # if '责任免除' not in RULES: continue
        # RULES = ['责任免除']
        for rule in RULES:
            cnt += 1
            st = time()
            print(f'processing material_path:{material_path}<<====>>{rule}...')
            rule_info_extract(rule,material_path, aibox)
            print(f'**{time() - st} finished material_path:{material_path}<<====>>{rule}...')

    return cnt

def rule_txt_merge(data_name:str='', retain_rules:list=[]) -> None:
    '''合并同rule的txt文件'''
    from utils import rule_clauses, get_rule, save_sample

    M_DIR = f'../data/{data_name}/materials'
    df = pd.read_json(f"../data/{data_name}/data.jsonl", lines=True)

    df['rule'] = df['rule'].apply(get_rule)

    cnt = 0

    tables = str.maketrans(word_map)

    for row in df.iloc[:].iterrows():
        cnt += 1
        rule, rule_id, material_id = row[1].rule, row[1].rule_id, row[1].material_id
        label = row[1].result if 'result' in df.columns else None

        if rule not in retain_rules: continue

        if material_id not in ['m_00059a']: continue

        material_path = f'{M_DIR}/{material_id}'

        logger.info(f"==============={cnt=} || {material_id=} || {rule=} || {label=}===============")

        module_content_list = []
        for file in os.listdir(material_path):
            path = f'{material_path}/{file}/{rule}.txt'
            logger.info(f"load data {path}...")
            if not os.path.exists(path): continue

            sample = open(path, 'r', encoding='utf-8').read()

            sample = sample.translate(tables)


            if len(re.sub(r'[^\u4e00-\u9fff]', '', sample)) <=4: continue

            # module_content_list.append('START@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            # module_content_list.append(f"mid::{material_id} >> {file}")
            module_content_list.append(sample)

            # module_content_list += disclaimer_data_split(sample)
            #
            # module_content_list.append('END@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')

        disclaimer_data_split(module_content_list)

    # save_path = f"../logs/{'#'.join(retain_rules)}_merge.txt"
    # print(save_path)
    # with open(save_path, 'w', encoding='utf-8') as fp:
    #     fp.write('\n******************************************************\n'.join(module_content_list))



    # with open(f'../logs/{retain_rules}_merge.txt', 'w', encoding='utf-8') as fp:
    #     for data in module_content_list:
    #         print(data)
    #         print("*"*150+'\n')
    #         fp.write(data)
            # fp.write('\n******************************************************\n'.join(datas))



if __name__ == '__main__':
    aibox = AiBox(mode='api', model='qw72')

    stime = time()
    cnt = rule_preprocess("测试 B 集")
    # print("** time:", cnt, time() - stime, (time() - stime) / cnt)
    # rule_txt_merge(data_name='测试 A 集', retain_rules=['责任免除'])