'''
用户构造文本比对模型的训练数据
'''

import sys
sys.path.append('../../')

import random
import pandas as pd
from src.support.utils import get_rule
from src.support.word_map_tools import word_map
from loguru import logger
import re, os
from difflib import SequenceMatcher

disclaimer_patterns = [
        r'下列.*不负.*赔偿',
        r'下列.*不负.*责任',
        r'下列.*不承担.*赔偿',
        r'下列.*不承担.*责任',
        r'下列.*不承担.*保险',
        r'下列财产不属于本保险合同的保险标',
        r'其他不属于.*不负责.*赔偿',
        r'第[一二三四五六七八九十]+条.+不负责',
        r'第[一二三四五六七八九十]+条.+责任免除',
        r'★投保人/被保险人未履行义务导致的责任免除★',
        r'因下列原因.*保险人',
        r'下列.*情形.*保险人',
        '[一二三四五六七八九十]、保险人.*不负责',
        '[一二三四五六七八九十]、保险人.*不承担',
        '《.+》责任免除事项',
        '下列属于其他险种保险责任.*费⽤.*责任',
        '[一二三四五六七八九十]、.+保险人不负责赔偿'
]

def diff_similarity(str1, str2):
    '''编辑距离相似度'''
    return SequenceMatcher(None, str1, str2).ratio()



def remove_similar_strings(strings, threshold=0.7):
    '''去除相似度超过阈值的字符串，只保留第一个'''
    unique_strings = []

    for current_str in strings:
        # 检查当前字符串是否与已保留的字符串中的任何一个相似
        is_similar = False
        for kept_str in unique_strings:
            if diff_similarity(current_str, kept_str) > threshold:
                is_similar = True
                break

        # 如果不相似，则保留
        if not is_similar:
            unique_strings.append(current_str)

    return unique_strings


def make_sample(pairA:str, pairB:str, output:str) -> str:
    prompt = f'''
你是一个专业的金融保险行业信息处理专家，需要对下面两份文本片段内容进行冲突分析。
- 只在两个文本中共有的、相同的某条条款进行比对，当具体内容有存在实质性差异，就视为冲突；如果是相同描述但不同表达，则不视为冲突。
- 当文本存在多个条款时，当其中一个文本存在其他条款而另一个文本未提及，则不视为冲突，即无需比较。
请开始你的分析：\n\n【文本1】：\n{pairA}\n\n【文本2】\n{pairB}\n
## 输出格式
- 如果存在冲突，就输出：<res>文本冲突</res>\n<冲突文本段>：xxx
- 如果不存在冲突，就输出：<res>文本一致</res>\n<冲突文本段>：无。'''
    system = '你是一个专业的金融保险行业信息处理专家，需要对下面两份文本片段进行冲突分析，不同文本段的部分关键信息可能会被多处定义，售卖平台需要保证这些定义的一致性,从而满足监管要求，同时保障客户的合法权益。'
    return {'instruction': system, 'input': prompt.replace('\t',''), 'output': output.replace('\t','')}


def data_for_train(data:str) -> list:
    '''构造文本比对训练数据
    1、正样例：在大块数据中，随机删除某些条款；在小块数据中，随机增加一些额外噪声
    2、负样例：在小块中，删除某些类型；或者数字；或者日期（数值型、类别型）
    '''
    out_template_true = '<res>文本一致</res>\n<冲突文本段>：'
    out_template_false = '<res>文本冲突</res>\n<冲突文本段>：'


    data = data.replace('<strong>', '').replace('</strong>', '').replace('（','(').replace('）',')').strip()
    data_split = re.sub(r'（释义[一|二|三|四|五|六|七|八|九|十]+）', '', data).split('```')

    blocks = []
    for _data in data_split:
        sub_data_lines = _data.split('\n')
        block = ''
        for line in sub_data_lines:
            if not line: continue
            # print(line, any(re.search(rule,line) for rule in patterns))
            if any(re.search(rule,line) for rule in disclaimer_patterns):
                blocks.append(block)
                block = line
            else:
                block += '\n' + line
        if block:
            # print(block)
            blocks.append(block)

    blocks = [block for block in blocks if len(block.split()) > 1 or len(block) >= 50]
    blocks = remove_similar_strings(blocks)

    train_pairs = []

    for block in blocks:
        '''基于block进行数据构造'''
        sents = re.split(r'\d+．|\([1-9|一|⼀|二|⼆|三|四|五|六|七|八|九|十]+\)|[1-9|一|⼀|二|⼆|三|四|五|六|七|八|九|十]、', block)
        if len(sents) < 3: continue

        # 大块的正样例构建
        choices = list(range(1, min(4, len(sents))))
        random_number = random.choice(choices)
        pairA = random.sample(sents, min(4, len(sents)))
        pairB = random.sample(pairA, random_number)
        output = '其中一个文本对比另一个文本，存在几个条款未提及情况，但在共有的、相同的条款中无实质性冲突。'
        true_sample = make_sample('\n'.join(pairA), '\n'.join(pairB), out_template_true+output)
        train_pairs.append(true_sample)

        # new_block = None
        # for i, sample in enumerate(pairB):
        #     if len(sample) <= 20 and sample.count('、') >= 2:
        #         tmp = sample.split('、')
        #         new_block = '、'.join(tmp[:1]+tmp[2:])
        #         break
        #
        # if new_block:
        #     pairB[i] = new_block
        #     if random_number == 1:
        #         out_put = f'文本2缺少{tmp[1]}，明显不一致，存在冲突。'
        #         false_sample = make_sample('\n'.join(pairA), '\n'.join(pairB), out_template_false + out_put)
        #     else:
        #         out_put = f'文本1缺少{tmp[1]}，明显不一致，存在冲突。'
        #         false_sample = make_sample('\n'.join(pairA), '\n'.join(pairB), out_template_false + out_put)
        #     train_pairs.append(false_sample)


        sample = sents[1]
        new_block = sents[0][-4:] + ' ' + sample + ' ' + sents[2][:8]
        output = '两文本在共有的、相同的条款中无实质性冲突。'
        if random_number == 1:
            true_sample = make_sample(sample, new_block, out_template_true+output)
        else:
            true_sample = make_sample(new_block, sample, out_template_true+output)
        train_pairs.append(true_sample)

        # 构造负样例
        data_for_false = [block] if random_number == 1 else sents

        for sample in data_for_false:
            if (day_match:=re.search(r'[1-9]+(天|日)', sample)):
                old_day = day_match.group()[:-1]
                new_day = int(old_day) + random_number + 10
                new_block = sample.replace(day_match.group(), str(new_day) + '天')
                out_put = f'文本1是{old_day}天，而文本2是{new_day}天，明显不一致，存在冲突。'

                false_sample = make_sample(sample, new_block, out_template_false + out_put)
                train_pairs.append(false_sample)
                break

            if (money_match:=re.search(r'[1-9]+元', sample)):
                old_money = money_match.group()[:-1]
                new_money = int(old_day) + random_number * 10
                new_block = sample.replace(money_match.group(), str(new_money) + '元')
                out_put = f'文本1是{old_money}元，而文本2是{new_money}元，明显不一致，存在冲突。'

                false_sample = make_sample(sample, new_block, out_template_false + out_put)
                train_pairs.append(false_sample)
                break


        for sample in sents:
            if sample.count('、') >= 2:
                tmp = sample.split('、')
                new_block = '、'.join(tmp[:1]+tmp[2:])
                if random_number == 1:
                    out_put = f'文本2缺少{tmp[1]}，明显不一致，存在冲突。'
                    false_sample = make_sample(sample, new_block, out_template_false + out_put)
                else:
                    out_put = f'文本1缺少{tmp[1]}，明显不一致，存在冲突。'
                    false_sample = make_sample(new_block, sample, out_template_false + out_put)
                train_pairs.append(false_sample)
                break

    return train_pairs

def rule_txt_merge(data_name:str='', retain_rules:list=[]) -> None:
    '''合并同rule的txt文件'''
    M_DIR = f'../data/{data_name}/materials'
    df = pd.read_json(f"../data/{data_name}/data.jsonl", lines=True)

    df['rule'] = df['rule'].apply(get_rule)

    cnt = 0

    tables = str.maketrans(word_map)

    all_pairs = []
    for row in df.iloc[:].iterrows():
        cnt += 1
        rule, rule_id, material_id = row[1].rule, row[1].rule_id, row[1].material_id
        label = row[1].result if 'result' in df.columns else None

        if rule not in retain_rules: continue

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

            module_content_list.append(sample)

        filter_datas = [sample for sample in module_content_list if len(sample) >=20 and any(k in sample for k in ['天','元','日'])]
        if not filter_datas:
            filter_datas = [max(module_content_list, key=len)]
        for longest_string in filter_datas[:1]:
            blocks = data_for_train(longest_string)
            all_pairs += blocks


    return all_pairs

if __name__ == '__main__':
    all_pairs = rule_txt_merge(data_name='raw_data_b', retain_rules=['责任免除'])
    pd.DataFrame(all_pairs).to_json('../../data/train_text_comp.jsonl',
                                    lines=True, orient='records', force_ascii=False)

    print(len(all_pairs))