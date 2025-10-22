import re
from tqdm import tqdm

from .patterns import disclaimer_patterns

from src.data_splits.tools import (
    zh_number_same_string,
    keep_zh_number,
    diff_similarity,
    ngram_similarity
)
split_sys = '\n' + '*' * 100 + '\n'

'''
责任免除切分逻辑：
1、先对同种情形的条款进行分大块：
    END@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    因下列原因造成被保险人住院治疗或医疗费用支出的，保险人不承担给付保险金责任：
    1. 投保人对被保险人的故意杀害或故意伤害；
    2. 被保险人故意自杀、自伤，但被保险人自杀时为无民事行为能力人的除外；
    START@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
2、再保留文本相似度高的大块，并组成后续“一致性比较”的文本对：
pair = (块1，块2)
1)块1： 
    因下列原因造成被保险人住院治疗或医疗费用支出的，保险人不承担给付保险金责任：
    1. 投保人对被保险人的故意杀害或故意伤害；
    2. 被保险人故意自杀、自伤，但被保险人自杀时为无民事行为能力人的除外；
2)块2：
    因下列原因造成被保险人住院治疗或医疗费用支出的，保险人不承担给付保险金责任：
    1. 投保人对被保险人的故意杀害或故意伤害；
    2. 被保险人故意自杀，但被保险人自杀时为无民事行为能力人的除外；
3、再将相似度高的大块，进行细分小块，得到每条条款，进一步计算小块之间的文本相似度，保留相似度高的小块文本对，作为“一致性比较”的文本对：
1）细分小块：
    1. 投保人对被保险人的故意杀害或故意伤害；
    2. 被保险人故意自杀、自伤，但被保险人自杀时为无民事行为能力人的除外；
    1. 投保人对被保险人的故意杀害或故意伤害；
    2. 被保险人故意自杀，但被保险人自杀时为无民事行为能力人的除外；
2）进行去重（完全一致的不必须进入后续的一致性对比）：
    1. 投保人对被保险人的故意杀害或故意伤害；
    2. 被保险人故意自杀、自伤，但被保险人自杀时为无民事行为能力人的除外；
    2. 被保险人故意自杀，但被保险人自杀时为无民事行为能力人的除外；
3）保留相似度高的小块文本对,输入到一致性对比模型：
    pair = (2. 被保险人故意自杀、自伤，但被保险人自杀时为无民事行为能力人的除外；, 
            2. 被保险人故意自杀，但被保险人自杀时为无民事行为能力人的除外；)
'''

def __disclaimer_data_presplit(data:str) -> list:
    '''责任免除分块'''
    data = data.replace('<strong>', '').replace('</strong>', '').replace('.','．')
    # data_split = re.sub(r'（释义[一|二|三|四|五|六|七|八|九|十]+）', '', data).split('```')

    data_split = re.sub(r'（释义[一|二|三|四|五|六|七|八|九|十]+）', '', data).split(split_sys)

    blocks = []
    for _data in data_split:
        _data = _data.replace('### 责任免除','').replace('**责任免除**','')
        if len(keep_zh_number(_data)) <= 10: continue

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
    return list(set(blocks))


def disclaimer_data_split(all_infos:list):
    '''责任免除分块'''
    inputs = []
    for info in all_infos:
        inputs += __disclaimer_data_presplit(info)

    inputs = list(dict.fromkeys(inputs))

    sim_blocks = []
    len_inputs = len(inputs)
    for i in tqdm(range(len_inputs)):
        for j in range(i+1, len_inputs):
            if zh_number_same_string(inputs[i], inputs[j]): continue

            score = diff_similarity(inputs[i], inputs[j])
            n_score = ngram_similarity(inputs[i], inputs[j])
            if score > 0.75 or n_score > 0.8:
                if score >= 0.75 and any(re.search(rule,inputs[i]) for rule in disclaimer_patterns) and any(re.search(rule,inputs[j]) for rule in disclaimer_patterns):
                    sents = (re.split(r'\d\．|\([1-9|一|⼀|二|⼆|三|四|五|六|七|八|九|十]+\)', inputs[i].replace('（', '(').replace('）', ')'))
                     + re.split(r'\d\．|\([1-9|一|⼀|二|⼆|三|四|五|六|七|八|九|十]+\)', inputs[j].replace('（', '(').replace('）', ')')) )

                    sents = [sent.strip() for sent in sents
                             if 60 >= len(keep_zh_number(sent)) >=4
                             and not any(re.search(rule, sent) for rule in disclaimer_patterns)]
                    sents = [sent for sent in sents if not re.search(r'第.+条.+责任免除事项')]
                    sents = list(set(sents))
                    for k1 in range(len(sents)):
                        if any(re.search(rule, sents[k1]) for rule in disclaimer_patterns): continue

                        for k2 in range(k1+1, len(sents)):
                            if zh_number_same_string(sents[k1], sents[k2]): continue
                            if any(re.search(rule, sents[k2]) for rule in disclaimer_patterns): continue

                            if (diff_similarity(sents[k1], sents[k2]) >= 0.7):
                                sim_blocks.append((sents[k1], sents[k2]))

                sim_blocks.append((inputs[i], inputs[j]))

    return sim_blocks



def extract_chinese(text):
    # 使用正则表达式提取所有中文字符
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    chinese_matches = chinese_pattern.findall(text)
    return ''.join(chinese_matches)


def remove_common_by_chinese(list1, list2):
    # 提取每个字符串的中文部分
    list1_chinese = [extract_chinese(s) for s in list1]
    list2_chinese = [extract_chinese(s) for s in list2]

    # 找出两个列表中相同的中文部分
    common_chinese = set(list1_chinese) & set(list2_chinese)

    # 移除list1和list2中中文部分在common_chinese中的字符串
    new_list1 = [s for s, ch in zip(list1, list1_chinese) if ch not in common_chinese]
    new_list2 = [s for s, ch in zip(list2, list2_chinese) if ch not in common_chinese]

    return new_list1, new_list2



def disclaimer_data_split(all_infos:list):
    '''责任免除分块'''
    inputs = []
    for info in all_infos:
        inputs.append(__disclaimer_data_presplit(info))

    sim_blocks = []

    len_inputs = len(inputs)
    for i in range(len_inputs):
        for j in range(i+1, len_inputs):
            blocks_i = inputs[i]
            blocks_j = inputs[j]
            for bi in blocks_i:
                if len(bi) < 6: continue
                for bj in blocks_j:
                    if len(bj) < 6: continue

                    if '房屋连续' in bi and '房屋连续' in bj:
                        print()
                    if zh_number_same_string(bi, bj): continue

                    score = diff_similarity(bi, bj)
                    n_score = ngram_similarity(bi, bj)
                    if score > 0.75 or n_score > 0.8:
                        if score >= 0.75 and any(re.search(rule,bi) for rule in disclaimer_patterns) and any(re.search(rule,bj) for rule in disclaimer_patterns):
                            sent1 = re.split(r'\d+．|\([1-9|一|⼀|二|⼆|三|四|五|六|七|八|九|十]+\)', bi.replace('（', '(').replace('）', ')').replace('.','．'))
                            sent2 = re.split(r'\d+．|\([1-9|一|⼀|二|⼆|三|四|五|六|七|八|九|十]+\)', bj.replace('（', '(').replace('）', ')').replace('.', '．'))

                            sent1 = [sent.strip().replace('\t','') for sent in sent1
                                     if 60 >= len(keep_zh_number(sent)) >=4
                                     and not any(re.search(rule, sent) for rule in disclaimer_patterns)]

                            sent2 = [sent.strip().replace('\t','') for sent in sent2
                                     if 60 >= len(keep_zh_number(sent)) >=4
                                     and not any(re.search(rule, sent) for rule in disclaimer_patterns)]

                            sent1, sent2 = remove_common_by_chinese(sent1, sent2)

                            for k1 in sent1:
                                if any(re.search(rule, k1) for rule in disclaimer_patterns): continue
                                for k2 in sent2:
                                    if any(re.search(rule, k2) for rule in disclaimer_patterns): continue

                                    if (diff_similarity(k1, k2) >= 0.7):
                                        sim_blocks.append((k1, k2))

                        #                 print(k1)
                        #                 print('vs')
                        #                 print(k2)
                        #                 print("#"*100)
                        # print(bi)
                        # print('vs')
                        # print(bj)
                        # print("#" * 200)

                        sim_blocks.append((bi, bj))

    sim_blocks = list(set(sim_blocks))
    sim_blocks = sorted(sim_blocks, key=lambda x: len(''.join(x)))

    return sim_blocks