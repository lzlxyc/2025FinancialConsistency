import re
from tqdm import tqdm

from .patterns import disclaimer_patterns

from .tools import (
    zh_same_string,
    diff_similarity,
    keep_only_chinese_strict,
    ngram_similarity
)

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
    blocks = []
    data_split = data.replace('<strong>','').replace('</strong>','').split('```')
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

    return blocks


def disclaimer_data_split(all_infos:list):
    '''责任免除分块'''
    inputs = []
    for info in all_infos:
        inputs += __disclaimer_data_presplit(info)
    inputs = [s for s in inputs if len(s.split()) > 1 or len(s) >= 50]
    inputs = list(dict.fromkeys(inputs))

    sim_blocks = []
    len_inputs = len(inputs)
    for i in tqdm(range(len_inputs)):
        for j in range(i+1, len_inputs):
            if zh_same_string(inputs[i], inputs[j]): continue

            score = diff_similarity(inputs[i], inputs[j])
            n_score = ngram_similarity(inputs[i], inputs[j])
            if score > 0.75 or n_score > 0.8:
                if score >= 0.75 and any(re.search(rule,inputs[i]) for rule in disclaimer_patterns) and any(re.search(rule,inputs[j]) for rule in disclaimer_patterns):
                    sents = (re.split(r'\d\.|\([1-9|一|二|三|四|五|六|七|八|九|十]+\)', inputs[i].replace('（', '(').replace('）', ')'))
                     + re.split(r'\d\.|\([1-9|一|二|三|四|五|六|七|八|九|十]+\)', inputs[j].replace('（', '(').replace('）', ')')) )

                    sents = [sent.strip() for sent in sents
                             if 60 >= len(keep_only_chinese_strict(sent)) >=4
                             and not any(re.search(rule, sent) for rule in disclaimer_patterns)]
                    sents = list(set(sents))
                    for k1 in range(len(sents)):
                        if any(re.search(rule, sents[k1]) for rule in disclaimer_patterns): continue

                        for k2 in range(k1+1, len(sents)):
                            if zh_same_string(sents[k1], sents[k2]): continue
                            if any(re.search(rule, sents[k2]) for rule in disclaimer_patterns): continue

                            if (diff_similarity
                                (sents[k1], sents[k2]) >= 0.7):
                                sim_blocks.append((sents[k1], sents[k2]))

                sim_blocks.append((inputs[i], inputs[j]))

    return sim_blocks