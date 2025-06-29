import re
from patterns import disclaimer_patterns

from tools import (
    zh_same_string,
    diff_similarity,
    keep_only_chinese_strict,
    ngram_similarity
)


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
    for i in range(len_inputs):
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