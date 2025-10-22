import re

split_sys = '```'
from .tools import (
    zh_number_same_string,
    diff_similarity,
    ngram_similarity
)

'''
保障时间切分逻辑：
'''


def process(text: str) -> str:
    return re.sub(r'\n+', '\n', text.replace('"', ''))


def __indemnity_time_data_presplit(data:str) -> list:
    blocks = []

    data = data.replace('.','．').replace('""','').replace('"','')

    for _data in data.split(split_sys):
        block = ''
        for line in _data.split('\n'):
            # if not line or '空字符串' in line or '未提及' in data: continue

            # print(f"{line=} {block=} {blocks=}")
            if re.search(r'^\d+\．|^（[1-9]+）|^（[一二三四五六七八九十]+）|^[①-⑨]', line.strip()):
                blocks.append(block)
                block = line
            elif len(block) <= 4 or '  -' in line:
                block += line
            else:
                blocks.append(block)
                block = line

        if len(block) >= 4: blocks.append(block)

    return blocks


def indemnity_time_data_split(all_infos:list) -> list:
    '''保障责任免除分块'''
    sim_blocks = []
    # 对大块进行配对
    for i in range(len(all_infos)):
        for j in range(i+1, len(all_infos)):
            b1, b2 = process(all_infos[i]), process(all_infos[j])
            if zh_number_same_string(b1, b2) or len(b1) <= 3 or len(b2) <= 3: continue
            for k1 in b1.split(split_sys):
                for k2 in b2.split(split_sys):
                    if zh_number_same_string(k1, k2) or len(k1) <= 3 or len(k2) <= 3: continue
                    sim_blocks.append((k1, k2))

    # 对于小块进行配对
    inputs = []
    for info in all_infos:
        blocks = __indemnity_time_data_presplit(info)
        blocks = [s for s in blocks if len(s.split()) > 1 or len(s) >= 50]
        blocks = list(dict.fromkeys(blocks))
        inputs.append(blocks)

    len_inputs = len(inputs)
    for i in range(len_inputs):
        for j in range(i+1, len_inputs):
            blocks_i = inputs[i]
            blocks_j = inputs[j]
            for bi in blocks_i:
                if len(bi) < 10: continue
                for bj in blocks_j:
                    if len(bj) < 10: continue
                    if zh_number_same_string(bi, bj): continue
                    score = diff_similarity(bi, bj)
                    n_score = ngram_similarity(bi, bj)
                    if score > 0.75 or n_score > 0.8:
                        sim_blocks.append((bi, bj))

    return sim_blocks
