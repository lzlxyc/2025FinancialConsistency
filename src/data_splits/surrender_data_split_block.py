import re
from src.data_splits.base_split_block import remove_blank_line,data_presplit

from src.data_splits.tools import (
    zh_same_string,
    diff_similarity,
    keep_only_chinese_strict,
    ngram_similarity
)


def process(text: str) -> str:
    return re.sub(r'\n+', '\n', text.replace('""', ''))

def remove_html_tags(text: str) -> str:
    return re.sub(r'<\s*\w+[^>]*>(.*?)<\s*/\s*\w+\s*>', r'\1', text, flags=re.DOTALL)

def __surrender_data_presplit(data: str,aibox,chunker,rule) -> list:
    return data_presplit(data,aibox,chunker,rule)


def surrender_data_split(all_infos:list,aibox,chunker,rule) -> list:
    '''保障责任免除分块'''
    sim_blocks = []
    all_infos = [remove_blank_line(remove_html_tags(infos)) for infos in all_infos]
    # inputs = []
    # for info in all_infos:
    #     blocks = data_presplit(info, aibox, chunker, rule)
    #     # print('for info in all_infos: is done')
    #     # blocks = [s for s in blocks if len(s.split()) > 1 or len(s) >= 50]
    #     blocks = list(dict.fromkeys(blocks))
    #     inputs.append(blocks)
    # for i in range(len(all_infos)):
    #     for j in range(i+1, len(all_infos)):
    #         b1, b2 = process(all_infos[i]), process(all_infos[j])
    #         if zh_same_string(b1, b2) or len(b1) <= 3 or len(b2) <= 3: continue
    #         sim_blocks.append((b1, b2))
    # print(len(inputs), len(all_infos))
    # len_inputs = len(inputs)
    # for i in range(len_inputs):
    #     for j in range(i+1, len_inputs):
    #         blocks_i = inputs[i]
    #         blocks_j = inputs[j]
    #         for bi in blocks_i:
    #             if len(bi) < 5: continue
    #             for bj in blocks_j:
    #                 if len(bj) < 5: continue
    #                 score = diff_similarity(bi, bj)
    #                 n_score = ngram_similarity(bi, bj)
    #                 # print(score,'\n',n_score,'111111111111\n',bi,'222222222222222\n',bj,'333333333333\n',)
    #                 if score > 0.1 or n_score > 0.1:
    #                     sim_blocks.append((bi, bj))

    for i in range((len(all_infos))):
        for j in range(i+1, len(all_infos)):
            if re.search(r'[\u4e00-\u9fff]', all_infos[i]) and re.search(r'[\u4e00-\u9fff]', all_infos[j]):
                sim_blocks.append((all_infos[i],all_infos[j]))
    return sim_blocks
