import re
from collections import Counter
import math
from difflib import SequenceMatcher

def preprocess(text):
    """文本预处理：去标点、转小写、分词"""
    return  re.sub(r'[^\w\s]', '', text.lower()).replace('\n','').strip()  # 移除标点



def generate_ngrams(words, n=2):
    """生成N-gram序列"""
    return [''.join(words[i:i + n]) for i in range(len(words) - n + 1)]



def ngram_similarity(text1:str, text2:str, n=2):
    """计算两份文件的N-gram相似度（余弦相似度）"""
    # 读取文件

    # 预处理并生成N-grams
    words1, words2 = preprocess(text1), preprocess(text2)
    if len(words1) <=3 or len(words2) <=3: return 0.0

    ngrams1, ngrams2 = generate_ngrams(words1, n), generate_ngrams(words2, n)

    same_len = len(set(ngrams1) & set(ngrams2))

    max_score = max(same_len / len(ngrams1), same_len / len(ngrams2))
    # print(max_score, same_len, len(ngrams1), len(ngrams2))
    # print(text1)
    # print('############################################')
    # print(text2)
    # print("*"*100)

    # 统计词频
    vec1, vec2 = Counter(ngrams1), Counter(ngrams2)

    # 获取所有唯一N-grams
    all_ngrams = set(vec1.keys()).union(set(vec2.keys()))

    # 构建向量
    vec1_full = [vec1.get(ngram, 0) for ngram in all_ngrams]
    vec2_full = [vec2.get(ngram, 0) for ngram in all_ngrams]

    # 计算余弦相似度
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1_full, vec2_full))
    magnitude1 = math.sqrt(sum(v ** 2 for v in vec1_full))
    magnitude2 = math.sqrt(sum(v ** 2 for v in vec2_full))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    score = dot_product / (magnitude1 * magnitude2)
    return 0.75*max_score +  0.25*score


def diff_score(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


def spilt_block_by_socre(all_infos:list, rule):
    inputs = []
    for info in all_infos:
        inputs += disclaimer_data_split(info)
    inputs = [s for s in inputs if len(s.split()) > 1 or len(s) >= 50]
    inputs = list(set(inputs))

    sim_blocks = []
    len_inputs = len(inputs)
    for i in range(len_inputs):
        for j in range(i+1, len_inputs):
            score = diff_score(inputs[i], inputs[j])
            n_score = ngram_similarity(inputs[i], inputs[j])
            # print(score, n_score)
            if score > 0.75 or n_score > 0.8:
                # print(score, n_score)
                # print(inputs[i])
                # print("=================================vs======================================")
                # print(inputs[j]+'\n\n')

                sim_blocks.append((inputs[i], inputs[j]))

    return sim_blocks



def disclaimer_data_split(data:str) -> list:
    '''责任免除分块'''
    patterns = [
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

    blocks = []
    data_split = data.split('```')
    for _data in data_split:
        sub_data_lines = _data.split('\n')
        block = ''
        for line in sub_data_lines:
            if not line: continue
            # print(line, any(re.search(rule,line) for rule in patterns))
            if any(re.search(rule,line) for rule in patterns):
                blocks.append(block)
                block = line
            else:
                block += '\n' + line
        if block:
            # print(block)
            blocks.append(block)

    return blocks









