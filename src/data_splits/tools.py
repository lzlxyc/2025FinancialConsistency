import re
from collections import Counter
import math
from difflib import SequenceMatcher



def keep_only_chinese_strict(text:str) ->str:
    '''只保留中文字符串'''
    return re.sub(r'[^\u4e00-\u9fff]', '', text)


def zh_same_string(text1:str, text2:str) -> bool:
    '''中文字符串文本对比'''
    return keep_only_chinese_strict(text1) == keep_only_chinese_strict(text2)


def preprocess(text):
    """文本预处理：去标点、转小写、分词"""
    return  re.sub(r'[^\w\s]', '', text.lower()).replace('\n','').strip()  # 移除标点


def ngram_similarity(text1:str, text2:str, n=2):
    """计算两份文件的N-gram相似度（余弦相似度）"""
    def generate_ngrams(words, n=2):
        """生成N-gram序列"""
        return [''.join(words[i:i + n]) for i in range(len(words) - n + 1)]

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


def diff_similarity(str1, str2):
    '''编辑距离相似度'''
    return SequenceMatcher(None, str1, str2).ratio()
















