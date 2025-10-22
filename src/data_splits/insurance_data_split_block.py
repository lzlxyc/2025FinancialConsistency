import re
from src.data_splits.base_split_block import remove_blank_line

from .tools import (
    zh_same_string,
    diff_similarity,
    keep_only_chinese_strict,
    ngram_similarity
)

'''
保障责任切分逻辑：
'''


def process(text: str) -> str:
    return re.sub(r'\n+', '\n', text.replace('""', ''))

def remove_html_tags(text: str) -> str:
    return re.sub(r'<\s*\w+[^>]*>(.*?)<\s*/\s*\w+\s*>', r'\1', text, flags=re.DOTALL)

def __insurance_data_presplit(data: str) -> list:
    blocks = []
    block = ''

    pattern = re.compile(r''' 
        ^#{2,6}\s*\d+(\.\d+)*\s+|               # markdown 目录编号标题，如 #### 1.3 投保年龄
        ^#{2,6}\s*第[一二三四五六七八九十百千万零〇]+条| # markdown 中文编号标题：### 第十五条
        ^#{2,6}\s*[\u4e00-\u9fa5]{2,}.*$|       # markdown 中文标题（无编号）
        ^[◆●■※▍•\*]+|                          # 特殊符号开头：◆职业要求、●注意事项等
        ^\*\*.+?\*\*[:：]?|                      # 加粗开头，如 **投保人条件：**
        ^【.+?】|                               # 【...】中文括号起始，如 【风险承受能力声明】
        ^\d+(\.\d+)+\s+|                        # 数字点格式，如 1.3 投保年龄
        ^\d+\.\s*|                              # 1.
        ^\d+\s+|                                # 1 （数字加空格） 新增支持纯数字加空格格式
        ^\d+）|                                 # 1）
        ^\d+、|                                 # 1、
        ^（\d+）|                              # （1）
        ^（[一二三四五六七八九十]+）|            # （一）
        ^[①-⑨]|                                # ①
        ^[一二三四五六七八九十]+、|              # 一、
        ^第[一二三四五六七八九十百千万零〇]+条|   # 第二十条等
        ^<[hp]\d?>第[一二三四五六七八九十]+条|   # <h4>第X条
        ^.{1,10}[:：]$                          # 冒号结尾短行（≤10字）
    ''', flags=re.VERBOSE)

    lines = data.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        is_short_colon = (line.endswith(':') or line.endswith('：')) and len(line) <= 10

        if pattern.search(line) or is_short_colon:
            if block:
                blocks.append(block.strip())
            block = line
        elif len(block) <= 4 or '  -' in line:
            block += ' ' + line
        else:
            blocks.append(block.strip())
            block = line

    if block:
        blocks.append(block.strip())

    # 回退策略：分块太少，按行拆并合并短句
    if len(blocks) <= 1:
        raw_lines = [line.strip() for line in data.split('\n') if line.strip()]
        blocks = []
        current = ''
        for line in raw_lines:
            if len(line) < 10:
                current += ' ' + line
            else:
                if current:
                    blocks.append(current.strip())
                current = line
        if current:
            blocks.append(current.strip())

    return blocks


def insurance_data_split(all_infos:list) -> list:
    '''保障责任免除分块'''
    sim_blocks = []
    # 对大块进行配对
    all_infos = [remove_blank_line(remove_html_tags(infos)) for infos in all_infos]

    for i in range(len(all_infos)):
        for j in range(i+1, len(all_infos)):
            b1, b2 = process(all_infos[i]), process(all_infos[j])
            if zh_same_string(b1, b2) or len(b1) <= 3 or len(b2) <= 3: continue
            sim_blocks.append((b1, b2))


    # 对于小块进行配对
    inputs = []
    for info in all_infos:
        blocks = __insurance_data_presplit(info)
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
                    if zh_same_string(bi, bj): continue
                    score = diff_similarity(bi, bj)
                    n_score = ngram_similarity(bi, bj)
                    if score > 0.75 or n_score > 0.8:
                        sim_blocks.append((bi, bj))

    return sim_blocks
