import regex as re
from src.data_splits.base_split_block import remove_blank_line

from src.data_splits.tools import (
    zh_same_string,
    diff_similarity,
    keep_only_chinese_strict,
    ngram_similarity
)

from src.block_text_comparison import logger
import time
'''
保障责任切分逻辑：
'''
all_pattern_list = []

def process(text: str) -> str:
    return re.sub(r'\n+', '\n', text.replace('""', ''))

def remove_html_tags(text: str) -> str:
    return re.sub(r'<\s*\w+[^>]*>(.*?)<\s*/\s*\w+\s*>', r'\1', text, flags=re.DOTALL)


def clean_pattern(p: str) -> str:
    # 转义中文括号
    p = p.replace('（', '[（]').replace('）', '[）]')

    # 转义加粗符号 **
    p = p.replace('**', r'\*\*')

    # 替换捕获组为非捕获组
    p = re.sub(r'\((?!\?)', r'(?:', p)

    # 去除尾部或头部多余的 `|`
    p = re.sub(r'^\|+', '', p)
    p = re.sub(r'\|+$', '', p)

    return p

def make_regular(text,aibox):
    System = f'''
    你是一个专业的正则表达式生成专家，任务是从保险条款或说明性文本中提取**段落起始规则**，以便进行换行分段处理。你必须遵循以下要求：

    1. 你的目标是生成一个或多个正则表达式，用于识别“每个段落的起始位置”；
    2. 请重点参考以下常见段落起始样式生成规则：
       - Markdown 标题：如 `## 第五条`、`### 1.3 投保年龄`
       - 中文条款编号：如 `第十五条`、`（一）`、`1）`、`1、`、`一、` 等
       - 特殊符号开头：如 `◆职业要求`、`●注意事项`
       - 加粗内容标题：如 `** 投保人条件 **`
       - 中文括号标题：如 `【风险承受能力声明】`
       - 数字编号开头：如 `1.3 投保年龄`、`1.`、`1 ）`
       - HTML 格式标题：如 `<h4>第一条`
       - 冒号结尾的短句（通常为小节标题）：如 `赔付条件：`
    3. 每条正则应以“换行符或段首”为起点，匹配每一个新段落的可能模式；
    4. 正则表达式中禁止使用不定长的 look-behind（如 `(?<=\n|\A)`），请使用非捕获组 `(?:^|\n)` 替代；
    5. 你的输出必须为标准 XML 格式，形如：<reg>正则表达式</reg>；
    6. 如果输入为空字符串，直接输出：<reg>null</reg>；
    7. 不要输出除 `<reg>...</reg>` 外的任何解释或内容。

    请基于上述要求，结合输入文本，精准生成适合的分段正则表达式。
    '''

    prompt = f'''
    请根据以下条款内容，生成正则表达式以实现分段：{text}
    '''
    pre_pattern = aibox.chat(prompt=prompt, system=System)

    # 从返回文本中提取所有正则表达式
    pattern_list = re.findall(r'<reg>(.*?)</reg>', pre_pattern, re.DOTALL)
    pattern_list = [clean_pattern(p) for p in pattern_list]
    logger.info('------------------------------------')
    logger.info(text)
    logger.info(pre_pattern)
    logger.info('====================================')

    return pattern_list

def __indemnity_responsibility_data_presplit(data: str, aibox) -> list:
    global all_pattern_list  # 声明使用全局变量
    blocks = []
    block = ''

    if not data.strip():
        return []

    # 调用 make_regular 并扩展 global all_pattern_list
    for i in range(5):
        try:
            new_patterns = make_regular(data, aibox)
            break
        except Exception as e:
            if i < 4:
                time.sleep(5)
            else:
                raise

    # 编译测试，剔除非法的正则表达式
    valid_new_patterns = []
    for p in new_patterns:
        try:
            re.compile(p)
            valid_new_patterns.append(p)
        except re.error as e:
            logger.warning(f"[正则编译失败] pattern: {p}\n错误信息: {e}")

    # 只加入通过编译的 pattern
    all_pattern_list.extend(p for p in valid_new_patterns if p not in all_pattern_list)

    # 合并所有正则表达式为一个整体 pattern
    if all_pattern_list:
        combined_pattern = '|'.join(f'(?:{p})' for p in all_pattern_list)
        try:
            pattern = re.compile(combined_pattern)
        except re.error as e:
            logger.error(f"[组合正则编译失败] pattern: {combined_pattern}\n错误信息: {e}")
            pattern = re.compile(r'^\s*$')  # fallback：无效匹配
    else:
        pattern = re.compile(r'^\s*$')  # fallback：无效匹配


    # 分行处理
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

    # 回退：按行合并短句
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

def indemnity_responsibility_data_split(all_infos:list,aibox) -> list:
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
        blocks = __indemnity_responsibility_data_presplit(info,aibox)
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
