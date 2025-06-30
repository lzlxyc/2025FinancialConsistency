import re

from src.data_splits.tools import (
    zh_same_string,
    diff_similarity,
    ngram_similarity
)

from tqdm import tqdm

def contains_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def translate2ENG(text,aibox,rule):
    translate2Eng_System = f'''
    你是一个专业的金融保险行业英文翻译专家，需要对下面的金融保险关于{rule}的条款准确翻译为英文并返回。你需要遵循以下规则：
    1.你的输出只有翻译后的英文，不要有任何其他输出，
    2.如果输入为''，直接返回null。
    '''
    prompt = f'''
    请对以下关于{rule}的条款进行准确翻译：{text}
    '''
    return aibox.chat(prompt=prompt, system=translate2Eng_System)

def data_presplit(data:str,aibox,chunker,rule:str) -> list:
    text = data
    if re.search(r'[\u4e00-\u9fff]', text) is None:
        return []
    clean_text = remove_blank_line(text)
    lines = clean_text.splitlines()  # 每行为一个中文条款句子

    # 遍历并翻译为英文
    translated_lines = []
    for line in tqdm(lines):
        eng = line
        for attempt in range(3):
            try:
                eng = translate2ENG(line,aibox,rule).strip()
                break
            except Exception as e:
                print(f"翻译失败，第 {attempt + 1} 次尝试: {e}")
        translated_lines.append(eng)
        # print(line)
        # print(eng)
    text_eng = "\n".join(translated_lines)
    chunks = chunker.chunk(text_eng)
    eng2zh_map = list(zip(translated_lines, lines))

    blocks = []
    for chunk in chunks:
        chunk_text = chunk.text.strip()
        # print(chunk_text)
        # 收集 chunk 中涉及到的中文原文
        chunk_zh = []
        for eng_line, zh_line in eng2zh_map:
            if eng_line in chunk_text:
                chunk_zh.append(zh_line)
        blocks.append("\n".join(chunk_zh))  # block 中是该chunk对应的中文内容
    return blocks
    #    print('==============================')
    # for i in blocks:
    #     print(i)
    #     print('----------------------')

def remove_blank_line(text: str) -> str:
    lines = text.splitlines()
    non_blank_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(non_blank_lines)

def data_split(all_infos:list,aibox,chunker,rule):
    inputs = []
    for info in all_infos:
        blocks = data_presplit(info,aibox,chunker,rule)
        blocks = [s for s in blocks if len(s.split()) > 1 or len(s) >= 50]
        blocks = list(dict.fromkeys(blocks))
        inputs.append(blocks)

    sim_blocks = []
    len_inputs = len(inputs)
    for i in range(len_inputs):
        for j in range(i+1, len_inputs):
            blocks_i = inputs[i]
            blocks_j = inputs[j]
            for bi in blocks_i:
                for bj in blocks_j:
                    if zh_same_string(bi, bj): continue
                    score = diff_similarity(bi, bj)
                    n_score = ngram_similarity(bi, bj)
                    if score > 0.75 or n_score > 0.8:
                        sim_blocks.append((bi, bj))

    return sim_blocks


if __name__  == '__main__':
    pass