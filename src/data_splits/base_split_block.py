import re

from src.data_splits.tools import (
    zh_number_same_string,
    diff_similarity,
    ngram_similarity
)



from tqdm import tqdm

def contains_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def remove_blank_line(text: str) -> str:
    lines = text.splitlines()
    chinese_lines = [line for line in lines if re.search(r'[\u4e00-\u9fff]', line)]
    return "\n".join(chinese_lines)

def remove_html_tags(text: str) -> str:
    return re.sub(r'<\s*\w+[^>]*>(.*?)<\s*/\s*\w+\s*>', r'\1', text, flags=re.DOTALL)

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
    eng2zh_map = []
    for line in tqdm(lines):
        sentences = re.split(r'(?<=[。；;])', line)
        sentences = [s.strip() for s in sentences if s.strip()]
        translated_sentences = []
        for sentence in sentences:
            eng = sentence
            for attempt in range(3):
                try:
                    eng = translate2ENG(sentence,aibox,rule).strip()
                    break
                except Exception as e:
                    print(f"翻译失败，第 {attempt + 1} 次尝试: {e}")
            eng2zh_map.append((eng,sentence))
            translated_sentences.append(eng)
        # print(line)
        # print(eng)
        text_eng = " ".join(translated_sentences)
        translated_lines.append(text_eng)
    text_eng = '\n'.join(translated_lines)

    chunks = chunker.chunk(text_eng)
    blocks = []
    for chunk in chunks:
        chunk_text = chunk.text.strip()
        # 收集 chunk 中涉及到的中文原文
        chunk_zh = []
        for eng_line, zh_line in eng2zh_map:
            if eng_line in chunk_text:
                chunk_zh.append(zh_line)
        # if len(chunk_zh) != 0:
        blocks.append(" ".join(chunk_zh))  # block 中是该chunk对应的中文内容

    def contains_chinese(text: str) -> bool:
        return re.search(r'[\u4e00-\u9fff]', text) is not None

    # 如果 blocks 是一个字符串列表
    if not any(contains_chinese(block) for block in blocks):
        for chunk in chunks:
            print(chunk.text.strip(), '+++++++++++++++++++++\n')
            print(eng2zh_map, '******************************\n')
    return blocks

def data_presplit_no_translate(data:str,chunker,rule:str) -> list:
    text = data
    if re.search(r'[\u4e00-\u9fff]', text) is None:
        return []
    clean_text = remove_blank_line(text)

    chunks = chunker.chunk(clean_text)
    blocks = []
    for chunk in chunks:
        chunk_text = chunk.text.strip()
        blocks.append(" ".join(chunk_text))  # block 中是该chunk对应的中文内容

    return blocks



def data_split(all_infos:list, chunker,rule):
    inputs = []
    for info in all_infos:
        blocks = data_presplit_no_translate(info,chunker,rule)
        # blocks = [s for s in blocks if len(s.split()) > 1 or len(s) >= 50]
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
                    if re.search(r'[\u4e00-\u9fff]', bi) and re.search(r'[\u4e00-\u9fff]', bj):
                        if zh_number_same_string(bi, bj): continue
                        score = diff_similarity(bi, bj)
                        n_score = ngram_similarity(bi, bj)
                        if score > 0.3 or n_score > 0.3:
                            sim_blocks.append((bi, bj))

    return sim_blocks

def data_no_split(all_infos:list) -> list:
    '''不分块 直接召回对比'''
    sim_blocks = []
    all_infos = [remove_blank_line(remove_html_tags(infos)) for infos in all_infos]

    for i in range((len(all_infos))):
        for j in range(i+1, len(all_infos)):
            if re.search(r'[\u4e00-\u9fff]', all_infos[i]) and re.search(r'[\u4e00-\u9fff]', all_infos[j]):
                sim_blocks.append((all_infos[i],all_infos[j]))
    return sim_blocks


if __name__  == '__main__':
    pass