import os
import pandas as pd
import glob, json, re
from tqdm import tqdm

def get_processed_paths(log_path):
    """
    读取 log 文件中所有处理过的路径（不区分 normal / abnormal），返回：
    - set: 所有路径
    - str: 最后一条路径
    """
    if not os.path.exists(log_path):
        return set(), None

    processed_paths = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 只处理包含路径的日志行
            if " - INFO - " in line:
                try:
                    log_parts = line.strip().split(" - INFO - ")
                    if len(log_parts) < 2:
                        continue
                    path = log_parts[1].split(" ***")[0].strip()
                    processed_paths.append(path)
                except Exception:
                    continue

    last_path = processed_paths[-1] if processed_paths else None
    return processed_paths, last_path



def get_chunk_list(lst, chunk_size=200, overlap=20):
    """Yield successive chunks with overlap."""
    step = chunk_size - overlap
    for i in range(0, len(lst), step):
        yield lst[i:i + chunk_size]

def do_align(client,model_name,memory,lines,_type):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system",
             "content": f"你是一个专业的保险行业的信息处理专家。你要做的事情要求如下："
                        f"1.user提供的的content中可能会有由于转换导致的换行或者排版错误，你需要理解内容后将应该在一个条款中的内容调整到一行。"
                        f"2.用户会在type词条告诉你这个是保险项目中的什么类型资料，你需要集基于资料的类型整理1中的内容"
                        f"3.如果该内容内无实质性意义内容，例如只有一个null或者几个数字等，则直接返回none"
                        f"4.由于每次输入都会有与上一次部分的重复内容，你需要结合上一次的memory，不输出重复内容。"
                        # f"4.如果本次有分块导致的残缺内容，不处理残缺内容行"
                        f"5.你的输出只包含原文调整格式后的条款，如果该条款原有序号则保留该序号,不要有其他输出。"},
            {
                "role": "user",
                "type": _type,
                "memory":memory,
                "content": "".join(lines),
            },
        ],
        extra_body={"enable_thinking": False},
    )
    return completion.choices[0].message.content

def clean_data(abnormal_format_path, raw_data_path, clean_data_path, log_path, client, model_name, logger):
    # 获取已处理路径列表 + 最后一条路径
    processed_paths, last_path = get_processed_paths(log_path)

    # 获取异常路径列表
    if os.path.exists(abnormal_format_path):
        with open(abnormal_format_path, 'r', encoding='utf-8') as f:
            abnormal_paths = set(line.strip() for line in f if line.strip())
    else:
        abnormal_paths = set()

    # 主逻辑开始
    jsonl_path = os.path.join(raw_data_path, "data.jsonl")
    for row in tqdm(pd.read_json(jsonl_path, lines=True).iloc[:].iterrows()):
        material_id = row[1].material_id
        for path in glob.glob(os.path.join(raw_data_path, "materials", material_id, "*", "*")):
            # 相对路径用作 log 中唯一标识
            rel_path = os.path.relpath(path, raw_data_path)
            if path in processed_paths and path != last_path:
                continue
            # 是否跳过（根据 log 和 last_path）


            with open(path, encoding='utf-8') as f:
                lines = f.readlines()

            target_path = os.path.join(clean_data_path, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            if path in abnormal_paths:
                _type = os.path.normpath(path).split(os.sep)[-2]
                logger.info(f"{path} ***abnormal***")
                content = ""
                memory = ''
                try:
                    for chunk_lines in get_chunk_list(lines):
                        res = do_align(client, model_name, memory, chunk_lines, _type)
                        memory = chunk_lines
                        if res:
                            content += res
                except Exception as e:
                    logger.error(f"❌ do_align 失败跳过: {rel_path} | 错误信息: {e}")
                    continue

                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                logger.info(f"{path}")
                with open(target_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)


if __name__ =='__main__':
    raw_data_path = os.path.join('../../','data','raw_data',"测试 A 集",)  # 替换为你的目录路径 05 clause 03notice
    clean_data_path = os.path.join('../../','data', 'clean_data', "测试 A 集", )
    # 遍历md_dir_path下的所有文件
    # 1 统计每一行有多少个字 然后将字数存下  然后取第一个非0的数字 统计数字0-9的频率统计一下判断该文件是否符合Benford 定律 主要判断1的占比是否在25-35%之间
    # 如果符合就跳过
    # 如果不符合，记录下改文件路径到os.path.join(raw_data_path,'abnormal_format.md')
    # sk-2f73f0e455d54bb9a756757df0c7ea2d
    pass