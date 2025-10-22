import re
import os
import logging
from tqdm import tqdm



def process_failed_paths(log_file_path, client, model_name, get_chunk_list, do_align,logger):
    failed_paths = []
    pattern = re.compile(r"❌ do_align 失败跳过: (.*?) \| 错误信息:")

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                failed_paths.append(match.group(1))


    for path in failed_paths:
        rel_path = path  # 原始路径，保留用于日志记录
        norm_path = os.path.normpath(path)
        _type = norm_path.split(os.sep)[-2]  # 倒数第二层目录作为type
        logger.info(f"{path} ***abnormal***")

        content = ""
        memory = ""

        # 读取文件内容
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"❌ 无法读取文件: {rel_path} | 错误信息: {e}")
            continue

        for chunk_lines in tqdm(get_chunk_list(lines, chunk_size=100, overlap=10)):
            try:
                res = do_align(client, model_name, memory, chunk_lines, _type)
            except Exception as e:
                logger.error(f"❌ do_align 失败跳过 chunk: {rel_path} | 错误信息: {e}")
                res = None

            memory = chunk_lines
            content += res if res else ''.join(chunk_lines)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

if __name__ =='__main__':
    log_path = 'your_log_file.log'
    failed = extract_failed_paths(log_path)
    print(failed)
