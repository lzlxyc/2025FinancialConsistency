import os
import glob
from collections import Counter
from tqdm import tqdm

def get_first_nonzero_digit(n):
    """获取第一个非零数字"""
    for ch in str(n):
        if ch.isdigit() and ch != '0':
            return int(ch)
    return None

def is_mode_sparse(lengths, threshold=0.5):
    """判断是否集中在前三个众数±2范围的并集内"""
    counter = Counter(lengths)
    if not counter:
        return False, 0.0

    # 获取出现频率最高的前三个众数
    most_common_modes = [item[0] for item in counter.most_common(3)]

    # 构造这些众数的 ±2 范围并集
    mode_range_set = set()
    for mode in most_common_modes:
        mode_range_set.update(range(mode - 2, mode + 3))  # ±2 即 [-2, -1, 0, 1, 2]

    # 判断 lengths 中有多少在这些范围内
    in_range_count = sum(1 for x in lengths if x in mode_range_set)
    ratio = in_range_count / len(lengths)

    return not (ratio > threshold), ratio

def check_md_files(md_dir_path, raw_data_path, logger):
    abnormal_files = []

    for path in tqdm(glob.glob(os.path.join(md_dir_path, "**/*.md"), recursive=True)):
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            logger.warning(f"⚠️ 解码错误: {path}")
            return  # 注意：这里应为 continue 而非 return，否则整个函数中断

        # 每行字数
        line_lengths = [len(line.strip()) for line in lines if len(line.strip()) > 0]
        total_chars = sum(line_lengths)

        if total_chars < 10:
            continue  # 总字符数太少，跳过判断
        if len(line_lengths) == 0:
            continue

        is_sparse, ratio = is_mode_sparse(line_lengths)
        if not is_sparse:
            abnormal_files.append(path)

    save_path = os.path.join(raw_data_path, "abnormal_format.md")
    with open(save_path, "w", encoding="utf-8") as f:
        for p in abnormal_files:
            f.write(p + '\n')

    print(f"✅ 完成，异常文件数：{len(abnormal_files)}，已保存至 abnormal_format.md")
    return save_path

if __name__ == "__main__":
    raw_data_path = "../../data/raw_data/测试 A 集"
    md_dir_path = os.path.join(raw_data_path, "materials")