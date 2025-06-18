import os
import logging

from format_check import check_md_files
from format_alignment import clean_data
from openai import OpenAI

model2key = {
    'qwen2.5-72b-instruct':'sk-026d537f247f4cf6a293652b2a803ff3',
    'qwen2.5-1.5b-instruct':'sk-2f73f0e455d54bb9a756757df0c7ea2d'
}

def setup_logger(log_path):
    logger = logging.getLogger("clean_data_logger")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
if __name__ == '__main__':
    model_name = 'qwen2.5-72b-instruct'
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        # https://bailian.console.aliyun.com/?tab=model#/api-key
        api_key=model2key[model_name],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    raw_data_path = os.path.join('../../', 'data', 'raw_data', "测试 A 集")
    clean_data_path = os.path.join('../../', 'data', 'clean_data'+model_name, "测试 A 集")
    os.makedirs(clean_data_path, exist_ok=True)
    md_dir_path = os.path.join(raw_data_path, "materials")
    log_file_path = os.path.join(clean_data_path, "process.log")


    logger = setup_logger(log_file_path)


    abnormal_format_path = check_md_files(md_dir_path = md_dir_path,
                                          raw_data_path = raw_data_path,
                                          logger = logger)

    clean_data(abnormal_format_path=abnormal_format_path,
               raw_data_path=raw_data_path,
               clean_data_path=clean_data_path,
               log_path=log_file_path,
               client = client,
               model_name = model_name,
               logger=logger)