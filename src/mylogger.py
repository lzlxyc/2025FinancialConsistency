import logging
import os

def setup_logger(log_file='app.log', overwrite=False):
    # 如果设置了 overwrite，则删除旧日志文件
    if overwrite and os.path.exists(log_file):
        os.remove(log_file)

    # 创建 logger 实例
    logger = logging.getLogger(log_file)  # 用 log_file 做唯一标识，避免多个 logger 混淆
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers = []

    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 🔑 使用追加模式 'a'
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


