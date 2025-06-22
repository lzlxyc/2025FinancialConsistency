import logging
import os

def setup_logger(log_file='app.log'):
    if os.path.exists(log_file):
        os.remove(log_file)

    # 创建 logger 实例
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别

    # 防止重复添加 handler（避免重复日志）
    if logger.handlers:
        logger.handlers = []

    # 创建 formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 创建文件 handler 并设置级别为 DEBUG
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 2. 创建控制台 handler 并设置级别为 INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 将 handlers 添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


