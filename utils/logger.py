import logging
import os
from datetime import datetime

def get_logger(log_dir, log_name):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # 避免重复输出
    if not logger.handlers:
        # 文件 handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)

        # 控制台 handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 输出格式
        formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger