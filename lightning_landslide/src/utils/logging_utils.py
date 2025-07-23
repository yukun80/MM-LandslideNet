# =============================================================================
# src/utils/logging_utils.py - 日志配置和工具
# =============================================================================

"""
日志配置和工具函数

提供统一的日志配置和相关工具函数，确保整个项目的日志输出一致性。
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
import colorlog


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    use_colors: bool = True,
) -> None:
    """
    设置项目的日志配置

    Args:
        level: 日志级别
        log_file: 日志文件路径（可选）
        format_string: 自定义格式字符串（可选）
        use_colors: 是否在控制台使用彩色日志
    """
    # 清除现有的handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 设置根日志级别
    root_logger.setLevel(level)

    # 默认格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_colors:
        try:
            # 彩色日志格式
            color_format = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s%(reset)s - %(message)s"
            formatter = colorlog.ColoredFormatter(
                color_format,
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        except ImportError:
            # 如果colorlog不可用，使用标准格式
            formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    else:
        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件handler（如果指定了文件路径）
    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level)

        # 文件日志不使用颜色
        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Logging to file: {log_file_path}")

    # 设置第三方库的日志级别
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info("Logging setup completed")


def get_project_logger(name: str) -> logging.Logger:
    """
    获取项目专用的logger

    Args:
        name: logger名称

    Returns:
        配置好的logger
    """
    return logging.getLogger(f"landslide_detection.{name}")


class TqdmLoggingHandler(logging.Handler):
    """
    与tqdm兼容的日志handler

    当使用tqdm进度条时，普通的日志输出可能会干扰进度条显示。
    这个handler确保日志输出与tqdm协调工作。
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            from tqdm import tqdm

            msg = self.format(record)
            tqdm.write(msg)
        except ImportError:
            # 如果tqdm不可用，使用标准输出
            print(self.format(record))

