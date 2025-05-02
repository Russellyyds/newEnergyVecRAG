import logging
import sys
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime


def setup_logger(name: str = None) -> logging.Logger:
    """
    Configure and return a logger.

    Args:
        name: Optional name for the logger

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f"app_{datetime.now().strftime('%Y%m%d')}.log"
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger