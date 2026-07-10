import json
import logging
import os
import sys
from pathlib import Path

CONFIG_FILE = "app_path.json"

def get_logger(name: str, log_pseudo: str = None, level: int = logging.DEBUG) -> logging.Logger:
    """
    Create a configured logger with console output and optional file output.
    
    Parameters:
        name (str): Logger name.
        log_pseudo (str, optional): Either a config key (looked up in app_path.json) or a direct file path.
                                    If it looks like a path (contains / or \ or ends in .log), it's treated as a path.
                                    Otherwise, it's treated as a config key.
        level (int): Logging level applied to the logger and its handlers.
    
    Returns:
        logger (logging.Logger): The configured logger.
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:  # avoid duplicate handlers on re-import
        return logger
    
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_pseudo:
        # Determine if log_pseudo is a config key or a direct path
        is_path = "/" in log_pseudo or "\\" in log_pseudo or log_pseudo.endswith(".log")
        
        if is_path:
            # Treat as direct file path
            log_file = log_pseudo
        else:
            # Treat as config key
            config = load_config()
            log_file = config.get(log_pseudo)
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

def load_config():
    """Load default values from app_path.json if it exists."""
    if not os.path.exists(CONFIG_FILE):
        return {}
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        return config
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load config from {CONFIG_FILE}: {e}")
        return {}

def save_config(config):
    """Save configuration to app_path.json."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {CONFIG_FILE}")
    except IOError as e:
        print(f"Warning: Could not save config to {CONFIG_FILE}: {e}")