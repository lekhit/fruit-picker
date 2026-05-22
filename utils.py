import os
import sys
import logging
import time
import traceback
import torch

def setup_logger(log_file="benchmark.log", verbose=True):
    """
    Sets up a robust logging configuration that streams to both stdout and a local log file.
    Includes timestamps, log levels, file names, and full traceback details for warnings/errors.
    """
    logger = logging.getLogger("AppleTrackerBenchmark")
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler for permanent debug logging
    try:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file {log_file} due to {e}. Logging to console only.")
        
    # Console handler for real-time visualization
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if not verbose else logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info("=== Logging System Initialized ===")
    logger.info(f"Diagnostics will be written to: {os.path.abspath(log_file)}")
    return logger

def get_vram_usage():
    """
    Returns current and peak VRAM usage in Megabytes (MB).
    Works on NVIDIA GPUs. If GPU is unavailable, returns (0.0, 0.0).
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        return {
            "allocated_mb": round(allocated, 2),
            "peak_mb": round(peak, 2),
            "reserved_mb": round(reserved, 2),
            "free_mb": round((torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)) - (torch.cuda.memory_allocated() / (1024 * 1024)), 2)
        }
    elif torch.backends.mps.is_available():
        # Apple Silicon MPS has no direct memory query API in PyTorch, but we can return basic placeholder
        return {
            "allocated_mb": 0.0,
            "peak_mb": 0.0,
            "reserved_mb": 0.0,
            "free_mb": 0.0,
            "note": "MPS Active"
        }
    else:
        return {
            "allocated_mb": 0.0,
            "peak_mb": 0.0,
            "reserved_mb": 0.0,
            "free_mb": 0.0,
            "note": "CPU Active"
        }

def log_exception(logger, msg, exception):
    """
    Convenience method to log an exception with its complete traceback.
    """
    tb = traceback.format_exc()
    logger.error(f"{msg}: {exception}\nTraceback:\n{tb}")

def print_gpu_diagnostics(logger):
    """
    Prints detailed GPU device details for debugging remote runs.
    """
    logger.info("--- GPU Diagnostics ---")
    cuda_avail = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        num_devices = torch.cuda.device_count()
        logger.info(f"CUDA Device Count: {num_devices}")
        for i in range(num_devices):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  Device {i}: {props.name}")
            logger.info(f"    Total Memory: {round(props.total_memory / (1024**3), 2)} GiB")
            logger.info(f"    Compute Capability: {props.major}.{props.minor}")
    else:
        mps_avail = torch.backends.mps.is_available()
        logger.info(f"MPS (Apple Silicon) Available: {mps_avail}")
    logger.info("-----------------------")
