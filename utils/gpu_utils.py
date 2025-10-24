"""
GPU utilities for H100/5090 optimization.
"""

import torch
from typing import Optional, Dict, Any
from utils.logger import logger


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Get the best available device (CUDA GPU or CPU).

    Args:
        force_cpu: Force CPU usage even if GPU available

    Returns:
        torch.device instance
    """
    if force_cpu:
        logger.info("Forcing CPU usage")
        return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

        # Print memory info
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i} Memory: {total_memory:.2f} GB")

        return device
    else:
        logger.warning("CUDA not available, using CPU")
        return torch.device("cpu")


def setup_mixed_precision() -> bool:
    """
    Check if mixed precision (FP16) training is supported.

    Returns:
        True if mixed precision is available
    """
    if torch.cuda.is_available():
        # Check if GPU supports FP16
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] >= 7:  # Volta and newer (includes H100, 5090)
            logger.info("Mixed precision (FP16) training available")
            return True
        else:
            logger.warning(f"GPU compute capability {compute_capability} does not support efficient FP16")
            return False
    else:
        logger.warning("Mixed precision requires CUDA")
        return False


def get_optimal_batch_size(
    sequence_length: int,
    feature_dim: int,
    base_batch_size: int = 256,
    max_memory_gb: float = 70.0
) -> int:
    """
    Estimate optimal batch size based on available GPU memory.

    Args:
        sequence_length: Length of input sequences
        feature_dim: Number of features per timestep
        base_batch_size: Starting batch size
        max_memory_gb: Maximum GPU memory to use (GB)

    Returns:
        Recommended batch size
    """
    if not torch.cuda.is_available():
        logger.warning("No GPU available, using small batch size")
        return 32

    # Get available memory
    device = torch.device("cuda")
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
    available_memory = total_memory * 0.8  # Leave 20% buffer

    # Estimate memory per sample (rough approximation)
    # FP32: 4 bytes, FP16: 2 bytes
    bytes_per_value = 2  # Assuming mixed precision
    memory_per_sample = (sequence_length * feature_dim * bytes_per_value) / 1e9

    # Calculate max batch size
    max_batch_size = int(min(max_memory_gb, available_memory) / memory_per_sample)

    # Round down to nearest power of 2
    optimal_batch_size = 2 ** int(torch.log2(torch.tensor(max_batch_size)))

    # Clamp to reasonable range
    optimal_batch_size = max(16, min(optimal_batch_size, 2048))

    logger.info(f"Available GPU memory: {available_memory:.2f} GB")
    logger.info(f"Est. memory per sample: {memory_per_sample * 1000:.2f} MB")
    logger.info(f"Recommended batch size: {optimal_batch_size}")

    return optimal_batch_size


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU cache cleared")


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with memory stats
    """
    if not torch.cuda.is_available():
        return {"available": False}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    return {
        "available": True,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "total_gb": total,
        "free_gb": total - allocated,
        "utilization": allocated / total
    }


def print_gpu_memory():
    """Print current GPU memory usage."""
    info = get_gpu_memory_info()
    if info["available"]:
        logger.info(f"GPU Memory: {info['allocated_gb']:.2f}/{info['total_gb']:.2f} GB ({info['utilization']:.1%})")
    else:
        logger.warning("No GPU available")


if __name__ == "__main__":
    # Test GPU utilities
    device = get_device()
    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"\nMixed precision available: {setup_mixed_precision()}")

        optimal_batch = get_optimal_batch_size(
            sequence_length=1000,
            feature_dim=500,
            base_batch_size=256
        )
        print(f"\nOptimal batch size (for 1000x500 sequences): {optimal_batch}")

        print_gpu_memory()
