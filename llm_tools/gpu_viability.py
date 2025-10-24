import llm_tools

from llm_tools.config.memory_config import Model
from llm_tools.utils.memory_utils import calculate_training_memory
from llm_tools.utils.hardware_utils import gpus_available

def is_gpu_viable(
    model: Model,
    batch_size: int,
    sequence_length: int,
    optimizer: str,
    trainable_parameters: float,
    gpu_name: str,
    num_gpus: int = 1,
    safety_margin: float = 0.9,
) -> bool:
    """Check if the specified GPU can handle training the model with given parameters."""

    if gpu_name not in gpus_available:
        raise ValueError(f"GPU '{gpu_name}' not found in available GPUs.")

    gpu_memory_GB = gpus_available[gpu_name]["memory_GB"]

    training_memory_bytes = calculate_training_memory(
        model,
        batch_size,
        sequence_length,
        optimizer,
        trainable_parameters,
        in_int=True,
    )["training_memory"]

    training_memory_gb = training_memory_bytes / (1024 ** 3)

    return training_memory_gb < gpu_memory_GB * safety_margin * num_gpus