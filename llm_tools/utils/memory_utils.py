from functools import lru_cache
from llm_tools.config.memory_config import (
    DATA_TYPES,
    PARAMETERS,
    DATA_TYPE_SIZES,
    OPTIMIZERS,
    Model
)

# lightweight drop-in cache replacement for streamlit.cache_data
cache_data = lru_cache(maxsize=None)


# ----------------- Memory Functions ----------------- #
@cache_data
def get_memory(*args, in_int=False):
    """Convert total memory from bytes to human-readable format."""
    total = 0
    warning = False
    for arg in args:
        if arg > 0:
            total += arg
        else:
            warning = True
    
    if in_int:
        return total

    # Convert bytes to human-readable format
    if total == 0:
        result = ""
    elif total < 1024:
        result = f"{total} Bytes"
    elif total < 1024**2:
        result = f"{total / 1024:.2f} KB"
    elif total < 1024**3:
        result = f"{total / (1024**2):.2f} MB"
    elif total < 1024**4:
        result = f"{total / (1024**3):.2f} GB"
    else:
        result = f"{total / (1024**4):.2f} TB"
    result += " * " if warning else ""
    return result


@cache_data
def get_model_weights(model_size, precision):
    """Calculate the memory required for model weights."""
    try:
        return model_size * DATA_TYPE_SIZES[precision] * (10**9)
    except:
        return 0


@cache_data
def get_kv_cache(
    precision, batch_size, sequence_length, hidden_size, num_hidden_layers
):
    """Calculate the memory required for key-value cache."""
    try:
        return (
            2
            * batch_size
            * sequence_length
            * num_hidden_layers
            * hidden_size
            * DATA_TYPE_SIZES[precision]
        )
    except:
        return 0


@cache_data
def get_activation_memory(
    batch_size: int, sequence_length: int, model: Model
):
    """
    Calculate the memory required for activations. It references this paper:
    https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf

    Let 
    s = sequence length
    b = batch size
    h = hidden size
    a = attention heads
    L = number of transformer layers

    Then, the number of activations for each step is:
    Attention:
    Q, K, V: sbhL
    QK^T: 2sbhaL
    Softmax: as^2bL
    Softmax dropout: as^2bL
    Attention over Values: as^2bL + sbhL

    MLP:
    I'm assuming this structure of the MLP:
    Linear layer, Activation function, Linear layer
    With the first linear layer transforming h dimensions into some higher
    dimension m and the second linear layer projecting it back.
    Then, the memory required is:
    1st Linear layer: sbhL
    Activation: sbmL
    2nd Linear layer: sbmL

    Total = (3 + 2a)sbhL + 3as^2bL + 2msbL
    """
    
    try:
        return (
            (
                (3 + 2 * model.num_attention_heads)
                * sequence_length
                * batch_size
                * model.hidden_size
                * model.num_hidden_layers
            ) + (
                3
                * model.num_attention_heads
                * sequence_length ** 2
                * batch_size
                * model.num_hidden_layers
            ) + (
                2
                * model.mlp_layer_size
                * sequence_length
                * batch_size
                * model.num_hidden_layers
            )
        ) * DATA_TYPE_SIZES[model.precision]
    except:
        return 0


@cache_data
def get_optimizer_memory(model_size, optimizer):
    """Calculate the memory required for optimizer."""
    try:
        return OPTIMIZERS[optimizer] * model_size * (10**9)
    except:
        return 0


@cache_data
def get_gradient_memory(model_size, precision):
    """Calculate the memory required for gradients."""
    precision = "float32"
    try:
        return DATA_TYPE_SIZES[precision] * model_size * (10**9)
    except:
        return 0


@cache_data
def calculate_inference_memory(
    model: Model,
    batch_size: int,
    sequence_length: int,
    in_int: bool = False,
):
    """Calculate the total memory required for inference using a Model object."""
    model_weights = get_model_weights(model.model_size, model.precision)
    kv_cache = get_kv_cache(
        model.precision,
        batch_size,
        sequence_length,
        model.hidden_size,
        model.num_hidden_layers,
    )
    activation_memory = get_activation_memory(batch_size, sequence_length, model) / \
        model.num_hidden_layers  # Only one layer's activations are stored at a time
    return {
        "model_weights": get_memory(model_weights, in_int=in_int),
        "kv_cache": get_memory(kv_cache, in_int=in_int),
        "activation_memory": get_memory(activation_memory, in_int=in_int),
        "inference_memory": get_memory(
            model_weights, kv_cache, activation_memory, in_int=in_int
        ),
    }

@cache_data
def calculate_training_memory(
    model: Model,
    batch_size: int,
    sequence_length: int,
    optimizer: str,
    trainable_parameters: int,
    in_int: bool = False
):
    """Calculate the total memory required for training."""
    model_weights = get_model_weights(model.model_size, model.precision)
    activation_memory = get_activation_memory(
        batch_size, sequence_length, model
    )
    optimizer_memory = (
        get_optimizer_memory(model.model_size, optimizer) * trainable_parameters / 100
    )
    gradients_memory = (
        get_gradient_memory(model.model_size, model.precision) * trainable_parameters / 100
    )

    return {
        "model_weights": get_memory(model_weights, in_int=in_int),
        "activation_memory": get_memory(activation_memory, in_int=in_int),
        "optimizer_memory": get_memory(optimizer_memory, in_int=in_int),
        "gradients_memory": get_memory(gradients_memory, in_int=in_int),
        "training_memory": get_memory(
            model_weights,
            activation_memory,
            optimizer_memory,
            gradients_memory,
            in_int=in_int,
        ),
    }
