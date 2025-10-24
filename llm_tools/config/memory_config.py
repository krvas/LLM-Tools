import os
import json
from functools import lru_cache

# Data type sizes in bytes
DATA_TYPE_SIZES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
}

# Optimizer memory multipliers
OPTIMIZERS = {
    "AdamW": 8,
    "Quantized AdamW": 2,
    "SGD": 4,
}

# Available data types
DATA_TYPES = list(DATA_TYPE_SIZES.keys())

# Model parameters mapping
PARAMETERS = {
    "model_size": "model_size",
    "precision": "torch_dtype",
    "hidden_size": "hidden_size",
    "num_hidden_layers": "num_hidden_layers",
    "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_key_value_heads",
    "mlp_layer_size": "intermediate_size",
    "max_sequence_length": "max_position_embeddings",
}


class Model:
    def __init__(self, model_size, torch_dtype, hidden_size,
                 num_hidden_layers, num_attention_heads,
                 intermediate_size, max_position_embeddings, **kwargs):
        self.model_size = model_size
        self.precision = torch_dtype
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_layer_size = intermediate_size
        self.max_sequence_length = max_position_embeddings
    
    def __repr__(self):
        return json.dumps({
            "1. model_size": self.model_size,
            "2. precision": self.precision,
            "3. hidden_size": self.hidden_size,
            "4. num_hidden_layers": self.num_hidden_layers,
            "5. num_attention_heads": self.num_attention_heads,
            "6. mlp_layer_size": self.mlp_layer_size,
            "7. max_sequence_length": self.max_sequence_length,
        }, indent=2, sort_keys=True)

@lru_cache(maxsize=None)
def load_predefined_models() -> dict:
    """Load model configurations from the 'predefined_models' folder."""
    models = {}
    for model_file in os.listdir(os.path.join("llm_tools", "models")):
        if model_file.endswith(".json"):
            with open(os.path.join("llm_tools", "models", model_file), "r") as f:
                try:
                    models[model_file[:-5]] = Model(**json.load(f))
                except Exception as e:
                    print(f"Error loading model from {model_file}: {e}")
    return models
