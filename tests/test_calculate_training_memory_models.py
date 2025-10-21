import json
from pathlib import Path
import llm_tools
import traceback
from llm_tools.utils.memory_utils import calculate_training_memory
from llm_tools.config.memory_config import OPTIMIZERS

MODELS_DIR = Path(llm_tools.__file__).parent / "models"
MODEL_FILES = sorted(MODELS_DIR.glob("*.json"))

def main():
    for model_path in MODEL_FILES:
        print(f"\n=== {model_path.name} ===")
        try:
            data = json.loads(model_path.read_text())

            model_size = data.get("model_size")
            precision = data.get("torch_dtype", data.get("precision", "float32"))
            batch_size = 1
            sequence_length = min(data.get("max_position_embeddings", 2048), 2048)
            hidden_size = data.get("hidden_size")
            num_hidden_layers = data.get("num_hidden_layers")
            num_attention_heads = data.get("num_attention_heads")
            optimizer = next(iter(OPTIMIZERS.keys()))
            trainable_parameters = 1.0
            mlp_layer_size = data.get("intermediate_size", (hidden_size or 0) * 4)

            result = calculate_training_memory(
                model_size,
                precision,
                batch_size,
                sequence_length,
                hidden_size,
                num_hidden_layers,
                num_attention_heads,
                optimizer,
                trainable_parameters,
                mlp_layer_size,
            )

            # pretty-print the result
            print(json.dumps(result, indent=2, sort_keys=True))
        except Exception:
            print("Error while processing", model_path.name)
            traceback.print_exc()

if __name__ == "__main__":
    main()