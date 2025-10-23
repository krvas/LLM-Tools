import traceback
from pathlib import Path
import sys
import random

import llm_tools
from llm_tools.gpu_viability import is_gpu_viable
from llm_tools.utils.hardware_utils import gpus_available
from llm_tools.utils.memory_utils import calculate_training_memory
from llm_tools.config.memory_config import OPTIMIZERS, load_predefined_models

# Ensure workspace root is on sys.path when running outside the devcontainer env
ROOT = Path(llm_tools.__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    optimizer = next(iter(OPTIMIZERS.keys()))
    print("model,gpu,viable,training_memory_GB,notes")

    models = load_predefined_models()
    for name, model in models.items():
        try:
            batch_size = 1
            # prefer model.max_sequence_length, fall back to 2048
            sequence_length = min(getattr(model, "max_sequence_length", getattr(model, "max_position_embeddings", 2048)), 2048)
            trainable_parameters = 1.0

            for gpu_name in gpus_available.keys():
                note = ""
                try:
                    mem_bytes = calculate_training_memory(
                        model,
                        batch_size,
                        sequence_length,
                        optimizer,
                        trainable_parameters,
                        in_int=True,
                    )["training_memory"]
                    mem_gb = round(mem_bytes / (1024 ** 3),1)
                    num_gpus = random.choice([1,2,4,8])

                    viable = is_gpu_viable(
                        model,
                        batch_size,
                        sequence_length,
                        optimizer,
                        trainable_parameters,
                        gpu_name,
                        num_gpus=num_gpus,
                        safety_margin=0.9,
                    )
                except Exception as e:
                    viable = False
                    mem_gb = "error"
                    note = f"calculation error: {e}"

                print(f"{name},{num_gpus}x {gpu_name},{viable},{mem_gb},{note}")
        except Exception as e:
            print(f"{name},,False,,failed to process model: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()