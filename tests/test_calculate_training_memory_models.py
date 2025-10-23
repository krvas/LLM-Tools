import json
import random
import traceback
from llm_tools.utils.memory_utils import calculate_training_memory
from llm_tools.config.memory_config import OPTIMIZERS, load_predefined_models

def main():
    models = load_predefined_models()  # returns dict[str, Model]
    for name, model in models.items():
        optimizer = random.choice(list(OPTIMIZERS.keys()))
        print(f"\n=== {name} with {optimizer} ===")
        try:
            batch_size = 1
            sequence_length = min(getattr(model, "max_sequence_length", 2048), 2048)
            trainable_parameters = 100
            # New signature expects a Model instance as the first arg
            result = calculate_training_memory(
                model,
                batch_size,
                sequence_length,
                optimizer,
                trainable_parameters,
            )

            print(json.dumps(result, indent=2, sort_keys=True))
        except Exception:
            print("Error while processing", name)
            traceback.print_exc()

if __name__ == "__main__":
    main()