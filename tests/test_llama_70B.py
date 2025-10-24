import json
from llm_tools.config.memory_config import load_predefined_models, OPTIMIZERS
from llm_tools.utils.memory_utils import calculate_training_memory

def main():
    models = load_predefined_models()
    optimizer = "Quantized AdamW"
    result = calculate_training_memory(
        models['Meta-Llama-3-70B-Instruct'],
        batch_size=1,
        sequence_length=2048,
        optimizer=optimizer,
        trainable_parameters=100,
    )
    print(models['Meta-Llama-3-70B-Instruct'])
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()