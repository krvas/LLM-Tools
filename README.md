# LLM-Tools — Quick README

A compact set of utilities for estimating model memory requirements and assessing GPU suitability for inference/training workloads.

## References
- Forked from https://github.com/manuelescobar-dev/LLM-Tools

## Quick setup
- Requires Python 3.x (available in the dev container).
- Requires Docker: the repository includes a .devcontainer/ configuration — use the devcontainer (e.g., VS Code Remote - Containers or the equivalent) to build and run the workspace with the provided environment.
- No additional install step needed to run the bundled scripts inside the dev container.

## Running the test scripts
- Run a single test script:
  - `python3 tests/some_test.py`
- Run all test scripts from a Unix shell:
  - `for f in tests/*.py; do python3 "$f"; done`

## What to look at
- memory_utils.py
  - `calculate_training_memory()`: estimates RAM required for training given model parameters and training config.
  - `calculate_inference_memory()`: estimates memory needed for inference under different precisions and batch sizes.
- gpu_viability.py
  - `is_gpu_viable()`: evaluates whether a given GPU (or combination) is suitable for the model and mode, returning a pass/fail and summary details.

These utilities are small and intended to be easy to call from scripts or the command line.

## Contributing
- Open issues or pull requests for improvements or bug fixes.
- Keep changes focused and include tests for new behaviour.