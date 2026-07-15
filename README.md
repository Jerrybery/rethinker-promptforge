# Rethinker + EmbodiedPromptForge

A robotic manipulation research project that rethinks failure cases via vision-language reasoning and forges embodied prompts.

## Quick start

```bash
# Install the package in editable mode
pip install -e .

# Run unit tests
pytest tests/unit/test_config_loader.py
```

## Project layout

```
configs/
  models.yaml   # Model endpoints and parameters (vLLM, DINO, AnyGrasp, CloudCritic)
  robot.yaml    # Robot hardware configuration and safety limits
src/rethinker_promptforge/
  __init__.py
  config.py     # YAML config loader
```

## Optional robot dependencies

Install the RoboTwin simulator dependency once the fork URL/version is known:

```bash
pip install -e ".[robot]"
```
