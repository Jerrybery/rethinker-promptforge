# User Prompt: Planner Agent v0

## Rethinker semantic analysis

{{rethinker_output}}

## Available DINO labels

{{dino_labels}}

## Action library

{{action_library}}

## Memory summary

{{memory_summary}}

## Previous feedback

{{previous_feedback}}

## Instructions

1. Read the Rethinker analysis above.
2. Choose `mission`, `pick`, and (when relevant) `place` using only labels listed in the DINO label set.
3. Respond with exactly one JSON object matching the schema in the system prompt.
4. Do not include Markdown prose outside the JSON object.

Return only valid JSON.
