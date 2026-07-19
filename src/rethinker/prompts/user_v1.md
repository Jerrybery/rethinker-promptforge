# User Prompt: Rethinker Agent v1

## Task goal

{{task_goal}}

## Current scene

Detected objects (from DINO vision model):

{{detections}}

## Memory summary

{{memory_summary}}

## Previous feedback

{{previous_feedback}}

## Instructions

1. Look at the RGB image and the detections above.
2. Consider the task goal, memory, and previous feedback.
3. Form an explicit hidden hypothesis about occluded or unobserved objects relevant to the task.
4. Choose the most appropriate high-level mission type.
5. Respond with exactly one JSON object matching the schema in the system prompt.
6. Do not include Markdown prose outside the JSON object.

Return only valid JSON.
