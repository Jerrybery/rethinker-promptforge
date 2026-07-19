# User Prompt: Monolithic Planner Baseline v0

## Task goal

{{task_goal}}

## Current scene

Detected objects (from DINO vision model):

{{detections}}

## Previous feedback

{{previous_feedback}}

## Instructions

1. Look at the RGB image and the detections above.
2. Consider the task goal and any previous feedback.
3. Choose the mission type and the pick/place target labels in one step.
4. Respond with exactly one JSON object matching the schema in the system prompt.
5. Do not include Markdown prose outside the JSON object.

Return only valid JSON.
