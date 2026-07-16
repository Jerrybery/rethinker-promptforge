"""Executor-specific schemas.

Extends the shared :class:`common.schema.ExecutorOutput` with execution
status, risk flags, and feedback so the ExecutorAgent can report structured
outcomes without fabricating low-level joint commands.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from common.schema import ExecutorOutput, Feedback


class ExecutorAgentOutput(ExecutorOutput):
    """Execution outcome returned by :class:`executor.agent.ExecutorAgent`.

    Inherits the low-level robot-command fields from ``ExecutorOutput`` but
    makes them optional, since the agent emits a success/failure/risk result
    and natural-language feedback rather than explicit joint angles.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool | None = None
    risk: bool = False
    feedback: Feedback | None = None
    status: str | None = None

    # Low-level command fields are optional for the agent result.
    step_index: int = Field(default=0, ge=0)
    joint_angles: list[float] = Field(default_factory=list)
    gripper_state: float = Field(default=0.0, ge=0.0, le=1.0)
