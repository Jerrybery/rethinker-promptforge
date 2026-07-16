"""Robot state representation.

Defines immutable Pydantic models for end-effector pose and full robot state,
including the latest RGB camera image as a numpy array.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Pose(BaseModel):
    """7-DOF end-effector pose.

    ``position`` is the Cartesian coordinate ``[x, y, z]`` in meters.
    ``orientation`` is a unit quaternion ``[x, y, z, w]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: list[float] = Field(..., min_length=3, max_length=3)
    orientation: list[float] = Field(..., min_length=4, max_length=4)

    @classmethod
    def from_list(cls, pose: list[float] | tuple[float, ...]) -> "Pose":
        """Build a ``Pose`` from a flat 7-vector ``[x, y, z, qx, qy, qz, qw]``."""
        if len(pose) != 7:
            raise ValueError(f"Expected 7-DOF pose, got {len(pose)} values")
        return cls(position=list(pose[:3]), orientation=list(pose[3:]))

    def to_list(self) -> list[float]:
        """Return the pose as a flat 7-vector ``[x, y, z, qx, qy, qz, qw]``."""
        return list(self.position) + list(self.orientation)

    def __repr__(self) -> str:
        px, py, pz = self.position
        ox, oy, oz, ow = self.orientation
        return (
            f"Pose(position=[{px:.3f}, {py:.3f}, {pz:.3f}], "
            f"orientation=[{ox:.3f}, {oy:.3f}, {oz:.3f}, {ow:.3f}])"
        )


class RobotState(BaseModel):
    """Snapshot of robot state at a single timestamp.

    Attributes:
        pose: Current 7-DOF end-effector pose.
        gripper: Normalized gripper opening in ``[0, 1]`` (1 = fully open).
        camera_image: Latest RGB camera image as a ``uint8`` numpy array.
        timestamp: Unix timestamp of the observation.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    pose: Pose
    gripper: float = Field(..., ge=0.0, le=1.0)
    camera_image: np.ndarray
    timestamp: float

    def __repr__(self) -> str:
        return (
            f"RobotState(pose={self.pose}, gripper={self.gripper:.2f}, "
            f"camera_image={self.camera_image.shape}, timestamp={self.timestamp:.3f})"
        )
