"""Task catalogue schemas.

Defines ``TaskDefinition``, a concrete Pydantic model that extends the shared
``TaskUnit`` with catalogue-specific validation and serialization helpers.
"""

from __future__ import annotations

from common.schema import TaskUnit


class TaskDefinition(TaskUnit):
    """A validated manipulation task entry loaded from a task catalogue.

    This is a thin subclass of :class:`common.schema.TaskUnit` so that the
    catalogue layer can add its own validators/fields in the future without
    changing the shared schema.
    """

    pass
