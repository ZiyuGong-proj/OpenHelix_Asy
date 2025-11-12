"""Utility helpers for working with language instructions."""

from __future__ import annotations

INSTRUCTION_REASONING_SUFFIX = " Let's think step by step"


def append_reasoning_suffix(text: str) -> str:
    """Append the reasoning suffix to an instruction if missing.

    The CALVIN instructions stored in the datasets often end with a trailing
    period.  To keep the original phrasing natural we strip the trailing period
    before appending the reasoning hint.  Repeated invocations of this helper
    are idempotent to avoid duplicating the suffix when the function is applied
    at multiple points of the pipeline.
    """

    stripped = text.rstrip()
    if stripped.endswith(INSTRUCTION_REASONING_SUFFIX.strip()):
        return stripped

    if stripped.endswith('.'):
        stripped = stripped[:-1]

    return f"{stripped}{INSTRUCTION_REASONING_SUFFIX}"

