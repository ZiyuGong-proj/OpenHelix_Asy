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


def format_step_reasoning_prompt(text: str, step_index: int) -> str:
    """Add step-aware guidance so the VLM reasons about the current frame.

    The helper preserves the standard reasoning suffix and adds an extra
    sentence that explicitly requests frame-specific, step-by-step reasoning.
    ``step_index`` is zero-based and converted to a human-readable index in the
    resulting text.
    """

    augmented = append_reasoning_suffix(text)
    step_number = step_index + 1

    if augmented.endswith((".", "?", "!")):
        separator = " "
    else:
        separator = ". "

    return (
        f"{augmented}{separator}"
        f"Focus on the observation at step {step_number} and describe what you see. "
        "Then reason step by step about the immediate action the robot should take based on this specific frame."
    )

