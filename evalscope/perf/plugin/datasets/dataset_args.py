"""Typed per-dataset argument schemas for the perf ``--dataset-args`` mechanism.

Each dataset plugin declares an ``args_schema`` (a subclass of
:class:`BaseDatasetArgs`).  The raw ``--dataset-args`` JSON is validated against
the plugin's own ``args_schema`` at plugin construction time (see
:class:`~evalscope.perf.plugin.datasets.base.DatasetPluginBase`), giving plugins
a strongly-typed, fail-fast accessor instead of scattered ``query_parameters.*``
reads.

This module must NOT import any dataset plugin module (they import
``Arguments``), to avoid an import cycle.  Importing :mod:`multi_turn_args` is
safe because it has no such dependency.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from evalscope.perf.multi_turn_args import MultiTurnArgs


class BaseDatasetArgs(BaseModel):
    """Base schema for ``--dataset-args``.

    Rejects unknown keys so a mistyped argument fails fast at parse time rather
    than being silently ignored.
    """

    model_config = ConfigDict(extra='forbid')


class TextLengthArgs(BaseDatasetArgs):
    """Input-length controls for real-text datasets (issue #1483)."""

    target_input_len: Optional[int] = None
    """Target input length in tokens.

    When set, prompts are actively fit to this length (see ``input_len_mode``)
    instead of being filtered by ``min/max_prompt_length``.  Requires a
    tokenizer (``--tokenizer-path``).  ``None`` (default) keeps the legacy
    length-filter behavior.
    """

    input_len_mode: Literal['cap', 'drop'] = 'cap'
    """Policy applied when a prompt is shorter than ``target_input_len``.

    - ``cap`` (default): truncate over-length prompts to the target; keep
      shorter prompts as-is (result <= target, preserves real semantics).
    - ``drop``: truncate over-length prompts; skip shorter ones so every
      yielded prompt is exactly ``target_input_len`` (drops data).
    """

    prefix_file: Optional[str] = None
    """Path to a UTF-8 plain-text file used as a long-context prefix (issue #1524).

    When set, the file content is tokenized once and, for every request, sliced
    to exactly ``target_input_len - len(prompt_tokens)`` tokens so the total
    input hits the target length with real low-entropy text.  If the file is
    shorter than the remaining budget it is repeated (tiled) to cover it.
    Requires ``target_input_len`` and is incompatible with
    ``input_len_mode='drop'``.
    """

    prefix_role: Literal['system', 'user'] = 'system'
    """Role used to inject the long prefix.

    - ``system`` (default): the prefix goes into a leading system message,
      matching real RAG traffic and inference-engine prefix-cache handling.
      Falls back to plain-text concatenation when no chat template is applied.
    - ``user``: the prefix is prepended to the (first) user message content.
    """

    @field_validator('target_input_len')
    @classmethod
    def _validate_target_input_len(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(f'target_input_len must be > 0, got {v}')
        return v

    @model_validator(mode='after')
    def _validate_prefix_file(self) -> 'TextLengthArgs':
        if self.prefix_file is not None and self.target_input_len is None:
            raise ValueError('`prefix_file` requires `target_input_len` to be set.')
        # `drop` already forces every prompt to exactly `target_input_len`, leaving a
        # zero prefix budget, so the prefix would silently never be injected.
        if self.prefix_file is not None and self.input_len_mode == 'drop':
            raise ValueError(
                "`prefix_file` is incompatible with `input_len_mode='drop'`: `drop` keeps only prompts that "
                "already fill `target_input_len`, so no prefix budget is left. Use `input_len_mode='cap'` "
                'and let the prefix fill the remaining budget.'
            )
        return self


class TextDatasetArgs(TextLengthArgs):
    """Args schema for single-turn real-text datasets.

    Used by ``openqa`` / ``longalpaca`` / ``share_gpt_*`` (single-turn) /
    ``line_by_line`` (plain-text path).
    """


class MMMUMultiImageDatasetArgs(BaseDatasetArgs):
    """Arguments for the MMMU multi-image performance dataset."""

    subset: str = 'Music'
    """MMMU subject/configuration to load."""

    min_images: int = 2
    """Minimum number of images required for a row to be included."""

    @field_validator('min_images')
    @classmethod
    def _validate_min_images(cls, v: int) -> int:
        if not 2 <= v <= 7:
            raise ValueError(f'min_images must be between 2 and 7, got {v}')
        return v


class MultiTurnDatasetArgs(MultiTurnArgs, BaseDatasetArgs):
    """Args schema for multi-turn datasets.

    Inherits every :class:`~evalscope.perf.multi_turn_args.MultiTurnArgs` field
    and ``sample_params()`` while adding fail-fast unknown-key rejection.  Note
    ``min_turns`` / ``max_turns`` remain top-level ``Arguments`` fields and are
    intentionally not folded here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
