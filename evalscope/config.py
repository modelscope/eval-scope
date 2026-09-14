# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501
import copy
import json
import os
from argparse import Namespace
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import ConfigDict, Field, SecretStr, field_validator, model_validator

from evalscope.agent.external.config import ExternalAgentConfig
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.dataset.dataset import validate_dataset_limit
from evalscope.api.model import GenerateConfig, Model, ModelAPI
from evalscope.constants import (
    DEFAULT_DATASET_CACHE_DIR,
    DEFAULT_WORK_DIR,
    EvalBackend,
    EvalType,
    HubType,
    JudgeStrategy,
    ModelTask,
)
from evalscope.utils.argument_utils import BaseArgument, parse_int_or_float
from evalscope.utils.deprecation_utils import deprecated_warning
from evalscope.utils.import_utils import check_import
from evalscope.utils.io_utils import dict_to_yaml, gen_hash, json_to_dict, safe_filename, yaml_to_dict
from evalscope.utils.logger import get_logger
from evalscope.version import __version__ as _evalscope_version

AgentConfigUnion = Annotated[
    Union[NativeAgentConfig, ExternalAgentConfig],
    Field(discriminator='mode'),
]

logger = get_logger()


def _secretize_api_keys(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return value
    if isinstance(value, list):
        return [_secretize_api_keys(item) for item in value]
    if not isinstance(value, dict):
        return value

    result = {}
    for key, val in value.items():
        if key == 'api_key' and isinstance(val, str):
            result[key] = SecretStr(val)
        else:
            result[key] = _secretize_api_keys(val)
    return result


# Default configurations
DEFAULT_IMAGE_GEN_CONFIG = {
    'height': 1024,
    'width': 1024,
    'num_inference_steps': 50,
    'guidance_scale': 9.0,
}

DEFAULT_TEXT_GEN_CHECKPOINT_CONFIG = {
    'max_tokens': 2048,
    'do_sample': False,
    'top_k': 50,
    'top_p': 1.0,
    'temperature': 1.0,
    'n': 1,
}

DEFAULT_TEXT_GEN_SERVICE_CONFIG = {
    'temperature': 0.0,
}

DEFAULT_MODEL_ARGS_CHECKPOINT = {
    'revision': 'master',
    'precision': 'torch.float16',
}

DEFAULT_API_EVAL_BATCH_SIZE = 8

REMOTE_API_EVAL_TYPES = frozenset(
    {
        EvalType.OPENAI_API,
        EvalType.OPENAI_RESPONSES_API,
        EvalType.ANTHROPIC_API,
        EvalType.LITELLM,
    }
)

DEPRECATED_EVAL_TYPE_ALIASES = {
    'checkpoint': EvalType.CHECKPOINT,
    'server': EvalType.OPENAI_API,
}


class SandboxTaskConfig(BaseArgument):
    """Unified sandbox configuration for both pooled (CodeExecutionSandboxMixin) and
    per-sample (EnclaveAgentEnvironment) execution paths.

    This is the forward-looking replacement for the legacy top-level
    ``TaskConfig.use_sandbox`` / ``sandbox_type`` / ``sandbox_manager_config``
    triplet.  :meth:`TaskConfig._init_default_sandbox_config` folds those
    legacy fields into ``self.sandbox`` once at construction time; afterwards
    all consumers read from ``self.sandbox`` and the legacy fields are
    untouched aliases retained solely for input compatibility.
    """

    enabled: bool = False
    """Whether to enable the sandbox subsystem for this task."""

    engine: str = 'docker'
    """Sandbox engine name.  One of ``'docker'`` / ``'volcengine'`` (or aliases
    accepted by :func:`evalscope.api.sandbox.resolve_engine`)."""

    default_config: Dict[str, Any] = Field(default_factory=dict)
    """Task-level overrides merged on top of ``BenchmarkMeta.sandbox_config``.
    The merged dict is passed to :func:`build_sandbox_config`.  Also acts as
    the default sandbox config for per-sample agent environments."""

    manager_config: Dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded to the ms_enclave manager constructor (e.g.
    ``base_url`` for a remote docker daemon, or volcengine credentials)."""

    pool_size: Optional[int] = None
    """Warm-pool size for pooled execution.  Defaults to ``eval_batch_size``
    when ``None``."""


class JudgeModelConfig(BaseArgument):
    """Transport and identity for one independently weighted judge model.

    Prompt and verdict semantics belong to :class:`JudgeContractConfig`, shared by every
    configured judge.  Letting model entries carry those fields made a multi-judge run compare
    incomparable answers while appearing to aggregate one metric.
    """

    model_config = ConfigDict(
        extra='forbid', arbitrary_types_allowed=True, protected_namespaces=(), validate_default=True
    )

    model_id: str
    judge_id: Optional[str] = None
    api_key: Optional[SecretStr] = None
    api_url: Optional[str] = None
    eval_type: Optional[str] = None
    model_args: Dict[str, Any] = Field(default_factory=dict)
    generation_config: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('model_args', mode='before')
    @classmethod
    def _secretize_model_args(cls, value: Any) -> Any:
        return _secretize_api_keys(value)


class JudgeContractConfig(BaseArgument):
    """Shared semantics for the generic Native Judge contract.

    Custom benchmark contracts keep their official prompt and schema in the adapter; these
    fields only configure the generic single-verdict mixin.
    """

    model_config = ConfigDict(
        extra='forbid', arbitrary_types_allowed=True, protected_namespaces=(), validate_default=True
    )

    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    score_mapping: Optional[Dict[str, float]] = None
    score_type: Literal['pattern', 'numeric'] = 'pattern'


class JudgeConfig(BaseArgument):
    """Typed configuration for Native LLM judging."""

    model_config = ConfigDict(
        extra='forbid', arbitrary_types_allowed=True, protected_namespaces=(), validate_default=True
    )

    strategy: Literal['auto', 'rule', 'llm', 'llm_recall'] = JudgeStrategy.AUTO
    models: List[JudgeModelConfig] = Field(default_factory=list)
    contract: JudgeContractConfig = Field(default_factory=JudgeContractConfig)
    repeats: int = Field(default=1, ge=1)
    position_swap: Literal['auto', 'on', 'off'] = 'auto'
    aggregation: Literal['mean', 'median', 'majority_vote'] = 'mean'
    min_valid_judges: int = Field(default=1, ge=1)

    @model_validator(mode='before')
    @classmethod
    def _normalize_models(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if isinstance(values.get('models'), dict):
            values['models'] = [values['models']]
        return values

    @model_validator(mode='after')
    def _validate_models(self) -> 'JudgeConfig':
        seen = set()
        model_ids = [model.model_id for model in self.models]
        for model in self.models:
            if model.judge_id is None:
                if model_ids.count(model.model_id) != 1:
                    raise ValueError(
                        f'Judge model_id {model.model_id!r} is configured more than once; each entry needs judge_id.'
                    )
                model.judge_id = model.model_id
            if model.judge_id in seen:
                raise ValueError(f'duplicate judge_id: {model.judge_id!r}')
            seen.add(model.judge_id)
        if self.min_valid_judges > len(self.models) and self.models:
            raise ValueError('judge.min_valid_judges cannot exceed the configured judge model count.')
        return self


class TaskConfig(BaseArgument):
    # Model-related arguments
    model: Optional[Union[str, Model, ModelAPI]] = None
    """The model to be evaluated. Can be a string path, Model object, or ModelAPI object."""

    model_id: Optional[str] = None
    """Unique identifier for the model. Auto-generated from model name if not provided."""

    model_args: Dict = Field(default_factory=dict)
    """Additional arguments to pass to the model during initialization."""

    model_task: str = ModelTask.TEXT_GENERATION
    """The type of task the model performs (e.g., text generation, image generation)."""

    # Template-related arguments
    chat_template: Optional[str] = None
    """Chat template to use for formatting conversations with the model."""

    # Dataset-related arguments
    datasets: List[str] = Field(default_factory=list)
    """List of dataset names to evaluate the model on."""

    dataset_args: Dict = Field(default_factory=dict)
    """Additional arguments to pass to datasets during loading."""

    dataset_dir: str = DEFAULT_DATASET_CACHE_DIR
    """Directory where datasets are cached locally."""

    dataset_hub: str = HubType.MODELSCOPE
    """Hub platform to download datasets from (e.g., ModelScope, HuggingFace)."""

    repeats: int = 1
    """Number of times to repeat the dataset items for k-metrics evaluation."""

    # Generation configuration arguments
    generation_config: Union[Dict, GenerateConfig] = Field(default_factory=dict)
    """Configuration parameters for text/image generation."""

    # Evaluation-related arguments
    eval_type: Optional[str] = None
    """Evaluation backend type. One of: 'llm_ckpt' (local checkpoint), 'openai_api',
    'openai_responses_api', 'anthropic_api', 'litellm', 'mock_llm', 'text2image', 'text2speech',
    'image_editing', 'custom'. Deprecated aliases normalized on input: 'checkpoint' -> 'llm_ckpt',
    'server' -> 'openai_api'."""

    eval_backend: str = EvalBackend.NATIVE
    """Backend framework to use for evaluation."""

    eval_config: Union[str, Dict, None] = None
    """Additional evaluation configuration parameters."""

    limit: Optional[Union[int, float]] = None
    """Maximum number of samples to evaluate. Can be int (count) or float (fraction)."""

    eval_batch_size: int = 1
    """Batch size / concurrency for evaluation, applied across all stages:
    - Inference: concurrent requests (openai_api mode) or batch size (llm_ckpt mode).
    - LLM-judge review (BatchReviewer Pass 1): number of concurrent threads.
    - batch_calculate_metrics (BatchReviewer Pass 2): number of samples per batch window.
    - Sandbox execution: worker pool size.

    Defaults to ``DEFAULT_API_EVAL_BATCH_SIZE`` when left unset for a remote API ``eval_type``.
    """

    # Cache and working directory arguments
    use_cache: Optional[str] = None
    """Path to a previous output directory (e.g. 'outputs/20260519_120000') to resume from.
    Reuses cached predictions and reviews matched by sample_id; set None to start fresh."""

    rerun_review: bool = False
    """When use_cache is set, force re-running review/scoring while reusing predictions.

    This is also the explicit override for a native cache identity mismatch;
    the resulting review is recorded under the current evaluation version.
    """

    work_dir: str = DEFAULT_WORK_DIR
    """Root directory for evaluation outputs (predictions/, reviews/, reports/, logs/).
    A timestamped subdirectory is appended unless `no_timestamp=True` or `use_cache` is set."""

    no_timestamp: bool = False
    """Do not add timestamp to the work_dir to avoid overwriting previous results."""

    enable_progress_tracker: bool = False
    """Whether to write a progress.json file tracking hierarchical evaluation progress.
    When True, each TqdmLogging instance auto-reports its stage to the file-backed
    ProgressTracker so the service layer can expose a real-time /progress endpoint."""

    # Debug and runtime mode arguments
    ignore_errors: bool = False
    """Whether to continue evaluation when encountering errors."""

    debug: bool = False
    """Enable debug mode for detailed logging and error reporting."""

    seed: Optional[int] = 42
    """Random seed for reproducible results."""

    api_url: Optional[str] = None
    """API endpoint URL for server-based model evaluation."""

    api_key: Optional[SecretStr] = SecretStr('EMPTY')
    """API key for authenticating with server-based models."""

    timeout: Optional[float] = None
    """[Deprecated] Use `generation_config.timeout` instead. Will be removed in v2.0.0.
    When set, value is forwarded to `generation_config.timeout` with a deprecation warning."""

    stream: Optional[bool] = None
    """[Deprecated] Use `generation_config.stream` instead. Will be removed in v2.0.0.
    When set, value is forwarded to `generation_config.stream` with a deprecation warning."""

    # Native judge configuration
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    """Typed configuration for Native LLM judging."""

    analysis_report: bool = False
    """Whether to generate detailed analysis reports after evaluation."""

    collect_perf: bool = True
    """Whether to collect per-request performance metrics (latency, TTFT, token usage)
    during evaluation. TTFT requires streaming (set `generation_config.stream=True`)."""

    # Sandbox configuration arguments
    sandbox: Optional[SandboxTaskConfig] = None
    """Unified sandbox configuration (preferred).  When set, takes precedence
    over the legacy ``use_sandbox`` / ``sandbox_type`` / ``sandbox_manager_config``
    fields which are kept as deprecated aliases."""

    use_sandbox: bool = False
    """[Deprecated] Use ``sandbox.enabled`` instead.  Kept as an alias for
    backward compatibility; will be removed in v2.0.0."""

    sandbox_type: Optional[str] = 'docker'
    """[Deprecated] Use ``sandbox.engine`` instead.  Kept as an alias for
    backward compatibility; will be removed in v2.0.0."""

    sandbox_manager_config: Optional[Dict] = Field(default_factory=dict)
    """[Deprecated] Use ``sandbox.manager_config`` instead.  Kept as an
    alias for backward compatibility; will be removed in v2.0.0."""

    # Agent configuration (native AgentLoop OR external-agent bridge,
    # discriminated by the ``mode`` field on the embedded config).
    agent_config: Optional[AgentConfigUnion] = None
    """Per-task agent configuration.

    Discriminated union driven by the ``mode`` field:

    * ``mode='native'`` (default) → :class:`NativeAgentConfig`; every
      DefaultDataAdapter-based benchmark routes inference through the
      :class:`AgentLoop`.
    * ``mode='external'`` → :class:`ExternalAgentConfig`; inference is
      delegated to a third-party CLI (claude-code, mock, ...) and the
      bridge captures the LLM traffic into the same :class:`AgentTrace`.

    AgentLoopAdapter subclasses keep benchmark defaults and consume supported
    explicit overrides from this field; adapters with bespoke loops may ignore
    it. ``dict`` inputs accept ``{'mode': 'external',
    'framework': 'claude-code'}`` style payloads."""

    evalscope_version: Optional[str] = _evalscope_version
    """EvalScope version used for the evaluation."""

    # --- Field validators (single-field logic) ---

    @field_validator('limit', mode='before')
    @classmethod
    def _validate_limit(cls, v: Any) -> Any:
        if v is not None:
            v = parse_int_or_float(v)
            v = validate_dataset_limit(v)
            if v == 0:
                return None
        return v

    @field_validator('eval_config', mode='before')
    @classmethod
    def _validate_eval_config(cls, v: Any) -> Any:
        if not v:
            return v
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            extension = os.path.splitext(v)[-1]
            if extension in ['.yaml', '.yml']:
                return yaml_to_dict(v)
            elif extension == '.json':
                return json_to_dict(v)
            else:
                try:
                    return json.loads(v)
                except Exception as e:
                    raise ValueError('eval_config string is not a valid json string or file path.') from e
        else:
            raise ValueError('eval_config should be a dict or a file path string.')

    @field_validator('agent_config', mode='before')
    @classmethod
    def _validate_agent_config(cls, v: Any) -> Any:
        return cls._coerce_agent_config(v)

    @staticmethod
    def _coerce_agent_config(v: Any) -> Any:
        if v is None or isinstance(v, (NativeAgentConfig, ExternalAgentConfig)):
            return v
        if isinstance(v, dict):
            mode = v.get('mode', 'native')
            if mode == 'external':
                return ExternalAgentConfig.model_validate(v)
            if mode == 'native':
                return NativeAgentConfig.model_validate(v)
            raise ValueError(f'`agent_config.mode` must be "native" or "external", got {mode!r}.')
        raise ValueError(
            f'`agent_config` must be a dict, NativeAgentConfig, ExternalAgentConfig or None, got {type(v).__name__}.'
        )

    @model_validator(mode='before')
    @classmethod
    def _migrate_deprecated_input(cls, data: Any) -> Any:
        """Fold deprecated top-level input keys that are pure renames onto their canonical
        field, once at the public boundary. Deprecations that need the coerced field value
        (timeout/stream/n, sandbox legacy fields) run after validation."""
        if not isinstance(data, dict):
            return data
        values = dict(data)
        cls._migrate_judge_keys(values)
        cls._migrate_eval_type_alias(values)
        return values

    @staticmethod
    def _migrate_judge_keys(values: Dict[str, Any]) -> None:
        if 'judge_worker_num' in values:
            raise ValueError('`judge_worker_num` has been removed; use `eval_batch_size`.')
        legacy_keys = {'judge_strategy', 'judge_model_args'} & set(values)
        if 'judge' in values and legacy_keys:
            raise ValueError('Use either `judge` or legacy judge_strategy/judge_model_args, not both.')
        if not legacy_keys:
            return
        legacy_args = values.pop('judge_model_args', None)
        if isinstance(legacy_args, dict):
            legacy_args = dict(legacy_args)
        strategy = values.pop('judge_strategy', JudgeStrategy.AUTO)
        if legacy_args and 'score_pattern' in legacy_args:
            raise ValueError(
                '`judge_model_args.score_pattern` is not supported by the JSON judge contract; '
                'use `judge.contract.score_mapping`.'
            )
        deprecated_warning(
            logger,
            '`judge_strategy` and `judge_model_args` are deprecated; use the typed `judge` configuration.',
        )
        semantic_keys = {'system_prompt', 'prompt_template', 'score_mapping', 'score_type'}
        contract = {key: legacy_args.pop(key) for key in semantic_keys if legacy_args and key in legacy_args}
        values['judge'] = {
            'strategy': strategy,
            'models': [legacy_args] if legacy_args else [],
            'contract': contract,
        }

    @staticmethod
    def _migrate_eval_type_alias(values: Dict[str, Any]) -> None:
        canonical = DEPRECATED_EVAL_TYPE_ALIASES.get(values.get('eval_type'))
        if canonical is None:
            return
        deprecated_warning(
            logger,
            f'`eval_type={values["eval_type"]!r}` is deprecated and will be removed in v2.0.0. '
            f'Use {canonical!r} instead.',
        )
        values['eval_type'] = canonical

    @field_validator('sandbox', mode='before')
    @classmethod
    def _validate_sandbox(cls, v: Any) -> Any:
        return cls._coerce_sandbox_config(v)

    # --- Model validator (cross-field logic, replaces __post_init__) ---

    @model_validator(mode='after')
    def _post_init(self) -> 'TaskConfig':
        self._init_model_and_id()
        self._init_default_eval_batch_size()
        self._init_default_generation_config()
        self._init_default_model_args()
        self._init_default_sandbox_config()
        self._parse_rag_eval_config()

        return self

    def _init_default_eval_batch_size(self) -> None:
        """Remote APIs serve requests concurrently, so raise the default there."""
        if 'eval_batch_size' not in self.model_fields_set and self.eval_type in REMOTE_API_EVAL_TYPES:
            self.eval_batch_size = DEFAULT_API_EVAL_BATCH_SIZE

    def _parse_rag_eval_config(self) -> None:
        """Parse eval_config into typed Pydantic models for RAGEval backend."""
        if self.eval_backend != EvalBackend.RAG_EVAL or not isinstance(self.eval_config, dict):
            return
        tool = self.eval_config.get('tool', '').lower()
        if tool == 'mteb':
            from evalscope.backend.rag_eval.mteb.arguments import MTEBToolConfig

            self.eval_config = MTEBToolConfig(**self.eval_config)
        elif tool == 'ragas':
            from evalscope.backend.rag_eval.ragas.arguments import RAGASToolConfig

            self.eval_config = RAGASToolConfig(**self.eval_config)
        elif tool == 'clip_benchmark':
            from evalscope.backend.rag_eval.clip_benchmark.arguments import ClipBenchmarkToolConfig

            self.eval_config = ClipBenchmarkToolConfig(**self.eval_config)

    def _init_model_and_id(self) -> None:
        # Set model to DummyCustomModel if not provided
        if self.model is None:
            logger.info('No model is provided, using DummyCustomModel for testing.')
            self.model = self.model_task
            self.eval_type = EvalType.MOCK_LLM

        # Select the matching service API when api_url is provided
        if self.api_url is not None and self.eval_type is None:
            self.eval_type = (
                EvalType.TEXT2IMAGE if self.model_task == ModelTask.IMAGE_GENERATION else EvalType.OPENAI_API
            )
            logger.info(f"api_url is provided, setting eval_type to '{self.eval_type}'.")

        # Set eval_type to CHECKPOINT if model is a string path and eval_type is not set
        if self.model and self.eval_type is None:
            logger.info('No eval_type is provided, setting eval_type to CHECKPOINT.')
            self.eval_type = EvalType.CHECKPOINT

        # Set model_id if not provided
        if not self.model_id:
            self.model_id = self._infer_model_id()

    def _infer_model_id(self) -> str:
        if isinstance(self.model, str):
            return safe_filename(os.path.basename(self.model))
        elif isinstance(self.model, Model):
            return safe_filename(self.model.name)
        elif isinstance(self.model, ModelAPI):
            return safe_filename(self.model.model_name)
        return 'dummy_model'

    def _init_default_generation_config(self) -> None:
        # 1. Set defaults if empty
        if not self.generation_config:
            self.generation_config = self._get_default_generation_config()

        # 2. Validate/Convert to GenerateConfig object
        self.generation_config = self._coerce_generation_config(self.generation_config)

        # 3. Sync batch size
        self.generation_config.batch_size = self.eval_batch_size

        # 4. Handle deprecations
        self._apply_typed_config_deprecations()

    def _get_default_generation_config(self) -> Dict:
        if self.model_task == ModelTask.IMAGE_GENERATION:
            return DEFAULT_IMAGE_GEN_CONFIG.copy()

        elif self.model_task == ModelTask.TEXT_GENERATION:
            if self.eval_type == EvalType.CHECKPOINT:
                return DEFAULT_TEXT_GEN_CHECKPOINT_CONFIG.copy()
            elif self.eval_type in (EvalType.OPENAI_API, EvalType.OPENAI_RESPONSES_API):
                return DEFAULT_TEXT_GEN_SERVICE_CONFIG.copy()

        return {}

    @staticmethod
    def _coerce_generation_config(value: Union[Dict, GenerateConfig]) -> GenerateConfig:
        if isinstance(value, GenerateConfig):
            return value
        return GenerateConfig.model_validate(value)

    @staticmethod
    def _coerce_sandbox_config(value: Any) -> Optional[SandboxTaskConfig]:
        if value is None or isinstance(value, SandboxTaskConfig):
            return value
        if isinstance(value, dict):
            return SandboxTaskConfig.model_validate(value)
        raise ValueError(f'`sandbox` must be a dict, SandboxTaskConfig or None, got {type(value).__name__}.')

    def _apply_typed_config_deprecations(self) -> None:
        assert isinstance(self.generation_config, GenerateConfig)

        if self.timeout is not None:
            deprecated_warning(
                logger,
                'The `timeout` parameter is deprecated and will be removed in v2.0.0. Use `generation_config.timeout` instead.',
            )
            self.generation_config.timeout = self.timeout

        if self.stream is not None:
            deprecated_warning(
                logger,
                'The `stream` parameter is deprecated and will be removed in v2.0.0. Use `generation_config.stream` instead.',
            )
            self.generation_config.stream = self.stream

        if self.generation_config.n is not None and self.generation_config.n > 1:
            self.repeats = self.generation_config.n
            self.generation_config.n = 1
            deprecated_warning(
                logger,
                'The `n` parameter in generation_config is deprecated and will be removed in v2.0.0. Use `TaskConfig.repeats` instead.',
            )

    def _init_default_model_args(self) -> None:
        if self.model_args:
            return
        if self.model_task == ModelTask.TEXT_GENERATION and self.eval_type == EvalType.CHECKPOINT:
            self.model_args = DEFAULT_MODEL_ARGS_CHECKPOINT.copy()

    def _init_default_sandbox_config(self) -> None:
        """Fold legacy sandbox fields onto ``self.sandbox`` and validate availability.

        Runs after validation so ``use_sandbox`` / ``sandbox_type`` / ``sandbox_manager_config``
        are already coerced to their declared types; consumers then read sandbox settings
        exclusively from ``self.sandbox``.
        """
        if self.sandbox is None:
            self.sandbox = self._build_sandbox_from_legacy_fields()
        elif self._legacy_sandbox_fields_set():
            deprecated_warning(
                logger,
                'Both `sandbox` and legacy sandbox fields '
                '(`use_sandbox` / `sandbox_type` / `sandbox_manager_config`) are set; '
                'the nested `sandbox` object takes precedence. The legacy fields will be '
                'removed in v2.0.0.',
            )

        if not self.sandbox.enabled:
            return

        check_import('ms_enclave', 'evalscope[sandbox]', raise_error=True)

    def _legacy_sandbox_fields_set(self) -> bool:
        return (
            bool(self.use_sandbox) or (self.sandbox_type not in (None, 'docker')) or bool(self.sandbox_manager_config)
        )

    def _build_sandbox_from_legacy_fields(self) -> SandboxTaskConfig:
        return SandboxTaskConfig(
            enabled=bool(self.use_sandbox),
            engine=self.sandbox_type or 'docker',
            manager_config=dict(self.sandbox_manager_config or {}),
        )

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override into base, returning a new dict."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = TaskConfig._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def update(self, other: Union['TaskConfig', dict]) -> None:
        if isinstance(other, TaskConfig):
            other = other._to_update_dict()
        other = _secretize_api_keys(other)
        current = self._to_update_dict()
        incoming_agent_config = other.get('agent_config')
        current_agent_config = current.get('agent_config')
        if isinstance(incoming_agent_config, dict) and isinstance(current_agent_config, dict):
            incoming_mode = incoming_agent_config.get('mode')
            if incoming_mode is not None and incoming_mode != current_agent_config.get('mode'):
                current['agent_config'] = {}
        merged = self._deep_merge(current, other)
        if isinstance(merged.get('generation_config'), dict):
            merged['generation_config'] = self._coerce_generation_config(merged['generation_config'])
        if isinstance(merged.get('sandbox'), dict):
            merged['sandbox'] = self._coerce_sandbox_config(merged['sandbox'])
        if isinstance(merged.get('agent_config'), dict):
            merged['agent_config'] = self._coerce_agent_config(merged['agent_config'])
        if isinstance(merged.get('judge'), dict):
            merged['judge'] = JudgeConfig.model_validate(merged['judge'])
        for key, value in merged.items():
            setattr(self, key, value)

    def _to_update_dict(self) -> dict:
        return self._serialize('update')

    def _serialize(self, purpose: str) -> dict:
        """Single serialization core shared by :meth:`to_dict` (``purpose='yaml'``) and
        :meth:`_to_update_dict` (``purpose='update'``).

        ``yaml`` renders JSON-native values for persistence; ``update`` keeps typed
        objects so :meth:`update` can deep-merge then re-coerce. Every special field
        that model_dump cannot round-trip is rendered here per purpose, so the two
        paths cannot drift.
        """
        json_mode = purpose == 'yaml'
        dump_mode = 'json' if json_mode else None
        special = {'model', 'generation_config', 'sandbox', 'agent_config'}
        result = self.model_dump(mode='json', exclude=special) if json_mode else self.model_dump(exclude=special)
        result['model'] = self._dump_model(json_mode)
        result['generation_config'] = self._dump_generation_config(mode=dump_mode)
        result['sandbox'] = self._dump_sandbox(purpose)
        result['agent_config'] = self._dump_agent_config(mode=dump_mode)
        return result

    def _dump_model(self, json_mode: bool) -> Any:
        if json_mode and isinstance(self.model, (Model, ModelAPI)):
            return self.model.__class__.__name__
        return self.model

    def _dump_sandbox(self, purpose: str) -> Any:
        if purpose == 'update':
            return self.sandbox
        return self.sandbox.model_dump(mode='json') if self.sandbox is not None else None

    def _dump_generation_config(self, mode: Optional[str] = None) -> Union[dict, GenerateConfig, None]:
        if not isinstance(self.generation_config, GenerateConfig):
            return self.generation_config

        kwargs = {'exclude_unset': True}
        if mode is not None:
            kwargs['mode'] = mode
        return self.generation_config.model_dump(**kwargs)

    def _dump_agent_config(
        self, mode: Optional[str] = None
    ) -> Union[dict, NativeAgentConfig, ExternalAgentConfig, None]:
        if not isinstance(self.agent_config, (NativeAgentConfig, ExternalAgentConfig)):
            return self.agent_config
        fields = set(self.agent_config.model_fields_set)
        fields.add('mode')
        kwargs = {'include': fields}
        if mode is not None:
            kwargs['mode'] = mode
        return self.agent_config.model_dump(**kwargs)

    def dump_yaml(self, output_dir: str, generated_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Dump the task configuration and optional generated runtime metadata to YAML."""
        task_cfg_file = os.path.join(output_dir, 'task_config.yaml')
        try:
            logger.info(f'Dump task config to {task_cfg_file}')
            payload = self.to_dict()
            if generated_metadata:
                payload.update(generated_metadata)
            dict_to_yaml(payload, task_cfg_file)
        except Exception as e:
            logger.warning(f'Failed to dump overall task config: {e}')

    def to_dict(self) -> dict:
        return self._serialize('yaml')


def _strip_generated_evaluation_metadata(task_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Remove output-only evaluation metadata before treating a snapshot as input."""
    result = copy.deepcopy(task_cfg)
    result.pop('resolved_benchmarks', None)
    result.pop('evaluation_identity', None)
    return result


def load_task_config_snapshot(config_file: str) -> Optional[Dict[str, Any]]:
    """Load a raw output task-config snapshot for cache identity validation."""
    if not os.path.isfile(config_file):
        return None
    config = yaml_to_dict(config_file)
    if not isinstance(config, dict):
        raise ValueError(f'Invalid task config snapshot at {config_file}: expected a mapping.')
    return config


def parse_task_config(task_cfg) -> TaskConfig:
    """Parse task configuration from various formats into a TaskConfig object."""
    if isinstance(task_cfg, TaskConfig):
        logger.info('Args: Task config is provided with TaskConfig type.')
    elif isinstance(task_cfg, dict):
        logger.info('Args: Task config is provided with dictionary type.')
        task_cfg = TaskConfig.from_dict(_strip_generated_evaluation_metadata(task_cfg))
    elif isinstance(task_cfg, Namespace):
        logger.info('Args: Task config is provided with CommandLine type.')
        task_cfg = TaskConfig.from_args(task_cfg)
    elif isinstance(task_cfg, str):
        extension = os.path.splitext(task_cfg)[-1]
        logger.info(f'Args: Task config is provided with {extension} file type.')
        if extension in ['.yaml', '.yml']:
            task_cfg = TaskConfig.from_dict(_strip_generated_evaluation_metadata(yaml_to_dict(task_cfg)))
        elif extension == '.json':
            task_cfg = TaskConfig.from_dict(_strip_generated_evaluation_metadata(json_to_dict(task_cfg)))
        else:
            raise ValueError('Args: Unsupported file extension.')
    else:
        raise ValueError('Args: Please provide a valid task config.')
    return task_cfg
