"""AIPerf-backed InferenceX AgentX MVP scenario.

This module intentionally delegates trace reconstruction and DAG replay to
AIPerf. EvalScope owns only configuration validation, dataset provenance,
process lifecycle and result normalization.
"""

import hashlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evalscope.api.dataset import download_dataset_snapshot
from evalscope.constants import HubType
from evalscope.utils import get_secret_value
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.perf.arguments import Arguments

logger = get_logger()

_AIPERF_VERSION = '0.12.0'
_DEFAULT_SEED = 20260707
_DEFAULT_DURATION = 1800.0
_SMOKE_DURATION = 60.0
_SMOKE_TRACE_LIMIT = 4
_AIPERF_SUMMARY = 'profile_export_aiperf.json'

_DATASETS = {
    '256k': {
        'modelscope_id': 'evalscope/cc-traces-weka-062126-256k',
        'modelscope_revision': 'a97a07a0474f9287023827b89a1478e52f1917da',
        'huggingface_id': 'semianalysisai/cc-traces-weka-062126-256k',
        'huggingface_revision': '8fecd2fc56694469f758f0afbbb6335ad3043740',
        'public_dataset': 'semianalysis_cc_traces_weka_062126_256k',
        'sha256': 'e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef',
    },
    'full': {
        'modelscope_id': 'evalscope/cc-traces-weka-062126',
        'modelscope_revision': 'c73c455b0ef068dc2cdf7e85b6cf24696925eea4',
        'huggingface_id': 'semianalysisai/cc-traces-weka-062126',
        'huggingface_revision': '23f152f6f0f9399a85901b89a6458def0ef16729',
        'public_dataset': 'semianalysis_cc_traces_weka_062126',
        'sha256': '29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276',
    },
}


class AgentXScenario(BaseModel):
    """Typed configuration for the AIPerf AgentX MVP scenario."""

    model_config = ConfigDict(extra='forbid')

    name: Literal['agentx'] = 'agentx'
    variant: Literal['256k', 'full'] = '256k'
    mode: Literal['benchmark', 'smoke'] = 'benchmark'
    max_context_length: Optional[int] = None
    trace_limit: Optional[int] = None
    num_gpus: Optional[int] = None
    engine: Optional[str] = None
    engine_version: Optional[str] = None
    hardware: Optional[str] = None
    tokenizer_trust_remote_code: bool = False
    tokenizer_revision: Optional[str] = None
    request_timeout_seconds: Optional[float] = Field(default=None, gt=0)
    benchmark_grace_period: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator('max_context_length', 'trace_limit', 'num_gpus')
    @classmethod
    def _positive_optional_int(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError('must be greater than zero')
        return value

    @model_validator(mode='after')
    def _validate_mode(self) -> 'AgentXScenario':
        if self.mode == 'benchmark' and self.trace_limit is not None:
            raise ValueError('trace_limit is only supported when mode is smoke')
        return self

    @property
    def effective_trace_limit(self) -> Optional[int]:
        """Return the resolved trace limit for this run."""
        if self.mode == 'smoke':
            return self.trace_limit or _SMOKE_TRACE_LIMIT
        return None


class AgentXMetric(BaseModel):
    """A schema-forward AIPerf metric block."""

    model_config = ConfigDict(extra='allow')

    unit: str
    avg: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    std: Optional[float] = None
    p1: Optional[float] = None
    p5: Optional[float] = None
    p10: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p90: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    count: Optional[int] = None
    sum: Optional[float] = None


class AgentXDatasetProvenance(BaseModel):
    """Resolved AgentX dataset identity."""

    source: str
    dataset_id: str
    revision: Optional[str] = None
    trace_file: str
    sha256: str
    verified: bool


class AgentXRunSummary(BaseModel):
    """EvalScope-owned summary alongside untouched AIPerf artifacts."""

    schema_version: str = '1.0'
    status: Literal['completed', 'failed', 'cancelled']
    scenario: AgentXScenario
    concurrency: int
    duration: float
    seed: int
    aiperf_version: Optional[str] = None
    aiperf_schema_version: Optional[str] = None
    dataset: AgentXDatasetProvenance
    submission_valid: bool
    aiperf_submission_valid: Optional[bool] = None
    aiperf_submission_invalid_reasons: List[str] = Field(default_factory=list)
    mirror_revalidated: bool = False
    metrics: Dict[str, AgentXMetric] = Field(default_factory=dict)
    derived_metrics: Dict[str, AgentXMetric] = Field(default_factory=dict)
    error_summary: List[Dict[str, Any]] = Field(default_factory=list)
    raw_artifact_dir: str
    raw_summary_file: Optional[str] = None
    command: List[str] = Field(default_factory=list)


def parse_agentx_scenario(value: Union[str, Dict[str, Any], AgentXScenario]) -> AgentXScenario:
    """Parse the CLI shorthand, JSON form, mapping, or typed scenario."""
    if isinstance(value, AgentXScenario):
        return value
    if isinstance(value, dict):
        return AgentXScenario.model_validate(value)
    if not isinstance(value, str):
        raise ValueError('--scenario must be "agentx", a JSON object, a dict, or AgentXScenario')
    if value == 'agentx':
        return AgentXScenario()
    try:
        return AgentXScenario.model_validate_json(value)
    except ValueError as e:
        raise ValueError('--scenario must be "agentx" or a valid AgentX JSON object') from e


def validate_agentx_arguments(args: 'Arguments') -> None:
    """Validate the narrow subset of Perf settings meaningful to AgentX."""
    scenario = args.scenario
    if scenario is None:
        return
    if args.api != 'openai':
        raise ValueError('AgentX supports only --api openai (OpenAI Chat Completions).')
    if args.open_loop or args.multi_turn or args.sla_auto_tune:
        raise ValueError('AgentX does not support --open-loop, --multi-turn, or --sla-auto-tune.')
    if args.rate != [-1] and args.rate != -1:
        raise ValueError('AgentX uses closed-loop session-tree concurrency; --rate is not supported.')
    if args.warmup_num != 0:
        raise ValueError('AgentX owns warmup through AIPerf; --warmup-num is not supported.')
    if args.dataset_args:
        raise ValueError('AgentX does not support --dataset-args.')
    if args.dataset != 'openqa':
        raise ValueError('AgentX chooses its Weka dataset from --scenario; do not pass --dataset.')
    if 'number' in args.model_fields_set:
        raise ValueError('AgentX is duration-based; --number is not supported.')
    if scenario.mode == 'benchmark' and args.duration is not None and args.duration < 900:
        raise ValueError('AgentX benchmark mode requires --duration >= 900 seconds.')
    if not args.tokenizer_path:
        raise ValueError('AgentX requires --tokenizer-path for Weka trace replay.')
    if not args.model:
        raise ValueError('--model is required for AgentX.')
    if any(value <= 0 for value in args.parallel):
        raise ValueError('--parallel values must be greater than zero for AgentX.')


def run_agentx_benchmark(args: 'Arguments', output_path: str) -> Dict[str, Dict[str, Any]]:
    """Run one AIPerf process per requested AgentX concurrency value."""
    _require_aiperf()
    scenario = args.scenario
    assert scenario is not None
    dataset = _resolve_dataset(args, scenario)
    seed = args.seed if args.seed is not None else _DEFAULT_SEED
    duration = _resolve_duration(args, scenario)
    base_path = Path(output_path) / f'agentx_{scenario.variant}'
    results: Dict[str, Dict[str, Any]] = {}

    failed_run: Optional[AgentXRunSummary] = None
    for index, concurrency in enumerate(args.parallel):
        run_path = base_path / f'parallel_{concurrency}'
        artifacts_path = run_path / 'aiperf'
        artifacts_path.mkdir(parents=True, exist_ok=True)
        command, env = _build_command(args, scenario, dataset, int(concurrency), duration, seed, artifacts_path)
        summary = _run_one_agentx(
            command=command,
            env=env,
            scenario=scenario,
            dataset=dataset,
            concurrency=int(concurrency),
            duration=duration,
            seed=seed,
            run_path=run_path,
            artifacts_path=artifacts_path,
        )
        _write_json(run_path / 'agentx_summary.json', summary.model_dump(mode='json'))
        results[f'parallel_{concurrency}'] = summary.model_dump(mode='json')
        if summary.status != 'completed':
            failed_run = summary
            break
        if index < len(args.parallel) - 1:
            logger.info(f'Sleeping for {args.sleep_interval} seconds before the next AgentX run...')
            time.sleep(args.sleep_interval)

    _write_json(base_path / 'agentx_sweep_summary.json', results)
    _print_summary(results)
    if failed_run is not None:
        raise RuntimeError(f'AgentX AIPerf run {failed_run.status}; inspect {failed_run.raw_artifact_dir}')
    return results


def _require_aiperf() -> None:
    if not (sys.version_info.major == 3 and 11 <= sys.version_info.minor < 14):
        raise RuntimeError('AgentX requires Python 3.11-3.13. Install with `pip install "evalscope[agentx]"`.')
    try:
        version = importlib.metadata.version('aiperf')
    except importlib.metadata.PackageNotFoundError as e:
        raise RuntimeError('AIPerf is required. Install with `pip install "evalscope[agentx]"`.') from e
    if version != _AIPERF_VERSION:
        raise RuntimeError(f'AgentX requires aiperf=={_AIPERF_VERSION}, found {version}.')


def _resolve_dataset(args: 'Arguments', scenario: AgentXScenario) -> AgentXDatasetProvenance:
    spec = _DATASETS[scenario.variant]
    source = HubType.LOCAL if args.dataset_path else (args.data_source or HubType.MODELSCOPE)
    if source not in {HubType.MODELSCOPE, HubType.HUGGINGFACE, HubType.LOCAL}:
        raise ValueError(f'Unsupported AgentX data source: {source}')

    if source == HubType.HUGGINGFACE:
        _verify_huggingface_revision(spec)
        return AgentXDatasetProvenance(
            source=source,
            dataset_id=spec['huggingface_id'],
            revision=spec['huggingface_revision'],
            trace_file='',
            sha256=spec['sha256'],
            verified=True,
        )

    if source == HubType.MODELSCOPE:
        root = download_dataset_snapshot(
            spec['modelscope_id'],
            data_source=HubType.MODELSCOPE,
            revision=spec['modelscope_revision'],
            allow_file_pattern='traces.jsonl',
        )
        dataset_id = spec['modelscope_id']
        revision = spec['modelscope_revision']
    else:
        root = download_dataset_snapshot(args.dataset_path, data_source=HubType.LOCAL)
        dataset_id = os.path.realpath(args.dataset_path)
        revision = None

    trace_file = Path(root) / 'traces.jsonl'
    if not trace_file.is_file():
        raise FileNotFoundError(f'AgentX dataset must contain traces.jsonl: {root}')
    digest = _sha256(trace_file)
    verified = digest == spec['sha256']
    if scenario.mode == 'benchmark' and not verified:
        raise ValueError(f'AgentX benchmark dataset hash mismatch for {trace_file}: {digest}')
    return AgentXDatasetProvenance(
        source=source,
        dataset_id=dataset_id,
        revision=revision,
        trace_file=str(trace_file),
        sha256=digest,
        verified=verified,
    )


def _verify_huggingface_revision(spec: Dict[str, str]) -> None:
    """Fail closed if the date-pinned public dataset head has moved."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(spec['huggingface_id'], revision=spec['huggingface_revision'], files_metadata=True)
    except Exception as e:
        raise RuntimeError('Unable to verify the AgentX Hugging Face dataset revision.') from e
    if info.sha != spec['huggingface_revision']:
        raise RuntimeError(
            f'AgentX Hugging Face revision changed: expected {spec["huggingface_revision"]}, got {info.sha}.'
        )
    trace = next((file for file in info.siblings if file.rfilename == 'traces.jsonl'), None)
    lfs_sha256 = trace.lfs.sha256 if trace is not None and trace.lfs is not None else None
    if lfs_sha256 != spec['sha256']:
        raise RuntimeError(
            f'AgentX Hugging Face traces.jsonl hash changed: expected {spec["sha256"]}, got {lfs_sha256}.'
        )


def _resolve_duration(args: 'Arguments', scenario: AgentXScenario) -> float:
    if args.duration is not None:
        return float(args.duration)
    return _SMOKE_DURATION if scenario.mode == 'smoke' else _DEFAULT_DURATION


def _build_command(
    args: 'Arguments',
    scenario: AgentXScenario,
    dataset: AgentXDatasetProvenance,
    concurrency: int,
    duration: float,
    seed: int,
    artifacts_path: Path,
) -> tuple[List[str], Dict[str, str]]:
    base_url = _split_chat_url(args.url)
    command = [
        sys.executable,
        '-m',
        'aiperf',
        'profile',
        '--scenario',
        'inferencex-agentx-mvp',
        '--url',
        base_url,
        '--endpoint-type',
        'chat',
        '--model',
        args.model,
        '--concurrency',
        str(concurrency),
        '--benchmark-duration',
        str(duration),
        '--random-seed',
        str(seed),
        '--use-server-token-count',
        '--artifact-dir',
        str(artifacts_path),
        '--ui',
        'simple',
    ]
    if args.tokenizer_path:
        command.extend(['--tokenizer', args.tokenizer_path])
    if scenario.tokenizer_trust_remote_code:
        command.append('--tokenizer-trust-remote-code')
    if scenario.tokenizer_revision:
        command.extend(['--tokenizer-revision', scenario.tokenizer_revision])
    if scenario.request_timeout_seconds is not None:
        command.extend(['--request-timeout-seconds', str(scenario.request_timeout_seconds)])
    if scenario.benchmark_grace_period is not None:
        command.extend(['--benchmark-grace-period', str(scenario.benchmark_grace_period)])
    if scenario.max_context_length:
        command.extend(['--max-context-length', str(scenario.max_context_length)])
    if scenario.mode == 'smoke':
        command.extend(['--num-dataset-entries', str(scenario.effective_trace_limit)])
    if dataset.source == HubType.HUGGINGFACE:
        command.extend(['--public-dataset', _DATASETS[scenario.variant]['public_dataset']])
    else:
        command.extend(['--hf-weka-dataset', str(Path(dataset.trace_file).parent)])
    if scenario.mode == 'smoke' or dataset.source != HubType.HUGGINGFACE:
        command.append('--unsafe-override')

    env = os.environ.copy()
    api_key = get_secret_value(args.api_key)
    if api_key:
        command.extend(['--api-key', str(api_key)])
    for key, value in args.headers.items():
        raw_value = str(get_secret_value(value))
        if api_key and key.lower() == 'authorization' and raw_value == f'Bearer {api_key}':
            continue
        command.extend(['--header', f'{key}:{raw_value}'])
    return command, env


def _run_one_agentx(
    command: List[str],
    env: Dict[str, str],
    scenario: AgentXScenario,
    dataset: AgentXDatasetProvenance,
    concurrency: int,
    duration: float,
    seed: int,
    run_path: Path,
    artifacts_path: Path,
) -> AgentXRunSummary:
    log_path = run_path / 'aiperf.log'
    redacted_command = _redact_command(command)
    status: Literal['completed', 'failed', 'cancelled'] = 'failed'
    process: Optional[subprocess.Popen[str]] = None
    fallback_error: Optional[str] = None
    try:
        with log_path.open('w', encoding='utf-8') as log_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
            return_code = process.wait()
        status = 'completed' if return_code == 0 else 'failed'
        if return_code != 0:
            fallback_error = f'AIPerf exited with return code {return_code}.'
    except KeyboardInterrupt:
        status = 'cancelled'
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    raw_summary = artifacts_path / _AIPERF_SUMMARY
    raw_data: Dict[str, Any] = {}
    if raw_summary.is_file():
        try:
            raw_data = json.loads(raw_summary.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            if status != 'cancelled':
                status = 'failed'
                fallback_error = 'AIPerf produced invalid profile_export_aiperf.json.'
    elif status != 'cancelled':
        status = 'failed'
        fallback_error = 'AIPerf did not produce profile_export_aiperf.json.'
    if raw_data.get('was_cancelled'):
        status = 'cancelled'
    summary = _normalize_summary(
        raw_data=raw_data,
        status=status,
        scenario=scenario,
        dataset=dataset,
        concurrency=concurrency,
        duration=duration,
        seed=seed,
        artifacts_path=artifacts_path,
        raw_summary=raw_summary if raw_summary.is_file() else None,
        command=redacted_command,
    )
    if fallback_error and not summary.error_summary:
        summary.error_summary = [{'message': fallback_error}]
    return summary


def _normalize_summary(
    raw_data: Dict[str, Any],
    status: Literal['completed', 'failed', 'cancelled'],
    scenario: AgentXScenario,
    dataset: AgentXDatasetProvenance,
    concurrency: int,
    duration: float,
    seed: int,
    artifacts_path: Path,
    raw_summary: Optional[Path],
    command: List[str],
) -> AgentXRunSummary:
    metadata = raw_data.get('metadata') or {}
    metrics = {
        name: AgentXMetric.model_validate(value)
        for name, value in raw_data.items()
        if isinstance(value, dict) and isinstance(value.get('unit'), str)
    }
    raw_valid = metadata.get('submission_valid')
    reasons = list(metadata.get('submission_invalid_reasons') or [])
    mirror_revalidated = (
        status == 'completed'
        and scenario.mode == 'benchmark'
        and dataset.source in {HubType.MODELSCOPE, HubType.LOCAL}
        and dataset.verified
        and raw_valid is False
        and reasons == ['unsafe_override']
    )
    submission_valid = bool(raw_valid) or mirror_revalidated
    if scenario.mode == 'smoke' or status != 'completed':
        submission_valid = False
        mirror_revalidated = False
    derived = _derive_per_gpu_metrics(metrics, scenario.num_gpus)
    return AgentXRunSummary(
        status=status,
        scenario=scenario,
        concurrency=concurrency,
        duration=duration,
        seed=seed,
        aiperf_version=raw_data.get('aiperf_version'),
        aiperf_schema_version=raw_data.get('schema_version'),
        dataset=dataset,
        submission_valid=submission_valid,
        aiperf_submission_valid=raw_valid if isinstance(raw_valid, bool) else None,
        aiperf_submission_invalid_reasons=reasons,
        mirror_revalidated=mirror_revalidated,
        metrics=metrics,
        derived_metrics=derived,
        error_summary=list(raw_data.get('error_summary') or []),
        raw_artifact_dir=str(artifacts_path),
        raw_summary_file=str(raw_summary) if raw_summary else None,
        command=command,
    )


def _derive_per_gpu_metrics(metrics: Dict[str, AgentXMetric], num_gpus: Optional[int]) -> Dict[str, AgentXMetric]:
    if not num_gpus:
        return {}
    derived: Dict[str, AgentXMetric] = {}
    for name in ('request_throughput', 'output_token_throughput'):
        metric = metrics.get(name)
        if metric is None or metric.avg is None:
            continue
        derived[f'{name}_per_gpu'] = AgentXMetric(
            unit=f'{metric.unit}/GPU',
            avg=metric.avg / num_gpus,
            source_metric=name,
            divisor=num_gpus,
            derived_by='evalscope',
        )
    return derived


def _split_chat_url(url: str) -> str:
    """Return the AIPerf base URL after validating the OpenAI chat endpoint."""
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError(f'AgentX requires an absolute OpenAI Chat Completions URL, got: {url}')
    expected = '/v1/chat/completions'
    if not parsed.path.rstrip('/').endswith(expected):
        raise ValueError(f'AgentX requires a URL ending with {expected}.')
    base_path = parsed.path.rstrip('/')[: -len(expected)]
    return urlunsplit((parsed.scheme, parsed.netloc, base_path, '', ''))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_command(command: List[str]) -> List[str]:
    redacted = []
    redact_next = False
    for part in command:
        if redact_next:
            redacted.append('<redacted>')
            redact_next = False
        elif part == '--api-key':
            redacted.append(part)
            redact_next = True
        elif ':' in part and part.split(':', 1)[0].lower() in {
            'authorization',
            'proxy-authorization',
            'x-api-key',
            'x-auth-token',
        }:
            redacted.append(f'{part.split(":", 1)[0]}:<redacted>')
        else:
            redacted.append(re.sub(r'\$\{EVALSCOPE_AGENTX_[^}]+\}', '<redacted>', part))
    return redacted


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


def _print_summary(results: Dict[str, Dict[str, Any]]) -> None:
    for name, result in results.items():
        metrics = result.get('metrics', {})
        highlights = []
        for metric_name in (
            'request_throughput',
            'output_token_throughput',
            'output_token_throughput_per_user',
            'time_to_first_token',
            'inter_token_latency',
            'request_latency',
            'request_count',
            'error_request_count',
        ):
            metric = metrics.get(metric_name)
            if metric and metric.get('avg') is not None:
                highlights.append(f'{metric_name}={metric["avg"]} {metric["unit"]}')
        logger.info(f'AgentX {name}: submission_valid={result["submission_valid"]}; ' + ', '.join(highlights))
