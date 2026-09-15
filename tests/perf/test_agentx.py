import json
from pathlib import Path

import pytest

from evalscope.constants import HubType
from evalscope.perf.arguments import Arguments
from evalscope.perf.scenarios import agentx as agentx_module
from evalscope.perf.scenarios.agentx import (
    AgentXDatasetProvenance,
    AgentXScenario,
    _build_command,
    _normalize_summary,
    _redact_command,
    _resolve_dataset,
    _run_one_agentx,
)


def _args(**kwargs) -> Arguments:
    kwargs.setdefault('tokenizer_path', 'test-tokenizer')
    return Arguments(
        model='test-model',
        url='http://localhost:8080/v1/chat/completions',
        **kwargs,
    )


class TestAgentXArguments:

    def test_shorthand_uses_benchmark_defaults(self):
        args = _args(scenario='agentx')

        assert args.scenario == AgentXScenario()
        assert args.parallel == [1]

    def test_json_scenario_is_parsed(self):
        args = _args(scenario='{"name":"agentx","mode":"smoke","trace_limit":2,"num_gpus":8}')

        assert args.scenario.mode == 'smoke'
        assert args.scenario.trace_limit == 2
        assert args.scenario.num_gpus == 8

    def test_request_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match='greater than 0'):
            AgentXScenario(request_timeout_seconds=0)

    @pytest.mark.parametrize(
        'kwargs, message',
        [
            ({'scenario': 'agentx', 'number': 1}, '--number'),
            ({'scenario': 'agentx', 'open_loop': True}, '--open-loop'),
            ({'scenario': 'agentx', 'warmup_num': 1}, '--warmup-num'),
            ({'scenario': 'agentx', 'duration': 60}, '>= 900'),
            ({'scenario': 'agentx', 'tokenizer_path': None}, '--tokenizer-path'),
        ],
    )
    def test_conflicting_perf_flags_are_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            _args(**kwargs)


class TestAgentXCommand:

    @pytest.mark.parametrize('grace', [None, 0, 600.5])
    @pytest.mark.parametrize('mode', ['smoke', 'benchmark'])
    def test_benchmark_grace_period_is_forwarded(
        self, tmp_path: Path, grace: float | None, mode: str
    ) -> None:
        args = _args(scenario=json.dumps({
            'name': 'agentx', 'mode': mode, 'benchmark_grace_period': grace,
            'request_timeout_seconds': 90.5,
        }))
        dataset = AgentXDatasetProvenance(
            source=HubType.LOCAL, dataset_id='test', revision=None,
            trace_file=str(tmp_path / 'traces.jsonl'), sha256='digest', verified=True,
        )
        command, _ = _build_command(
            args, args.scenario, dataset, concurrency=1, duration=1800,
            seed=20260707, artifacts_path=tmp_path,
        )
        assert command.count('--benchmark-grace-period') == int(grace is not None)
        if grace is not None:
            assert float(command[command.index('--benchmark-grace-period') + 1]) == grace
        assert command[command.index('--request-timeout-seconds') + 1] == '90.5'
        assert command[command.index('--benchmark-duration') + 1] == '1800'

    @pytest.mark.parametrize('grace', [-1, float('inf'), float('nan')])
    def test_benchmark_grace_period_rejects_invalid_values(self, grace: float) -> None:
        with pytest.raises(ValueError):
            AgentXScenario(benchmark_grace_period=grace)

    @pytest.mark.parametrize(
        'scenario, expected',
        [
            ('agentx', False),
            ('{"name":"agentx","tokenizer_trust_remote_code":false}', False),
            ('{"name":"agentx","tokenizer_trust_remote_code":true}', True),
        ],
    )
    def test_tokenizer_trust_is_opt_in(self, tmp_path: Path, scenario: str, expected: bool) -> None:
        args = _args(scenario=scenario)
        dataset = AgentXDatasetProvenance(
            source=HubType.LOCAL,
            dataset_id='test',
            revision=None,
            trace_file=str(tmp_path / 'traces.jsonl'),
            sha256='digest',
            verified=True,
        )
        command, _ = _build_command(
            args, args.scenario, dataset, concurrency=1, duration=1800,
            seed=20260707, artifacts_path=tmp_path,
        )
        assert args.scenario.tokenizer_trust_remote_code is expected
        assert command.count('--tokenizer-trust-remote-code') == int(expected)

    @pytest.mark.parametrize(
        'scenario, expected_revision, expected_timeout',
        [
            ('agentx', None, None),
            (
                '{"name":"agentx","tokenizer_revision":"v1.0.0","request_timeout_seconds":90.5}',
                'v1.0.0',
                90.5,
            ),
        ],
    )
    def test_tokenizer_revision_and_request_timeout_are_forwarded(
        self,
        tmp_path: Path,
        scenario: str,
        expected_revision: str | None,
        expected_timeout: float | None,
    ) -> None:
        args = _args(scenario=scenario)
        dataset = AgentXDatasetProvenance(
            source=HubType.LOCAL,
            dataset_id='test',
            revision=None,
            trace_file=str(tmp_path / 'traces.jsonl'),
            sha256='digest',
            verified=True,
        )
        command, _ = _build_command(
            args, args.scenario, dataset, concurrency=1, duration=1800, seed=20260707, artifacts_path=tmp_path
        )

        assert args.scenario.tokenizer_revision == expected_revision
        assert args.scenario.request_timeout_seconds == expected_timeout
        assert command.count('--tokenizer-revision') == int(expected_revision is not None)
        assert command.count('--request-timeout-seconds') == int(expected_timeout is not None)
        if expected_revision is not None:
            assert command[command.index('--tokenizer-revision') + 1] == expected_revision
        if expected_timeout is not None:
            assert command[command.index('--request-timeout-seconds') + 1] == str(expected_timeout)

    def test_local_mirror_command_uses_weka_hf_and_redacts_secrets(self, tmp_path):
        args = _args(scenario='agentx', api_key='secret-token', headers={'X-API-Key': 'other-secret'})
        dataset = AgentXDatasetProvenance(
            source=HubType.MODELSCOPE,
            dataset_id='evalscope/cc-traces-weka-062126-256k',
            revision='revision',
            trace_file=str(tmp_path / 'traces.jsonl'),
            sha256='digest',
            verified=True,
        )

        command, _ = _build_command(
            args, args.scenario, dataset, concurrency=8, duration=1800, seed=20260707, artifacts_path=tmp_path
        )

        assert '--hf-weka-dataset' in command
        assert '--unsafe-override' in command
        assert command.count('--unsafe-override') == 1
        assert command[command.index('--url') + 1] == 'http://localhost:8080'
        assert '--endpoint' not in command
        assert 'secret-token' in command
        assert 'other-secret' in ' '.join(command)
        assert 'secret-token' not in ' '.join(_redact_command(command))
        assert 'other-secret' not in ' '.join(_redact_command(command))

    def test_huggingface_command_uses_official_alias(self, tmp_path):
        args = _args(scenario='agentx', data_source='huggingface')
        dataset = AgentXDatasetProvenance(
            source=HubType.HUGGINGFACE,
            dataset_id='semianalysisai/cc-traces-weka-062126-256k',
            revision='revision',
            trace_file='',
            sha256='digest',
            verified=True,
        )

        command, _ = _build_command(
            args, args.scenario, dataset, concurrency=1, duration=1800, seed=20260707, artifacts_path=tmp_path
        )

        assert 'semianalysis_cc_traces_weka_062126_256k' in command
        assert '--hf-weka-dataset' not in command


class TestAgentXNormalization:

    def _dataset(self) -> AgentXDatasetProvenance:
        return AgentXDatasetProvenance(
            source=HubType.MODELSCOPE,
            dataset_id='mirror',
            revision='revision',
            trace_file='traces.jsonl',
            sha256='digest',
            verified=True,
        )

    def test_verified_mirror_revalidates_only_unsafe_override(self, tmp_path):
        raw = {
            'aiperf_version': '0.12.0',
            'schema_version': '1.5',
            'metadata': {'submission_valid': False, 'submission_invalid_reasons': ['unsafe_override']},
            'output_token_throughput': {'unit': 'tokens/sec', 'avg': 80.0},
            'time_to_first_token': {'unit': 'ms', 'p90': 12.0},
        }

        summary = _normalize_summary(
            raw, 'completed', AgentXScenario(num_gpus=8), self._dataset(), 8, 1800, 20260707, tmp_path, None, []
        )

        assert summary.submission_valid is True
        assert summary.aiperf_submission_valid is False
        assert summary.mirror_revalidated is True
        assert summary.derived_metrics['output_token_throughput_per_gpu'].avg == 10
        assert summary.metrics['time_to_first_token'].avg is None

    def test_verified_mirror_preserves_upstream_validity(self, tmp_path):
        summary = _normalize_summary(
            {'metadata': {'submission_valid': True}},
            'completed',
            AgentXScenario(),
            self._dataset(),
            1,
            1800,
            20260707,
            tmp_path,
            None,
            [],
        )

        assert summary.submission_valid is True
        assert summary.aiperf_submission_valid is True
        assert summary.mirror_revalidated is False

    def test_smoke_and_runtime_failure_are_never_valid(self, tmp_path):
        raw = {'metadata': {'submission_valid': False, 'submission_invalid_reasons': ['unsafe_override']}}
        smoke = _normalize_summary(
            raw, 'completed', AgentXScenario(mode='smoke'), self._dataset(), 1, 60, 20260707, tmp_path, None, []
        )
        failed = _normalize_summary(
            raw, 'failed', AgentXScenario(), self._dataset(), 1, 1800, 20260707, tmp_path, None, []
        )

        assert smoke.submission_valid is False
        assert failed.submission_valid is False


class TestAgentXProcessLifecycle:

    def test_failed_run_without_aiperf_summary_has_diagnostic(self, monkeypatch, tmp_path):
        class CompletedStdout:

            def __iter__(self):
                return iter(())

        class FailedProcess:

            def __init__(self):
                self.stdout = CompletedStdout()

            def wait(self, timeout=None):
                return 1

        process = FailedProcess()
        monkeypatch.setattr(agentx_module.subprocess, 'Popen', lambda *args, **kwargs: process)
        dataset = AgentXDatasetProvenance(
            source=HubType.MODELSCOPE,
            dataset_id='mirror',
            revision='revision',
            trace_file='traces.jsonl',
            sha256='digest',
            verified=True,
        )

        summary = _run_one_agentx(
            command=['aiperf'],
            env={},
            scenario=AgentXScenario(mode='smoke'),
            dataset=dataset,
            concurrency=1,
            duration=60,
            seed=20260707,
            run_path=tmp_path,
            artifacts_path=tmp_path / 'aiperf',
        )

        assert summary.status == 'failed'
        assert summary.error_summary == [{'message': 'AIPerf did not produce profile_export_aiperf.json.'}]

    def test_interrupted_run_without_aiperf_summary_is_cancelled(self, monkeypatch, tmp_path):
        class InterruptedStdout:

            def __iter__(self):
                raise KeyboardInterrupt
                yield ''

        class InterruptedProcess:

            def __init__(self):
                self.stdout = InterruptedStdout()
                self.signal = None

            def poll(self):
                return None

            def send_signal(self, signal):
                self.signal = signal

            def wait(self, timeout=None):
                return -2

        process = InterruptedProcess()
        monkeypatch.setattr(agentx_module.subprocess, 'Popen', lambda *args, **kwargs: process)
        dataset = AgentXDatasetProvenance(
            source=HubType.MODELSCOPE,
            dataset_id='mirror',
            revision='revision',
            trace_file='traces.jsonl',
            sha256='digest',
            verified=True,
        )

        summary = _run_one_agentx(
            command=['aiperf'],
            env={},
            scenario=AgentXScenario(mode='smoke'),
            dataset=dataset,
            concurrency=1,
            duration=60,
            seed=20260707,
            run_path=tmp_path,
            artifacts_path=tmp_path / 'aiperf',
        )

        assert summary.status == 'cancelled'
        assert summary.raw_summary_file is None
        assert process.signal == agentx_module.signal.SIGINT


class TestAgentXDatasetValidation:

    def test_unverified_local_data_is_allowed_only_for_smoke(self, tmp_path):
        (tmp_path / 'traces.jsonl').write_text(json.dumps({'not': 'a real trace'}) + '\n', encoding='utf-8')
        smoke_args = _args(scenario='{"name":"agentx","mode":"smoke"}', dataset_path=str(tmp_path))
        result = _resolve_dataset(smoke_args, smoke_args.scenario)

        assert result.verified is False
        benchmark_args = _args(scenario='agentx', dataset_path=str(tmp_path))
        with pytest.raises(ValueError, match='hash mismatch'):
            _resolve_dataset(benchmark_args, benchmark_args.scenario)
