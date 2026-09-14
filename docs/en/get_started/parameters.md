# Parameters

Run `evalscope eval --help` to get the full list of parameters.

## Environment Variables

The following environment variables can be set before launch to control global default behavior:

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `EVALSCOPE_CACHE` | Root cache directory for EvalScope, used to store datasets, intermediate evaluation files, etc. | `~/.cache/evalscope` |
| `EVALSCOPE_LANGUAGE` | Global default language, affects output language for reports, etc. (`en` or `zh`) | `en` |
| `EVALSCOPE_HEARTBEAT_INTERVAL` | Heartbeat reporting interval (seconds) | `60` |
| `MODELSCOPE_CACHE` | Root cache directory for ModelScope models and datasets | `~/.cache/modelscope/hub` |
| `DATASET_TF_BATCH_SIZE` | Batch size for dataset transformation | `100` |

## Model Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--model` | `str` | Name of the model to be evaluated<br>• ModelScope model ID (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)<br>• Local model path (e.g., `/path/to/model`)<br>• Model ID for API service (e.g., `Qwen2.5-0.5B-Instruct`) | - |
| `--model-id` | `str` | Alias for the evaluated model, used in reports | Last part of `model` |
| `--api-url` | `str` | Model API endpoint, supports OpenAI-compatible and OpenAI Responses API roots<br>Example: `http://127.0.0.1:8000/v1` or `https://api.openai.com/v1` | `None` |
| `--api-key` | `str` | Model API endpoint key | `EMPTY` |
| `--model-args` | `str` | Model loading parameters, comma-separated `key=value` or JSON string<br>• `revision`: Model revision<br>• `precision`: Model precision<br>• `device_map`: Device allocation | `revision=master`<br>`precision=torch.float16`<br>`device_map=auto` |
| `--model-task` | `str` | Model task type | `text_generation`<br>(Options: `image_generation`) |
| `--chat-template` | `str` | Model inference template, supports Jinja template string | `None` (uses transformers default) |

**Example:**
```bash
# key=value format
--model-args revision=master,precision=torch.float16,device_map=auto

# JSON string format
--model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}'
```

## Model Inference Parameters

The `--generation-config` parameter supports the following options (comma-separated `key=value` or JSON string):

| Parameter | Type | Description | Supported Backends |
|-----------|------|-------------|--------------------|
| `timeout` | `int`/`float` | Request timeout (seconds) | All |
| `retries` | `int` | Number of retries, default is 5. | OpenAI-compatible |
| `retry_interval` | `int` | Retry interval (seconds), default is 10. | OpenAI-compatible |
| `stream` | `bool` | Whether to return responses in streaming mode | All |
| `max_tokens` | `int` | Maximum number of tokens generated | All |
| `top_p` | `float` | Nucleus sampling; only considers tokens accounting for top_p probability mass | All |
| `temperature` | `float` | Sampling temperature, range 0~2; higher means more randomness | All |
| `stop_seqs` | `list[str]` | Sequences that trigger stop generation; the returned text does not include the stop sequence | All |
| `frequency_penalty` | `float` | Range -2.0~2.0; positive values penalize repeated tokens | OpenAI-compatible |
| `presence_penalty` | `float` | Range -2.0~2.0; positive values penalize already appeared tokens | OpenAI-compatible |
| `repetition_penalty` | `float` | Exponential penalty applied to existing tokens. 1.0 means no penalty | OpenAI-compatible, HuggingFace, vLLM |
| `logit_bias` | `dict` | Mapping of token IDs to bias values (-100~100)<br>Example: `"42=10,43=-10"` | OpenAI-compatible |
| `seed` | `int` | Random seed | OpenAI-compatible |
| `do_sample` | `bool` | Whether to use sampling strategy (otherwise greedy decoding) | Transformers |
| `top_k` | `int` | Sample next token from the top_k most likely candidates | Anthropic, Google, HuggingFace, vLLM, SGLang |
| `logprobs` | `bool` | Whether to return log probabilities for output tokens | OpenAI-compatible, HuggingFace, llama-cpp-python |
| `top_logprobs` | `int` | Return the top N tokens and their probabilities (range 0~20) | OpenAI-compatible, HuggingFace |
| `parallel_tool_calls` | `bool` | Whether to support parallel tool calls | OpenAI, Groq |
| `response_schema` | `dict` | Request structured output (JSON Schema); the output still needs to be validated | OpenAI, Google, Mistral |
| `reasoning_effort` | `str` | Reasoning effort level, passed through to the server as-is (e.g. `none` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max`); the accepted values are decided by the model and the server | OpenAI-compatible |
| `reasoning_tokens` | `int` | Maximum tokens budget for reasoning (thinking budget) | Anthropic Claude |
| `reasoning_summary` | `str` | Reasoning summary verbosity. One of `concise` / `detailed` / `auto` | OpenAI reasoning series |
| `reasoning_history` | `str` | How to encode prior-turn assistant `reasoning_content` in multi-turn requests. One of `reasoning_field` (default; pass as independent top-level field, works for DeepSeek V4 thinking, Qwen3 thinking), `think_tag` (embed as `<think>...</think>` in content string, legacy Together/Groq compatible), `none` (strip entirely; **required for DeepSeek R1 legacy** which forbids `reasoning_content` in requests) | OpenAI-compatible |
| `extra_body` | `dict` | Extra request body for OpenAI-compatible services | OpenAI-compatible services |
| `extra_query` | `dict` | Extra query parameters for OpenAI-compatible services | OpenAI-compatible services |
| `extra_headers` | `dict` | Extra headers for OpenAI-compatible services | OpenAI-compatible services |
| `height` | `int` | For image generation models, specifies image height | Image generation models |
| `width` | `int` | For image generation models, specifies image width | Image generation models |
| `num_inference_steps` | `int` | For image models, number of inference steps | Image generation models |
| `guidance_scale` | `float` | For image models, guidance scale | Image generation models |

**Example:**
```bash
# key=value format
--generation-config do_sample=true,temperature=0.5

# JSON string format (supports more complex parameters)
--generation-config '{"do_sample":true,"temperature":0.5,"extra_body": {"chat_template_kwargs":{"enable_thinking": false}}}'
```

## Dataset Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--datasets` | `list[str]` | Dataset name list, space-separated<br>Refer to [Dataset List](./supported_dataset/llm.md) | - |
| `--dataset-dir` | `str` | Dataset download path | `~/.cache/modelscope/datasets` |
| `--dataset-hub` | `str` | Dataset source | `modelscope`<br>(Options: `huggingface`) |
| `--limit` | `int`/`float` | Maximum samples to evaluate per dataset<br>• int: First N samples<br>• float: First N% of samples<br>Example: `100` or `0.1` | `None` (evaluate all) |
| `--repeats` | `int` | Number of times to repeat inference on the same sample | `1` |
| `--dataset-args` | `str` | Dataset configuration parameters (JSON string), see table below | `{}` |

### dataset-args Configuration Options

`--dataset-args` is a JSON string; each dataset can be configured with the following parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_id` | `str` | ModelScope dataset ID or local path |
| `local_path` | `str` | Local dataset path, deprecated, please use `dataset_id` |
| `review_timeout` | `float` | Timeout for evaluation samples (seconds), recommended for code tasks |
| `prompt_template` | `str` | Prompt template, example: `Question: {query}\nAnswer:` |
| `system_prompt` | `str` | System prompt |
| `subset_list` | `list[str]` | List of dataset subsets to evaluate |
| `few_shot_num` | `int` | Number of few-shot examples |
| `few_shot_random` | `bool` | Whether to randomly sample few-shot data |
| `shuffle` | `bool` | Whether to shuffle the data |
| `shuffle_choices` | `bool` | Whether to shuffle choice order (multiple-choice only) |
| `metric_list` | `list[str\|dict]` | Metric list. Use canonical names such as `accuracy`; legacy aliases such as `acc` are normalized for compatibility. |
| `aggregation` | `str` | Aggregation method for evaluation results, default is `mean`. Options: `mean_and_pass_at_k`, `mean_and_vote_at_k`, `mean_and_pass_hat_k` (all require setting `repeats=k`).<br>• `pass_at_k`: Probability that the same sample passes at least once in k generations (e.g., set `repeats=5` for `humaneval`)<br>• `vote_at_k`: Scoring by voting on k results for the same sample<br>• `pass_hat_k`: Probability that the same sample passes all k times (e.g., set `repeats=3` for `tau2_bench`) |
| `filters` | `dict` | Output filters<br>• `remove_until`: Remove content before specified string<br>• `extract`: Extract regex-matched content |
| `force_redownload` | `bool` | Whether to force re-download the dataset |
| `extra_params` | `dict` | Dataset-related extra parameters, refer to [dataset documentation](./supported_dataset/index.md), specify `{<param_name>:<value>}` as needed, where the type (`type`) and choices (`choices`) of `value` depend on the specific parameter. For SWE-bench agentic and similar benchmarks, see [Agent Evaluation](../user_guides/agent/native.md#swe-bench-agentic-benchmarks) |
| `sandbox_config` | `dict` | Sandbox configuration (see Sandbox Parameters below) |

**sandbox_config Options:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `image` | `str` | Docker image name | `python:3.11-slim` |
| `network_enabled` | `bool` | Whether to enable networking | `true` |
| `tools_config` | `dict` | Tool configuration dictionary | `{'shell_executor': {}, 'python_executor': {}}` |

**Example:**
```bash
--datasets gsm8k arc ifeval hle \
--dataset-args '{
  "gsm8k": {
    "few_shot_num": 4,
    "few_shot_random": false
  },
  "arc": {
    "dataset_id": "/path/to/arc"
  },
  "ifeval": {
    "filters": {
      "remove_until": "</think>"
    }
  },
  "hle": {
    "extra_params": {
      "include_multi_modal": false
    }
  }
}'
```

## Evaluation Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--eval-type` | `str` | Evaluation type<br>• `llm_ckpt`: Local model inference (transformers)<br>• `openai_api`: OpenAI-compatible Chat Completions API service<br>• `openai_responses_api`: OpenAI official Responses API service<br>• `anthropic_api`: Anthropic Claude API service<br>• `litellm`: LiteLLM multi-provider routing (supports 100+ LLM providers)<br>• `text2image`: Text-to-image model (local diffusers or an OpenAI-compatible Images API service)<br>• `text2speech`: Text-to-speech model service<br>• `image_editing`: Image editing model<br>• `mock_llm`: Simulated inference (for verification)<br>• `custom`: Custom evaluation type | `None` (auto-detect) |
| `--eval-batch-size` | `int` | Evaluation batch size, applies to the following stages:<br>• Inference: concurrent requests (remote API modes) or batch size (`llm_ckpt` mode)<br>• LLM-judge review: number of concurrent threads<br>• `batch_calculate_metrics`: number of samples per batch window | `1` (`8` for remote API modes: `openai_api`, `openai_responses_api`, `anthropic_api`, `litellm`) |
| `--eval-backend` | `str` | Evaluation backend<br>• `Native`: Default backend<br>• `OpenCompass`: LLM evaluation<br>• `VLMEvalKit`: Multimodal model evaluation<br>• `RAGEval`: RAG/Embedding/Reranker/CLIP evaluation<br>• `ThirdParty`: Special task evaluation | `Native` |
| `--eval-config` | `str` | Configuration file path for non-Native backends | - |

```{seealso}
Refer to the [other backend usage guide](../user_guides/backend/index.md)
```

## Judge Parameters

Native LLM judging is configured through one typed `judge` object. In Python/YAML use `judge={...}`; on the CLI use
`--judge '<JSON object>'`.

```python
TaskConfig(
    model='MODEL',
    datasets=['simple_qa'],
    judge={
        'strategy': 'llm',
        'models': {
            'model_id': 'JUDGE_MODEL',
            'api_url': 'OPENAI_COMPATIBLE_URL',
            'api_key': 'JUDGE_API_KEY',
            'generation_config': {'temperature': 0.0, 'retries': 3},
        },
        'repeats': 1,
        'position_swap': 'auto',
        'aggregation': 'mean',
        'min_valid_judges': 1,
    },
)
```

`models` accepts one object or a list of objects. A list enables independent judges; every entry needs a unique
`judge_id` when the same `model_id` occurs more than once. `judge_id` defaults to a unique `model_id`.

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `strategy` | `auto\|rule\|llm\|llm_recall` | `auto` follows benchmark policy; `llm_recall` judges only rule-based misses and takes `max(rule, judge)`. | `auto` |
| `models` | `object\|list[object]` | One or more Judge model configurations. `model_id` is required for reproducible review caching. | `[]` |
| `repeats` | `int >= 1` | Independent verdict observations per Judge, distinct from transport retries. | `1` |
| `position_swap` | `auto\|on\|off` | `auto` preserves the benchmark's official position-swap policy. | `auto` |
| `aggregation` | `mean\|median\|majority_vote` | Cross-observation aggregation for ordinary metrics. | `mean` |
| `min_valid_judges` | `int >= 1` | Minimum valid Judge verdicts required for a metric. | `1` |

Each entry in `models` supports `judge_id`, `model_id`, `api_key`, `api_url`, `eval_type`, `model_args`, and
`generation_config`. Provider-specific model initialization options belong in `model_args`; transport retry belongs
in `generation_config.retries`.

`judge.contract` configures the generic single-verdict judge only: `system_prompt`, `prompt_template`,
`score_mapping`, and `score_type`. `pattern` asks the Judge for a JSON verdict label selected from `score_mapping`;
`numeric` asks for a JSON score in `[0, 1]`. The framework appends the JSON-format instruction, parses the normal
model response once, and never uses constrained decoding, regex score extraction, or corrective follow-up prompts.
An invalid reply is unavailable and excluded from the metric rather than reported as zero.

Reports include `JudgeSummary` with coverage, failure counts, and disagreement for samples reviewed by an LLM.
When an adapter resolves a sample through a deterministic judge short-circuit, its score metadata records
`judge_skipped=true` and `judge_skip_reason`; the web review panel labels it as rule-based scoring instead of an
LLM verdict. Native evaluations require an exact cached evaluation identity match before reusing predictions and
reviews. With `rerun_review=True`, predictions are reused and the review cache is atomically replaced after
success; it is the explicit override for an identity mismatch, and the generated config records the prediction
source under the current evaluation version.

`judge_strategy` and a single mapping `judge_model_args` remain accepted only as a deprecated input migration. The
removed `judge_worker_num` and `score_pattern` are rejected.

## Sandbox Parameters

EvalScope manages sandbox settings via the nested `--sandbox` configuration (mapped to `SandboxTaskConfig`).

### --sandbox Options

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `enabled` | `bool` | Whether to enable the sandbox | `false` |
| `engine` | `str` | Sandbox engine: `docker`, `volcengine`, etc. | `docker` |
| `default_config` | `dict` | Task-level sandbox config; merged with `BenchmarkMeta.sandbox_config`, and used as the default per-sample environment config in Agent mode | `{}` |
| `manager_config` | `dict` | Forwarded to the ms_enclave manager (e.g. `base_url` for remote docker, volcengine credentials) | `{}` |
| `pool_size` | `int \| None` | Warmup pool size for pooled execution; falls back to `eval_batch_size` when `None` | `None` |

For full usage including local and remote manager examples, see [Sandbox Environment Usage](../user_guides/sandbox.md).

## Agent Parameters

`--agent-config` / `agent_config` enables [Agent Evaluation](../user_guides/agent/index.md): once set, all benchmarks based on `DefaultDataAdapter` switch to [native AgentLoop](../user_guides/agent/native.md) inference, or delegate to a third-party CLI such as Claude Code / Codex via the [external Agent Bridge](../user_guides/agent/bridge.md). `AgentLoopAdapter` subclasses (such as `swe_bench_*_agentic`) keep their benchmark defaults but accept supported explicit overrides such as strategy, step limit, and tools.

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--agent-config` | `dict \| NativeAgentConfig` | Global Agent configuration (see below) | `None` (Agent mode disabled) |

### agent-config Options

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `strategy` | `str` | Strategy name: `function_calling` / `react` / `swe_bench_toolcall` / `swe_bench_backticks` | `function_calling` |
| `tools` | `list[str]` | Tool whitelist: `bash` / `python_exec` (`submit` is auto-injected by the strategy) | `[]` |
| `environment` | `str \| None` | Agent command environment, such as `local` or `docker` | `None` |
| `environment_extra` | `dict` | Agent environment constructor options; Docker images belong in `sandbox_config.image` | `{}` |
| `max_steps` | `int` | Hard upper bound of loop iterations | `10` |
| `kwargs` | `dict` | Strategy constructor kwargs, e.g. `{'system_prompt': '...'}` | `{}` |

```{seealso}
For full usage, examples and Trace visualization, see [Agent Evaluation](../user_guides/agent/index.md).
```

## Other Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--work-dir` | `str` | Evaluation output path (see directory structure below) | `./outputs` |
| `--no-timestamp` | `bool` | Do not add timestamp to work_dir | `false` |
| `--use-cache` | `str` | Reuse local cache path (e.g., `outputs/20241210_194434`)<br>Reuses inference and evaluation results | `None` |
| `--rerun-review` | `bool` | Used with `--use-cache`: re-runs review/scoring from cached predictions and atomically replaces the review cache after success | `false` |
| `--enable-progress-tracker` | `bool` | Whether to enable progress tracking, writing hierarchical evaluation progress to `progress.json` in real time, queryable via the service API | `false` |
| `--collect-perf` | `bool` | Collect per-request performance metrics (latency, TTFT, token usage) and write them into the evaluation report. TTFT requires `--generation-config stream=true`. Use `--no-collect-perf` to disable | `true` |
| `--seed` | `int` | Random seed | `42` |
| `--debug` | `bool` | Whether to enable debug mode | `false` |
| `--ignore-errors` | `bool` | Whether to ignore errors during generation | `false` |
| `--dry-run` | `bool` | Dry run to check parameters without executing inference | `false` |

### work-dir Directory Structure Example

```text
./outputs/{timestamp}/
├── configs/
│   └── task_config_b6f42c.yaml      # Task configuration
├── logs/
│   └── eval_log.log                 # Evaluation log
├── predictions/
│   └── {model_id}/
│       └── {dataset}.jsonl          # Model inference results
├── reports/
│   └── {model_id}/
│       └── {dataset}.json           # Evaluation report
├── reviews/
│   └── {model_id}/
│       └── {dataset}.jsonl          # Evaluation result details
└── progress.json                    # Progress tracking file (generated when --enable-progress-tracker is enabled)
```

Example `progress.json` format:

```json
{
  "status": "running",
  "pipeline": "eval",
  "total_count": 14042,
  "processed_count": 5200,
  "percent": 37.03,
  "stage": {
    "name": "Evaluating", "label": "mmlu",
    "current": 1, "total": 3, "status": "running",
    "children": [
      {"name": "Predicting", "label": "mmlu@test", "current": 1000, "total": 1000, "status": "completed", "children": []},
      {"name": "Reviewing",  "label": "mmlu@test", "current": 320,  "total": 1000, "status": "running",  "children": []}
    ]
  },
  "updated_at": "2026-03-09T10:05:42Z"
}
```
