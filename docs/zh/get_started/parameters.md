# 参数说明

执行 `evalscope eval --help` 可获取全部参数说明。

## 环境变量

以下环境变量可在启动前设置，用于控制全局默认行为：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `EVALSCOPE_CACHE` | EvalScope 缓存根目录，用于存放数据集、评测中间文件等 | `~/.cache/evalscope` |
| `EVALSCOPE_LANGUAGE` | 全局默认语言，影响报告等输出语言（`en` 或 `zh`） | `en` |
| `EVALSCOPE_HEARTBEAT_INTERVAL` | 评测进度心跳上报间隔（秒） | `60` |
| `MODELSCOPE_CACHE` | ModelScope 模型与数据集缓存根目录 | `~/.cache/modelscope/hub` |
| `DATASET_TF_BATCH_SIZE` | 数据集转换的批处理大小 | `100` |

## 模型参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--model` | `str` | 被评测的模型名称<br>• ModelScope模型ID（如`Qwen/Qwen2.5-0.5B-Instruct`）<br>• 本地模型路径（如`/path/to/model`）<br>• API服务的模型ID（如`Qwen2.5-0.5B-Instruct`） | - |
| `--model-id` | `str` | 模型别名，用于报告展示 | `model`的最后一部分 |
| `--api-url` | `str` | 模型API端点，支持OpenAI兼容格式和OpenAI Responses API根路径<br>示例：`http://127.0.0.1:8000/v1` 或 `https://api.openai.com/v1` | `None` |
| `--api-key` | `str` | 模型API端点密钥 | `EMPTY` |
| `--model-args` | `str` | 模型加载参数，逗号分隔的`key=value`或JSON字符串<br>• `revision`: 模型版本<br>• `precision`: 模型精度<br>• `device_map`: 设备分配 | `revision=master`<br>`precision=torch.float16`<br>`device_map=auto` |
| `--model-task` | `str` | 模型任务类型 | `text_generation`<br>（可选：`image_generation`） |
| `--chat-template` | `str` | 模型推理模板，支持Jinja模板字符串（仅本地推理支持该参数） | `None`（使用transformers默认） |

**示例：**
```bash
# key=value形式
--model-args revision=master,precision=torch.float16,device_map=auto

# JSON字符串形式
--model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}'
```

## 模型推理参数

`--generation-config` 参数支持以下配置项（逗号分隔的`key=value`或JSON字符串）：

| 参数 | 类型 | 说明 | 支持的后端 |
|------|------|------|------------|
| `timeout` | `int`/`float` | 请求超时时间（秒） | 所有 |
| `retries` | `int` | 重试次数，默认为5 | OpenAI兼容 |
| `retry_interval` | `int` | 重试间隔时间（秒），默认10 | OpenAI兼容 |
| `stream` | `bool` | 是否流式返回响应 | 所有 |
| `max_tokens` | `int` | 最大生成token数量 | 所有 |
| `top_p` | `float` | Nucleus采样，考虑概率质量为top_p的token | 所有 |
| `temperature` | `float` | 采样温度，范围0~2，越高越随机 | 所有 |
| `stop_seqs` | `list[str]` | 触发停止生成的序列列表，返回文本不包含该序列 | 所有 |
| `frequency_penalty` | `float` | 范围-2.0~2.0，正值惩罚重复token | OpenAI兼容 |
| `presence_penalty` | `float` | 范围-2.0~2.0，正值惩罚已出现token | OpenAI兼容 |
| `repetition_penalty` | `float` | 对已生成token施加指数惩罚，1.0 表示不惩罚 | OpenAI兼容、HuggingFace、vLLM |
| `logit_bias` | `dict` | token id到偏置值的映射（-100~100）<br>示例：`"42=10,43=-10"` | OpenAI兼容 |
| `seed` | `int` | 随机种子 | OpenAI兼容 |
| `do_sample` | `bool` | 是否采用采样策略（否则贪婪解码） | Transformers |
| `top_k` | `int` | 从top_k最可能的词中采样 | Anthropic、Google、HuggingFace、vLLM、SGLang |
| `logprobs` | `bool` | 是否返回输出token的对数概率 | OpenAI兼容、HuggingFace、llama-cpp-python |
| `top_logprobs` | `int` | 返回概率最高的前N个token（范围0~20） | OpenAI兼容、HuggingFace |
| `parallel_tool_calls` | `bool` | 工具调用是否支持并行 | OpenAI、Groq |
| `response_schema` | `dict` | 请求结构化输出（JSON Schema），仍需对输出做校验 | OpenAI、Google、Mistral |
| `reasoning_effort` | `str` | reasoning 努力程度，原样透传给服务端（如 `none` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max`），合法取值由具体模型和服务端决定 | OpenAI兼容 |
| `reasoning_tokens` | `int` | reasoning 最大 token 预算（thinking budget） | Anthropic Claude |
| `reasoning_summary` | `str` | reasoning 摘要级别，可选 `concise` / `detailed` / `auto` | OpenAI reasoning 系列 |
| `reasoning_history` | `str` | 多轮对话中如何编码上一轮 assistant 的 `reasoning_content`。可选值：`reasoning_field`（默认，作为独立顶层字段透传，适配 DeepSeek V4 thinking、Qwen3 thinking 等）、`think_tag`（编码为 `<think>...</think>` 塞进 content 字符串，兼容旧版 Together / Groq 等部署）、`none`（完全剥离，DeepSeek R1 等禁止回传 `reasoning_content` 的 legacy 模型必须显式设此值） | OpenAI兼容 |
| `extra_body` | `dict` | 向OpenAI兼容服务发送的额外请求体 | OpenAI兼容服务 |
| `extra_query` | `dict` | 向OpenAI兼容服务发送的额外查询参数 | OpenAI兼容服务 |
| `extra_headers` | `dict` | 向OpenAI兼容服务发送的额外请求头 | OpenAI兼容服务 |
| `height` | `int` | 图像生成模型专用，指定图像高度 | 图像生成模型 |
| `width` | `int` | 图像生成模型专用，指定图像宽度 | 图像生成模型 |
| `num_inference_steps` | `int` | 图像生成模型专用，推理步数 | 图像生成模型 |
| `guidance_scale` | `float` | 图像生成模型专用，指导尺度 | 图像生成模型 |

**示例：**
```bash
# key=value形式
--generation-config do_sample=true,temperature=0.5

# JSON字符串形式（支持更复杂参数）
--generation-config '{"do_sample":true,"temperature":0.5,"extra_body": {"chat_template_kwargs":{"enable_thinking": false}}}'
```

## 数据集参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--datasets` | `list[str]` | 数据集名称列表，空格分隔<br>参考[数据集列表](./supported_dataset/llm.md) | - |
| `--dataset-dir` | `str` | 数据集下载路径 | `~/.cache/modelscope/datasets` |
| `--dataset-hub` | `str` | 数据集下载源 | `modelscope`<br>（可选：`huggingface`） |
| `--limit` | `int`/`float` | 每个数据集最大评测数据量<br>• int：评测前N条数据<br>• float：评测前N%数据<br>示例：`100`或`0.1` | `None`（全部评测） |
| `--repeats` | `int` | 重复推理一个样例多次 | `1` |
| `--dataset-args` | `str` | 数据集配置参数（JSON字符串），详见下表 | `{}` |

### dataset-args 配置项

`--dataset-args` 为JSON字符串，每个数据集可配置以下参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `dataset_id` | `str` | 数据集modelscope id/本地路径 |
| `local_path` | `str` | 本地数据集路径，已废弃，请使用`dataset_id` |
| `review_timeout` | `float` | 评测样本超时时间（秒），代码类任务建议设置 |
| `prompt_template` | `str` | Prompt模板，示例：`Question: {query}\nAnswer:` |
| `system_prompt` | `str` | 系统prompt |
| `subset_list` | `list[str]` | 评测数据子集列表 |
| `few_shot_num` | `int` | few-shot示例数量 |
| `few_shot_random` | `bool` | 是否随机采样few-shot数据 |
| `shuffle` | `bool` | 是否打乱数据 |
| `shuffle_choices` | `bool` | 是否打乱选项顺序（仅多选题） |
| `metric_list` | `list[str\|dict]` | 指标列表。应使用 `accuracy` 等规范名称；`acc` 等旧别名仅为兼容用途并会被规范化。 |
| `aggregation` | `str` | 评测结果聚合方式，默认`mean`。可选：`mean_and_pass_at_k`、`mean_and_vote_at_k`、`mean_and_pass_hat_k`（均需设置`repeats=k`）。<br>• `pass_at_k`：同一样例生成k次至少一次通过的概率（如`humaneval`设`repeats=5`）<br>• `vote_at_k`：对同一样例k次结果投票后计分<br>• `pass_hat_k`：同一样例k次全部通过的概率（如`tau2_bench`设`repeats=3`） |
| `filters` | `dict` | 输出过滤器<br>• `remove_until`: 过滤指定字符串之前的内容<br>• `extract`: 提取正则匹配的内容 |
| `force_redownload` | `bool` | 是否强制重新下载数据集 |
| `extra_params` | `dict` | 数据集相关的**额外参数**，具体参考[各数据集说明](./supported_dataset/index.md)，指定`{<param_name>:<value>}`即可, `value`的类型(`type`)和选择范围(`choices`)根据具体参数而定。SWE-bench agentic 等基准的扩展参数请参见 [Agent 评测](../user_guides/agent/native.md#用例swe-bench-agentic) |
| `sandbox_config` | `dict` | Sandbox配置（详见下方Sandbox参数） |

**sandbox_config 配置项：**

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `image` | `str` | Docker镜像名称 | `python:3.11-slim` |
| `network_enabled` | `bool` | 是否启用网络 | `true` |
| `tools_config` | `dict` | 工具配置字典 | `{'shell_executor': {}, 'python_executor': {}}` |

**示例：**
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

## 评测参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--eval-type` | `str` | 评测类型<br>• `llm_ckpt`: 本地模型推理（transformers）<br>• `openai_api`: OpenAI兼容Chat Completions API服务<br>• `openai_responses_api`: OpenAI官方Responses API服务<br>• `anthropic_api`: Anthropic Claude API服务<br>• `litellm`: LiteLLM多厂商路由（支持100+ LLM服务商）<br>• `text2image`: 文本转图像模型（本地 diffusers 或 OpenAI 兼容 Images API 服务）<br>• `text2speech`: 文本转语音模型服务<br>• `image_editing`: 图像编辑模型<br>• `mock_llm`: 模拟推理（功能验证）<br>• `custom`: 自定义评测类型 | `None`（自动判断） |
| `--eval-batch-size` | `int` | 评测批量大小，作用于以下阶段：<br>• 推理阶段：并发请求数（远程 API 模式）或批量大小（`llm_ckpt`模式）<br>• LLM-judge 评审阶段：并发线程数<br>• batch_calculate_metrics 阶段：每批次处理的样本数 | `1`（`openai_api`、`openai_responses_api`、`anthropic_api`、`litellm` 等远程 API 模式为`8`） |
| `--eval-backend` | `str` | 评测后端<br>• `Native`: 默认后端<br>• `OpenCompass`: 大语言模型评测<br>• `VLMEvalKit`: 多模态模型评测<br>• `RAGEval`: RAG/Embedding/Reranker/CLIP评测<br>• `ThirdParty`: 特殊任务评测 | `Native` |
| `--eval-config` | `str` | 非Native后端的配置文件路径 | - |

## Judge参数

Native LLM Judge 通过一个 typed `judge` 对象配置：Python/YAML 使用 `judge={...}`，CLI 使用
`--judge '<JSON object>'`。

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

`models` 可传单个对象或对象列表。列表表示独立 Judge；重复的 `model_id` 必须显式指定不同的
`judge_id`，唯一 `model_id` 默认同时作为 `judge_id`。

| 字段 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `strategy` | `auto\|rule\|llm\|llm_recall` | `auto` 遵循 benchmark 策略；`llm_recall` 仅复核规则漏判，并取 `max(rule, judge)`。 | `auto` |
| `models` | `object\|list[object]` | 一个或多个 Judge 模型配置；为保证 review cache 可复现，必须给出 `model_id`。 | `[]` |
| `repeats` | `int >= 1` | 每个 Judge 的独立判分观测次数，不等同于 transport retry。 | `1` |
| `position_swap` | `auto\|on\|off` | `auto` 保持 benchmark 官方的位置交换策略。 | `auto` |
| `aggregation` | `mean\|median\|majority_vote` | 普通指标的跨观测聚合方式。 | `mean` |
| `min_valid_judges` | `int >= 1` | 一个指标所需的最少有效 Judge verdict 数。 | `1` |

`models` 的每项支持 `judge_id`、`model_id`、`api_key`、`api_url`、`eval_type`、`model_args` 与
`generation_config`。provider 私有的模型初始化参数放入 `model_args`；transport 重试放入
`generation_config.retries`。

`judge.contract` 仅配置通用单 verdict Judge：`system_prompt`、`prompt_template`、`score_mapping` 和
`score_type`。`pattern` 要求 Judge 在 JSON 中返回 `score_mapping` 之一；`numeric` 要求 JSON 分数位于
`[0, 1]`。框架会在 prompt 中追加 JSON 格式要求，只解析一次普通模型回复；不使用 constrained decoding、
正则提分或纠正性追问。无效回复显示为 unavailable，并从指标中排除，而非记为 0。

对于经过 LLM 判定的样本，报告包含 `JudgeSummary`：覆盖率、失败计数与分歧。当 adapter 通过确定性的
Judge 短路直接判定样本时，得分 metadata 会记录 `judge_skipped=true` 和 `judge_skip_reason`；Web review
面板会将其标为规则直接判分，而非 LLM verdict。native 评测复用 prediction 和 review 前要求缓存的评测身份完全匹配。
设置 `rerun_review=True` 可复用 prediction 并重算 review，新的 review 文件只有成功后才原子替换旧文件；它也是身份
不匹配时唯一的显式覆盖开关，生成的配置会在当前评测版本下记录 prediction 来源。

旧 `judge_strategy` 和单个 mapping `judge_model_args` 仅保留一轮输入迁移并会告警。已删除的
`judge_worker_num` 和 `score_pattern` 会明确报错。

## Sandbox参数

EvalScope 使用嵌套的 `--sandbox` 配置（对应 `SandboxTaskConfig`）统一管理沙箱设置。

### --sandbox 配置项

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `enabled` | `bool` | 是否启用沙箱 | `false` |
| `engine` | `str` | 沙箱引擎，可选 `docker`、`volcengine` 等 | `docker` |
| `default_config` | `dict` | 任务级沙箱配置，将与 `BenchmarkMeta.sandbox_config` 合并；同时作为 Agent 模式中每个样本环境的默认配置 | `{}` |
| `manager_config` | `dict` | 转发给 ms_enclave manager 构造函数的参数（如远端 docker daemon 的 `base_url`、volcengine 凭证等） | `{}` |
| `pool_size` | `int \| None` | 池化执行的预热池大小，`None` 时与 `eval_batch_size` 对齐 | `None` |

完整使用方法（含本地与远端管理器配置示例）请参考 [沙箱环境使用](../user_guides/sandbox.md)。

## Agent 参数

`--agent-config` / `agent_config` 用于启用 [Agent 评测](../user_guides/agent/index.md)：当设置后，所有基于 `DefaultDataAdapter` 的基准会改用 [内置 AgentLoop](../user_guides/agent/native.md) 进行推理，或通过 [外部 Agent Bridge](../user_guides/agent/bridge.md) 转交给 Claude Code / Codex 等成品 CLI。`AgentLoopAdapter` 子类（如 `swe_bench_*_agentic`）保留 benchmark 默认值，同时接受其支持的显式覆盖，例如策略、步数和工具。

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--agent-config` | `dict \| NativeAgentConfig` | Agent 全局配置，详见下表 | `None`（关闭 Agent 模式） |

### agent-config 配置项

| 字段 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `strategy` | `str` | 策略名称：`function_calling` / `react` / `swe_bench_toolcall` / `swe_bench_backticks` | `function_calling` |
| `tools` | `list[str]` | 工具白名单：`bash` / `python_exec`（`submit` 由策略自动注入） | `[]` |
| `environment` | `str \| None` | Agent 命令执行环境，例如 `local` 或 `docker` | `None` |
| `environment_extra` | `dict` | Agent 环境构造参数；Docker 镜像放在 `sandbox_config.image` | `{}` |
| `max_steps` | `int` | 循环迭代硬上限 | `10` |
| `kwargs` | `dict` | 策略构造参数，例如 `{'system_prompt': '...'}` | `{}` |

```{seealso}
完整使用说明、用例与 Trace 可视化请参见 [Agent 评测](../user_guides/agent/index.md)。
```

## 其他参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--work-dir` | `str` | 评测输出路径（详见下方目录结构） | `./outputs` |
| `--no-timestamp` | `bool` | 是否不在工作目录中添加时间戳 | `false` |
| `--use-cache` | `str` | 复用本地缓存路径（如`outputs/20241210_194434`）<br>重用推理结果和评测结果 | `None` |
| `--rerun-review` | `bool` | 配合 `--use-cache` 使用：基于 predictions 缓存重跑评测/打分，并仅在成功后原子替换 reviews 缓存 | `false` |
| `--enable-progress-tracker` | `bool` | 是否开启进度追踪，将层级评测进度实时写入`progress.json`，可通过服务接口查询 | `false` |
| `--collect-perf` | `bool` | 采集每次推理请求的性能指标（延迟、TTFT、Token 用量），汇总后写入评测报告。采集 TTFT 需开启 `--generation-config stream=true`；使用 `--no-collect-perf` 可禁用 | `true` |
| `--seed` | `int` | 随机种子 | `42` |
| `--debug` | `bool` | 是否开启调试模式 | `false` |
| `--ignore-errors` | `bool` | 是否忽略生成过程中的错误 | `false` |
| `--dry-run` | `bool` | 预检参数，不执行推理，只打印参数 | `false` |

### work-dir 目录结构示例

```text
./outputs/{timestamp}/
├── configs/
│   └── task_config_b6f42c.yaml      # 任务配置
├── logs/
│   └── eval_log.log                 # 评测日志
├── predictions/
│   └── {model_id}/
│       └── {dataset}.jsonl          # 模型推理结果
├── reports/
│   └── {model_id}/
│       └── {dataset}.json           # 评测报告
├── reviews/
│   └── {model_id}/
│       └── {dataset}.jsonl          # 评测结果详情
└── progress.json                    # 进度追踪文件（启用--enable-progress-tracker时生成）
```

`progress.json` 文件格式示例：

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
