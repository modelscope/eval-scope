# AgentX 服务性能基准

AgentX 用于衡量推理服务处理长编码 Agent 会话时的性能，不评估模型能力或任务正确率。

## 安装

AgentX 要求 Python 3.11--3.13，并且不包含在默认 Perf extra 中：

```bash
pip install 'evalscope[agentx]'
```

## 服务前置条件

默认 256K workload 需要支持流式 OpenAI Chat Completions 的服务端，并且实际上下文窗口至少为 256K tokens。32K 服务无法运行默认 trace。使用 vLLM 时，请设置 `--max-model-len 262144`，并在运行 AgentX 前通过 `/v1/models` 返回的 `max_model_len` 核验。

## 快速开始

建议先执行短 smoke 测试。它使用一个活跃会话，结果不能与公开成绩比较：

```bash
evalscope perf \
  --scenario '{"name":"agentx","mode":"smoke"}' \
  --model YOUR_MODEL \
  --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions \
  --parallel 1
```

## 正式运行

默认运行使用已校验的 ModelScope 256K 镜像、固定 seed 和 30 分钟时长：

```bash
evalscope perf \
  --scenario agentx \
  --model YOUR_MODEL \
  --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions \
  --parallel 8
```

`parallel` 是同时运行的 AgentX 会话数。一个会话可能发出多个请求，因此实际请求并发可能更高。

仅在需要与 AIPerf 上游结果兼容时使用 `--data-source huggingface`：

```bash
evalscope perf --scenario agentx --data-source huggingface \
  --model YOUR_MODEL --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions --parallel 8
```

## 等待在途请求完成

在场景 JSON 中设置 `benchmark_grace_period`，可以控制发送时长结束后 AIPerf 额外等待在途响应的秒数。例如，将 `--scenario agentx` 替换为：

```bash
--scenario '{"name":"agentx","benchmark_grace_period":600}'
```

取值须为有限的非负数；零也会显式透传。省略时保留 AIPerf 默认值（0.12.0 中为 30 秒）。该参数不会改变 `--duration` 或单个请求的 `request_timeout_seconds`。在此期间完成的响应会计入指标，阶段总耗时可能超过发送时长。对比不同引擎时请使用相同设置，并检查阶段日志中的取消请求数。

## 高级配置

JSON 形式可用于选择其他 workload 变体、配置请求时间参数或记录部署元数据。`full` 变体需要服务端支持更大的上下文窗口：

```bash
evalscope perf \
  --scenario '{"name":"agentx","variant":"full","num_gpus":8,"engine":"vllm"}' \
  --model YOUR_MODEL --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions --parallel 8
```

## 结果

每个并发点会生成 `agentx_summary.json`，整体汇总为 `agentx_sweep_summary.json`。原始 benchmark 文件保存在相邻的 `aiperf/` 目录，便于排查问题。

smoke、取消、失败、缩短或哈希不符的运行，`submission_valid` 均为 false。需要可比较结果时，请使用符合要求的数据集和时长执行正式运行。

## 兼容性

首版固定 `aiperf==0.12.0` 及其 `inferencex-agentx-mvp` 场景。它不等同于当前 InferenceX AgentX v1.0 榜单口径，后者的 profiling 和 warmup 规则已变化。AgentX 指标不能作为模型质量或 Agent 任务得分。
