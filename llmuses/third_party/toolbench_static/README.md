
## Description
We evaluate the effectiveness of tool learning benchmark: [ToolBench](https://arxiv.org/pdf/2307.16789) (Qin et al.,2023b). The task involve integrating API calls to accomplish tasks, where the agent must accurately select the appropriate API and compose necessary API requests.

Moreover, we partition the test set of ToolBench into in-domain and out-of-domain based on whether the tools used in the test instances have been seen during training.

This division allows us to evaluate performance in both in-distribution and out-of-distribution scenarios. We call this dataset to be `ToolBench-Static`.

For more details, please refer to: [Small LLMs Are Weak Tool Learners: A Multi-LLM Agent](https://arxiv.org/abs/2401.07324)

## Dataset

- Dataset statistics:
  - Number of in_domain: 1588
  - Number of out_domain: 781

## Usage

### Installation

```bash
pip install llmuses -U
pip install ms-swift -U
pip install rouge -U
```


### Download the dataset

```bash
wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/toolbench-static/data.zip
```


### Unzip the dataset

```bash
unzip data.zip
# The dataset will be unzipped to the `/path/to/data/toolbench_static` folder
```


### Task configuration

There are two ways to configure the task: dict and yaml.

1. Configuration with dict:

```python
your_task_config = {
    'infer_args': {
        'model_name_or_path': '/path/to/model_dir',
        'model_type': 'qwen2-7b-instruct',
        'data_path': 'data/toolbench_static',
        'output_dir': 'output_res',
        'deploy_type': 'swift',
        'max_new_tokens': 2048,
        'num_infer_samples': None
    },
    'eval_args': {
        'input_path': 'output_res',
        'output_path': 'output_res'
    }
}
```
- Arguments:
  - `model_name_or_path`: The path to the model local directory.
  - `model_type`: The model type, refer to [模型类型列表](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md)
  - `data_path`: The path to the dataset directory contains `in_domain.json` and `out_of_domain.json` files.
  - `output_dir`: The path to the output directory. Default to `output_res`.
  - `deploy_type`: The deploy type, default to `swift`.
  - `max_new_tokens`: The maximum number of tokens to generate.
  - `num_infer_samples`: The number of samples to infer. Default to `None`, which means infer all samples.
  - `input_path`: The path to the input directory for evaluation, should be the same as `output_dir` of `infer_args`.
  - `output_path`: The path to the output directory for evaluation.


2. Configuration with yaml:

```yaml
infer_args:
  model_name_or_path: /path/to/model_dir      # absolute path is recommended
  model_type: qwen2-7b-instruct
  data_path: /path/to/data/toolbench_static   # absolute path is recommended
  deploy_type: swift
  max_new_tokens: 2048
  num_infer_samples: null
  output_dir: output_res
eval_args:
  input_path: output_res
  output_path: output_res
```
refer to [config_default.yaml](config_default.yaml) for more details.


### Run the task

```python
```python
from llmuses.third_party.toolbench_static import run_task

# Run the task with dict configuration
run_task(task_cfg=your_task_config)

# Run the task with yaml configuration
run_task(task_cfg='/path/to/your_task_config.yaml')
```


### Results and metrics

- Metrics:
  - `Plan.EM`: The agent’s planning decisions at each step for using tools invocation, generating answer, or giving up. Exact match score.
  - `Act.EM`: Action exact match score, including the tool name and arguments.
  - `HalluRate`（lower is better）: The hallucination rate of the agent's answers at each step.
  - `Avg.F1`: The average F1 score of the agent's tools calling at each step.
  - `R-L`: The Rouge-L score of the agent's answers at each step.

Generally, we focus on `Act.EM`, `HalluRate` and `Avg.F1` metrics.
