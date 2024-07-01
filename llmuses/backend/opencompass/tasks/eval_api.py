# Copyright (c) Alibaba, Inc. and its affiliates.
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask


with read_base():
    from llmuses.backend.opencompass.tasks.eval_datasets import datasets

# 1. Get datasets
# Note: OpenAI API format evaluation needs a special humaneval postprocessor
for _dataset in datasets:
    if _dataset['path'] == 'openai_humaneval':
        from opencompass.datasets.humaneval import humaneval_gpt_postprocess
        _dataset['eval_cfg']['pred_postprocessor']['type'] = humaneval_gpt_postprocess


# 2. Get models, only for placeholder, you should fill in the real model information from command line
# See more templates in `opencompass.cli.arguments.ApiModelConfig`
models = []

# 3. Get infer config
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=4,
        task=dict(type=OpenICLInferTask)),
)
