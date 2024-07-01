# Copyright (c) Alibaba, Inc. and its affiliates.

"""
1. Installation
eval-scope: pip install llmuses>=0.4.0
ms-opencompass: pip install ms-opencompass

2. Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip

3. Deploy model serving

4. Run eval task
"""
from llmuses.backend.opencompass import OpenCompassBackendManager
from llmuses.run import run_task
from llmuses.summarizer import Summarizer


def run_swift_eval():

    # List all datasets
    # e.g.  ['mmlu', 'WSC', 'DRCD', 'chid', 'gsm8k', 'AX_g', 'BoolQ', 'cmnli', 'ARC_e', 'ocnli_fc', 'summedits', 'MultiRC', 'GaokaoBench', 'obqa', 'math', 'agieval', 'hellaswag', 'RTE', 'race', 'ocnli', 'strategyqa', 'triviaqa', 'WiC', 'COPA', 'piqa', 'nq', 'mbpp', 'csl', 'Xsum', 'CB', 'tnews', 'ARC_c', 'afqmc', 'eprstmt', 'ReCoRD', 'bbh', 'CMRC', 'AX_b', 'siqa', 'storycloze', 'humaneval', 'cluewsc', 'winogrande', 'lambada', 'ceval', 'bustm', 'C3', 'lcsts']
    print(f'** All datasets from OpenCompass backend: {OpenCompassBackendManager.list_datasets()}')

    # Prepare the config
    """
    Attributes:
        `eval_backend`: Default to 'OpenCompass'
        `datasets`: list, refer to `OpenCompassBackendManager.list_datasets()`
        `models`: list of dict, each dict must contain `path` and `openai_api_base` 
                `path`: reuse the value of '--model_type' in the command line `swift deploy`
                `openai_api_base`: the base URL of swift model serving
        `work_dir`: str, the directory to save the evaluation results、logs and summaries. Default to 'outputs/default'
                
        Refer to `opencompass.cli.arguments.ApiModelConfig` for other optional attributes.
    """
    # Option 1: Use dict format
    # Args:
    #   path: The path of the model, it means the `model_type` for swift, e.g. 'llama3-8b-instruct'
    #   is_chat: True for chat model, False for base model
    #   key: The OpenAI api-key of the model api, default to 'EMPTY'
    #   openai_api_base: The base URL of the OpenAI API, it means the swift model serving URL.
    task_cfg = dict(
        eval_backend='OpenCompass',
        eval_config={'datasets': ['mmlu', 'ceval', 'ARC_c', 'gsm8k'],
                     'models': [
                         {'path': 'llama3-8b-instruct', 'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions'},
                         {'path': 'llama3-8b', 'is_chat': False, 'key': 'EMPTY', 'openai_api_base': 'http://127.0.0.1:8001/v1/completions'}
                     ],
                     'work_dir': 'outputs/llama3_eval_result',
                     'limit': 5,  # Could be int/float/str, e.g. 5 or 5.0 or `[10:20]`
                     },
    )

    # Option 2: Use yaml file
    # task_cfg = 'examples/tasks/default_eval_swift_openai_api.yaml'

    # Option 3: Use json file
    # task_cfg = 'examples/tasks/default_eval_swift_openai_api.json'

    # Run task
    run_task(task_cfg=task_cfg)

    # [Optional] Get the final report with summarizer
    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>>The report list: {report_list}')


if __name__ == '__main__':
    run_swift_eval()
