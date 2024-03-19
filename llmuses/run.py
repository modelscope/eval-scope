# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa

import json
import argparse
from typing import Union
import torch        # noqa

from llmuses.constants import DEFAULT_ROOT_CACHE_DIR
from llmuses.evaluator import Evaluator
from llmuses.evaluator.evaluator import HumanevalEvaluator
from llmuses.models.custom import CustomModel
from llmuses.utils import import_module_util
from llmuses.utils.logger import get_logger

logger = get_logger()

"""
Run evaluation for LLMs.
"""

BENCHMARK_PATH_PREFIX = 'llmuses.benchmarks.'
MEMBERS_TO_IMPORT = ['DATASET_ID', 'SUBSET_LIST', 'DataAdapterClass', 'ModelAdapterClass']


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation on benchmarks for LLMs.')

    parser.add_argument('--model',
                        help='The model id on modelscope, or local model dir.',
                        type=str,
                        required=True)
    parser.add_argument('--model-type',
                        type=str,
                        help='The type for evaluating. '
                             'service - for APIs, TO-DO'
                             'checkpoint - for models on ModelScope or local model dir, '
                             'custom - for custom models.'
                             '         Need to set `--model` to llmuses.models.custom.CustomModel format.'
                             'default to `checkpoint`.',
                        required=False,
                        default='checkpoint',
                        )
    parser.add_argument('--model-args',
                        type=str,
                        help='The model args, should be a string.',
                        required=False,
                        default='revision=None,precision=torch.float16,device_map=auto'
                        )
    parser.add_argument('--generation-config',
                        type=str,
                        help='The generation config, should be a string.',
                        required=False,
                        default='do_sample=False,repetition_penalty=1.0,max_new_tokens=512',
                        )
    parser.add_argument('--datasets',
                        help='Dataset id list, align to the module name in llmuses.benchmarks',
                        type=str,
                        nargs='+',
                        required=True)
    parser.add_argument('--dataset-args',
                        type=json.loads,
                        help='The dataset args, should be a json string. The key of dict should be aligned to datasets,'
                             'e.g. {"humaneval": {"local_path": "/to/your/path"}}',
                        required=False,
                        default='{}')
    parser.add_argument('--dataset-dir',
                        help='The datasets dir. Use to specify the local datasets or datasets cache dir.'
                             'See --dataset-hub for more details.',
                        required=False,
                        default=DEFAULT_ROOT_CACHE_DIR)
    parser.add_argument('--dataset-hub',
                        help='The datasets hub, can be `ModelScope` or `HuggingFace` or `Local`. '
                             'Default to `ModelScope`.'
                             'If `Local`, the --dataset-dir should be local input data dir.'
                             'Otherwise, the --dataset-dir should be the cache dir for datasets.',
                        required=False,
                        default='ModelScope')
    parser.add_argument('--outputs',
                        help='Outputs dir.',
                        required=False,
                        default='outputs')
    parser.add_argument('--work-dir',
                        help='The root cache dir.',
                        required=False,
                        default=DEFAULT_ROOT_CACHE_DIR)
    parser.add_argument('--limit',
                        type=int,
                        help='Max evaluation samples num for each subset. Default to None, which means no limit.',
                        default=None)
    parser.add_argument('--debug',
                        help='Debug mode, will print information for debugging.',
                        action='store_true',
                        default=False)
    parser.add_argument('--dry-run',
                        help='Dry run in single processing mode.',
                        action='store_true',
                        default=False)
    parser.add_argument('--mem-cache',
                        help='To use memory cache or not.',
                        action='store_true',
                        default=False)
    parser.add_argument('--stage',      # TODO
                        help='The stage of evaluation pipeline, '
                             'can be `all`, `infer`, `review`, `report`. Default to `all`.',
                        type=str,
                        default='all')

    args = parser.parse_args()

    return args


def parse_str_args(str_args: str) -> dict:
    assert isinstance(str_args, str), 'args should be a string.'
    arg_list: list = str_args.strip().split(',')
    arg_list = [arg.strip() for arg in arg_list]
    arg_dict: dict = dict([arg.split('=') for arg in arg_list])

    final_args = dict()
    for k, v in arg_dict.items():
        try:
            final_args[k] = eval(v)
        except:
            if v.lower() == 'true':
                v = True
            if v.lower() == 'false':
                v = False
            final_args[k] = v

    return final_args


def run_task(task_cfg: dict):

    logger.info(task_cfg)

    model_args: dict = task_cfg.get('model_args',
                                    {'revision': None, 'precision': torch.float16, 'device_map': 'auto'})
    # Get the GLOBAL default config (infer_cfg) for prediction
    generation_config: dict = task_cfg.get('generation_config',
                                           {'do_sample': False,
                                            'repetition_penalty': 1.0,
                                            'max_length': 2048,
                                            'max_new_tokens': 512,
                                            'temperature': 0.3,
                                            'top_k': 50,
                                            'top_p': 0.8, }
                                           )
    dataset_args: dict = task_cfg.get('dataset_args', {})
    dry_run: bool = task_cfg.get('dry_run', False)
    model: Union[str, CustomModel] = task_cfg.get('model', None)
    eval_type: str = task_cfg.get('eval_type', 'checkpoint')
    datasets: list = task_cfg.get('datasets', None)
    work_dir: str = task_cfg.get('work_dir', DEFAULT_ROOT_CACHE_DIR)
    outputs: str = task_cfg.get('outputs', 'outputs')
    mem_cache: bool = task_cfg.get('mem_cache', False)
    dataset_hub: str = task_cfg.get('dataset_hub', 'ModelScope')
    dataset_dir: str = task_cfg.get('dataset_dir', DEFAULT_ROOT_CACHE_DIR)
    stage: str = task_cfg.get('stage', 'all')                               # TODO: to be implemented
    limit: int = task_cfg.get('limit', None)
    debug: str = task_cfg.get('debug', False)

    if model is None or datasets is None:
        raise ValueError('** Args: Please provide model and datasets. **')

    model_precision = model_args.get('precision', torch.float16)
    if isinstance(model_precision, str):
        model_precision = eval(model_precision)

    # Get model args
    if dry_run:
        from llmuses.models.dummy_chat_model import DummyChatModel
        model_id: str = 'dummy'
        model_revision: str = 'v1.0.0'
    elif eval_type == 'custom':
        model_id: str = None
        model_revision: str = None
    else:
        model_id: str = model
        model_revision: str = model_args.get('revision', None)
        if model_revision == 'None':
            model_revision = eval(model_revision)

    for dataset_name in datasets:
        # Get imported_modules
        imported_modules = import_module_util(BENCHMARK_PATH_PREFIX, dataset_name, MEMBERS_TO_IMPORT)

        if dataset_name == 'humaneval' and dataset_args.get('humaneval', {}).get('local_path') is None:
            raise ValueError('Please specify the local problem path of humaneval dataset in --dataset-args,'
                             'e.g. {"humaneval": {"local_path": "/to/your/path"}}, '
                             'And refer to https://github.com/openai/human-eval/tree/master#installation to install it,'
                             'Note that you need to enable the execution code in the human_eval/execution.py first.')

        if dry_run:
            from llmuses.models.dummy_chat_model import DummyChatModel
            model_adapter = DummyChatModel(model_cfg=dict())
        elif eval_type == 'custom':
            if not isinstance(model, CustomModel):
                raise ValueError('Please provide a custom model instance '
                                 'in format of llmuses.models.custom.CustomModel.')
            from llmuses.models.model_adapter import CustomModelAdapter
            model_adapter = CustomModelAdapter(custom_model=model)
        else:
            # Init model adapter
            device_map = model_args.get('device_map', 'auto') if torch.cuda.is_available() else None
            model_adapter = imported_modules['ModelAdapterClass'](model_id=model_id,
                                                                  model_revision=model_revision,
                                                                  device_map=device_map,
                                                                  torch_dtype=model_precision,
                                                                  cache_dir=work_dir)

        if dataset_name == 'humaneval':
            problem_file: str = dataset_args.get('humaneval', {}).get('local_path')

            evaluator = HumanevalEvaluator(problem_file=problem_file,
                                           model_id=model_id,
                                           model_revision=model_revision,
                                           model_adapter=model_adapter,
                                           outputs_dir=outputs,
                                           is_custom_outputs_dir=False, )
        else:
            dataset_name_or_path: str = dataset_args.get(dataset_name, {}).get('local_path') or imported_modules[
                'DATASET_ID']

            # Init data adapter
            few_shot_num: int = dataset_args.get(dataset_name, {}).get('few_shot_num', None)
            few_shot_random: bool = dataset_args.get(dataset_name, {}).get('few_shot_random', True)
            data_adapter = imported_modules['DataAdapterClass'](few_shot_num=few_shot_num,
                                                                few_shot_random=few_shot_random)

            evaluator = Evaluator(
                dataset_name_or_path=dataset_name if dataset_hub == 'Local' else dataset_name_or_path,
                subset_list=imported_modules['SUBSET_LIST'],
                data_adapter=data_adapter,
                model_adapter=model_adapter,
                use_cache=mem_cache,
                root_cache_dir=work_dir,
                outputs_dir=outputs,
                is_custom_outputs_dir=False,
                datasets_dir=dataset_dir,
                datasets_hub=dataset_hub,
                stage=stage,
                eval_type=eval_type,
            )

        infer_cfg = generation_config or {}
        infer_cfg.update(dict(limit=limit))
        evaluator.eval(infer_cfg=infer_cfg, debug=debug)


def main():
    args = parse_args()

    # Get task_cfg
    task_cfg = {
        'model_args': parse_str_args(args.model_args),
        'generation_config': parse_str_args(args.generation_config),
        'dataset_args': args.dataset_args,
        'dry_run': args.dry_run,
        'model': args.model,
        'eval_type': args.eval_type,
        'datasets': args.datasets,
        'work_dir': args.work_dir,
        'outputs': args.outputs,
        'mem_cache': args.mem_cache,
        'dataset_hub': args.dataset_hub,
        'dataset_dir': args.dataset_dir,
        'stage': args.stage,
        'limit': args.limit,
        'debug': args.debug
    }

    run_task(task_cfg)


if __name__ == '__main__':
    # Usage: python3 llmuses/run.py --model ZhipuAI/chatglm2-6b --datasets mmlu hellaswag --limit 10
    # Usage: python3 llmuses/run.py --model qwen/Qwen-1_8B --generation-config do_sample=false,temperature=0.0 --datasets ceval --dataset-args '{"ceval": {"few_shot_num": 0}}' --limit 10
    main()
