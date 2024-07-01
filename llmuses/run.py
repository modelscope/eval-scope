# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa
import copy
import json
import argparse
import os.path
from typing import Union, List
import torch        # noqa

from llmuses.config import TaskConfig
from llmuses.constants import DEFAULT_ROOT_CACHE_DIR
from llmuses.evaluator import Evaluator
from llmuses.evaluator.evaluator import HumanevalEvaluator
from llmuses.models.custom import CustomModel
from llmuses.utils import import_module_util, yaml_to_dict, make_outputs_dir, gen_hash, json_to_dict, EvalBackend
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
                        # required=True,
                        required=False,
                        )
    parser.add_argument('--model-type',
                        help='Deprecated. See `--template-type`',
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument('--template-type',
                        type=str,
                        help='The template type for generation, should be a string.'
                             'Refer to `https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md` for more details.',
                        # required=True,
                        required=False,
                        )
    parser.add_argument('--eval-type',
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
                        # required=True,
                        required=False,
                        )
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
                        help='Outputs dir. Default to `outputs`, which means dump to current path: ./outputs',
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
    parser.add_argument('--use-cache',
                        help='To reuse the cache or not. Default to `true`.',
                        type=str,
                        default='false')
    parser.add_argument('--stage',
                        help='The stage of evaluation pipeline, '
                             'can be `all`, `infer`, `review`. Default to `all`.',
                        type=str,
                        default='all')

    parser.add_argument('--eval-backend',
                        help='The evaluation backend to use. Default to None.'
                             'can be `Native`, `OpenCompass` and `ThirdParty`. '
                             'Default to `OpenCompass`.',
                        type=str,
                        default=EvalBackend.OPEN_COMPASS.value,
                        required=False)

    parser.add_argument('--eval-config',
                        help='The eval task config file path for evaluation backend, should be a yaml or json file.',
                        type=str,
                        default=None,
                        required=False)

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


def run_task(task_cfg: Union[str, dict, TaskConfig, List[TaskConfig]]) -> Union[dict, List[dict]]:

    if isinstance(task_cfg, list):
        eval_results = []
        for one_task_cfg in task_cfg:
            eval_results.append(run_task(one_task_cfg))
        return eval_results

    if isinstance(task_cfg, TaskConfig):
        task_cfg = task_cfg.to_dict()
    elif isinstance(task_cfg, str):
        if task_cfg.endswith('.yaml'):
            task_cfg = yaml_to_dict(task_cfg)
        elif task_cfg.endswith('.json'):
            task_cfg = json_to_dict(task_cfg)
        else:
            raise ValueError(f'Unsupported file format: {task_cfg}, should be a yaml or json file.')
    elif isinstance(task_cfg, dict):
        logger.info('** Args: Task config is provided with dictionary type. **')
    else:
        raise ValueError('** Args: Please provide a valid task config. **')

    # Check and run evaluation backend
    if task_cfg.get('eval_backend'):
        eval_backend = task_cfg.get('eval_backend')
        eval_config: Union[str, dict] = task_cfg.get('eval_config')

        assert eval_config is not None, 'Please provide eval_config for specific evaluation backend.'

        if eval_backend == EvalBackend.OPEN_COMPASS.value:
            from llmuses.backend.opencompass import OpenCompassBackendManager
            oc_backend_manager = OpenCompassBackendManager(config=eval_config)
            oc_backend_manager.run()

        return dict()

    # Get the output task config
    output_task_cfg = copy.copy(task_cfg)
    logger.info(output_task_cfg)

    model_args: dict = task_cfg.get('model_args',
                                    {'revision': 'default', 'precision': torch.float16, 'device_map': 'auto'})
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
    model_type: str = task_cfg.get('model_type', None)
    template_type: str = task_cfg.get('template_type', None)
    eval_type: str = task_cfg.get('eval_type', 'checkpoint')
    datasets: list = task_cfg.get('datasets', None)
    work_dir: str = task_cfg.get('work_dir', DEFAULT_ROOT_CACHE_DIR)
    outputs: str = task_cfg.get('outputs', 'outputs')
    mem_cache: bool = task_cfg.get('mem_cache', False)
    use_cache: bool = task_cfg.get('use_cache', True)
    dataset_hub: str = task_cfg.get('dataset_hub', 'ModelScope')
    dataset_dir: str = task_cfg.get('dataset_dir', DEFAULT_ROOT_CACHE_DIR)
    stage: str = task_cfg.get('stage', 'all')
    limit: int = task_cfg.get('limit', None)
    debug: str = task_cfg.get('debug', False)

    if model is None or datasets is None:
        if not task_cfg.get('eval_backend'):
            raise ValueError('** Args: Please provide model and datasets. **')

    if model_type:
        logger.warning('** DeprecatedWarning: `--model-type` is deprecated, please use `--template-type` instead.')

    model_precision = model_args.get('precision', torch.float16)
    if isinstance(model_precision, str):
        model_precision = eval(model_precision)

    if mem_cache:
        logger.warning('** DeprecatedWarning: `--mem-cache` is deprecated, please use `--use-cache` instead.')

    logger.info(f'** Set use_cache to {use_cache}.')

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
        model_revision: str = model_args.get('revision', 'default')

    # Get outputs directory
    if isinstance(model_id, str) and os.path.isdir(os.path.expanduser(model_id)):
        # get the output_model_id when model_id is a local model dir
        output_model_id: str = gen_hash(model_id)
    else:
        output_model_id: str = model_id
    if outputs == 'outputs':
        outputs = make_outputs_dir(root_dir=os.path.join(work_dir, 'outputs'),
                                   datasets=datasets,
                                   model_id=output_model_id,
                                   model_revision=model_revision,)

    eval_results = dict()
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
                raise ValueError(f'Expected llmuses.models.custom.CustomModel, but got {type(model)}.')
            from llmuses.models.model_adapter import CustomModelAdapter
            model_adapter = CustomModelAdapter(custom_model=model)
        else:
            # Init model adapter
            device_map = model_args.get('device_map', 'auto') if torch.cuda.is_available() else None
            model_adapter = imported_modules['ModelAdapterClass'](model_id=model_id,
                                                                  model_revision=model_revision,
                                                                  device_map=device_map,
                                                                  torch_dtype=model_precision,
                                                                  cache_dir=work_dir,
                                                                  template_type=template_type)

        if dataset_name == 'humaneval':
            problem_file: str = dataset_args.get('humaneval', {}).get('local_path')

            evaluator = HumanevalEvaluator(problem_file=problem_file,
                                           model_id=model_id,
                                           model_revision=model_revision,
                                           model_adapter=model_adapter,
                                           outputs_dir=outputs,
                                           is_custom_outputs_dir=False, )
        else:
            # TODO: CHECK dataset_args
            dataset_name_or_path: str = dataset_args.get(dataset_name, {}).get('local_path') or imported_modules[
                'DATASET_ID']

            in_prompt_template: str = dataset_args.get(dataset_name, {}).get('prompt_template', '')

            # Init data adapter
            few_shot_num: int = dataset_args.get(dataset_name, {}).get('few_shot_num', None)
            few_shot_random: bool = dataset_args.get(dataset_name, {}).get('few_shot_random', True)
            data_adapter = imported_modules['DataAdapterClass'](few_shot_num=few_shot_num,
                                                                few_shot_random=few_shot_random,
                                                                prompt_template=in_prompt_template,)

            in_subset_list: list = dataset_args.get(dataset_name, {})\
                .get('subset_list', imported_modules['SUBSET_LIST'])
            logger.info(f'\n** Evaluating on subsets for {dataset_name}: {in_subset_list}\n')

            evaluator = Evaluator(
                dataset_name_or_path=dataset_name_or_path,
                subset_list=in_subset_list,
                data_adapter=data_adapter,
                model_adapter=model_adapter,
                use_cache=use_cache,
                root_cache_dir=work_dir,
                outputs_dir=outputs,
                is_custom_outputs_dir=outputs != 'outputs',
                datasets_dir=dataset_dir,
                datasets_hub=dataset_hub,
                stage=stage,
                eval_type=eval_type,
                overall_task_cfg=output_task_cfg,
            )

        infer_cfg = generation_config or {}
        infer_cfg.update(dict(limit=limit))
        res_dict: dict = evaluator.eval(infer_cfg=infer_cfg, debug=debug)

        eval_results[dataset_name] = res_dict

    return eval_results


def main():
    args = parse_args()

    # Get task_cfg
    use_cache: bool = False if args.use_cache.lower() == 'false' else True
    task_cfg = {
        'model_args': parse_str_args(args.model_args),
        'generation_config': parse_str_args(args.generation_config),
        'dataset_args': args.dataset_args,
        'dry_run': args.dry_run,
        'model': args.model,
        'template_type': args.template_type,
        'eval_type': args.eval_type,
        'datasets': args.datasets,
        'work_dir': args.work_dir,
        'outputs': args.outputs,
        'mem_cache': args.mem_cache,
        'use_cache': use_cache,
        'dataset_hub': args.dataset_hub,
        'dataset_dir': args.dataset_dir,
        'stage': args.stage,
        'limit': args.limit,
        'debug': args.debug,

        'eval_backend': args.eval_backend,
        'eval_config': args.eval_config,
    }

    run_task(task_cfg)


if __name__ == '__main__':
    # Usage: python3 llmuses/run.py --model ZhipuAI/chatglm2-6b --datasets mmlu hellaswag --limit 10
    # Usage: python3 llmuses/run.py --model qwen/Qwen-1_8B --generation-config do_sample=false,temperature=0.0 --datasets ceval --dataset-args '{"ceval": {"few_shot_num": 0}}' --limit 10
    main()
