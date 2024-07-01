# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Union
from copy import deepcopy

from llmuses.third_party.toolbench_static.infer import InferArgs, run_infer
from llmuses.third_party.toolbench_static.eval import EvalArgs, run_eval
from llmuses.utils import yaml_to_dict, get_logger, json_to_dict

logger = get_logger()


def run_task(task_cfg: Union[str, dict]):

    if isinstance(task_cfg, str):
        if task_cfg.endswith('.yaml'):
            task_cfg: dict = yaml_to_dict(task_cfg)
        elif task_cfg.endswith('.json'):
            task_cfg: dict = json_to_dict(task_cfg)
        else:
            raise ValueError(f'Unsupported file format: {task_cfg}, should be yaml or json file.')

    # Run inference for each domain
    infer_args: dict = task_cfg['infer_args']
    for domain in ['in_domain', 'out_of_domain']:
        domain_infer_args = deepcopy(infer_args)
        domain_infer_args.update({'data_path': os.path.join(infer_args['data_path'], f'{domain}.json')})
        domain_infer_args.update({'output_dir': os.path.join(infer_args['output_dir'], domain)})

        task_infer_args = InferArgs(**domain_infer_args)
        print(f'**Run infer config: {task_infer_args}')
        run_infer(task_infer_args)

    # Run evaluation for each domain
    eval_args: dict = task_cfg['eval_args']
    for domain in ['in_domain', 'out_of_domain']:
        domain_eval_args = deepcopy(eval_args)
        domain_eval_args.update({'input_path': os.path.join(eval_args['input_path'], domain)})
        domain_eval_args.update({'output_path': os.path.join(eval_args['output_path'], domain)})

        task_eval_args = EvalArgs(**domain_eval_args)
        print(f'**Run eval config: {task_eval_args}')
        run_eval(task_eval_args)


if __name__ == '__main__':
    # task_cfg_file = 'config_default.yaml'
    task_cfg_file = 'config_default.json'

    run_task(task_cfg=task_cfg_file)
