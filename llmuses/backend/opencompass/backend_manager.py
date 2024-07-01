# Copyright (c) Alibaba, Inc. and its affiliates.
from enum import Enum
from typing import Optional, Union
import subprocess
from dataclasses import asdict
import tempfile

from llmuses.utils import is_module_installed, get_module_path, get_valid_list
from llmuses.backend.base import BackendManager
from llmuses.backend.opencompass.api_meta_template import get_template
from llmuses.utils.logger import get_logger

logger = get_logger()


class CmdMode(Enum):

    # The basic mode is to run the command directly,
    # e.g. `python -m run --models model1 model2 --datasets dataset1 dataset2`
    BASIC = 'basic'

    # The script mode is to run the command with a script,
    # e.g. `python -m run your_config_script.py`
    SCRIPT = 'script'


class RunMode(Enum):

    # The command mode is to run the command directly with the command line.
    CMD = 'cmd'

    # The function mode is to run the command with a function call -- run_task().
    FUNCTION = 'function'


class OpenCompassBackendManager(BackendManager):

    def __init__(self, config: Union[str, dict], **kwargs):
        """
        The backend manager for OpenCompass.

        Args:
            config: Union[str, dict], the configuration yaml-file or the configuration dictionary.
                attributes:
                    datasets: list, the datasets.
                    models: list, the models.
                    work_dir (Optional): str, the working directory. Default to None, which means the current directory.
                    dry_run (Optional): bool, the dry-run flag. Default to False.
                    debug (Optional): bool, the debug flag. Default to False.
                    reuse (Optional): str, reuse previous outputs & results. Default to None.
                    generation_kwargs (Optional): dict, the generation config. Default to {}.
                    limit (Optional): int or float or str, the limit of the number of examples. Default to None.
                        if limit is a string, it should be in the format of '[start:end]'.

                example:
                    # TODO: add demo config
                    config = dict(
                        datasets=[mmlu, ceval],
                        models=[...],
                        ...
                    )

            **kwargs: the keyword arguments.
        """

        self._check_env()
        super().__init__(config, **kwargs)

        from opencompass.cli.arguments import Arguments as OpenCompassArguments
        self.args = OpenCompassArguments(**self.config_d)

    @property
    def cmd(self):
        return self.get_cmd()

    @staticmethod
    def _check_env():
        if is_module_installed('opencompass'):
            logger.info('Please make sure you have installed the `ms-opencompass`: `pip install ms-opencompass`')
        else:
            raise ModuleNotFoundError('Please install the `ms-opencompass` first: `pip install ms-opencompass`')

    @staticmethod
    def get_restore_arg(arg_name: str, arg_val: bool):
        if arg_val:
            return f'--{arg_name}'
        else:
            return ''

    @staticmethod
    def get_arg_with_default(arg_name: str, arg_val: Optional[str] = None):
        if arg_val:
            return f'--{arg_name} {arg_val}'
        else:
            return ''

    def load_task_template(self):
        """
        Load the initial OpenCompass task template from task config file.

        Returns:
            (mmengine.config.config.Config), the initial task template config.
        """
        from opencompass.utils.run import get_config_from_arg

        template_config_path = get_module_path('llmuses.backend.opencompass.tasks.eval_api')
        self.args.config = template_config_path
        return get_config_from_arg(self.args)

    @staticmethod
    def list_datasets(return_details: bool = False):
        from opencompass.utils.run import get_config_from_arg
        from dataclasses import dataclass

        @dataclass
        class TempArgs:
            config: str
            accelerator: str = None

        template_config_path = get_module_path('llmuses.backend.opencompass.tasks.eval_api')
        template_cfg = get_config_from_arg(TempArgs(config=template_config_path))

        # e.g. ['mmlu', 'ceval', 'openai_humaneval', ...]
        dataset_show_names = list(set([_dataset['dataset_name'] for _dataset in template_cfg.datasets]))

        if return_details:
            return dataset_show_names, template_cfg.datasets
        else:
            return dataset_show_names

    def get_task_args(self):
        return self.args

    def get_cmd(self, cmd_mode: str = CmdMode.BASIC):

        if cmd_mode == CmdMode.BASIC:
            assert self.args.datasets, 'The datasets are required.'
            assert self.args.models, 'The models are required.'

            cmd_str = f'python -m run_oc ' \
                      f'--models {" ".join(self.args.models)} ' \
                      f'--datasets {" ".join(self.args.datasets)} ' \
                      f'{self.get_restore_arg("dry-run", self.args.dry_run)} ' \
                      f'{self.get_arg_with_default("work-dir", self.args.work_dir)}'

        elif cmd_mode == CmdMode.SCRIPT:
            assert self.args.config, 'The script file is required.'
            cmd_str = f'python -m run_oc {self.args.config}'
        else:
            raise ValueError(f'Unsupported command mode: {cmd_mode}')

        return cmd_str

    def run(self, run_mode: str = RunMode.FUNCTION):
        """
        The entry function to run the OpenCompass task.

        Args:
            run_mode: str, the running mode, e.g. 'function' or 'cmd'.

        Returns:
            None
        """
        if run_mode == RunMode.FUNCTION:
            from opencompass.cli.main import run_task
            from opencompass.cli.arguments import ApiModelConfig

            assert isinstance(self.args.models, list) and len(self.args.models) > 0, 'The models are required.'

            tmp_model_d: dict = self.args.models[0]
            assert 'path' in tmp_model_d and 'openai_api_base' in tmp_model_d, \
                f"Got invalid model config: {tmp_model_d}. \nTo get valid format: " \
                "{'path': 'qwen-7b-chat', 'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions'}"

            # Get valid datasets
            dataset_names = self.args.datasets       # e.g. ['mmlu', 'ceval']
            dataset_names_all, real_dataset_all = self.list_datasets(return_details=True)

            if not dataset_names:
                logger.warning(f'No datasets are specified in the config. Use all the datasets: {dataset_names_all}')
                valid_dataset_names = dataset_names_all
            else:
                valid_dataset_names, invalid_dataset_names = get_valid_list(dataset_names, dataset_names_all)
                if len(invalid_dataset_names) > 0:
                    logger.error(f'Invalid datasets: {invalid_dataset_names}, '
                                 f'refer to the following list to get proper dataset name: {dataset_names_all}')
                assert len(valid_dataset_names) > 0, f'No valid datasets. ' \
                                                     f'To get the valid datasets, please refer to {dataset_names_all}'

            valid_datasets = [_dataset for _dataset in real_dataset_all if _dataset['dataset_name'] in valid_dataset_names]
            for _dataset in valid_datasets:
                _dataset.pop('dataset_name')
                _dataset['reader_cfg']['test_range'] = self.args.limit

            # Get valid models
            models = []
            for model_d in self.args.models:
                # model_d: {'path': 'qwen-7b-chat',
                #           'meta_template': 'default-api-meta-template-oc',   # Optional
                #           'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions'}
                # Note: 'meta_template' can be a dict or a string, default is None

                if 'meta_template' in model_d and isinstance(model_d['meta_template'], str):
                    model_d['meta_template'] = get_template(model_d['meta_template'])

                # set the 'abbr' as the 'path' if 'abbr' is not specified
                model_d['abbr'] = model_d['path']

                model_config = ApiModelConfig(**model_d)
                models.append(asdict(model_config))

            # Load the initial task template and override configs
            template_cfg = self.load_task_template()
            template_cfg.datasets = valid_datasets
            template_cfg.models = models

            # Dump task config to a temporary file
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.py', mode='w')
            template_cfg.dump(tmp_file.name)
            # logger.info(f'The task config is dumped to: {tmp_file.name}')
            self.args.config = tmp_file.name

            # Submit the task
            logger.info(f'*** Run task with config: {self.args.config} \n')
            run_task(self.args)

        # TODO: add more arguments for the command line
        elif run_mode == RunMode.CMD:
            subprocess.run(self.cmd, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            raise ValueError(f'Unsupported run mode: {run_mode}')


if __name__ == '__main__':

    # OpenCompassBackendManager.list_datasets()
    # ['mmlu', 'WSC', 'DRCD', 'chid', 'gsm8k', 'AX_g', 'BoolQ', 'cmnli', 'ARC_e', 'ocnli_fc', 'summedits', 'MultiRC', 'GaokaoBench', 'obqa', 'math', 'agieval', 'hellaswag', 'RTE', 'race', 'flores', 'ocnli', 'strategyqa', 'triviaqa', 'WiC', 'COPA', 'commonsenseqa', 'piqa', 'nq', 'mbpp', 'csl', 'Xsum', 'CB', 'tnews', 'ARC_c', 'afqmc', 'eprstmt', 'ReCoRD', 'bbh', 'TheoremQA', 'CMRC', 'AX_b', 'siqa', 'storycloze', 'humaneval', 'cluewsc', 'winogrande', 'lambada', 'ceval', 'bustm', 'C3', 'lcsts']

    # 'meta_template': 'default-api-meta-template-oc',
    # models: llama3-8b-instruct, qwen-7b-chat
    oc_backend_manager = OpenCompassBackendManager(
        config={'datasets': ['mmlu', 'ceval', 'ARC_c', 'gsm8k'],
                'models': [{'path': 'llama3-8b-instruct', 'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions'}],
                'limit': 5
                }
    )
    oc_backend_manager.run()
