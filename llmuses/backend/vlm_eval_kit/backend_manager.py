from typing import Optional, Union
from llmuses.utils import is_module_installed, get_module_path, get_valid_list, yaml_to_dict, json_to_dict
from llmuses.backend.base import BackendManager
from llmuses.utils.logger import get_logger
from functools import partial
import subprocess
from dataclasses import dataclass

logger = get_logger()


class RunMode:

    # The command mode is to run the command directly with the command line.
    CMD = 'cmd'

    # The function mode is to run the command with a function call -- run_task().
    FUNCTION = 'function'


class VLMEvalKitBackendManager(BackendManager):
    def __init__(self, config: Union[str, dict], **kwargs):
        """BackendManager for VLM Evaluation Kit

        Args:
            config (Union[str, dict]): the configuration yaml-file or the configuration dictionary
        """
        self._check_env()
        super().__init__(config, **kwargs)

        from vlmeval.utils.arguments import Arguments as VLMEvalArguments
        print(self.config_d)
        self.args = VLMEvalArguments(**self.config_d)
        
        self.valid_models = self.list_supported_VLMs()
        self.valid_model_names = list(self.valid_models.keys())
        self.valid_datasets = self.list_supported_datasets()
        
        self._check_valid()
        print(self.args)
        # process model path / tokenizer path
    

    def _check_valid(self):
        # Ensure not both model and datasets are empty
        if not self.args.data or not self.args.model:
            raise ValueError('** Args: Please provide model and datasets. **')
        
        
        # Check datasets
        valid_datasets, invalid_datasets = get_valid_list(self.args.data, self.valid_datasets)
        assert len(invalid_datasets) == 0, f'Invalid datasets: {invalid_datasets}, ' \
                            f'refer to the following list to get proper dataset name: {self.valid_datasets}'
        
        # Check model
        if isinstance(self.args.model[0], dict):
            model_names = [model['name'] for model in self.args.model]
            valid_model_names, invalid_model_names = get_valid_list(model_names, self.valid_model_names)
            assert len(invalid_model_names) == 0, f'Invalid models: {invalid_model_names}, ' \
                                f'refer to the following list to get proper model name: {self.valid_model_names}'
            for model_cfg in self.args.model:
                model_name = model_cfg['name']
                if 'path' in model_cfg:
                    model_class = self.valid_models[model_name]
                    self.valid_models[model_name] = partial(model_class, model_path=model_cfg['path'])
            
                if 'tokenizer_path' in model_cfg:
                    model_class = self.valid_models[model_name]
                    self.valid_models[model_name] = partial(model_class, tokenizer_path=model_cfg['path'])
            
            self.args.model = valid_model_names
                
        elif isinstance(self.args.model[0], str):
            valid_model_names, invalid_model_names = get_valid_list(self.args.model, self.valid_model_names)
            assert len(invalid_model_names) == 0, f'Invalid models: {invalid_model_names}, ' \
                                f'refer to the following list to get proper model name: {self.valid_model_names}'
            

    @property
    def cmd(self):
        return self.get_cmd()
    
    @staticmethod
    def list_supported_VLMs():
        from vlmeval.config import supported_VLM
        return supported_VLM
    
    @staticmethod
    def list_supported_datasets():
        from vlmeval.dataset import SUPPORTED_DATASETS
        return SUPPORTED_DATASETS
    

    @staticmethod
    def _check_env():
        if is_module_installed('vlmeval'):
            logger.info('Please make sure you have installed the `ms-vlmeval`: `pip install ms-vlmeval`')
        else:
            raise ModuleNotFoundError('Please install the `ms-vlmeval` first: `pip install ms-vlmeval`')

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

    def get_cmd(self):
        assert self.args.data, 'The datasets are required.'
        assert self.args.model, 'The models are required.'

        cmd_str = f'python -m vlmeval ' \
            f'--model {" ".join(self.args.model)} ' \
            f'--data {" ".join(self.args.data)} ' \
            f'{self.get_restore_arg("verbose", self.args.verbose)} ' \
            f'{self.get_restore_arg("ignore", self.args.ignore)} ' \
            f'{self.get_restore_arg("rerun", self.args.rerun)} ' \
            f'{self.get_arg_with_default("work-dir", self.args.work_dir)} ' \
            f'{self.get_arg_with_default("limit", self.args.limit)} ' \
            f'{self.get_arg_with_default("mode", self.args.mode)} ' \
            f'{self.get_arg_with_default("nproc", self.args.nproc)} ' \
            f'{self.get_arg_with_default("judge", self.args.judge)} ' \
            f'{self.get_arg_with_default("retry", self.args.retry)} '
        
        return cmd_str

    def run(self, run_mode: str = RunMode.FUNCTION):
        if run_mode == RunMode.CMD:
            logger.info(f'** Run command: {self.cmd}')
            try:
                subprocess.run(self.cmd, check=True, ext=True,shell=True,)
            except subprocess.CalledProcessError as e:
                logger.error(f'** Run command failed: {e.stderr}')
                raise

        elif run_mode == RunMode.FUNCTION:
            from vlmeval.run import run_task
            logger.info(f'*** Run task with config: {self.args} \n')
            run_task(self.args)
            
        else:
            raise NotImplementedError
