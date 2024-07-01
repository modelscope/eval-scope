# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Union

from llmuses.utils import yaml_to_dict


class BackendManager:
    def __init__(self, config: Union[str, dict], **kwargs):
        """
        BackendManager is the base class for the evaluation backend manager.
        It provides the basic configuration parsing, command generation, task submission, and result fetching.

        config: str or dict, the configuration of the evaluation backend.
            could be a string of the path to the configuration file (yaml), or a dictionary.
        """
        if isinstance(config, str):
            self.config_d = yaml_to_dict(config)
        else:
            self.config_d = config

        self.kwargs = kwargs

    def run(self, *args, **kwargs):
        """
        Run the evaluation backend.
        """
        raise NotImplementedError
