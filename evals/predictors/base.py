# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Any

from evals.constants import PredictorMode, PredictorKeys


class Predictor(ABC):

    # TODO:
    #   1. Multi-thread calling to be supported
    #   2. Async calling to be supported

    def __init__(self, mode=PredictorMode.REMOTE, **kwargs):
        self.mode: str = mode
        self.model: Any = None

        if self.mode == PredictorMode.LOCAL:
            local_model_cfg = {}
            if PredictorKeys.LOCAL_MODEL in kwargs:
                local_model_cfg = kwargs.pop(PredictorKeys.LOCAL_MODEL)
            self._init_local_model(**local_model_cfg)

    def __call__(self, **kwargs):
        return self.predict(**kwargs)

    @abstractmethod
    def predict(self, **kwargs) -> dict:
        if self.mode == PredictorMode.LOCAL:
            return self._run_local_inference(**kwargs)
        elif self.mode == PredictorMode.REMOTE:
            return self._run_remote_inference(**kwargs)
        else:
            raise ValueError(f"Invalid predictor mode: {self.mode}")

    @abstractmethod
    def _run_local_inference(self, **kwargs) -> dict:
        ...

    @abstractmethod
    def _run_remote_inference(self, **kwargs) -> dict:
        ...

    def _init_local_model(self, **kwargs):
        if not kwargs:
            raise ValueError(f"Local model config is empty")

        ...

