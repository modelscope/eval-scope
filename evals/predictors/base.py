# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod

from evals.constants import PredictorMode


class Predictor(ABC):
    # TODO: support loading model from disk and calling remote service

    def __init__(self, mode=PredictorMode.REMOTE, **kwargs):
        self.mode = mode

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

