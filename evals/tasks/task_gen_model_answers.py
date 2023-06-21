# Copyright (c) Alibaba, Inc. and its affiliates.

from evals.tasks import BaseTask
from evals.utils.logger import get_logger
from evals.utils.utils import jsonl_to_list

logger = get_logger()


class TaskGenModelAnswers(BaseTask):

    def __init__(self, question_file: str, **kwargs):
        super(TaskGenModelAnswers, self).__init__(**kwargs)

        self.question_list = jsonl_to_list(question_file)

    def run(self, pred_cfg: dict):
        logger.info(f'pred_cfg: {pred_cfg}')
        pass
