# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random
import time
from functools import partial

import pandas as pd
from evals.constants import ArenaWinner, ArenaMode, EvalTaskConfig, FnCompletionParser
from evals.evaluator import BaseReviewer
from evals.predictors.openai_gpt_predictor import OpenaiGptPredictor
from evals.utils.arena_utils import get_battle_pairs, merge_ques_ans, shuffle_pairwise_preferences
from evals.utils.logger import get_logger
from evals.utils.utils import jsonl_dump_data, jsonl_to_list, random_seeded_choice
from evals.utils import completion_parsers

logger = get_logger()


class AutoReviewerGpt4(BaseReviewer):
    """
    Auto-review target answers(models) pairwise with GPT-4.

    Args:
        prompt_file: path to prompt templates file.
        answer_file_list: list of paths to answer files.
        review_file: path to review result file.
        reviewer_args: config for reviewer(GPT-4).

    Examples:
        >>> from evals.evaluator.auto_reviewer_gpt4 import AutoReviewerGpt4
        >>> input_kwargs = dict(prompt_file='/path/to/prompt_file.jsonl', answer_file_list=['/path/to/ans1_file.jsonl',
            '/path/to/ans2_file.jsonl', ...], review_file='/path/to/review_file.jsonl',
            reviewer_args={'max_tokens': 1024, 'temperature': 0.2})
        >>> auto_reviewer = AutoReviewerGpt4(**input_kwargs)
        >>> auto_reviewer.run()

    """

    MODEL_NAME = 'gpt-4'

    def __init__(self, prompt_file: str, answer_file_list: list, baseline_file: str, reference_file: str,
                 review_file: str, reviewer_args: dict, cache_file: str, **kwargs):
        super().__init__(**kwargs)

        self.review_file = review_file
        self.prompt_list = jsonl_to_list(prompt_file)
        self.answer_list = [
            jsonl_to_list(answer_file) for answer_file in answer_file_list
        ]
        self.reference_list = jsonl_to_list(reference_file) if reference_file else []
        self.cache_list = jsonl_to_list(cache_file) if cache_file and os.path.isfile(cache_file) else []

        self.reviewer_args = reviewer_args if reviewer_args \
            else self._get_default_args()
        
        self.mode = self.reviewer_args.pop('mode', ArenaMode.PAIRWISE_ALL)
        if self.mode == ArenaMode.PAIRWISE_BASELINE:
            assert baseline_file is not NotImplemented
            self.answer_list.append(jsonl_to_list(baseline_file))
            self.baseline_idx = len(self.answer_list) - 1

        self.is_randomize_output_order = self.reviewer_args.pop(
            EvalTaskConfig.IS_RANDOMIZE_OUTPUT_ORDER, False)
        self.seed = self.reviewer_args.pop(EvalTaskConfig.SEED, 123)

        fn_completion_parser = self.reviewer_args.pop(
            EvalTaskConfig.FN_COMPLETION_PARSER, FnCompletionParser.LMSYS_PARSER
        )
        completion_parser_kwargs = self.reviewer_args.pop(
            EvalTaskConfig.COMPLETION_PARSER_KWARGS, {}
        )
        if isinstance(fn_completion_parser, str):
            fn_completion_parser = getattr(completion_parsers, fn_completion_parser)

        self.fn_completion_parser = partial(
            fn_completion_parser, **completion_parser_kwargs
        )
        self.gpt_predictor = OpenaiGptPredictor(**self.reviewer_args)

    @staticmethod
    def _get_default_args():
        return dict(
            max_tokens=1024,
            temperature=0.2,
            mode=ArenaMode.PAIRWISE_ALL,
            is_randomize_output_order=True,
            fn_completion_parser=FnCompletionParser.LMSYS_PARSER,
            # completion_parser_kwargs=dict(output_format="[[rating_a,rating_b]]")
            seed=123
        )

    @staticmethod
    def gen_prompt(prompts_list: list, type: str, category: str, ques: str, ans1: str,
                   ans2: str, ans_ref: str = None):
        """
        Generate prompt for Auto-reviewer with GPT-4.
        """

        # Default to general category (idx 0)
        target_prompt_dict = prompts_list[0]
        for item in prompts_list:
            is_category_match = category in item['category'] if isinstance(item['category'], list) else item['category'] == category
            is_type_match = item.get('type', 'pairwise') == type
            if is_category_match and is_type_match:
                target_prompt_dict = item
                break

        sys_prompt = target_prompt_dict['system_prompt']
        prompt_template = target_prompt_dict['prompt_template']
        defaults = target_prompt_dict.get('defaults', dict({}))
        output_format = target_prompt_dict.get('output_format', '[[rating_a,rating_b]]')

        user_prompt = prompt_template.format(
            question=ques, answer_a=ans1, answer_b=ans2, ref_answer_1=ans_ref, **defaults)

        return sys_prompt, user_prompt, output_format

    def get_answer_dummy(self, sys_prompt: str, user_prompt: str, output_format) -> list:
        logger.info('Get dummy scores for input prompt ...')
        if output_format == '[[rating_a,rating_b]]':
            ans_list = [round(random.random(), 2), round(random.random(), 2)]
            return ' '.join(str(element) for element in ans_list)
        elif output_format == '[[A]]':
            return random.choice(['[[A]]', '[[B]]', '[[C]]'])

    def get_answer(self, sys_prompt: str, user_prompt: str) -> list:

        input_msg = dict(sys_prompt=sys_prompt, user_prompt=user_prompt)
        input_msg.update(self.reviewer_args)

        # Call GPT-4 predictor
        resp = self.gpt_predictor.predict(**input_msg)
        ans_text = resp['ans_text']
        # model_id = resp['model_id']

        return ans_text

    def get_reviews(self, item: pd.Series) -> dict:

        input_msg = dict(
            ques=item[0]['text'],
            category=item[0]['category'],
            ans1=item[0]['answer'],
            ans2=item[1]['answer'])

        model_a = item[0]['model_id']
        model_b = item[1]['model_id']

        if self.reference_list:
            ans_ref = next((ref for ref in self.reference_list if ref.get('question_id') == item[0]['question_id']), None)
            assert ans_ref['answer']
            input_msg['ans_ref'] = ans_ref['answer']

        sys_prompt, user_prompt, output_format = AutoReviewerGpt4.gen_prompt(
            prompts_list=self.prompt_list,
            type='single' if self.mode == ArenaMode.SINGLE else 'pairwise',
            **input_msg
        )

        # TODO: ONLY FOR TEST
        review_text = self.get_answer_dummy(sys_prompt, user_prompt, output_format)  

        # review_text = self.get_answer(sys_prompt, user_prompt)

        winner = self.fn_completion_parser(review_text, output_format)

        review_result = dict(
            model_a=model_a,
            model_b=model_b,
            win=winner,
            anony=True,
            tstamp=time.time(),
            language=item[0].get('language', 'NA'),
            question_id=item[0]['question_id'],
            category=input_msg['category'],
            question=input_msg['ques'],
            review_text=review_text)
        return review_result

    def run(self):
        print('Run battles for models ...')

        os.makedirs(os.path.dirname(self.review_file), exist_ok=True)

        merge_key = 'question_id'
        merged_ans_df = merge_ques_ans(self.answer_list, merge_key=merge_key)
        merged_ans_df = merged_ans_df.drop(columns=['question_id'])

        if self.mode == ArenaMode.PAIRWISE_ALL:
            battle_pairs = get_battle_pairs(merged_ans_df.columns)
        elif self.mode == ArenaMode.PAIRWISE_BASELINE:
            battle_pairs = get_battle_pairs(merged_ans_df.columns, self.baseline_idx)
        else:
            raise Exception(f'NotSupported mode: {self.mode}')

        res_list = self.cache_list
        for t in battle_pairs:
            pair_df = merged_ans_df[list(t)]
            if self.is_randomize_output_order:
                pair_df.columns = ['output_1', 'output_2']
                pair_df["is_switched_outputs"] = pair_df.apply(
                    lambda x: random_seeded_choice(
                        seed="is_switched_outputs" + x[0]["text"] + str(self.seed),
                        choices=[False, True],
                    ),
                    axis=1,
                )
                pair_df = shuffle_pairwise_preferences(
                    pair_df, pair_df["is_switched_outputs"]
                )
            
            for index, row in pair_df.iterrows():
                model_a = row[0]['model_id']
                model_b = row[1]['model_id']
                question = row[0]['text']
                if any(r['model_a'] == model_a and r['model_b'] == model_b and r['question'] == question for r in res_list):
                    logger.info(f"Use cache review for {model_a} vs {model_b} ...")
                    continue

                row_result = self.get_reviews(row)
                res_list.append(row_result)
                jsonl_dump_data(res_list, self.review_file)
