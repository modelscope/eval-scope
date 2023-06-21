# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean

from evals.constants import MetricsConstant
from evals.metrics.bundled_rouge_score import rouge_scorer
from evals.preprocess.tokenizers.gpt2_tokenizer import DummyTokenizer
from tqdm import tqdm

HERE = Path(__file__).absolute().parent

logger = logging.getLogger(__name__)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                  tokenizer=DummyTokenizer())


def compute_rouge_score(predict_l, reference_l):
    assert len(predict_l) == len(reference_l)
    if len(predict_l) == 0:
        tmp_d = dict()
        for key in MetricsConstant.ROUGE_KEYS:
            tmp_d[key] = 0
        return tmp_d

    result = defaultdict(list)
    for p, r in tqdm(zip(predict_l, reference_l)):
        one_sample = compute_rouge_score_one_sample(p, r)
        for rouge_key in MetricsConstant.ROUGE_KEYS:
            result[rouge_key].append(one_sample[rouge_key])
    rlt = {}
    for rouge_key in MetricsConstant.ROUGE_KEYS:
        rlt[rouge_key] = mean(result[rouge_key]) * 100 if rouge_key in result \
            else MetricsConstant.INVALID_VALUE
    return rlt


def compute_rouge_score_one_sample(predict, reference):
    result = dict()
    for p, r in zip(predict, reference):
        score = scorer.score(p, r)
        result['rouge-1-r'] = score['rouge1'].recall
        result['rouge-1-p'] = score['rouge1'].precision
        result['rouge-1-f'] = score['rouge1'].fmeasure
        result['rouge-2-r'] = score['rouge2'].recall
        result['rouge-2-p'] = score['rouge2'].precision
        result['rouge-2-f'] = score['rouge2'].fmeasure
        result['rouge-l-r'] = score['rougeL'].recall
        result['rouge-l-p'] = score['rougeL'].precision
        result['rouge-l-f'] = score['rougeL'].fmeasure

    return result


def _to_table(final_result) -> str:
    table = []
    # step 1. table header
    all_tasks = ['', 'total']
    all_tasks.extend(final_result['all_tasks'].split(','))
    table.append('\t'.join(all_tasks))

    # step 2. table row
    for rouge_key in MetricsConstant.ROUGE_KEYS:
        row = [rouge_key]
        for task in all_tasks:
            if not task:
                continue
            elif task == 'total':
                row.append(
                    f'{final_result["total"]["rouge"][rouge_key] :0.2f}')
            else:
                row.append(
                    f'{final_result["tasks"][task]["rouge"][rouge_key] :0.2f}')
        table.append('\t'.join(row))

    return '\n'.join(table)


def run_rouge_eval(data_l, md_level=2, report_metric_key='rouge-l-f'):
    print(f"{'#' * md_level} Rouge Eval")
    for data in tqdm(data_l):
        data['rouge'] = compute_rouge_score_one_sample(
            data['gen_tok_str'], data['reference_tok_str'])
    task_data_d = defaultdict(list)
    for data in data_l:
        for task in data['task_tags']:
            task_data_d[task].append(data)

    total_rouge = mean([data['rouge'][report_metric_key] for data in data_l])
    print(f'[total], count: {len(data_l)}, {report_metric_key}: '
          f'{total_rouge * 100:0.2f}%')

    for task, task_data in task_data_d.items():
        task_rouge = mean(
            [data['rouge'][report_metric_key] for data in task_data])
        print(
            f'[{task}], count: {len(task_data_d[task])}, {report_metric_key}: '
            f'{task_rouge * 100:0.2f}%')
