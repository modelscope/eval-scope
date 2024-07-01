# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) OpenCompass.

import functools
import importlib
import importlib.util
import os
import re
import json
import random
import sys
from typing import Any, Union, Dict, Tuple, List
import hashlib
import torch.nn.functional as F

import jsonlines as jsonl
import yaml

from llmuses.constants import DumpMode, OutputsStructure
from llmuses.utils.logger import get_logger

logger = get_logger()

TEST_LEVEL_LIST = [0, 1]

# Example: export TEST_LEVEL_LIST=0,1
TEST_LEVEL_LIST_STR = 'TEST_LEVEL_LIST'


def test_level_list():
    global TEST_LEVEL_LIST
    if TEST_LEVEL_LIST_STR in os.environ:
        TEST_LEVEL_LIST = [
            int(x) for x in os.environ[TEST_LEVEL_LIST_STR].split(',')
        ]

    return TEST_LEVEL_LIST


def jsonl_to_list(jsonl_file):
    """
    Read jsonl file to list.

    Args:
        jsonl_file: jsonl file path.

    Returns:
        list: list of lines. Each line is a dict.
    """
    res_list = []
    with jsonl.open(jsonl_file, mode='r') as reader:
        for line in reader.iter(
                type=dict, allow_none=True, skip_invalid=False):
            res_list.append(line)
    return res_list


def jsonl_to_reader(jsonl_file):
    """
    Read jsonl file to reader object.

    Args:
        jsonl_file: jsonl file path.

    Returns:
        reader: jsonl reader object.
    """
    with jsonl.open(jsonl_file, mode='r') as reader:
        return reader


def jsonl_to_csv():
    pass


def dump_jsonl_data(data_list, jsonl_file, dump_mode=DumpMode.OVERWRITE):
    """
    Dump data to jsonl file.

    Args:
        data_list: data list to be dumped.  [{'a': 'aaa'}, ...]
        jsonl_file: jsonl file path.
        dump_mode: dump mode. It can be 'overwrite' or 'append'.
    """
    if not jsonl_file:
        raise ValueError('output file must be provided.')

    jsonl_file = os.path.expanduser(jsonl_file)

    if dump_mode == DumpMode.OVERWRITE:
        dump_mode = 'w'
    elif dump_mode == DumpMode.APPEND:
        dump_mode = 'a'
    with jsonl.open(jsonl_file, mode=dump_mode) as writer:
        writer.write_all(data_list)
    logger.info(f'Dump data to {jsonl_file} successfully.')


def yaml_to_dict(yaml_file) -> dict:
    """
    Read yaml file to dict.
    """
    with open(yaml_file, 'r') as f:
        try:
            stream = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f'{e}')
            raise e

    return stream


def dict_to_yaml(d: dict, yaml_file: str):
    """
    Dump dict to yaml file.
    """
    with open(yaml_file, 'w') as f:
        yaml.dump(d, f, default_flow_style=False)
    logger.info(f'Dump data to {yaml_file} successfully.')


def json_to_dict(json_file) -> dict:
    """
    Read json file to dict.
    """
    with open(json_file, 'r') as f:
        try:
            stream = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f'{e}')
            raise e

    return stream


def get_obj_from_cfg(eval_class_ref: Any, *args, **kwargs) -> Any:
    module_name, spliter, cls_name = eval_class_ref.partition(':')

    try:
        obj_cls = importlib.import_module(module_name)
    except ImportError as e:
        logger.error(f'{e}')
        raise e

    if spliter:
        for attr in cls_name.split('.'):
            obj_cls = getattr(obj_cls, attr)

    return functools.partial(obj_cls, *args, **kwargs)


def markdown_table(header_l, data_l):
    md_str = f'| {" | ".join(header_l)} |'
    md_str += f'\n| {" | ".join(["---"] * len(header_l))} |'
    for data in data_l:
        if isinstance(data, str):
            data = [data]
        assert len(data) <= len(header_l)
        tmp = data + [''] * (len(header_l) - len(data))
        md_str += f'\n| {" | ".join(tmp)} |'
    return md_str


def random_seeded_choice(seed: Union[int, str, float], choices, **kwargs):
    """Random choice with a (potentially string) seed."""
    return random.Random(seed).choices(choices, k=1, **kwargs)[0]


def gen_hash(name: str):
    return hashlib.md5(name.encode(encoding='UTF-8')).hexdigest()


def dict_torch_dtype_to_str(d: Dict[str, Any]) -> dict:
    """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.

        Refer to: https://github.com/huggingface/transformers/pull/16065/files for details.
        """
    if d.get('torch_dtype', None) is not None and not isinstance(d['torch_dtype'], str):
        d['torch_dtype'] = str(d['torch_dtype']).split('.')[1]

    for value in d.values():
        if isinstance(value, dict):
            dict_torch_dtype_to_str(value)

    return d


class ResponseParser:

    @staticmethod
    def parse_first_capital(text: str) -> str:
        for t in text:
            if t.isupper():
                return t
        return ''

    @staticmethod
    def parse_last_capital(text: str) -> str:
        for t in text[::-1]:
            if t.isupper():
                return t
        return ''

    @staticmethod
    def parse_first_option_with_choices(text: str, options: list) -> str:
        """
        Find first valid option for text.

        Args:
            text: The text to parse.
            options: The options to find. e.g. ['A', 'B', 'C', 'D']
        """
        options_concat = '|'.join([str(i) for i in options])

        patterns = [
            f'答案是?\s?([{options_concat}])',
            f'答案是?\s?：([{options_concat}])',
            f'答案是?\s?:([{options_concat}])',
            f'答案应该?是\s?([{options_concat}])',
            f'答案应该?选\s?([{options_concat}])',
            f'答案为\s?([{options_concat}])',
            f'答案选\s?([{options_concat}])',
            f'选择?\s?([{options_concat}])',
            f'故选?\s?([{options_concat}])'
            f'只有选?项?\s?([{options_concat}])\s?是?对',
            f'只有选?项?\s?([{options_concat}])\s?是?错',
            f'只有选?项?\s?([{options_concat}])\s?不?正确',
            f'只有选?项?\s?([{options_concat}])\s?错误',
            f'说法不?对选?项?的?是\s?([{options_concat}])',
            f'说法不?正确选?项?的?是\s?([{options_concat}])',
            f'说法错误选?项?的?是\s?([{options_concat}])',
            f'([{options_concat}])\s?是正确的',
            f'([{options_concat}])\s?是正确答案',
            f'选项\s?([{options_concat}])\s?正确',
            f'所以答\s?([{options_concat}])',
            f'1.\s?([{options_concat}])[.。$]?$',
            f'所以\s?([{options_concat}][.。$]?$)',
            f'所有\s?([{options_concat}][.。$]?$)',
            f'[\s，：:,]([{options_concat}])[。，,\.]?$',
            f'[\s，,：:][故即]([{options_concat}])[。\.]?$',
            f'[\s，,：:]因此([{options_concat}])[。\.]?$',
            f'[是为。]\s?([{options_concat}])[。\.]?$',
            f'因此\s?([{options_concat}])[。\.]?$',
            f'显然\s?([{options_concat}])[。\.]?$',
            f'答案是\s?(\S+)(?:。|$)',
            f'答案应该是\s?(\S+)(?:。|$)',
            f'答案为\s?(\S+)(?:。|$)',
            f'答案是(.*?)[{options_concat}]',
            f'答案为(.*?)[{options_concat}]',
            f'固选(.*?)[{options_concat}]',
            f'答案应该是(.*?)[{options_concat}]',
            f'[Tt]he answer is [{options_concat}]',
            f'[Tt]he correct answer is [{options_concat}]',
            f'[Tt]he correct answer is:\n[{options_concat}]',
            f'(\s|^)[{options_concat}][\s。，,\.$]',  # noqa
            f'[{options_concat}]',
            f'^选项\s?([{options_concat}])',
            f'^([{options_concat}])\s?选?项',
            f'(\s|^)[{options_concat}][\s。，,：:\.$]',
            f'(\s|^)[{options_concat}](\s|$)',
            f'1.\s?(.*?)$',
        ]

        regexes = [re.compile(pattern) for pattern in patterns]
        for regex in regexes:
            match = regex.search(text)
            if match:
                outputs = match.group(0)
                for i in options:
                    if i in outputs:
                        return i
        return ''

    @staticmethod
    def parse_first_option(text: str) -> str:
        """
        Find first valid option for text.

        Args:
            text: The text to parse.
        """
        patterns = [
            r"[Aa]nswer:\s*(\w+)",
            r"[Tt]he correct answer is:\s*(\w+)",
            r"[Tt]he correct answer is:\n\s*(\w+)",
            r"[Tt]he correct answer is:\n\n-\s*(\w+)",
            r"[Tt]he answer might be:\n\n-\s*(\w+)",
            r"[Tt]he answer is \s*(\w+)",
        ]

        regexes = [re.compile(pattern) for pattern in patterns]
        for regex in regexes:
            match = regex.search(text)
            if match:
                return match.group(1)
        return ''

    @staticmethod
    def parse_first_capital_multi(text: str) -> str:
        match = re.search(r'([A-D]+)', text)
        if match:
            return match.group(1)
        return ''

    @staticmethod
    def parse_last_option(text: str, options: str) -> str:
        match = re.findall(rf'([{options}])', text)
        if match:
            return match[-1]
        return ''


def make_outputs_dir(root_dir: str, datasets: list, model_id: str, model_revision: str):
    # model_revision = model_revision if model_revision is not None else 'none'
    # now = datetime.datetime.now()
    # format_time = now.strftime('%Y%m%d_%H%M%S')
    # outputs_name = format_time + '_' + 'default' + '_' + model_id.replace('/', '_') + '_' + model_revision
    # outputs_dir = os.path.join(work_dir, outputs_name)
    # dataset_name = dataset_id.replace('/', '_')
    # outputs_dir = os.path.join(work_dir, dataset_name)

    if not model_id:
        model_id = 'default'
    model_id = model_id.replace('/', '_')

    if not model_revision:
        model_revision = 'default'

    outputs_dir = os.path.join(root_dir,
                               f"eval_{'-'.join(datasets)}_{model_id}_{model_revision}")

    return outputs_dir


def process_outputs_structure(outputs_dir: str, is_make: bool = True) -> dict:
    logs_dir = os.path.join(outputs_dir, 'logs')
    predictions_dir = os.path.join(outputs_dir, 'predictions')
    reviews_dir = os.path.join(outputs_dir, 'reviews')
    reports_dir = os.path.join(outputs_dir, 'reports')
    configs_dir = os.path.join(outputs_dir, 'configs')

    if is_make:
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)
        os.makedirs(reviews_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

    outputs_structure = {
        OutputsStructure.LOGS_DIR: logs_dir,
        OutputsStructure.PREDICTIONS_DIR: predictions_dir,
        OutputsStructure.REVIEWS_DIR: reviews_dir,
        OutputsStructure.REPORTS_DIR: reports_dir,
        OutputsStructure.CONFIGS_DIR: configs_dir,
    }

    return outputs_structure


def import_module_util(import_path_prefix: str, module_name: str, members_to_import: list) -> dict:
    """
    Import module utility function.

    Args:
        import_path_prefix: e.g. 'llmuses.benchmarks.'
        module_name: The module name to import. e.g. 'mmlu'
        members_to_import: The members to import.
            e.g. ['DATASET_ID', 'SUBJECT_MAPPING', 'SUBSET_LIST', 'DataAdapterClass']

    Returns:
        dict: imported modules map. e.g. {'DATASET_ID': 'mmlu', 'SUBJECT_MAPPING': {...}, ...}
    """
    imported_modules = {}
    module = importlib.import_module(import_path_prefix + module_name)
    for member_name in members_to_import:
        imported_modules[member_name] = getattr(module, member_name)

    return imported_modules


def normalize_score(score: Union[float, dict], keep_num: int = 4) -> Union[float, dict]:
    """
    Normalize score.

    Args:
        score: input score, could be float or dict. e.g. 0.12345678 or {'acc': 0.12345678, 'f1': 0.12345678}
        keep_num: number of digits to keep.

    Returns:
        Union[float, dict]: normalized score. e.g. 0.1234 or {'acc': 0.1234, 'f1': 0.1234}
    """
    if isinstance(score, float):
        score = round(score, keep_num)
    elif isinstance(score, dict):
        score = {k: round(v, keep_num) for k, v in score.items()}
    else:
        logger.warning(f'Unknown score type: {type(score)}')

    return score


def split_str_parts_by(text: str, delimiters: List[str]):
    """Split the text field into parts.
    Args:
        text: A text to be split.
        delimiters: The delimiters.
    Returns:
        The split text in list of dicts.
    """
    all_start_chars = [d[0] for d in delimiters]
    all_length = [len(d) for d in delimiters]

    text_list = []
    last_words = ''

    while len(text) > 0:
        for char_idx, char in enumerate(text):
            match_index = [
                idx for idx, start_char in enumerate(all_start_chars)
                if start_char == char
            ]
            is_delimiter = False
            for index in match_index:
                if text[char_idx:char_idx
                        + all_length[index]] == delimiters[index]:
                    if last_words:
                        if text_list:
                            text_list[-1]['content'] = last_words
                        else:
                            text_list.append({
                                'key': '',
                                'content': last_words
                            })
                    last_words = ''
                    text_list.append({'key': delimiters[index]})
                    text = text[char_idx + all_length[index]:]
                    is_delimiter = True
                    break
            if not is_delimiter:
                last_words += char
            else:
                break
        if last_words == text:
            text = ''

    text_list[-1]['content'] = last_words
    return text_list


def calculate_loss_scale(response: str,
                         use_loss_scale=False
                         ) -> Tuple[List[str], List[float]]:
    """Calculate the loss scale by splitting the agent response.
    This algorithm comes from paper: https://arxiv.org/pdf/2309.00986.pdf
    Agent response format:
    ```text
        Thought: you should always think about what to do
        Action: the action to take, should be one of the above tools[fire_recognition,
            fire_alert, call_police, call_fireman]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
    ```
    Args:
        response: The response text
        use_loss_scale: Use weighted loss. With this, some part of the loss will be enhanced to improve performance.
    Returns:
        A tuple of agent response parts and their weights.
    """
    if 'Action:' in response and 'Observation:' in response and use_loss_scale:
        agent_keyword = [
            'Action:', 'Action Input:', 'Thought:', 'Final Answer:',
            'Observation:'
        ]
        agent_parts = split_str_parts_by(response, agent_keyword)
        weights = []
        agent_content = []
        for c in agent_parts:
            if c['key'] in ('Action:', 'Action Input:'):
                weights += [2.0]
                weights += [2.0]
            elif c['key'] in ('Thought:', 'Final Answer:', ''):
                weights += [1.0]
                weights += [1.0]
            elif c['key'] in ('Observation:', ):
                weights += [2.0]
                weights += [0.0]
            agent_content.append(c['key'])
            agent_content.append(c['content'])
        return agent_content, weights
    else:
        return [response], [1.0]


def get_bucket_sizes(max_length: int) -> List[int]:
    return [max_length // 4 * (i + 1) for i in range(4)]


def _get_closet_bucket(bucket_sizes, data_length):
    """Select the one from bucket_sizes that is closest in distance to
    data_length. This is required for TorchAcc.
    """
    cloest_length = sys.maxsize
    for b in bucket_sizes:
        if b == data_length or ((b < cloest_length) and (b > data_length)):
            cloest_length = b

    if cloest_length == sys.maxsize:
        bucket_sizes.append(data_length)
        cloest_length = data_length

    return cloest_length


def pad_and_split_batch(padding_to, input_ids, attention_mask, labels,
                        loss_scale, max_length, tokenizer, rank, world_size):
    if padding_to is None:
        longest_len = input_ids.shape[-1]
        bucket_sizes = get_bucket_sizes(max_length)
        bucket_data_length = _get_closet_bucket(bucket_sizes, longest_len)
        padding_length = bucket_data_length - input_ids.shape[1]
        input_ids = F.pad(input_ids, (0, padding_length), 'constant',
                          tokenizer.pad_token_id)
        attention_mask = F.pad(attention_mask, (0, padding_length), 'constant',
                               0)
        if loss_scale:
            loss_scale = F.pad(loss_scale, (0, padding_length), 'constant', 0.)
        labels = F.pad(labels, (0, padding_length), 'constant', -100)

    # manully split the batch to different DP rank.
    batch_size = input_ids.shape[0] // world_size
    if batch_size > 0:
        start = rank * batch_size
        end = (rank + 1) * batch_size
        input_ids = input_ids[start:end, :]
        attention_mask = attention_mask[start:end, :]
        labels = labels[start:end, :]
        if loss_scale:
            loss_scale = loss_scale[start:end, :]
    return input_ids, attention_mask, labels, loss_scale


def get_dist_setting() -> Tuple[int, int, int, int]:
    """return rank, local_rank, world_size, local_world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', 1))
    return rank, local_rank, world_size, local_world_size


def use_torchacc() -> bool:
    return os.getenv('USE_TORCHACC', '0') == '1'


def is_module_installed(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def get_module_path(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec and spec.origin:
        return os.path.abspath(spec.origin)
    else:
        raise ValueError(f'Cannot find module: {module_name}')


def get_valid_list(input_list, candidate_list):
    """
    Get the valid and invalid list from input_list based on candidate_list.
    Args:
        input_list: The input list.
        candidate_list: The candidate list.

    Returns:
        valid_list: The valid list.
        invalid_list: The invalid list.
    """
    return [i for i in input_list if i in candidate_list], \
           [i for i in input_list if i not in candidate_list]


def get_latest_folder_path(work_dir):
    from datetime import datetime
    # Get all subdirectories in the work_dir
    folders = [f for f in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, f))]

    # Get the timestamp（YYYYMMDD_HHMMSS）
    timestamp_pattern = re.compile(r'^\d{8}_\d{6}$')

    # Filter out the folders
    timestamped_folders = [f for f in folders if timestamp_pattern.match(f)]

    if not timestamped_folders:
        print(f'>> No timestamped folders found in {work_dir}!')
        return None

    # timestamp parser
    def parse_timestamp(folder_name):
        return datetime.strptime(folder_name, "%Y%m%d_%H%M%S")

    # Find the latest folder
    latest_folder = max(timestamped_folders, key=parse_timestamp)

    return os.path.join(work_dir, latest_folder)


def csv_to_list(file_path: str) -> List[dict]:
    import csv

    with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        result = [row for row in csv_reader]

    return result
