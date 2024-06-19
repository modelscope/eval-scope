# Quick Start
A stress testing tool that focuses on large language models and can be customized to support various data set formats and different API protocol formats.
## Usage

### Command line  
```bash
llmuses perf --help
usage: llmuses <command> [<args>] perf [-h] --model MODEL [--url URL] [--connect-timeout CONNECT_TIMEOUT] [--read-timeout READ_TIMEOUT] [-n NUMBER] [--parallel PARALLEL] [--rate RATE]
                                       [--log-every-n-query LOG_EVERY_N_QUERY] [--headers KEY1=VALUE1 [KEY1=VALUE1 ...]] [--wandb-api-key WANDB_API_KEY] [--name NAME] [--debug] [--tokenizer-path TOKENIZER_PATH]
                                       [--api API] [--max-prompt-length MAX_PROMPT_LENGTH] [--min-prompt-length MIN_PROMPT_LENGTH] [--prompt PROMPT] [--query-template QUERY_TEMPLATE] [--dataset DATASET]
                                       [--dataset-path DATASET_PATH] [--frequency-penalty FREQUENCY_PENALTY] [--logprobs] [--max-tokens MAX_TOKENS] [--n-choices N_CHOICES] [--seed SEED] [--stop STOP] [--stream]
                                       [--temperature TEMPERATURE] [--top-p TOP_P]

options:
  -h, --help            show this help message and exit
  --model MODEL         The test model name.
  --url URL
  --connect-timeout CONNECT_TIMEOUT
                        The network connection timeout
  --read-timeout READ_TIMEOUT
                        The network read timeout
  -n NUMBER, --number NUMBER
                        How many requests to be made, if None, will will send request base dataset or prompt.
  --parallel PARALLEL   Set number of concurrency request, default 1
  --rate RATE           Number of requests per second. default None, if it set to -1,then all the requests are sent at time 0. Otherwise, we use Poisson process to synthesize the request arrival times. Mutual exclusion
                        with parallel
  --log-every-n-query LOG_EVERY_N_QUERY
                        Logging every n query.
  --headers KEY1=VALUE1 [KEY1=VALUE1 ...]
                        Extra http headers accepts by key1=value1 key2=value2. The headers will be use for each query.You can use this parameter to specify http authorization and other header.
  --wandb-api-key WANDB_API_KEY
                        The wandb api key, if set the metric will be saved to wandb.
  --name NAME           The wandb db result name and result db name, default: {model_name}_{current_time}
  --debug               Debug request send.
  --tokenizer-path TOKENIZER_PATH
                        Specify the tokenizer weight path, used to calculate the number of input and output tokens,usually in the same directory as the model weight.
  --api API             Specify the service api, current support [openai|dashscope]you can define your custom parser with python, and specify the python file path, reference api_plugin_base.py,
  --max-prompt-length MAX_PROMPT_LENGTH
                        Maximum input prompt length
  --min-prompt-length MIN_PROMPT_LENGTH
                        Minimum input prompt length.
  --prompt PROMPT       Specified the request prompt, all the query will use this prompt, You can specify local file via @file_path, the prompt will be the file content.
  --query-template QUERY_TEMPLATE
                        Specify the query template, should be a json string, or local file,with local file, specified with @local_file_path,will will replace model and prompt in the template.
  --dataset DATASET     Specify the dataset [openqa|longalpaca|line_by_line]you can define your custom dataset parser with python, and specify the python file path, reference dataset_plugin_base.py,
  --dataset-path DATASET_PATH
                        Path to the dataset file, Used in conjunction with dataset. If dataset is None, each line defaults to a prompt.
  --frequency-penalty FREQUENCY_PENALTY
                        The frequency_penalty value.
  --logprobs            The logprobs.
  --max-tokens MAX_TOKENS
                        The maximum number of tokens can be generated.
  --n-choices N_CHOICES
                        How may chmpletion choices to generate.
  --seed SEED           The random seed.
  --stop STOP           The stop generating tokens.
  --stop-token-ids      Set the stop token ids.
  --stream              Stream output with SSE.
  --temperature TEMPERATURE
                        The sample temperature.
  --top-p TOP_P         Sampling top p.

```
### The result:
```bash
 Total requests: 10
 Succeed requests: 10
 Failed requests: 0
 Average QPS: 0.256
 Average latency: 3.859
 Throughput(average output tokens per second): 23.317
 Average time to first token: 0.007
 Average input tokens per request: 21.800
 Average output tokens per request: 91.100
 Average time per output token: 0.04289
 Average package per request: 93.100
 Average package latency: 0.042
 Percentile of time to first token: 
     p50: 0.0021
     p66: 0.0023
     p75: 0.0025
     p80: 0.0030
     p90: 0.0526
     p95: 0.0526
     p98: 0.0526
     p99: 0.0526
 Percentile of request latency: 
     p50: 3.9317
     p66: 3.9828
     p75: 4.0153
     p80: 7.2801
     p90: 7.7003
     p95: 7.7003
     p98: 7.7003
     p99: 7.7003
```
### Request parameter  
You can set request parameter's in query-template and with (--stop,--stream,--temperature, etc), the argument parameter will replace or add to the request.
#### request with parameters
Sample request llama3 vllm openai compatible interface.
```bash
llmuses perf --url 'http://127.0.0.1:8000/v1/chat/completions' --parallel 128 --model 'qwen' --log-every-n-query 10 --read-timeout=120 --dataset-path './datasets/open_qa.jsonl' -n 1 --max-prompt-length 128000 --api openai --stream --stop '<|im_end|>' --dataset openqa --debug
```
llmuses perf 'http://host:port/v1/chat/completions' --parallel 128 --model 'qwen' --log-every-n-query 10 --read-timeout=120 -n 10000 --max-prompt-length 128000 --tokenizer-path "THE_PATH_TO_TOKENIZER/Qwen1.5-32B/" --api openai --query-template '{"model": "%m", "messages": [{"role": "user","content": "%p"}], "stream": true,"skip_special_tokens": false,"stop": ["<|im_end|>"]}' --dataset openqa --dataset-path 'THE_PATH_TO_DATASETS/open_qa.jsonl'

#### query-template usage.
When you need to process more complex requests, you can use templates to simplify the command line.
If both the template and the parameter are present, the value in the parameter will prevail.
Query-template example:
```bash
llmuses perf --url 'http://127.0.0.1:8000/v1/chat/completions' --parallel 12 --model 'llama3' --log-every-n-query 10 --read-timeout=120 -n 1 --max-prompt-length 128000 --api openai --query-template '{"model": "%m", "messages": [], "stream": true, "stream_options":{"include_usage": true},"n": 3, "stop_token_ids": [128001, 128009]}' --dataset openqa --dataset-path './datasets/open_qa.jsonl'
```
For messages, the dataset processor message will replace messages in the query-template.

#### Start the client
```bash
# test openai service
llmuses perf --url 'https://api.openai.com/v1/chat/completions' --parallel 1 --headers 'Authorization=Bearer YOUR_OPENAI_API_KEY' --model 'gpt-4o' --dataset-path 'THE_DATA_TO/open_qa.jsonl'  --log-every-n-query 10 --read-timeout=120  -n 100 --max-prompt-length 128000 --api openai --stream --dataset openqa
```

```bash
##### open qa dataset and 
#### dataset address: https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/blob/main/open_qa.jsonl
llmuses perf --url 'http://IP:PORT/v1/chat/completions' --parallel 1 --model 'qwen' --log-every-n-query 1 --read-timeout=120 -n 1000 --max-prompt-length 128000 --tokenizer-path "THE_PATH_TO_TOKENIZER/Qwen1.5-32B/" --api openai --query-template '{"model": "%m", "messages": [{"role": "user","content": "%p"}], "stream": true,"skip_special_tokens": false,"stop": ["<|im_end|>"]}' --dataset openqa --dataset-path 'THE_PATH_TO_DATASETS/open_qa.jsonl'
```

#### How to log metrics to wandb
--wandb-api-key 'your_wandb_api_key'  --name 'name_of_wandb_and_result_db'  

![wandb sample](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/wandb_sample.png)

#### How to debug
--debug 
with --debug option, we will output the request and response.

#### How to analysis result.
The tool will save all data during the test to the sqlite3 database, including requests and responses. You can analyze the test data after the test.
```python
import sqlite3
import base64
import pickle
import json
result_db_path = 'db_name.db'
con = sqlite3.connect(result_db_path)
query_sql = "SELECT request, response_messages, prompt_tokens, completion_tokens \
                FROM result WHERE success='True'"
# how to save base64.b64encode(pickle.dumps(benchmark_data["request"])).decode("ascii"), 
with con:
    rows = con.execute(query_sql).fetchall()
    if len(rows) > 0:
        for row in rows:
            request = row[0]
            responses = row[1]
            request = base64.b64decode(request)
            request = pickle.loads(request)
            responses = base64.b64decode(responses)
            responses = pickle.loads(responses)
            response_content = ''
            for response in responses:
                response = json.loads(response)
                response_content += response['choices'][0]['delta']['content']
            print('prompt: %s, tokens: %s, completion: %s, tokens: %s' % (request['messages'][0]['content'], row[2], response_content, row[3]))
```

#### Api supported
Currently supports openai, dashscope, zhipu API request. You can specify api with --api.
You can custom your request with --query-template, you can specify a json string as:
'{"model": "%m", "messages": [{"role": "user","content": "%p"}], "stream": true,"skip_special_tokens": false,"stop": ["<|im_end|>"]}'
or a local file with @to_query_template_path. We will replace %m with model and %p with prompt.

#### How to extend API
To extend api you can create sub class of `ApiPluginBase`, annotation with @register_api("name_of_api")
with build_request build request via model, prompt, and
query_template. you can reference opanai_api.py
parse_responses return number_of_prompt_tokens and number_of_completion_tokens.
```python
class ApiPluginBase:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        
    @abstractmethod
    def build_request(self, messages: List[Dict], param: QueryParameters)->Dict:
        """Build a api request body.

        Args:
            messages (List[Dict]): The messages generated by dataset.
            param (QueryParameters): The query parameters.

        Raises:
            NotImplementedError: Not implemented.

        Returns:
            Dict: The api request body.
        """
        raise NotImplementedError
    
    @abstractmethod
    def parse_responses(self, 
                        responses: List, 
                        request: Any=None,
                        **kwargs:Any) -> Tuple[int, int]:
        """Parser responses and return number of request and response tokens.

        Args:
            responses (List[bytes]): List of http response body, for stream output,
                there are multiple responses, each is bytes, for general only one. 
            request (Any): The request body.

        Returns:
            Tuple: (Number of prompt_tokens and number of completion_tokens).
        """
        raise NotImplementedError  
```

#### Dataset supported
Currently supports line by line, longalpaca and openqa data set.
line by line with each line as a prompt.
longalpaca will get item['instruction'] as prompt.
openqa will get item['question'] as prompt.

#### How to extension dataset.
To extend api you can create sub class of `DatasetPluginBase`, annotation with @register_dataset('name_of_dataset')
implement build_prompt api return a prompt.
```python
class DatasetPluginBase:
    def __init__(self, query_parameters: QueryParameters):
        """Build data set plugin

        Args:
            dataset_path (str, optional): The input dataset path. Defaults to None.
        """
        self.query_parameters = query_parameters

    def __next__(self):
        for item in self.build_messages():
            yield item
        raise StopIteration

    def __iter__(self):
        return self.build_messages()
    
    @abstractmethod
    def build_messages(self)->Iterator[List[Dict]]:
        """Build the request.

        Raises:
            NotImplementedError: The request is not impletion.

        Yields:
            Iterator[List[Dict]]: Yield request messages.
        """
        raise NotImplementedError
    
    def dataset_line_by_line(self, dataset: str)->Iterator[str]:
        """Get content line by line of dataset.

        Args:
            dataset (str): The dataset path.

        Yields:
            Iterator[str]: Each line of file.
        """
        with open(dataset, 'r', encoding='utf-8') as f:
            for line in f:
                yield line
    
    def dataset_json_list(self, dataset: str)->Iterator[Dict]:
        """Read data from file which is list of requests.
           Sample: https://huggingface.co/datasets/Yukang/LongAlpaca-12k

        Args:
            dataset (str): The dataset path.

        Yields:
            Iterator[Dict]: The each request object.
        """
        with open(dataset, 'r', encoding='utf-8') as f:
            content = f.read()
        data = json.loads(content)
        for item in data:
            yield item      
```