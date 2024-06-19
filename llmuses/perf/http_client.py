"""LLM performance benchmark client.
"""
import argparse
import asyncio
from dataclasses import dataclass
import functools
import json
import logging
import platform
import signal
import sqlite3
import os
import time
import base64
import pickle
import importlib.util
import sys
from typing import List, Dict, Optional
from datetime import datetime, timezone
import aiohttp
from http import HTTPStatus
import aiohttp
import numpy as np
from llmuses.perf.plugin_registry import api_registry, dataset_registry
from llmuses.perf.query_parameters import QueryParameters
from llmuses.perf.server_sent_event import ServerSentEvent
# for plugin registry
from llmuses.perf.dashscope_api import DashScopeApiPlugin
from llmuses.perf.openai_api import OpenaiPlugin
from llmuses.perf.datasets.line_by_line import LineByLineDatasetPlugin
from llmuses.perf.datasets.longalpaca_12k import LongAlpacaDatasetPlugin
from llmuses.perf.datasets.openqa import OpenqaDatasetPlugin
from llmuses.perf.custom_api import CustomPlugin
from llmuses.perf._logging import logger

__all__ = [
    DashScopeApiPlugin,
    OpenaiPlugin,
    CustomPlugin,
    LineByLineDatasetPlugin,
    LongAlpacaDatasetPlugin,
    OpenqaDatasetPlugin,
]

_query_send_completed = False
_data_process_completed = False
_table_name = "result"

UNLIMITED_RATE = -1


async def on_request_start(session, context, params):
    logger.debug(f'Starting request: <{params}>')


async def on_request_chunk_sent(session, context, params):
    logger.debug(f'Request body: {params}')


async def on_response_chunk_received(session, context, params):
    logger.debug(f'Response info: <{params}>')


class AioHttpClient:
    def __init__(self,
                 url: str,
                 conn_timeout: int = 120,
                 read_timeout: int = 120,
                 headers: Dict = None,
                 debug: bool = False):
        # one client only has one connection
        client_timeout = aiohttp.ClientTimeout(total=read_timeout + conn_timeout,
                                               connect=conn_timeout,
                                               sock_read=read_timeout)
        self.debug = debug
        if debug:
            logger.setLevel(level=logging.DEBUG)
            trace_config = aiohttp.TraceConfig()
            trace_config.on_request_start.append(on_request_start)
            trace_config.on_request_chunk_sent.append(on_request_chunk_sent)
            # not support server sent event(stream=true)
            trace_config.on_response_chunk_received.append(on_response_chunk_received)
        self.client = aiohttp.ClientSession(trace_configs=[trace_config] if debug else [],
                                            connector=aiohttp.TCPConnector(limit=1),
                                            timeout=client_timeout)
        ua = "modelscope_bench"
        self.headers = {"user-agent": ua}
        if headers:
            self.headers.update(headers)
        self.url = url

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.close()

    async def aio_call(self):
        response = self._handle_request()
        if self.stream:
            return (item async for item in response)
        else:
            result = await response.__anext__()
            try:
                await response.__anext__()
            except StopAsyncIteration:
                pass
            return result

    async def _handle_stream(self, response):
        is_error = False
        status_code = response.status
        async for line in response.content:
            if line:
                line = line.decode("utf8")
                line = line.rstrip("\n").rstrip("\r")
                if self.debug:
                    logger.debug(line)
                sse_msg = ServerSentEvent.decode(line)
                if not sse_msg:
                    continue
                if sse_msg.event and sse_msg.event == "error":  # dashscope error
                    is_error = True

                if sse_msg.data:
                    if sse_msg.data.startswith("[DONE]"):  # openai api completed
                        break
                    yield (is_error, status_code, sse_msg.data)
                    # yield data

    async def _handle_response(self, response: aiohttp.ClientResponse):
        if (response.status == HTTPStatus.OK and "text/event-stream" in response.content_type):
            async for is_error, status_code, data in self._handle_stream(response):
                yield (is_error, status_code, data)
        elif response.status == HTTPStatus.OK and "application/json" in response.content_type:
            content = await response.json()
            if 'object' in content and content['object'] == 'error':
                yield(True, content['code'], content['message'])
            else:
                yield (False, HTTPStatus.OK, json.dumps(content))            
        elif response.status == HTTPStatus.OK:
            content = await response.read()
            yield (False, HTTPStatus.OK, content)
        else:
            if "application/json" in response.content_type:
                error = await response.json()
                yield (True, response.status, json.dumps(error))
            elif "text/event-stream" in response.content_type:
                async for _, _, data in self._handle_stream(response):
                    error = json.loads(data)
                yield (True, response.status, error)
            else:
                msg = await response.read()
                yield (True, response.status, msg.decode('utf-8'))

    async def post(self, body):
        try:
            headers = {"Content-Type": "application/json", **self.headers}
            response = await self.client.request("POST",
                                                 url=self.url,
                                                 json=body,
                                                 headers=headers)
            async with response:
                async for rsp in self._handle_response(response):
                    yield rsp
        except aiohttp.ClientConnectorError as e:
            logger.error(e)
            raise e
        except Exception as e:
            logger.error(e)
            raise e


def dynamic_import_module(dynamic_module_file_path: str):
    """Dynamic import input output process python file.

    Args:
        dynamic_module_file_path (str): The absolute path of the 
            input output process python path, or name of the format, 
            system support openai, dashscope format.
    """
    module_name = 'module_request_response_parser'

    dynamic_module_spec = importlib.util.spec_from_file_location(module_name, dynamic_module_file_path)
    dynamic_module = importlib.util.module_from_spec(dynamic_module_spec)
    sys.modules[module_name] = dynamic_module
    dynamic_module_spec.loader.exec_module(dynamic_module)
    return dynamic_module

def get_query_template(args):
    if args.query_template.startswith('@'):
        # read from file
        with open(args.query_template[1:], 'r') as f:
            content = f.read()
            return content.strip()
    return args.query_template.strip()

async def dispatch_requests_worker(request_queue: asyncio.Queue, args):
    query_generator_class = api_registry(args.api)
    if not query_generator_class:
        print('Can not find query generator: %s'%args.api)
    query_generator = query_generator_class(args.tokenizer_path)
    total_query_counter = 0
    query_parameters = QueryParameters(args)
    if args.prompt is not None:
        if args.prompt.startswith("@"):  # read local as prompt, same as curl --data
            with open(args.prompt, 'r', encoding='utf-8') as f:
                prompt = f.read()
        else:
            prompt = args.prompt
        messages = {'role': 'user', 'content': prompt}
        request = query_generator.build_request(messages, query_parameters)
        if args.number is None:
            await request_queue.put(request)
        else:
            for i in range(args.number):
                if args.rate == UNLIMITED_RATE:
                    await request_queue.put(request)
                else:
                    interval = np.random.exponential(1.0 / args.rate)
                    # The next request will be sent after the interval.
                    await asyncio.sleep(interval)
                    await request_queue.put(request)
    elif args.dataset_path is not None:
        # Ensure sufficient quantity of queries.
        while True:
            message_generator_class = dataset_registry.get_class(args.dataset)
            if not message_generator_class:
                print('Can not find dataset: %s plugin.'%(args.dataset))
                sys.exit(1)
            message_generator = message_generator_class(query_parameters)
            for messages in message_generator:
                request = query_generator.build_request(messages, query_parameters)
                if request is None:
                    continue
                await request_queue.put(request)
                total_query_counter += 1
                if args.number is not None:
                    if total_query_counter >= args.number:
                        break
                if args.rate == UNLIMITED_RATE:  # on rate limit
                    continue
                # Sample the request interval from the exponential distribution.
                # from vllm
                interval = np.random.exponential(1.0 / args.rate)
                # The next request will be sent after the interval.
                await asyncio.sleep(interval)
            if args.number is None:
                break
            elif total_query_counter >= args.number:
                break
    else:
        raise Exception("Prompt or dataset is required!")
    return total_query_counter


class BenchmarkData(dict):
    """Benchmark info, two parts
    1. query info.
       prompt length
    2. response info
       start send time
       list of package_receive_time
           package info.
       response complete time
           total response info(response tokens)       
    """

    def __init__(self):
        pass


def calculate_query_stream_metric(benchmark_data):
    first_chunk_latency = benchmark_data["chunk_times"][0] - benchmark_data["start_time"]  # the first chunk latency
    n_chunks = len(benchmark_data["chunk_times"]) - 2  # minus first and last chunk.
    n_chunks_time = benchmark_data["chunk_times"][-2] - benchmark_data["chunk_times"][0]  # -2 to last chunk
    return (first_chunk_latency, n_chunks, n_chunks_time)


async def statistic_benchmark_metric_worker(benchmark_data_queue: asyncio.Queue, args):
    """Statistics of performance metrics based on performance data
    """
    n_succeed_queries = 0
    n_failed_queries = 0
    total_first_chunk_latency = 0
    total_latency = 0.0
    n_total_chunks = 0
    n_total_prompt_tokens = 0
    n_total_completion_tokens = 0
    qps = 0
    concurrency = args.parallel
    start_time = None
    total_chunks_time = 0.0
    avg_latency = -1
    avg_first_chunk_latency = -1
    avg_token_per_seconds = -1
    avg_time_per_token = -1
    n_avg_chunks = -1
    avg_chunk_time = -1
    avg_prompt_tokens = -1
    avg_completion_tokens = -1
    total_time = 1  # avoid divide by zero
    n_total_queries = 0
    # avg generate tps generated tokens / time
    # avg chunk time, first latency - avg_chunk_time == first latency, 去掉第一个和最后一个，第一个和prefill合并了，最后一个生成token可能比较短
    # avg prefill tps
    # prefill time = 首包时间-avg_chunk_time
    # n-tokens-per-trunk
    n_benchmark_result = 0
    api_plugin_class = api_registry(args.api)
    if not api_plugin_class:
        print('Can not find query generator: %s'%args.api)
    api_plugin = api_plugin_class(args.tokenizer_path)
    utc_dt = datetime.now(timezone.utc)
    current_time = utc_dt.astimezone().strftime("%Y_%m_%d_%H_%M_%S_%f")
    if args.name:
        result_db_path = os.path.join('./', args.name)
    else:
        result_db_path = "./%s_benchmark_%s.db" % (args.model, current_time)
    print('Save the result to : %s'%result_db_path)
    if os.path.exists(result_db_path):
        print('The db file exist, delete it and start again!.')
        sys.exit(1)
        
    con = sqlite3.connect(result_db_path)

    db_cur = con.cursor()
    # create table
    # TPS output tokens per second
    # tpot Time per ooutput token
    db_cur.execute("CREATE TABLE %s(request, start_time, chunk_times, success, \
                   response_messages, completed_time, latency, first_chunk_latency, \
                   n_chunks, chunk_time, prompt_tokens, completion_tokens)" % _table_name)
    if args.wandb_api_key is not None:
        import wandb
        name = args.name if args.name is not None else '%s_%s' % (args.model, current_time)
        wandb.init(
            project="perf_benchmark",
            name=name,
            # track run metadata
            config={
                "model": args.model,
                "time": current_time
            })
        os.environ["WANDB_SILENT"] = "true"

    while True:
        try:
            benchmark_data = benchmark_data_queue.get_nowait()
            benchmark_data_queue.task_done()
            n_benchmark_result += 1
        except asyncio.QueueEmpty as e:
            if _data_process_completed:
                break
            await asyncio.sleep(1)
            continue
        if start_time is None:
            start_time = benchmark_data["start_time"]  # start time with first request start time
        # total requests
        total_time = time.perf_counter() - start_time

        if benchmark_data["success"]:
            n_succeed_queries += 1
            n_query_trunks = len(benchmark_data["chunk_times"])
            query_latency = benchmark_data["completed_time"] - benchmark_data["start_time"]
            if n_query_trunks > 1:
                query_first_chunk_latency, query_n_chunks, query_n_chunks_time = calculate_query_stream_metric(
                    benchmark_data)
            else:
                query_first_chunk_latency = query_latency     # not stream mode, query latency is equal total latency
                query_n_chunks = 1
                query_n_chunks_time = query_latency

            n_query_prompt_tokens, n_query_completion_tokens = api_plugin.parse_responses(
                benchmark_data["response_messages"], 
                request=benchmark_data["request"])
            n_total_prompt_tokens += n_query_prompt_tokens
            n_total_completion_tokens += n_query_completion_tokens

            total_first_chunk_latency += query_first_chunk_latency
            total_latency += query_latency
            n_total_chunks += query_n_chunks
            total_chunks_time += query_n_chunks_time

            # calc average
            avg_first_chunk_latency = total_first_chunk_latency / n_succeed_queries
            # average latency
            avg_latency = total_latency / n_succeed_queries
            # average generate chunks
            if n_query_trunks > 1:
                n_avg_chunks = n_total_chunks / n_succeed_queries + 2     # we remove the frist and last chunk.
            else:
                n_avg_chunks = n_total_chunks / n_succeed_queries
            avg_chunk_time = total_chunks_time / n_total_chunks
            avg_prompt_tokens = n_total_prompt_tokens / n_succeed_queries
            avg_completion_tokens = n_total_completion_tokens / n_succeed_queries
            # avg generate tps generated tokens / time
            avg_token_per_seconds = n_total_completion_tokens / total_time
            avg_time_per_token = total_time / n_total_completion_tokens
            # save the benchmark data to database.
            # save data to dist.
            insert_sql = "INSERT INTO %s VALUES('%s', %s, '%s', '%s', '%s', %s, %s, %s, %s, %s, %s, %s)" % (
                _table_name,
                base64.b64encode(pickle.dumps(benchmark_data["request"])).decode("ascii"),
                benchmark_data["start_time"],
                json.dumps(benchmark_data["chunk_times"]),
                benchmark_data["success"],
                base64.b64encode(pickle.dumps(benchmark_data["response_messages"])).decode("ascii"),
                benchmark_data["completed_time"],
                query_latency,
                query_first_chunk_latency,
                query_n_chunks,
                query_n_chunks_time,
                n_query_prompt_tokens,
                n_query_completion_tokens
            )
        else:
            n_failed_queries += 1
            # save the benchmark data to database.
            # save data to dist.
            insert_sql = "INSERT INTO %s(request, start_time, chunk_times, success, response_messages, completed_time)\
                VALUES('%s', %s, '%s', '%s', '%s', %s)" % (
                _table_name,
                base64.b64encode(pickle.dumps(benchmark_data["request"])).decode("ascii"),
                benchmark_data["start_time"],
                json.dumps(benchmark_data["chunk_times"]),
                benchmark_data["success"],
                base64.b64encode(pickle.dumps(benchmark_data["response_messages"])).decode("ascii"),
                benchmark_data["completed_time"]
            )
        n_total_queries = float(n_succeed_queries + n_failed_queries)  # float for calc
        qps = n_succeed_queries / total_time
        db_cur.execute(insert_sql)
        con.commit()
        default_ndigits = 3
        message = {"Time": round(total_time, default_ndigits),
                   "concurrency": concurrency,
                   "completed": int(n_total_queries),
                   "succeed": n_succeed_queries,
                   "failed": n_failed_queries,
                   "qps": round(qps, default_ndigits),
                   "latency": round(avg_latency, default_ndigits),
                   "time to first token": round(avg_first_chunk_latency, default_ndigits),
                   "throughput(output tokens per second)": round(avg_token_per_seconds, default_ndigits),
                   "time per output token": round(avg_time_per_token, 5),
                   "package per request": round(n_avg_chunks, default_ndigits),
                   "time per package": round(avg_chunk_time, default_ndigits),
                   "input tokens per request": round(avg_prompt_tokens, default_ndigits),
                   "output tokens per request": round(avg_completion_tokens, default_ndigits)}
        if args.wandb_api_key is not None:
            wandb.log(message)
        if int(n_total_queries) % args.log_every_n_query == 0:
            msg = json.dumps(message)
            msg = msg[1:-1].replace('"', '')
            logger.info(msg)
    con.commit()
    con.close()
    return (total_time, n_total_queries,
            n_succeed_queries, n_failed_queries,
            qps, avg_latency, avg_first_chunk_latency,
            n_avg_chunks, avg_chunk_time,
            avg_prompt_tokens, avg_completion_tokens,
            avg_token_per_seconds, avg_time_per_token,
            result_db_path)


def summary_result(expected_number_of_queries,
                   total_time,
                   n_total_queries,
                   n_succeed_queries,
                   n_failed_queries,
                   qps,
                   avg_latency,
                   avg_first_chunk_latency,
                   n_avg_chunks,
                   avg_chunk_time,
                   avg_prompt_tokens,
                   avg_completion_tokens,
                   avg_token_per_seconds,
                   avg_time_per_token,
                   result_db_path, args):

    print("Benchmarking summary: ")
    print(" Time taken for tests: %.3f seconds" % total_time)
    print(" Expected number of requests: %s" % expected_number_of_queries)
    print(" Number of concurrency: %d" % args.parallel)
    print(" Total requests: %d" % n_total_queries)
    print(" Succeed requests: %d" % n_succeed_queries)
    print(" Failed requests: %d" % n_failed_queries)
    print(" Average QPS: %.3f" % qps)
    print(" Average latency: %.3f" % avg_latency)
    print(" Throughput(average output tokens per second): %.3f" % avg_token_per_seconds)
    print(" Average time to first token: %.3f" % avg_first_chunk_latency)
    print(" Average input tokens per request: %.3f" % avg_prompt_tokens)
    print(" Average output tokens per request: %.3f" % avg_completion_tokens)
    print(" Average time per output token: %.5f" % avg_time_per_token)
    print(" Average package per request: %.3f" % n_avg_chunks)
    print(" Average package latency: %.3f" % avg_chunk_time)

    con = sqlite3.connect(result_db_path)
    query_sql = "SELECT start_time, chunk_times, success, \
                   completed_time, latency, first_chunk_latency, \
                   n_chunks, chunk_time, prompt_tokens, completion_tokens \
                       FROM %s WHERE success='True' ORDER BY first_chunk_latency ASC" % _table_name

    percentiles = [50, 66, 75, 80, 90, 95, 98, 99]
    with con:
        rows = con.execute(query_sql).fetchall()
        n_success_queries = len(rows)
        if len(rows) > len(percentiles):
            print(" Percentile of time to first token: ")
            for percentile in percentiles:
                idx = (int)(n_success_queries * percentile / 100)
                row = rows[idx]
                print("     p%s: %.4f" % (percentile, row[5] if row[5] is not None else float("inf")))
                # print(row)
            print(" Percentile of request latency: ")
            latency_index = 4
            rows.sort(key=lambda x: x[latency_index])
            for percentile in percentiles:
                idx = (int)(n_success_queries * percentile / 100)
                row = rows[idx]
                print("     p%s: %.4f" % (percentile, row[latency_index]
                      if row[latency_index] is not None else float("inf")))
        else:
            print(" Too little data to calculate quantiles!")
    con.close()


async def send_requests_worker(task_id, request_queue: asyncio.Queue, benchmark_data_queue: asyncio.Queue, args):
    client = AioHttpClient(args.url,
                           conn_timeout=args.connect_timeout,
                           read_timeout=args.read_timeout,
                           headers=args.headers,
                           debug=args.debug)
    async with client:
        while True:
            # Get a request out of the queue.
            try:
                request = request_queue.get_nowait()
                request_queue.task_done()
            except asyncio.QueueEmpty as e:
                if _query_send_completed:
                    break
                await asyncio.sleep(0.01)
                continue    # keep polling querys
            benchmark_data = BenchmarkData()
            benchmark_data["request"] = request
            benchmark_data["start_time"] = time.perf_counter()
            benchmark_data["chunk_times"] = []
            benchmark_data["success"] = False
            collected_messages = []
            try:
                async for (is_error, state_code, response_data) in client.post(request):
                    if is_error or state_code != HTTPStatus.OK:
                        logger.error("Request: %s failed, state_code: %s, data: %s" %
                                     (request, state_code, response_data))
                        break
                    else:
                        if response_data:
                            collected_messages.append(response_data)  # save the message
                            logger.debug(response_data)
                            benchmark_data["chunk_times"].append(time.perf_counter())

                benchmark_data["response_messages"] = collected_messages
                benchmark_data["completed_time"] = time.perf_counter()
                benchmark_data["success"] = not is_error
                await benchmark_data_queue.put(benchmark_data)
            except BaseException as e:
                if response_data:
                    collected_messages.append(response_data)  # save the message
                benchmark_data["response_messages"] = collected_messages
                benchmark_data["completed_time"] = time.perf_counter()
                await benchmark_data_queue.put(benchmark_data)
                logger.error("Request query: %s exception, response: %s" % (request, response_data))
                logger.exception(e)

def signal_handler(signal_name, loop):
    print("Got signal %s: exit" % signal_name)
    loop.stop()
    

async def benchmark(args) -> None:
    # add SIGINT and SIGTERM handler
    loop = asyncio.get_running_loop()
    for signal_name in {'SIGINT', 'SIGTERM'}:
        loop.add_signal_handler(
            getattr(signal, signal_name),
            functools.partial(signal_handler, signal_name, loop))
        
    request_tasks: List[asyncio.Task] = []
    # Queues can be used to distribute workload between several concurrent tasks
    # Create a queue that we will use to store our "workload".
    request_queue = asyncio.Queue()
    benchmark_data_queue = asyncio.Queue()
    dispatch_task = asyncio.create_task(dispatch_requests_worker(request_queue, args))
    statistic_benchmark_metric_task = asyncio.create_task(statistic_benchmark_metric_worker(benchmark_data_queue, args))
    for idx, task in enumerate(range(args.parallel)):
        task = asyncio.create_task(send_requests_worker(idx, request_queue, benchmark_data_queue, args))
        request_tasks.append(task)

    expected_number_of_queries = await dispatch_task  # wait for dispatch task complete
    await request_queue.join()
    global _query_send_completed
    _query_send_completed = True
    await asyncio.gather(*request_tasks, return_exceptions=True)
    await benchmark_data_queue.join()  # wait for all query is processed
    global _data_process_completed
    _data_process_completed = True
    (total_time, n_total_queries,
     n_succeed_queries, n_failed_queries,
     qps, avg_latency,
     avg_first_chunk_latency, n_avg_chunks,
     avg_chunk_time, avg_prompt_tokens,
     avg_completion_tokens, avg_token_per_seconds,
     avg_time_per_token, result_db_path) = await statistic_benchmark_metric_task

    summary_result(expected_number_of_queries, total_time, n_total_queries, n_succeed_queries,
                   n_failed_queries, qps, avg_latency, avg_first_chunk_latency,
                   n_avg_chunks, avg_chunk_time,
                   avg_prompt_tokens, avg_completion_tokens,
                   avg_token_per_seconds, avg_time_per_token,
                   result_db_path, args)
    await asyncio.sleep(0.250)


def process_number(input):
    try:
        return int(input)
    except ValueError:
        try:
            return float(input)
        except ValueError:
            return input


# from: https://gist.github.com/vadimkantorov/37518ff88808af840884355c845049ea
class ParseKVAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for each in values:
            try:
                key, value = each.split("=")
                if value.lower() == 'bool_true':
                    value = True
                if value.lower() == 'bool_false':
                    value = False
                
                value = process_number(value)
                getattr(namespace, self.dest)[key] = value
            except ValueError as ex:
                message = "\nTraceback: {}".format(ex)
                message += "\nError on '{}' || It should be 'key=value'".format(
                    each)
                raise argparse.ArgumentError(self, str(message))


def run_perf_benchmark(args):
    asyncio.run(benchmark(args))


def add_argument(parser: argparse.ArgumentParser):
    parser.add_argument("--model", type=str, required=True,
                        help="The test model name.")
    parser.add_argument("--url", type=str, default="localhost")
    parser.add_argument("--connect-timeout", type=int, default=120,
                        help="The network connection timeout")
    parser.add_argument("--read-timeout", type=int, default=120,
                        help="The network read timeout")
    parser.add_argument("-n", "--number", type=int, default=None,
                        help="How many requests to be made, if None, "
                        "will will send request base dataset or prompt.")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Set number of concurrency request, default 1")
    parser.add_argument("--rate", type=int, default=UNLIMITED_RATE,
                        help="Number of requests per second. default None, if it set to -1,"
                        "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.  Mutual exclusion with parallel")
    parser.add_argument("--log-every-n-query", type=int, default=10,
                        help="Logging every n query.")
    parser.add_argument("--headers", nargs="+", dest="headers",
                        action=ParseKVAction,
                        help="Extra http headers accepts by key1=value1 key2=value2. "
                             "The headers will be use for each query."
                             "You can use this parameter to specify http authorization and other header.",
                        metavar="KEY1=VALUE1")
    parser.add_argument("--wandb-api-key", type=str, default=None,
                        help="The wandb api key, if set the metric will be saved to wandb.")
    parser.add_argument("--name", type=str,
                        help="The wandb db result name and result db name, default: {model_name}_{current_time}")
    parser.add_argument("--debug", action='store_true', default=False,
                        help='Debug request send.')
    parser.add_argument("--tokenizer-path", type=str, required=False, default=None,
                        help="Specify the tokenizer weight path, used to calculate the number of input and output tokens,"
                        "usually in the same directory as the model weight. If service return usage will use usage info.")
    parser.add_argument("--api",
                        type=str,
                        default="openai",
                        help="Specify the service api, current support [openai|dashscope]"
                             "you can define your custom parser with python, and specify the python file path, "
                             "reference api_plugin_base.py,")
    parser.add_argument("--max-prompt-length", type=int, default=sys.maxsize,
                        help="Maximum input prompt length")
    parser.add_argument("--min-prompt-length", type=int, default=0,
                        help="Minimum input prompt length.")
    parser.add_argument("--prompt", type=str, required=False, default=None,
                        help="Specified the request prompt, all the query will use this prompt, "
                        "You can specify local file via @file_path, the prompt will be "
                        "the file content.")
    parser.add_argument("--query-template",
                        type=str,
                        default=None,
                        help="Specify the query template, should be a json string, or local file,"
                             "with local file, specified with @local_file_path,"
                             "will will replace model and prompt in the template.")
    parser.add_argument("--dataset",
                        type=str,
                        default='line_by_line',
                        help="Specify the dataset [openqa|longalpaca|line_by_line]"
                             "you can define your custom dataset parser with python, and specify the python file path, "
                             "reference dataset_plugin_base.py,")
    parser.add_argument("--dataset-path", type=str, required=False,
                        help="Path to the dataset file, Used in conjunction with dataset. "
                             "If dataset is None, each line defaults to a prompt.")
    
    parser.add_argument("--frequency-penalty", type=float, help="The frequency_penalty value.", default= None)
    parser.add_argument("--logprobs", action='store_true', help="The logprobs.", default=None)
    parser.add_argument("--max-tokens", type=int, help="The maximum number of tokens can be generated.", default=None)
    parser.add_argument("--n-choices", type=int, help="How may chmpletion choices to generate.", default=None)
    parser.add_argument("--seed", type=int, help="Rhe random seed.", default=None)
    parser.add_argument("--stop", nargs='*', help="The stop tokens.", default=None)
    parser.add_argument("--stop-token-ids", nargs='*', help="Set the stop token ids.", default=None)
    parser.add_argument("--stream", action='store_true', help="Stream output with SSE, Automatically add stream_option.include_usage with openai interface.", default=None)
    parser.add_argument("--temperature", type=float, help="The sample temperature.", default=None)
    parser.add_argument("--top-p", type=float, help="Sampling top p.", default=None)

if __name__ == "__main__":
    # for windows raise RuntimeError: Event loop is closed
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(
        description="Benchmark LLM service performance.")
    add_argument(parser)
    args = parser.parse_args()
    run_perf_benchmark(args)
