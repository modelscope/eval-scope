import base64
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import pytest
from openai import APIStatusError

from evalscope.api.messages import ContentImage
from evalscope.api.model import GenerateConfig, get_model, get_model_with_task_config
from evalscope.config import TaskConfig
from evalscope.constants import EvalType, ModelTask


class _ImageServiceHandler(BaseHTTPRequestHandler):

    requests: list[dict[str, Any]] = []
    image_requests: list[str] = []
    image_bytes = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='
    )
    response_kind = 'base64'

    def _send_json(self, status: int, response: dict[str, Any]) -> None:
        body = json.dumps(response).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('x-request-id', 'request-id')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        content_length = int(self.headers.get('Content-Length', '0'))
        payload = json.loads(self.rfile.read(content_length))
        self.requests.append({
            'path': self.path,
            'headers': dict(self.headers),
            'payload': payload,
        })

        if self.response_kind == 'error':
            self._send_json(503, {'error': {'message': 'image backend unavailable', 'type': 'server_error'}})
            return
        if self.response_kind == 'retry' and len(self.requests) == 1:
            self._send_json(429, {'error': {'message': 'rate limited', 'type': 'rate_limit_error'}})
            return

        if self.response_kind in {'base64', 'retry', 'data_uri', 'invalid_base64'}:
            image_base64 = base64.b64encode(self.image_bytes).decode('utf-8')
            if self.response_kind == 'data_uri':
                image_base64 = f'data:image/png;base64,{image_base64}'
            elif self.response_kind == 'invalid_base64':
                image_base64 = 'not-valid-base64'
            response = {
                'created': 123,
                'data': [{
                    'b64_json': image_base64,
                    'revised_prompt': 'A revised prompt',
                }],
                'usage': {
                    'input_tokens': 4,
                    'output_tokens': 8,
                    'total_tokens': 12,
                },
            }
        elif self.response_kind == 'url':
            host, port = self.server.server_address
            response = {'created': 123, 'data': [{'url': f'http://{host}:{port}/generated.png'}]}
        elif self.response_kind == 'invalid_url':
            response = {'created': 123, 'data': [{'url': '/tmp/generated.png'}]}
        else:
            response = {'created': 123, 'data': []}
        self._send_json(200, response)

    def do_GET(self) -> None:
        self.image_requests.append(self.path)
        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(self.image_bytes)))
        self.end_headers()
        self.wfile.write(self.image_bytes)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@contextmanager
def _image_service(response_kind: str = 'base64') -> Iterator[str]:
    _ImageServiceHandler.requests = []
    _ImageServiceHandler.image_requests = []
    _ImageServiceHandler.response_kind = response_kind
    server = ThreadingHTTPServer(('127.0.0.1', 0), _ImageServiceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f'http://{host}:{port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_remote_text2image_uses_openai_compatible_endpoint(caplog: pytest.LogCaptureFixture) -> None:
    with _image_service() as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1/images/generations',
            api_key='test-key',
            config=GenerateConfig(
                width=1024,
                height=1024,
                n=1,
                quality='low',
                vendor_option='fast',
            ),
            model_args={
                'pipeline_cls': 'FluxPipeline',
                'use_safetensors': True,
                'variant': 'fp16',
            },
            memoize=False,
        )
        output = model.generate('Draw a red panda reading a book.')

    request = _ImageServiceHandler.requests[0]
    assert request['path'] == '/v1/images/generations'
    assert request['payload'] == {
        'model': 'test-image-model',
        'prompt': 'Draw a red panda reading a book.',
        'n': 1,
        'quality': 'low',
        'size': '1024x1024',
        'vendor_option': 'fast',
    }
    assert request['headers']['Authorization'] == 'Bearer test-key'
    assert 'pipeline_cls' not in request['payload']
    assert 'use_safetensors' not in request['payload']
    assert 'variant' not in request['payload']
    assert 'Ignoring model_args unsupported by the OpenAI client' in caplog.text
    assert 'pipeline_cls, use_safetensors, variant' in caplog.text

    content = output.message.content[0]
    assert isinstance(content, ContentImage)
    assert base64.b64decode(content.image) == _ImageServiceHandler.image_bytes
    assert output.metadata == {
        'created': 123,
        'request_id': 'request-id',
        'revised_prompt': 'A revised prompt',
        'usage': {
            'input_tokens': 4,
            'output_tokens': 8,
            'total_tokens': 12,
        },
    }


def test_remote_text2image_downloads_url_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('EVALSCOPE_API_KEY', raising=False)
    with _image_service(response_kind='url') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='EMPTY',
            memoize=False,
        )
        output = model.generate('Draw a lighthouse in a storm.')

    request = _ImageServiceHandler.requests[0]
    assert request['path'] == '/v1/images/generations'
    assert request['headers']['Authorization'] == 'Bearer EMPTY'
    assert _ImageServiceHandler.image_requests == ['/generated.png']
    content = output.message.content[0]
    assert isinstance(content, ContentImage)
    assert base64.b64decode(content.image) == _ImageServiceHandler.image_bytes


def test_remote_text2image_empty_key_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'environment-key')
    monkeypatch.delenv('EVALSCOPE_API_KEY', raising=False)
    with _image_service() as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='EMPTY',
            memoize=False,
        )
        model.generate('Draw a cabin in the snow.')

    assert _ImageServiceHandler.requests[0]['headers']['Authorization'] == 'Bearer environment-key'


def test_remote_text2image_normalizes_chat_completions_url() -> None:
    with _image_service() as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1/chat/completions',
            api_key='test-key',
            memoize=False,
        )
        model.generate('Draw a paper boat.')

    assert _ImageServiceHandler.requests[0]['path'] == '/v1/images/generations'


def test_remote_text2image_retries_transient_errors(caplog: pytest.LogCaptureFixture) -> None:
    with _image_service(response_kind='retry') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(retries=2, retry_interval=0),
            model_args={'max_retries': 3},
            memoize=False,
        )
        output = model.generate('Draw a sunrise.')

    assert len(_ImageServiceHandler.requests) == 2
    assert 'use generation_config.retries instead' in caplog.text
    content = output.message.content[0]
    assert isinstance(content, ContentImage)
    assert base64.b64decode(content.image) == _ImageServiceHandler.image_bytes


def test_remote_text2image_zero_retries_still_sends_request() -> None:
    with _image_service() as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(retries=0, retry_interval=0),
            memoize=False,
        )
        output = model.generate('Draw a single request.')

    assert len(_ImageServiceHandler.requests) == 1
    assert isinstance(output.message.content[0], ContentImage)


def test_remote_text2image_preserves_error_body() -> None:
    with _image_service(response_kind='error') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(retries=1, retry_interval=0),
            memoize=False,
        )
        with pytest.raises(APIStatusError) as exc_info:
            model.generate('Draw an unavailable image.')

    assert exc_info.value.body == {'message': 'image backend unavailable', 'type': 'server_error'}
    assert len(_ImageServiceHandler.requests) == 1


@pytest.mark.parametrize('response_kind', ['base64', 'data_uri'])
def test_remote_text2image_accepts_base64_variants(response_kind: str) -> None:
    with _image_service(response_kind=response_kind) as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            memoize=False,
        )
        output = model.generate('Draw a blue circle.')

    content = output.message.content[0]
    assert isinstance(content, ContentImage)
    assert base64.b64decode(content.image) == _ImageServiceHandler.image_bytes


def test_remote_text2image_rejects_invalid_base64() -> None:
    with _image_service(response_kind='invalid_base64') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            memoize=False,
        )
        with pytest.raises(ValueError, match='invalid base64 image data'):
            model.generate('Draw invalid image data.')


def test_remote_text2image_rejects_non_http_url() -> None:
    with _image_service(response_kind='invalid_url') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            memoize=False,
        )
        with pytest.raises(ValueError, match='non-HTTP image URL'):
            model.generate('Draw a local file path.')


def test_remote_text2image_rejects_empty_data() -> None:
    with _image_service(response_kind='empty') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            memoize=False,
        )
        with pytest.raises(ValueError, match='non-empty data list'):
            model.generate('Draw an empty response.')


def test_image_generation_service_is_selected_from_task_config() -> None:
    with _image_service() as service_url:
        task_config = TaskConfig(
            model='test-image-model',
            model_task=ModelTask.IMAGE_GENERATION,
            api_url=f'{service_url}/v1',
        )
        model = get_model_with_task_config(task_config)
        output = model.generate('Draw a mountain reflected in a lake.')

    assert task_config.eval_type == EvalType.TEXT2IMAGE
    assert _ImageServiceHandler.requests[0]['path'] == '/v1/images/generations'
    content = output.message.content[0]
    assert isinstance(content, ContentImage)
    assert base64.b64decode(content.image) == _ImageServiceHandler.image_bytes


def test_local_text2image_still_checks_aigc_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    checked_modules: list[str] = []

    def reject_local_dependencies(
        module_name: list[str],
        package: str,
        raise_error: bool,
        feature_name: str,
    ) -> None:
        checked_modules.extend(module_name)
        raise ImportError('local dependencies are unavailable')

    monkeypatch.setattr('evalscope.models.text2image_model.check_import', reject_local_dependencies)

    with pytest.raises(ImportError, match='local dependencies are unavailable'):
        get_model(
            model='Qwen-Image',
            eval_type=EvalType.TEXT2IMAGE,
            memoize=False,
        )

    assert checked_modules == ['torch', 'torchvision', 'diffusers']
