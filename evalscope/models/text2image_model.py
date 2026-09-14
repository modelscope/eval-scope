from __future__ import annotations

import base64
import binascii
import importlib
import os
import time
from logging import getLogger
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai.types import ImagesResponse

from evalscope.api.messages import ChatMessage, ContentImage
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils.argument_utils import get_secret_value, get_supported_params
from evalscope.utils.function_utils import retry_call
from evalscope.utils.import_utils import check_import
from evalscope.utils.io_utils import PIL_to_base64, bytes_to_base64
from evalscope.utils.model_utils import get_device
from evalscope.utils.uri_utils import data_uri_to_base64, file_as_data, is_http_url

logger = getLogger()


class Text2ImageAPI(ModelAPI):
    """Text-to-image provider for local pipelines and OpenAI-compatible services."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        self.base_url = (base_url or '').rstrip('/')
        self._use_service = bool(self.base_url)
        if self._use_service:
            self._init_service(api_key=api_key, model_args=model_args)
        else:
            self._init_local_model(model_args=model_args)

    def _init_local_model(self, model_args: Dict[str, Any]) -> None:
        check_import(
            ['torch', 'torchvision', 'diffusers'],
            package='evalscope[aigc]',
            raise_error=True,
            feature_name='text2image',
        )
        import torch

        model_path = model_args.pop('model_path', None)
        torch_dtype = model_args.pop('precision', None) or model_args.pop('torch_dtype', None)
        device_map = model_args.pop('device_map', None)
        # torch dtype
        DTYPE_MAP = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16, 'auto': 'auto'}

        if isinstance(torch_dtype, str) and torch_dtype != 'auto':
            torch_dtype = DTYPE_MAP.get(torch_dtype, torch.float32)
        self.torch_dtype = torch_dtype
        self.device = device_map or get_device()

        self.pipeline_cls = model_args.pop('pipeline_cls', None)
        # default to DiffusionPipeline if not specified
        if self.pipeline_cls is None:
            if 'flux' in self.model_name.lower():
                self.pipeline_cls = 'FluxPipeline'
            else:
                self.pipeline_cls = 'DiffusionPipeline'

        model_name_or_path = model_path or self.model_name

        # from modelscope import pipeline_cls
        module = getattr(importlib.import_module('modelscope'), self.pipeline_cls)
        logger.info(f'Loading model {model_name_or_path} with {self.pipeline_cls} ...')

        self.model = module.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            **model_args,
        )

        self.model.to(self.device)

    def _init_service(self, api_key: Optional[str], model_args: Dict[str, Any]) -> None:
        self.base_url = self._normalize_service_base_url(self.base_url)
        self.api_key = self._resolve_service_api_key(api_key)

        client_params = get_supported_params(OpenAI) - {'api_key', 'base_url'}
        ignored_model_args = sorted(set(model_args) - client_params)
        if ignored_model_args:
            logger.warning(
                'Ignoring model_args unsupported by the OpenAI client in text2image service mode: '
                f'{", ".join(ignored_model_args)}'
            )
        client_args = {key: value for key, value in model_args.items() if key in client_params}
        if client_args.get('max_retries') not in {None, 0}:
            logger.warning(
                'Ignoring model_args.max_retries in text2image service mode; use generation_config.retries instead.'
            )
        client_args['max_retries'] = 0

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **client_args,
        )
        self._service_request_params = get_supported_params(self.client.images.generate)

    @staticmethod
    def _normalize_service_base_url(base_url: str) -> str:
        return base_url.rstrip('/').removesuffix('/images/generations').removesuffix('/chat/completions')

    @staticmethod
    def _resolve_service_api_key(api_key: Optional[str]) -> str:
        if api_key and api_key != 'EMPTY':
            return api_key
        return os.getenv('OPENAI_API_KEY') or os.getenv('EVALSCOPE_API_KEY') or api_key or 'EMPTY'

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if self._use_service:
            return self._generate_service(input=input, config=config)

        start_time = time.monotonic()

        # prepare generator
        kwargs: Dict[str, Any] = {}
        if config.height is not None:
            kwargs['height'] = config.height
        if config.width is not None:
            kwargs['width'] = config.width
        if config.num_inference_steps is not None:
            kwargs['num_inference_steps'] = config.num_inference_steps
        if config.guidance_scale is not None:
            kwargs['guidance_scale'] = config.guidance_scale
        # update with extra model parameters
        kwargs.update(config.model_extra)

        # assume the first text as prompt
        prompt = input[0].text
        # get the first image as output
        image = self.model(prompt=prompt, **kwargs).images[0]

        # Emit a data URI so downstream consumers (e.g. an LLM-judge request)
        # treat the image as inline base64 instead of a local file path.
        image_base64 = PIL_to_base64(image, add_header=True)

        return ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice.from_content(content=[ContentImage(image=image_base64)])],
            time=time.monotonic() - start_time,
        )

    def _generate_service(self, input: List[ChatMessage], config: GenerateConfig) -> ModelOutput:
        start_time = time.monotonic()
        request = self._service_request(input=input, config=config)
        response = retry_call(
            self.client.images.generate,
            retries=max(config.retries or 1, 1),
            sleep_interval=config.retry_interval or 0,
            **request,
        )
        image_base64, metadata = self._parse_service_response(response)

        return ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice.from_content(content=[ContentImage(image=image_base64)])],
            time=time.monotonic() - start_time,
            metadata=metadata,
        )

    def _service_request(self, input: List[ChatMessage], config: GenerateConfig) -> Dict[str, Any]:
        if not input or not input[0].text:
            raise ValueError('Text-to-image generation requires a non-empty prompt.')

        request: Dict[str, Any] = {
            'model': self.model_name,
            'prompt': input[0].text,
        }
        if config.n is not None:
            request['n'] = config.n
        if config.width is not None or config.height is not None:
            if config.width is None or config.height is None:
                raise ValueError('Both width and height are required when setting the image size.')
            request['size'] = f'{config.width}x{config.height}'

        request.update(config.model_extra or {})
        extra_body = dict(get_secret_value(config.extra_body) or {})
        for key in list(request):
            if key not in self._service_request_params:
                extra_body[key] = request.pop(key)
        if extra_body:
            request['extra_body'] = extra_body
        if config.extra_query is not None:
            request['extra_query'] = get_secret_value(config.extra_query)
        if config.extra_headers is not None:
            request['extra_headers'] = get_secret_value(config.extra_headers)
        if config.timeout is not None:
            request['timeout'] = config.timeout
        return request

    def _parse_service_response(self, response: ImagesResponse) -> tuple[str, Dict[str, Any]]:
        if not response.data:
            raise ValueError('Image generation service response must contain a non-empty data list.')

        image = response.data[0]
        image_b64_json = image.b64_json
        image_url = image.url
        if isinstance(image_b64_json, str) and image_b64_json:
            image_base64 = data_uri_to_base64(image_b64_json)
            try:
                base64.b64decode(image_base64, validate=True)
            except (binascii.Error, ValueError, TypeError) as ex:
                raise ValueError('Image generation service returned invalid base64 image data.') from ex
        elif isinstance(image_url, str) and image_url:
            if not is_http_url(image_url):
                raise ValueError('Image generation service returned a non-HTTP image URL.')
            image_bytes, _ = file_as_data(image_url)
            image_base64 = bytes_to_base64(image_bytes)
        else:
            raise ValueError('Image generation service response must include b64_json or url.')

        response_payload = response.model_dump(exclude_none=True)
        metadata = {key: value for key, value in response_payload.items() if key != 'data'}
        request_id = getattr(response, '_request_id', None)
        if request_id:
            metadata['request_id'] = request_id
        if image.revised_prompt is not None:
            metadata['revised_prompt'] = image.revised_prompt
        return image_base64, metadata
