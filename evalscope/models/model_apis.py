from evalscope.api.model import ModelAPI
from evalscope.api.registry import register_model_api
from evalscope.utils.deprecation_utils import deprecated
from evalscope.utils.import_utils import check_import


@register_model_api(name='mock_llm')
def mockllm() -> type[ModelAPI]:
    from .mockllm import MockLLM

    return MockLLM


@register_model_api(name='openai_api')
def openai_api() -> type[ModelAPI]:
    from .openai_compatible import OpenAICompatibleAPI

    return OpenAICompatibleAPI


@register_model_api(name='openai_responses_api')
def openai_responses_api() -> type[ModelAPI]:
    from .openai_responses import OpenAIResponsesAPI

    return OpenAIResponsesAPI


@register_model_api(name='anthropic_api')
def anthropic_api() -> type[ModelAPI]:
    check_import('anthropic', package='anthropic', raise_error=True, feature_name='anthropic_api')

    from .anthropic_compatible import AnthropicCompatibleAPI

    return AnthropicCompatibleAPI


@register_model_api(name='litellm')
def litellm_api() -> type[ModelAPI]:
    check_import('litellm', package='litellm', raise_error=True, feature_name='litellm')

    from .litellm_compatible import LiteLLMAPI

    return LiteLLMAPI


@register_model_api(name='server')
@deprecated(since='1.0.0', remove_in='2.0.0', alternative='openai_api')
def server() -> type[ModelAPI]:
    from .openai_compatible import OpenAICompatibleAPI

    return OpenAICompatibleAPI


@register_model_api(name='llm_ckpt')
def llm_ckpt() -> type[ModelAPI]:
    check_import('torch', package='torch', raise_error=True, feature_name='llm_ckpt')

    from .modelscope import ModelScopeAPI

    return ModelScopeAPI


@register_model_api(name='checkpoint')
@deprecated(since='1.0.0', remove_in='2.0.0', alternative='llm_ckpt')
def checkpoint() -> type[ModelAPI]:
    check_import('torch', package='torch', raise_error=True, feature_name='llm_ckpt')

    from .modelscope import ModelScopeAPI

    return ModelScopeAPI


@register_model_api(name='text2image')
def text2image() -> type[ModelAPI]:
    from .text2image_model import Text2ImageAPI

    return Text2ImageAPI


@register_model_api(name='image_editing')
def image_editing() -> type[ModelAPI]:
    check_import(
        ['torch', 'torchvision', 'diffusers'], package='evalscope[aigc]', raise_error=True, feature_name='image_editing'
    )

    from .image_edit_model import ImageEditAPI

    return ImageEditAPI


@register_model_api(name='text2speech')
def text2speech() -> type[ModelAPI]:
    from .text2speech_model import Text2SpeechAPI

    return Text2SpeechAPI
