from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional

from PIL import Image

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.datasets.dataset_args import MMMUMultiImageDatasetArgs
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.io_utils import PIL_to_base64


@register_dataset('mmmu_multi_image')
class MMMUMultiImageDatasetPlugin(DatasetPluginBase):
    """Build real multi-image stress-test requests from the MMMU validation set."""

    args_schema = MMMUMultiImageDatasetArgs
    dataset_id = 'AI-ModelScope/MMMU'
    max_images = 7

    def __init__(self, query_parameters: Arguments):
        if query_parameters.tokenize_prompt:
            raise ValueError(
                '--tokenize-prompt is not supported with the mmmu_multi_image dataset. '
                'The dataset produces multimodal messages that cannot be represented as a flat token-ID list.'
            )
        super().__init__(query_parameters)

    @staticmethod
    def _to_pil_image(value: Any) -> Optional[Image.Image]:
        """Normalize common dataset image representations to a PIL image."""
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict):
            raw_bytes = value.get('bytes')
            if raw_bytes:
                with Image.open(BytesIO(raw_bytes)) as image:
                    return image.copy()
            path = value.get('path')
            if path:
                with Image.open(path) as image:
                    return image.copy()
        return None

    def _collect_image_urls(self, item: Dict[str, Any]) -> List[str]:
        """Collect non-empty ``image_1`` ... ``image_7`` fields in source order."""
        image_urls: List[str] = []
        for index in range(1, self.max_images + 1):
            image = self._to_pil_image(item.get(f'image_{index}'))
            if image is None:
                continue
            image_urls.append(PIL_to_base64(image, add_header=True))
        return image_urls

    @staticmethod
    def _build_prompt(item: Dict[str, Any]) -> str:
        """Keep MMMU question text and options without changing benchmark semantics."""
        prompt = str(item['question'])
        options = item.get('options')
        if options and options != '[]':
            prompt = f'{prompt}\nOptions: {options}'
        return prompt

    def build_messages(self) -> Iterator[List[Dict]]:
        dataset = self.load_hub_dataset(
            dataset_id=self.dataset_id,
            split='validation',
            subset=self.dataset_args.subset,
        )

        for item in dataset:
            image_urls = self._collect_image_urls(item)
            if len(image_urls) < self.dataset_args.min_images:
                continue

            message = self.create_message(text=self._build_prompt(item), image_urls=image_urls)
            yield [message]
