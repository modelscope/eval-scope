from io import BytesIO

import pytest
from PIL import Image

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.mmmu_multi_image import MMMUMultiImageDatasetPlugin


def _args(dataset_args=None, **kwargs) -> Arguments:
    return Arguments(
        model='test-model',
        url='http://localhost:8080/v1/chat/completions',
        dataset='mmmu_multi_image',
        dataset_args=dataset_args,
        **kwargs,
    )


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (2, 2), 'white').save(buffer, format='PNG')
    return buffer.getvalue()


class TestMMMUMultiImageDataset:
    def test_builds_one_message_with_multiple_images_in_source_order(self, monkeypatch):
        plugin = MMMUMultiImageDatasetPlugin(_args({'subset': 'Music', 'min_images': 2}))
        rows = [
            {
                'question': 'Compare <image 1> and <image 2>.',
                'options': "['same', 'different']",
                'image_1': Image.new('RGB', (2, 2), 'red'),
                'image_2': {'bytes': _image_bytes()},
                'image_3': None,
            }
        ]
        load_args = {}

        def fake_load_hub_dataset(dataset_id, split='train', subset='default'):
            load_args.update(dataset_id=dataset_id, split=split, subset=subset)
            return rows

        monkeypatch.setattr(plugin, 'load_hub_dataset', fake_load_hub_dataset)

        requests = list(plugin.build_messages())

        assert load_args == {
            'dataset_id': 'AI-ModelScope/MMMU',
            'split': 'validation',
            'subset': 'Music',
        }
        assert len(requests) == 1
        message = requests[0][0]
        assert message['role'] == 'user'
        assert message['content'][0] == {
            'type': 'text',
            'text': "Compare <image 1> and <image 2>.\nOptions: ['same', 'different']",
        }
        assert [part['type'] for part in message['content']] == ['text', 'image_url', 'image_url']
        assert all(part['image_url']['url'].startswith('data:image/') for part in message['content'][1:])

    def test_skips_rows_below_minimum_image_count(self, monkeypatch):
        plugin = MMMUMultiImageDatasetPlugin(_args({'min_images': 2}))
        monkeypatch.setattr(
            plugin,
            'load_hub_dataset',
            lambda **_: [
                {
                    'question': 'Single image only',
                    'options': '[]',
                    'image_1': Image.new('RGB', (2, 2), 'white'),
                }
            ],
        )

        assert list(plugin.build_messages()) == []

    def test_min_images_is_validated(self):
        with pytest.raises(Exception, match='min_images must be between 2 and 7'):
            MMMUMultiImageDatasetPlugin(_args({'min_images': 1}))

    def test_rejects_tokenized_prompt_mode(self):
        with pytest.raises(ValueError, match='not supported with the mmmu_multi_image dataset'):
            MMMUMultiImageDatasetPlugin(_args(tokenize_prompt=True, tokenizer_path='Qwen/Qwen2.5-0.5B-Instruct'))
