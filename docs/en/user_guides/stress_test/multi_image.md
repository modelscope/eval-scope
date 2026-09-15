# Multi-image stress testing

EvalScope supports two complementary ways to benchmark requests that contain multiple images:

- `mmmu_multi_image` builds real multi-image requests from the open-source [MMMU](https://modelscope.cn/datasets/AI-ModelScope/MMMU/summary) dataset.
- `line_by_line` accepts OpenAI-style messages or complete request bodies, so you can replay your own multi-image samples without adding a dataset plugin.

## Real multi-image data with MMMU

`mmmu_multi_image` loads the MMMU validation split and keeps only rows that contain at least the configured number of images. Images are sent in `image_1` ... `image_7` order in one user message.

```bash
evalscope perf \
  --model your-vl-model \
  --url http://localhost:8000/v1/chat/completions \
  --dataset mmmu_multi_image \
  --dataset-args '{"subset":"Music","min_images":2}' \
  --parallel 4 \
  --number 100
```

Dataset arguments:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `subset` | `str` | `Music` | MMMU subject/configuration to load |
| `min_images` | `int` | `2` | Minimum number of images required for a row; must be between 2 and 7 |

The built-in mode is intended for real multimodal traffic rather than accuracy evaluation. Use the regular `evalscope eval --datasets mmmu` path when you need MMMU benchmark scoring.

## Custom multi-image data

For private or constructed datasets, use `line_by_line`. Each non-empty line can already be an OpenAI-style messages array or a complete request body. Put multiple `image_url` parts in the same user message.

For example, save the following as one JSON line in `multi_image.jsonl`:

```json
[{"role":"user","content":[{"type":"text","text":"Compare image 1 and image 2."},{"type":"image_url","image_url":{"url":"https://example.com/image-1.jpg"}},{"type":"image_url","image_url":{"url":"https://example.com/image-2.jpg"}}]}]
```

Then run:

```bash
evalscope perf \
  --model your-vl-model \
  --url http://localhost:8000/v1/chat/completions \
  --dataset line_by_line \
  --dataset-path multi_image.jsonl \
  --parallel 4 \
  --number 100
```

The model server must be able to access HTTP image URLs. You can also use API-compatible data URLs when your serving backend supports them.
