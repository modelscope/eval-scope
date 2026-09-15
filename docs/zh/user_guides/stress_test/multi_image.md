# 多图输入压测

EvalScope 提供两种互补方式，对一次请求包含多张图片的场景进行压测：

- `mmmu_multi_image`：从开源 [MMMU](https://modelscope.cn/datasets/AI-ModelScope/MMMU/summary) 数据集中构造真实多图请求。
- `line_by_line`：直接接受 OpenAI 风格的 messages 或完整请求体，因此无需新增数据集插件即可压测自建多图样本。

## 使用 MMMU 构造真实多图请求

`mmmu_multi_image` 加载 MMMU 的 validation split，并仅保留图片数量不少于配置值的样本。图片按照 `image_1` ... `image_7` 的顺序放入同一个 user message 中。

```bash
evalscope perf \
  --model your-vl-model \
  --url http://localhost:8000/v1/chat/completions \
  --dataset mmmu_multi_image \
  --dataset-args '{"subset":"Music","min_images":2}' \
  --parallel 4 \
  --number 100
```

数据集参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `subset` | `str` | `Music` | 要加载的 MMMU 学科/configuration |
| `min_images` | `int` | `2` | 样本最少图片数，范围为 2–7 |

该模式用于构造真实多模态压测流量，而不是计算 MMMU 准确率。如果需要正式的 MMMU 评测得分，请使用常规的 `evalscope eval --datasets mmmu` 流程。

## 使用自建多图数据

对于私有或自行构造的数据集，可以直接使用 `line_by_line`。每个非空行本身就可以是 OpenAI 风格的 messages 数组或完整请求体；只需在同一个 user message 中放入多个 `image_url` 内容块。

例如，将下面内容作为一行 JSON 保存到 `multi_image.jsonl`：

```json
[{"role":"user","content":[{"type":"text","text":"Compare image 1 and image 2."},{"type":"image_url","image_url":{"url":"https://example.com/image-1.jpg"}},{"type":"image_url","image_url":{"url":"https://example.com/image-2.jpg"}}]}]
```

然后执行：

```bash
evalscope perf \
  --model your-vl-model \
  --url http://localhost:8000/v1/chat/completions \
  --dataset line_by_line \
  --dataset-path multi_image.jsonl \
  --parallel 4 \
  --number 100
```

模型服务必须能够访问 HTTP 图片 URL；如果服务端支持，也可以使用兼容 API 的 data URL。
