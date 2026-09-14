# MADQA


## 概述

MADQA（Multimodal Agentic Document QA）评估文档检索智能体在基于异构 PDF 文档集合的人工编写问题上的表现。EvalScope 加载 ModelScope 镜像 `evalscope/MADQA`。

## 任务描述

- **任务类型**：智能体驱动的多模态文档问答
- **输入**：针对通过检索工具访问的一组 PDF 文档提出的自然语言问题
- **输出**：包含简洁答案值及文档/页码引用的 JSON 格式列表
- **领域**：涵盖金融、法律、参考资料、技术文档和公共记录等多个领域的异构真实世界文档

## 关键特性

- 包含 800 份 PDF 文档上的 2,250 个问题；公开的 ModelScope 镜像提供 1,550 个训练问题、200 个开发集问题和 500 个隐藏标签的测试问题
- 利用官方证据标注，将问题划分为单页、跨页和跨文档三类
- 当通过 `TaskConfig.agent_config` 提供检索工具时，支持原生和外部 EvalScope 智能体运行
- 保留官方定义的答案与引用 JSON 格式规范，包括可选的检索步骤计数

## 评估说明

- 默认的 `dev` 划分是公开评分的数据集。镜像中的 `test` 划分故意省略了答案和证据标签，无法在本地评分
- 报告官方确定性指标：ANLS*、ANLS* >= 0.5 的准确率、文档 F1 和页面 F1
- 当预测结果包含 `iterations` 字段或原生智能体轨迹记录了检索工具调用时，额外报告 Kuiper 统计量和浪费努力比率（Wasted Effort Ratio）
- 官方可选的 Gemini 语义准确性模式未启用：其固定的 Gemini 评判器和已发布的校准无法移植到 EvalScope 的评判器配置中
- EvalScope 未捆绑官方的 BM25/OCR 检索基线。请配置兼容的原生工具、MCP 服务器，或一个能够访问已发布文档 URL 的外部智能体
- [论文](https://arxiv.org/abs/2603.12180) | [GitHub](https://github.com/OxRML/MADQA)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `madqa` |
| **数据集ID** | [evalscope/MADQA](https://modelscope.cn/datasets/evalscope/MADQA/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2603.12180) |
| **标签** | `Agent`, `MultiModal`, `MultiTurn`, `QA`, `Retrieval` |
| **指标** | `anls`, `accuracy`, `document_f1`, `page_f1`, `kuiper_statistic`, `wasted_effort_ratio` |
| **默认示例数** | 0-shot |
| **评估划分** | `dev` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 200 |
| 提示词长度（平均） | 680.35 字符 |
| 提示词长度（最小/最大） | 625 / 806 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `single` | 163 | 677.05 | 625 | 806 |
| `cross_page` | 18 | 686.11 | 638 | 739 |
| `cross_doc` | 19 | 703.21 | 644 | 776 |

## 样例示例

**子集**: `single`

```json
{
  "input": [
    {
      "id": "b3dc7205",
      "content": "You are a document QA assistant with access to document-retrieval tools. The answer is contained in the document collection. Search iteratively when evidence is incomplete, then give concise answer values and exact PDF filename/page citations."
    },
    {
      "id": "1e14dcbf",
      "content": "What are the CAD Standards Guide requirements when it comes to numbered street name conventions?\n\nUse the available document-retrieval tools to find evidence before answering. Return only a JSON object:\n{\"answer\": [\"short answer\"], \"citations\": [{\"document\": \"filename.pdf\", \"page\": 1}], \"iterations\": 0}\n\nThe answer must be a list of concise values. Cite every document page used. Set `iterations` to the number of retrieval steps when it is known."
    }
  ],
  "target": "[[\"For First Street through Twelfth Street, spell-out the number\", \"For 28th Street and above (e.g. 44th Street), use numbers\", \"For Three Mile Road, Four Mile Road, etc., spell-out the number\"], [\"spell-out numbers for First Street through Twelfth Street\", \"use numbers for 28th Street and above (e.g. 44th Street)\", \"spell-out numbers for Three Mile Road, Four Mile Road, etc.\"]]",
  "id": 0,
  "group_id": 0,
  "subset_key": "single",
  "metadata": {
    "id": "dev/0",
    "question": "What are the CAD Standards Guide requirements when it comes to numbered street name conventions?",
    "evidence": [
      {
        "document": "24514009.pdf",
        "page": 7
      }
    ],
    "document_category": "Guide",
    "domain": "Reference",
    "hop_type": "single"
  }
}
```

## 提示模板

**系统提示：**
```text
You are a document QA assistant with access to document-retrieval tools. The answer is contained in the document collection. Search iteratively when evidence is incomplete, then give concise answer values and exact PDF filename/page citations.
```

**提示模板：**
```text
{question}

Use the available document-retrieval tools to find evidence before answering. Return only a JSON object:
{{"answer": ["short answer"], "citations": [{{"document": "filename.pdf", "page": 1}}], "iterations": 0}}

The answer must be a list of concise values. Cite every document page used. Set `iterations` to the number of retrieval steps when it is known.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets madqa \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['madqa'],
    dataset_args={
        'madqa': {
            # subset_list: ['single', 'cross_page', 'cross_doc']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
