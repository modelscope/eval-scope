# MADQA


## Overview

MADQA (Multimodal Agentic Document QA) evaluates document-retrieval agents on human-authored
questions grounded in a heterogeneous collection of PDF documents. EvalScope loads the ModelScope
mirror, `evalscope/MADQA`.

## Task Description

- **Task Type**: Agentic multimodal document question answering
- **Input**: A natural-language question over a collection of PDF documents accessed through retrieval tools
- **Output**: A concise list of answer values and document/page citations in JSON
- **Domain**: Heterogeneous real-world documents across financial, legal, reference, technical, and public-record domains

## Key Features

- Contains 2,250 questions over 800 PDF documents; the public ModelScope mirror provides 1,550 train, 200 dev, and 500 hidden-label test questions
- Separates single-page, cross-page, and cross-document questions using the official evidence annotations
- Supports native and external EvalScope agent runs when retrieval tools are supplied through `TaskConfig.agent_config`
- Preserves the official answer-and-citation JSON contract, including an optional retrieval-step count

## Evaluation Notes

- The default `dev` split is the public scored split. The mirrored `test` split intentionally omits answer and evidence labels and cannot be scored locally
- Reports official deterministic metrics: ANLS*, accuracy at ANLS* >= 0.5, document F1, and page F1
- Kuiper statistic and Wasted Effort Ratio are additionally reported when predictions provide `iterations` or a native agent trace records retrieval tool calls
- The official optional Gemini semantic-accuracy mode is not enabled: its fixed Gemini judge and published calibration are not portable to EvalScope judge configurations
- EvalScope does not bundle the official BM25/OCR retrieval baseline. Configure compatible native tools, MCP servers, or an external agent that can access the published document URLs
- [Paper](https://arxiv.org/abs/2603.12180) | [GitHub](https://github.com/OxRML/MADQA)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `madqa` |
| **Dataset ID** | [evalscope/MADQA](https://modelscope.cn/datasets/evalscope/MADQA/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2603.12180) |
| **Tags** | `Agent`, `MultiModal`, `MultiTurn`, `QA`, `Retrieval` |
| **Metrics** | `anls`, `accuracy`, `document_f1`, `page_f1`, `kuiper_statistic`, `wasted_effort_ratio` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `dev` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 200 |
| Prompt Length (Mean) | 680.35 chars |
| Prompt Length (Min/Max) | 625 / 806 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `single` | 163 | 677.05 | 625 | 806 |
| `cross_page` | 18 | 686.11 | 638 | 739 |
| `cross_doc` | 19 | 703.21 | 644 | 776 |

## Sample Example

**Subset**: `single`

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

## Prompt Template

**System Prompt:**
```text
You are a document QA assistant with access to document-retrieval tools. The answer is contained in the document collection. Search iteratively when evidence is incomplete, then give concise answer values and exact PDF filename/page citations.
```

**Prompt Template:**
```text
{question}

Use the available document-retrieval tools to find evidence before answering. Return only a JSON object:
{{"answer": ["short answer"], "citations": [{{"document": "filename.pdf", "page": 1}}], "iterations": 0}}

The answer must be a list of concise values. Cite every document page used. Set `iterations` to the number of retrieval steps when it is known.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets madqa \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

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
            # subset_list: ['single', 'cross_page', 'cross_doc']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
