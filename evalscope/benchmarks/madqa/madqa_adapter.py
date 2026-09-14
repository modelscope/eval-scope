import json
from typing import Any, Dict, List

from evalscope.api.agent import EventType
from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import anls_star, citation_f1, derive_hop_type, parse_prediction

DATASET_ID = 'evalscope/MADQA'
HOP_TYPES = ['single', 'cross_page', 'cross_doc']
PROMPT_TEMPLATE = """{question}

Use the available document-retrieval tools to find evidence before answering. Return only a JSON object:
{{"answer": ["short answer"], "citations": [{{"document": "filename.pdf", "page": 1}}], "iterations": 0}}

The answer must be a list of concise values. Cite every document page used. Set `iterations` to the number of retrieval steps when it is known."""
SYSTEM_PROMPT = """You are a document QA assistant with access to document-retrieval tools. The answer is contained in the document collection. Search iteratively when evidence is incomplete, then give concise answer values and exact PDF filename/page citations."""
DESCRIPTION = """
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
"""


@register_benchmark(
    BenchmarkMeta(
        name='madqa',
        pretty_name='MADQA',
        dataset_id=DATASET_ID,
        tags=[Tags.AGENT, Tags.MULTI_MODAL, Tags.MULTI_TURN, Tags.RETRIEVAL, Tags.QA],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2603.12180',
        metric_list=['anls', 'accuracy', 'document_f1', 'page_f1', 'kuiper_statistic', 'wasted_effort_ratio'],
        primary_metric='accuracy',
        few_shot_num=0,
        train_split=None,
        eval_split='dev',
        subset_list=HOP_TYPES,
        prompt_template=PROMPT_TEMPLATE,
        system_prompt=SYSTEM_PROMPT,
        evaluation_version='v1.0',
    )
)
class MADQAAdapter(AgentAdapter):
    """Adapter for the official MADQA answer, citation, and effort protocol."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a labeled MADQA question into an agent-evaluation sample."""
        answer_variants = record.get('answer_variants') or []
        evidence = record.get('evidence') or []
        if not answer_variants or not evidence:
            raise ValueError(
                'MADQA requires labeled questions. Use the default dev split; the public test split has no answer or evidence labels.'
            )

        question = str(record['question']).strip()
        hop_type = derive_hop_type(evidence)
        return Sample(
            input=question,
            target=json.dumps(answer_variants, ensure_ascii=False),
            subset_key=hop_type,
            metadata={
                'id': record.get('id', ''),
                'question': question,
                'evidence': evidence,
                'document_category': record.get('document_category', ''),
                'domain': record.get('domain', ''),
                'hop_type': hop_type,
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Normalize a model response to the MADQA JSON output contract."""
        return json.dumps(parse_prediction(prediction), ensure_ascii=False)

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Score a response with MADQA's deterministic answer and citation metrics."""
        response = parse_prediction(filtered_prediction)
        answer_variants = json.loads(reference)
        evidence = (task_state.metadata or {}).get('evidence') or []
        anls = anls_star(response['answer'], answer_variants)
        accuracy = float(anls >= 0.5)
        document_f1 = citation_f1(response['citations'], evidence, level='document')
        page_f1 = citation_f1(response['citations'], evidence, level='page')
        steps = self._steps(response, task_state)
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={
                'anls': anls,
                'accuracy': accuracy,
                'document_f1': document_f1,
                'page_f1': page_f1,
            },
            metadata={'steps': steps, 'citations': response['citations']},
            main_score_name='accuracy',
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Add official effort-calibration metrics to standard per-sample means."""
        scores = super().aggregate_scores(sample_scores)
        effort_results = [
            {
                'correct': bool(sample_score.score.value.get('accuracy')),
                'steps': self._score_steps(sample_score),
            }
            for sample_score in sample_scores
            if self._score_steps(sample_score) > 0
        ]
        if not effort_results:
            return scores

        correct_steps = [item['steps'] for item in effort_results if item['correct']]
        incorrect_steps = [item['steps'] for item in effort_results if not item['correct']]
        if correct_steps and incorrect_steps:
            scores.append(
                AggScore(
                    score=sum(incorrect_steps) / len(incorrect_steps) / (sum(correct_steps) / len(correct_steps)),
                    metric_name='wasted_effort_ratio',
                    aggregation='identity',
                    num=len(effort_results),
                )
            )

        correctness = [float(item['correct']) for item in sorted(effort_results, key=lambda item: item['steps'])]
        mean_correctness = sum(correctness) / len(correctness)
        if 0 < mean_correctness < 1:
            cumulative = 0.0
            deviations = []
            for value in correctness:
                cumulative += value - mean_correctness
                deviations.append(cumulative)
            scores.append(
                AggScore(
                    score=max(deviations) - min(deviations),
                    metric_name='kuiper_statistic',
                    aggregation='identity',
                    num=len(effort_results),
                )
            )
        return scores

    @staticmethod
    def _steps(response: Dict[str, Any], task_state: TaskState) -> int:
        if response['iterations'] > 0:
            return response['iterations']
        if task_state.agent_trace is None:
            return 0
        return sum(
            event.type == EventType.TOOL_CALL and event.payload.get('name') != 'submit'
            for event in task_state.agent_trace.events
        )

    @staticmethod
    def _score_steps(sample_score: SampleScore) -> int:
        metadata = sample_score.score.metadata or {}
        steps = metadata.get('steps', 0)
        return steps if isinstance(steps, int) and steps > 0 else 0
