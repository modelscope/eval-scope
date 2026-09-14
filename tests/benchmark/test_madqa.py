import json
import unittest

from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.madqa.utils import anls_star, citation_f1, parse_prediction
from evalscope.config import TaskConfig


class TestMADQAUtils(unittest.TestCase):

    def test_anls_star_matches_unordered_lists_and_best_variant(self) -> None:
        score = anls_star(
            ['second answer', 'first answer'],
            [['wrong answer'], ['first answer', 'second answer']],
        )

        self.assertEqual(score, 1.0)

    def test_anls_star_penalizes_missing_or_hallucinated_items(self) -> None:
        self.assertEqual(anls_star(['one'], [['one', 'two']]), 0.5)
        self.assertEqual(anls_star(['one', 'two'], [['one']]), 0.5)

    def test_parse_prediction_normalizes_citation_aliases(self) -> None:
        prediction = parse_prediction(
            'Reasoning\n{"answer": "value", "citations": [{"file": "source.pdf", "page": "2"}], "iterations": 3}'
        )

        self.assertEqual(prediction, {
            'answer': ['value'],
            'citations': [{'document': 'source.pdf', 'page': 2}],
            'iterations': 3,
        })

    def test_parse_prediction_uses_official_search_history_for_effort(self) -> None:
        prediction = parse_prediction('{"answer": ["value"], "citations": [], "search_history": [{}, {}]}')

        self.assertEqual(prediction['iterations'], 2)

    def test_citation_f1_uses_sets_at_each_official_level(self) -> None:
        evidence = [
            {'document': 'a.pdf', 'page': 1},
            {'document': 'a.pdf', 'page': 2},
            {'document': 'b.pdf', 'page': 1},
        ]
        citations = [{'document': 'a.pdf', 'page': 1}, {'document': 'b.pdf', 'page': 2}]

        self.assertEqual(citation_f1(citations, evidence, level='document'), 1.0)
        self.assertEqual(citation_f1(citations, evidence, level='page'), 0.4)


class TestMADQAAdapter(unittest.TestCase):

    @staticmethod
    def _adapter():
        return get_benchmark('madqa', TaskConfig(model='mock', datasets=['madqa']))

    @staticmethod
    def _record() -> dict:
        return {
            'id': 'dev/0',
            'question': 'Which values are correct?',
            'answer_variants': [['first', 'second'], ['1st', '2nd']],
            'evidence': [{'document': 'a.pdf', 'page': 1}, {'document': 'b.pdf', 'page': 2}],
            'document_category': 'Report',
            'domain': 'Financial',
        }

    def test_record_to_sample_preserves_official_metadata(self) -> None:
        sample = self._adapter().record_to_sample(self._record())

        self.assertEqual(sample.subset_key, 'cross_doc')
        self.assertEqual(json.loads(sample.target), self._record()['answer_variants'])
        self.assertEqual(sample.metadata['evidence'], self._record()['evidence'])

    def test_pipeline_scores_answer_citations_and_iterations(self) -> None:
        adapter = self._adapter()
        sample = adapter.record_to_sample(self._record())
        task_state = TaskState(
            model='mock',
            sample=sample,
            output=ModelOutput.from_content(
                model='mock',
                content='{"answer": ["second", "first"], "citations": [{"document": "a.pdf", "page": 1}, {"document": "b.pdf", "page": 2}], "iterations": 2}',
            ),
            completed=True,
        )

        sample_score = adapter.calculate_metrics(task_state)

        self.assertEqual(sample_score.score.value, {
            'anls': 1.0,
            'accuracy': 1.0,
            'document_f1': 1.0,
            'page_f1': 1.0,
        })
        self.assertEqual(sample_score.score.metadata['steps'], 2)

    def test_aggregate_scores_adds_official_effort_metrics(self) -> None:
        adapter = self._adapter()
        sample_scores = [
            SampleScore(
                score=Score(value={'anls': 1.0, 'accuracy': 1.0, 'document_f1': 1.0, 'page_f1': 1.0}, metadata={'steps': 2}),
                sample_id=0,
                group_id=0,
            ),
            SampleScore(
                score=Score(value={'anls': 0.0, 'accuracy': 0.0, 'document_f1': 0.0, 'page_f1': 0.0}, metadata={'steps': 4}),
                sample_id=1,
                group_id=1,
            ),
        ]

        aggregated = {(score.metric_name, score.aggregation): score.score for score in adapter.aggregate_scores(sample_scores)}

        self.assertEqual(aggregated[('wasted_effort_ratio', 'identity')], 2.0)
        self.assertEqual(aggregated[('kuiper_statistic', 'identity')], 0.5)


if __name__ == '__main__':
    unittest.main()
