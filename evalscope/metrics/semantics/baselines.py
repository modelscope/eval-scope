"""Common metric semantics baselines.

``SEMANTIC_BASELINES`` is the shared vocabulary the benchmark catalog and the legacy mapping
build on: a benchmark entry references a baseline by key and only overrides the fields that
actually differ, so the 200+ benchmark declarations stay one line each.

Conventions enforced here:

- Keys equal the ``semantic_id`` of the declaration, named ``{domain}.{concept}.{unit}``.
- Diagnostic baselines always use ``direction=none``.
- Quality and performance baselines use ``kind=quality``. Report-level primary selection is stored
  separately on the report.
- Bounded ratios in [0, 1] render as percent with ``display_multiplier=100``; official 0-100
  scales render as percent with ``display_multiplier=1``.

The two helpers below carry the shared display fields, so a baseline states only what makes it
different from the others. That is why the table can be read as a vocabulary rather than as a
wall of repeated display settings, and why a change to how percentages render is one edit.
"""

from typing import Dict, Optional

from evalscope.api.metric.semantics import MetricDirection, MetricDisplayKind, MetricKind, MetricSemantics, ValueRange

#: Value range of a ratio metric.
_RATIO_RANGE = ValueRange(min=0.0, max=1.0)

#: Value range of an official 0-100 point scale.
_POINTS_100_RANGE = ValueRange(min=0.0, max=100.0)

#: Decimals used by every percent-rendered baseline.
_PERCENT_PRECISION = 1


def _percent(
    semantic_id: str,
    metric_name: str,
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
    kind: MetricKind = MetricKind.QUALITY,
    value_range: ValueRange = _RATIO_RANGE,
    display_multiplier: float = 100.0,
) -> MetricSemantics:
    """Declare a bounded metric rendered as a percentage.

    Args:
        semantic_id: Identifier of the declaration; must equal its key in the table.
        metric_name: Display name of the metric.
        direction: Optimization direction. Error rates pass ``LOWER_IS_BETTER``.
        kind: Intrinsic metric classification. Diagnostics must also pass ``direction=NONE``.
        value_range: Bounds of the stored value. Official 0-100 scales pass
            ``_POINTS_100_RANGE``.
        display_multiplier: Scale applied for display only: ``100`` for a ``[0, 1]`` ratio, ``1``
            for a value already expressed in points.

    Returns:
        The validated baseline declaration.
    """
    return MetricSemantics(
        semantic_id=semantic_id,
        metric_name=metric_name,
        kind=kind,
        direction=direction,
        value_range=value_range,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=display_multiplier,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
    )


def _plain_number(
    semantic_id: str,
    metric_name: str,
    display_precision: int,
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
    kind: MetricKind = MetricKind.QUALITY,
    raw_unit: Optional[str] = None,
    display_unit: Optional[str] = None,
) -> MetricSemantics:
    """Declare an unbounded metric rendered as a plain number.

    Args:
        semantic_id: Identifier of the declaration; must equal its key in the table.
        metric_name: Display name of the metric.
        display_precision: Decimals of the displayed value.
        direction: Optimization direction. Latencies pass ``LOWER_IS_BETTER``.
        kind: Intrinsic metric classification. Diagnostics must also pass ``direction=NONE``.
        raw_unit: Unit of the stored value, when it has one.
        display_unit: Unit appended after a space, when the displayed value carries one.

    Returns:
        The validated baseline declaration.
    """
    return MetricSemantics(
        semantic_id=semantic_id,
        metric_name=metric_name,
        kind=kind,
        direction=direction,
        raw_unit=raw_unit,
        value_range=None,
        display_kind=MetricDisplayKind.NUMBER,
        display_unit=display_unit,
        display_precision=display_precision,
    )


SEMANTIC_BASELINES: Dict[str, MetricSemantics] = {
    # --- quality: bounded ratios, higher is better ---------------------------------------
    'quality.accuracy.ratio': _percent('quality.accuracy.ratio', 'Accuracy'),
    'quality.f1.ratio': _percent('quality.f1.ratio', 'F1'),
    'quality.precision.ratio': _percent('quality.precision.ratio', 'Precision'),
    'quality.recall.ratio': _percent('quality.recall.ratio', 'Recall'),
    'quality.exact_match.ratio': _percent('quality.exact_match.ratio', 'Exact Match'),
    'quality.pass_at_k.ratio': _percent('quality.pass_at_k.ratio', 'Pass@k'),
    'quality.score.ratio': _percent('quality.score.ratio', 'Score'),
    'quality.coverage.ratio': _percent('quality.coverage.ratio', 'Coverage'),
    'quality.win_rate.ratio': _percent('quality.win_rate.ratio', 'Win Rate'),
    'quality.iou.ratio': _percent('quality.iou.ratio', 'IoU'),
    # --- quality: text generation overlap and similarity ----------------------------------
    'quality.bleu.ratio': _percent('quality.bleu.ratio', 'BLEU'),
    'quality.rouge.ratio': _percent('quality.rouge.ratio', 'ROUGE'),
    'quality.meteor.ratio': _percent('quality.meteor.ratio', 'METEOR'),
    'quality.similarity.ratio': _percent('quality.similarity.ratio', 'Similarity'),
    #: CIDEr is a consensus score that is not bounded by 1, so it renders as a plain number.
    'quality.cider.unbounded': _plain_number('quality.cider.unbounded', 'CIDEr', display_precision=3),
    # --- quality: bounded error rates, lower is better -----------------------------------
    'quality.wer.ratio': _percent('quality.wer.ratio', 'WER', MetricDirection.LOWER_IS_BETTER),
    'quality.cer.ratio': _percent('quality.cer.ratio', 'CER', MetricDirection.LOWER_IS_BETTER),
    'quality.mer.ratio': _percent('quality.mer.ratio', 'MER', MetricDirection.LOWER_IS_BETTER),
    #: Share of failed or hallucinated outcomes: a graded result, unlike the diagnostic
    #: parse-status shares, so it keeps a direction.
    'quality.error_rate.ratio': _percent(
        'quality.error_rate.ratio',
        'Error Rate',
        MetricDirection.LOWER_IS_BETTER,
    ),
    # --- quality: official scales and unbounded judge scores ------------------------------
    'quality.score.points_100': _percent(
        'quality.score.points_100',
        'Score',
        value_range=_POINTS_100_RANGE,
        display_multiplier=1.0,
    ),
    'quality.judge_score.unbounded': _plain_number(
        'quality.judge_score.unbounded',
        'Judge Score',
        display_precision=2,
    ),
    'quality.effort_calibration.unbounded': _plain_number(
        'quality.effort_calibration.unbounded',
        'Effort Calibration',
        display_precision=4,
        direction=MetricDirection.LOWER_IS_BETTER,
    ),
    'quality.effort_ratio.unbounded': _plain_number(
        'quality.effort_ratio.unbounded',
        'Effort Ratio',
        display_precision=4,
        direction=MetricDirection.LOWER_IS_BETTER,
    ),
    #: Score assigned by a scoring model (aesthetic / preference / alignment scorers) whose
    #: scale is defined by the model rather than by the benchmark.
    'quality.model_score.unbounded': _plain_number(
        'quality.model_score.unbounded',
        'Model Score',
        display_precision=4,
    ),
    # --- perf: latency, lower is better ---------------------------------------------------
    'perf.latency.seconds': _plain_number(
        'perf.latency.seconds',
        'Latency',
        display_precision=3,
        direction=MetricDirection.LOWER_IS_BETTER,
        raw_unit='s',
        display_unit='s',
    ),
    'perf.latency.milliseconds': _plain_number(
        'perf.latency.milliseconds',
        'Latency',
        display_precision=2,
        direction=MetricDirection.LOWER_IS_BETTER,
        raw_unit='ms',
        display_unit='ms',
    ),
    # --- perf: throughput, higher is better -----------------------------------------------
    'perf.throughput.tokens_per_second': _plain_number(
        'perf.throughput.tokens_per_second',
        'Token Throughput',
        display_precision=2,
        raw_unit='tok/s',
        display_unit='tok/s',
    ),
    'perf.throughput.requests_per_second': _plain_number(
        'perf.throughput.requests_per_second',
        'Request Throughput',
        display_precision=2,
        raw_unit='req/s',
        display_unit='req/s',
    ),
    # --- diagnostic: never carries a direction nor a comparison group ----------------------
    'diagnostic.count.items': _plain_number(
        'diagnostic.count.items',
        'Count',
        display_precision=0,
        direction=MetricDirection.NONE,
        kind=MetricKind.DIAGNOSTIC,
    ),
    'diagnostic.parse_status.ratio': _percent(
        'diagnostic.parse_status.ratio',
        'Parse Status',
        MetricDirection.NONE,
        kind=MetricKind.DIAGNOSTIC,
    ),
    'diagnostic.unspecified': _plain_number(
        'diagnostic.unspecified',
        'Unspecified',
        display_precision=4,
        direction=MetricDirection.NONE,
        kind=MetricKind.DIAGNOSTIC,
    ),
}
"""Baseline identifier -> semantics. Keys equal the declared ``semantic_id``."""

__all__ = ['SEMANTIC_BASELINES']
