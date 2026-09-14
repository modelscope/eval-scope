"""Canonical metric semantics registry and read-old migration manifest.

``METRIC_DEFINITIONS`` is the only table used by the v2 resolver. It is keyed by canonical metric
name and deliberately contains no aggregation prefixes or dynamic ``k`` variants. The
``LEGACY_METRIC_MIGRATIONS`` table is a read-old manifest used only to migrate adapter output and
historical reports; aliases in it never participate in v2 resolution.

A canonical name needs no read-old entry: a v1 report carrying it already resolves through
``METRIC_DEFINITIONS`` to the same semantics, so restating it here would only be a second place to
forget when its display fields change. Declare a name here when reading it old differs from
resolving it new -- a non-canonical spelling, or a name whose historical meaning differs from the
current one (``error_rate``).
"""

from typing import Dict, Tuple

from evalscope.api.metric.semantics import MetricDirection
from evalscope.metrics.semantics.aliases import read_old_baselines
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES
from evalscope.metrics.semantics.entry import BASELINE_TABLE_LOCATION, MetricEntry

#: Where to declare a canonical metric name, used in audit and validation messages.
METRIC_NAME_TABLE_LOCATION = 'evalscope/metrics/semantics/catalog.py::METRIC_DEFINITIONS'

#: Where to declare a read-old alias, used in validation messages.
LEGACY_MIGRATION_TABLE_LOCATION = 'evalscope/metrics/semantics/catalog.py::LEGACY_METRIC_MIGRATIONS'

#: Where to declare a benchmark level collision override, used in validation messages.
BENCHMARK_OVERRIDE_TABLE_LOCATION = 'evalscope/metrics/semantics/catalog.py::BENCHMARK_METRIC_OVERRIDES'

LEGACY_METRIC_MIGRATIONS: Dict[str, MetricEntry] = {
    # --- quality ratios: one line each, reused by every benchmark ------------------------
    # Bounded [0, 1] ratios rendered as percent, higher is better.
    'mean_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_multi_choice_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_number_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_unit_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    # Accuracy over one kind of UI target, reported alongside the overall accuracy
    # (screenspot_pro).
    'mean_text_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_icon_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_Center_ACC': MetricEntry(baseline='quality.accuracy.ratio'),
    # hallusion_bench prefixes each accuracy with the aggregation bucket. The buckets come from
    # the data, so only the `Overall_` ones can be declared; the per-category rows degrade.
    'Overall_aAcc': MetricEntry(baseline='quality.accuracy.ratio'),
    'Overall_fAcc': MetricEntry(baseline='quality.accuracy.ratio'),
    'Overall_qAcc': MetricEntry(baseline='quality.accuracy.ratio'),
    # Grounding accuracy at a fixed IoU threshold (refcoco).
    'mean_ACC@0.1': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_ACC@0.3': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_ACC@0.5': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_ACC@0.7': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_ACC@0.9': MetricEntry(baseline='quality.accuracy.ratio'),
    # Rubric dimensions of the plawbench `case_analysis` subset, reported next to the overall
    # `mean_acc` as point ratios over the same rubric.
    'mean_conclusion_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_fact_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_reasoning_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_law_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    # Instruction following: strict / loose at prompt and instruction level.
    'mean_prompt_level_strict': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_prompt_level_loose': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_inst_level_strict': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_inst_level_loose': MetricEntry(baseline='quality.accuracy.ratio'),
    # Graded-answer benchmarks (simple_qa, browsecomp, chinese_simple_qa) report the share of
    # correct answers as `is_correct`.
    'mean_is_correct': MetricEntry(baseline='quality.accuracy.ratio'),
    # Agent style task completion ratio (miniwob, wide_search).
    'mean_success_rate': MetricEntry(baseline='quality.accuracy.ratio'),
    # --- exact match ---------------------------------------------------------------------
    'mean_em': MetricEntry(baseline='quality.exact_match.ratio'),
    # Tool-use benchmarks score the action and the plan by exact match (tool_bench).
    'mean_Act.EM': MetricEntry(baseline='quality.exact_match.ratio'),
    'mean_Plan.EM': MetricEntry(baseline='quality.exact_match.ratio'),
    # --- pass ratios ---------------------------------------------------------------------
    'mean_pass_rate': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'mean_pass_at_k': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'mean_pass_hat_k': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'Pass@1': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'pass@1': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'mean_strict_pass': MetricEntry(baseline='quality.pass_at_k.ratio'),
    # --- F1 / precision / recall ---------------------------------------------------------
    'mean_f1': MetricEntry(baseline='quality.f1.ratio'),
    'mean_F1': MetricEntry(baseline='quality.f1.ratio'),
    'mean_boundary_precision': MetricEntry(baseline='quality.precision.ratio'),
    # --- text generation overlap and similarity ------------------------------------------
    'Bleu_1': MetricEntry(baseline='quality.bleu.ratio'),
    'Bleu_2': MetricEntry(baseline='quality.bleu.ratio'),
    'Bleu_3': MetricEntry(baseline='quality.bleu.ratio'),
    'Bleu_4': MetricEntry(baseline='quality.bleu.ratio'),
    'mean_Bleu_1': MetricEntry(baseline='quality.bleu.ratio'),
    'mean_Bleu_2': MetricEntry(baseline='quality.bleu.ratio'),
    'mean_Bleu_3': MetricEntry(baseline='quality.bleu.ratio'),
    'mean_Bleu_4': MetricEntry(baseline='quality.bleu.ratio'),
    'mean_BLEU': MetricEntry(baseline='quality.bleu.ratio'),
    'mean_bleu': MetricEntry(baseline='quality.bleu.ratio'),
    'mean_ROUGE_L': MetricEntry(baseline='quality.rouge.ratio'),
    'mean_Rouge': MetricEntry(baseline='quality.rouge.ratio'),
    'mean_Rouge-L': MetricEntry(baseline='quality.rouge.ratio'),
    'mean_METEOR': MetricEntry(baseline='quality.meteor.ratio'),
    'mean_CIDEr': MetricEntry(baseline='quality.cider.unbounded'),
    'mean_bert_score': MetricEntry(baseline='quality.similarity.ratio'),
    'mean_comet': MetricEntry(baseline='quality.similarity.ratio'),
    'Semantic Consistency': MetricEntry(baseline='quality.similarity.ratio'),
    'Perceptual Similarity': MetricEntry(baseline='quality.similarity.ratio'),
    # --- speech recognition error rates: lower is better ---------------------------------
    'mean_wer': MetricEntry(baseline='quality.wer.ratio'),
    'mean_audio_wer': MetricEntry(baseline='quality.wer.ratio'),
    'mean_cer': MetricEntry(baseline='quality.cer.ratio'),
    'mean_mer': MetricEntry(baseline='quality.mer.ratio'),
    # --- graded failure rates: lower is better -------------------------------------------
    'mean_error_rate': MetricEntry(baseline='quality.error_rate.ratio'),
    'mean_HalluRate': MetricEntry(baseline='quality.error_rate.ratio'),
    'mean_distractor_leakage': MetricEntry(baseline='quality.error_rate.ratio'),
    # --- bounded quality scores ----------------------------------------------------------
    'mean_score': MetricEntry(baseline='quality.score.ratio'),
    'mean_partial_credit': MetricEntry(baseline='quality.score.ratio'),
    'mean_submission_ready': MetricEntry(baseline='quality.score.ratio'),
    'mean_required_coverage': MetricEntry(baseline='quality.coverage.ratio'),
    'mean_coverage_score': MetricEntry(baseline='quality.coverage.ratio'),
    # --- official 0-100 scales -----------------------------------------------------------
    'mean_eq_bench_score': MetricEntry(baseline='quality.score.points_100'),
    # --- win rates -----------------------------------------------------------------------
    'mean_winrate': MetricEntry(baseline='quality.win_rate.ratio'),
    # --- judge scores (unbounded) --------------------------------------------------------
    'mean_total_score': MetricEntry(baseline='quality.judge_score.unbounded'),
    'mean_normalized_score': MetricEntry(baseline='quality.score.ratio'),
    'mean_avg_score': MetricEntry(baseline='quality.judge_score.unbounded'),
    'mean_net_match_score': MetricEntry(baseline='quality.judge_score.unbounded'),
    # --- diagnostics: distribution shares and raw counts carry no direction --------------
    # The one name whose historical meaning differs from the current one: a v1 `error_rate` was a
    # parse-status share, while a current run grades it as a quality error rate. This divergence is
    # why the read-old manifest exists at all.
    'error_rate': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'mean_is_incorrect': MetricEntry(baseline='diagnostic.parse_status.ratio', display_name='Incorrect rate'),
    'mean_is_not_attempted': MetricEntry(baseline='diagnostic.parse_status.ratio', display_name='Not attempted rate'),
    # --- legacy names: only produced by report files written before the semantics -------
    # contract. Safe to drop once no report of that vintage is expected to be opened again.
    'WeightedScorePercent': MetricEntry(baseline='quality.score.points_100'),
    'AverageOutputTps': MetricEntry(baseline='perf.throughput.tokens_per_second'),
    # Names emitted by earlier revisions of adapters that have since renamed their metrics.
    # Kept so an archived report still renders with the right direction and unit instead of
    # degrading to a bare number.
    'mean_precision': MetricEntry(baseline='quality.precision.ratio'),
    'mean_recall': MetricEntry(baseline='quality.recall.ratio'),
    'mean_f1_score': MetricEntry(baseline='quality.f1.ratio'),
    'mean_simple_pass_rate': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'mean_simple_partial_credit': MetricEntry(baseline='quality.score.ratio'),
    'mean_simple_error_rate': MetricEntry(baseline='quality.error_rate.ratio'),
    'mean_compliance_score': MetricEntry(baseline='quality.score.ratio'),
    'mean_max_score': MetricEntry(baseline='quality.judge_score.unbounded'),
    # Document parsing: TEDS and CDM are similarity scores, edit distances are error rates.
    'overall_CH': MetricEntry(baseline='quality.score.ratio'),
    'overall_EN': MetricEntry(baseline='quality.score.ratio'),
    'table_TEDS': MetricEntry(baseline='quality.similarity.ratio'),
    'table_TEDS_structure_only': MetricEntry(baseline='quality.similarity.ratio'),
    'display_formula_CDM': MetricEntry(baseline='quality.similarity.ratio'),
    'text_block_Edit_dist': MetricEntry(baseline='quality.error_rate.ratio'),
    'display_formula_Edit_dist': MetricEntry(baseline='quality.error_rate.ratio'),
    'table_Edit_dist': MetricEntry(baseline='quality.error_rate.ratio'),
    'reading_order_Edit_dist': MetricEntry(baseline='quality.error_rate.ratio'),
    # Run cost breakdowns: observations, not grades.
    'mean_total_tokens': MetricEntry(baseline='diagnostic.count.items'),
    'mean_total_model_input_tokens': MetricEntry(baseline='diagnostic.count.items'),
    'mean_total_model_output_tokens': MetricEntry(baseline='diagnostic.count.items'),
    'mean_total_wall_time_s': MetricEntry(baseline='diagnostic.unspecified', raw_unit='s', display_precision=2),
    'mean_total_model_time_s': MetricEntry(baseline='diagnostic.unspecified', raw_unit='s', display_precision=2),
    'mean_total_tool_time_s': MetricEntry(baseline='diagnostic.unspecified', raw_unit='s', display_precision=2),
    'mean_total_other_time_s': MetricEntry(baseline='diagnostic.unspecified', raw_unit='s', display_precision=2),
}
LEGACY_METRIC_MIGRATIONS.update(
    {name: MetricEntry(baseline=baseline) for name, baseline in read_old_baselines().items()}
)
"""Final report metric name -> catalog entry, reused by every benchmark.

Seeded from the metric names this repository actually emits, so the common names render with the
right direction and unit. A name that is not here degrades to a diagnostic rather than blocking.
"""

# Canonical v2 names are declared independently from the read-old manifest above. This keeps
# aliases from participating in new-result resolution or determining which semantics wins when
# several historical spellings collapse to the same identity.
_CANONICAL_NAMES_BY_BASELINE = {
    'diagnostic.count.items': (
        'avg_reason_lens',
        'count_finish_reason_tool_call',
        'count_finish_reason_tool_calls',
        'count_successful_tool_call',
        'no_answer_num',
        'total_model_input_tokens',
        'total_model_output_tokens',
        'total_tokens',
    ),
    'diagnostic.parse_status.ratio': (
        'inference_error_rate',
        'maybe_ratio',
        'official_mean_correct_with_excessive_answers',
        'official_mean_fully_incorrect_items',
        'rate_empty_auto_rater_response',
        'rate_empty_model_response',
        'rate_invalid_auto_rater_response',
        'yes_ratio',
    ),
    'perf.throughput.tokens_per_second': ('average_output_tps',),
    'quality.accuracy.ratio': (
        'accuracy',
        'cell_acc',
        'conclusion_acc',
        'correct_acc',
        'easy_puzzle_acc',
        'error_acc',
        'fact_acc',
        'hard_puzzle_acc',
        'icon_acc',
        'inst_level_loose',
        'inst_level_strict',
        'is_correct',
        'large_puzzle_acc',
        'law_acc',
        'math_acc',
        'medium_puzzle_acc',
        'multi_choice_acc',
        'number_acc',
        'numeric_match',
        'official_mean_all_answers_correct',
        'overall_a_acc',
        'overall_f_acc',
        'overall_q_acc',
        'param_default_accept_rate',
        'param_immutable_reject_rate',
        'process_acc',
        'prompt_level_loose',
        'prompt_level_strict',
        'puzzle_acc',
        'reasoning_acc',
        'relaxed_acc',
        'schema_accuracy',
        'tool_calls_match_rate',
        'language_following_success_rate',
        'repeat_ngram_pass_rate',
        'scenario_check_pass_rate',
        'small_puzzle_acc',
        'success_rate',
        'task_averaged_acc',
        'text_acc',
        'unit_acc',
        'xl_puzzle_acc',
    ),
    'quality.bleu.ratio': ('bleu',),
    'quality.cer.ratio': ('cer',),
    'quality.cider.unbounded': ('cider',),
    'quality.coverage.ratio': ('coverage_score', 'required_coverage'),
    'quality.error_rate.ratio': (
        'display_formula_edit_dist',
        'distractor_leakage',
        'error_rate',
        'error_only_reasoning_rate',
        'hallucination_rate',
        'reading_order_edit_dist',
        'simple_error_rate',
        'table_edit_dist',
        'text_block_edit_dist',
    ),
    'quality.exact_match.ratio': ('act_em', 'exact_match', 'plan_em'),
    'quality.f1.ratio': (
        'f1',
        'f1_macro',
        'f1_micro',
        'f1_weighted',
        'official_mean_f1_score',
        'simple_f1_score',
        'task_averaged_f1',
        'tool_call_f1',
    ),
    'quality.iou.ratio': ('iou',),
    'quality.judge_score.unbounded': (
        'communication_quality',
        'completeness',
        'context_awareness',
        'instruction_following',
        'judge_score',
        'max_score',
        'net_match_score',
    ),
    'quality.mer.ratio': ('mer',),
    'quality.meteor.ratio': ('meteor',),
    'quality.model_score.unbounded': (
        'blipv2_score',
        'clipscore',
        'fga_blip2_score',
        'hps_v2_1_score',
        'hpsv2_score',
        'image_reward_score',
        'mps',
        'pick_score',
        'vqa_model_score',
    ),
    'quality.pass_at_k.ratio': (
        'main_problem_pass_rate',
        'pass_1',
        'pass_at_k',
        'pass_hat_k',
        'pass_rate',
        'simple_pass_rate',
        'strict_pass',
        'subproblem_pass_rate',
    ),
    'quality.precision.ratio': ('boundary_precision', 'official_mean_precision', 'precision'),
    'quality.recall.ratio': ('official_mean_recall', 'recall'),
    'quality.rouge.ratio': ('rouge', 'rouge_l'),
    'quality.score.points_100': ('eq_bench_score', 'weighted_score_percent'),
    'quality.score.ratio': (
        'clipped_score',
        'compliance_score',
        'contains_all',
        'contains_any',
        'mrcr_score',
        'normalized_score',
        'overall_ch',
        'overall_en',
        'overall_mrcr_score',
        'partial_credit',
        'simple_partial_credit',
        'submission_ready',
        'vqa_score',
    ),
    'quality.similarity.ratio': (
        'anls',
        'trigger_similarity',
        'bert_score',
        'comet',
        'display_formula_cdm',
        'perceptual_similarity',
        'sem_score',
        'semantic_consistency',
        'ssim',
        'table_teds',
        'table_teds_structure_only',
    ),
    'quality.wer.ratio': ('audio_wer', 'wer'),
    'quality.win_rate.ratio': ('win_rate',),
}

METRIC_DEFINITIONS: Dict[str, MetricEntry] = {
    name: MetricEntry(baseline=baseline) for baseline, names in _CANONICAL_NAMES_BY_BASELINE.items() for name in names
}
METRIC_DEFINITIONS.update(
    {
        # Three-way answer grading exposes these distribution shares as diagnostics. They remain
        # non-primary and directionless, but deserve report labels rather than internal identities,
        # which is why they are declared here instead of in the table above.
        'is_incorrect': MetricEntry(baseline='diagnostic.parse_status.ratio', display_name='Incorrect rate'),
        'is_not_attempted': MetricEntry(baseline='diagnostic.parse_status.ratio', display_name='Not attempted rate'),
        'bias_ratio': MetricEntry(
            baseline='quality.error_rate.ratio',
            metric_name='Bias Ratio',
        ),
        'document_f1': MetricEntry(baseline='quality.f1.ratio', metric_name='Document F1'),
        'page_f1': MetricEntry(baseline='quality.f1.ratio', metric_name='Page F1'),
        'kuiper_statistic': MetricEntry(
            baseline='quality.effort_calibration.unbounded',
            metric_name='Kuiper Statistic',
        ),
        'wasted_effort_ratio': MetricEntry(
            baseline='quality.effort_ratio.unbounded',
            metric_name='Wasted Effort Ratio',
        ),
        # $OneMillion-Bench reports this normalized rubric-weighted score as its primary metric.
        'expert_score': MetricEntry(baseline='quality.score.ratio', metric_name='Expert Score'),
        'total_model_time': MetricEntry(baseline='diagnostic.unspecified', raw_unit='s', display_precision=2),
        'total_other_time': MetricEntry(baseline='diagnostic.unspecified', raw_unit='s', display_precision=2),
        'total_tool_time': MetricEntry(baseline='diagnostic.unspecified', raw_unit='s', display_precision=2),
        'total_wall_time': MetricEntry(baseline='diagnostic.unspecified', raw_unit='s', display_precision=2),
        # Peak signal-to-noise ratio is reported in decibels and has no upper bound (the
        # implementation caps a perfect match at 100.0), so it renders as a plain number with a unit
        # rather than as a percentage.
        'psnr': MetricEntry(
            baseline='quality.model_score.unbounded',
            metric_name='PSNR',
            raw_unit='dB',
            display_unit='dB',
            display_precision=2,
        ),
        # LPIPS is a perceptual *distance*: a smaller value means the images are more alike, which is
        # the opposite of every other scorer sharing this baseline.
        'lpips': MetricEntry(
            baseline='quality.model_score.unbounded',
            metric_name='LPIPS',
            direction=MetricDirection.LOWER_IS_BETTER,
        ),
    }
)

AGGREGATION_SEMANTICS: Dict[Tuple[str, str], MetricEntry] = {
    # An `accuracy` aggregated by pass@k / pass^k measures a pass ratio, not accuracy, so the
    # display name and semantics follow the aggregation rather than the base metric. Any `k` is
    # represented by `dimensions.k` and therefore needs no catalog enumeration.
    #
    # Only genuine meaning changes belong here. `pass_rate` / `pass_at_k` already resolve to
    # `quality.pass_at_k.ratio` through METRIC_DEFINITIONS, and `vote_at_k` does not change what
    # any of these metrics measures, so neither is restated.
    ('accuracy', 'pass_at_k'): MetricEntry(baseline='quality.pass_at_k.ratio'),
    ('accuracy', 'pass_hat_k'): MetricEntry(baseline='quality.pass_at_k.ratio'),
}

BENCHMARK_METRIC_OVERRIDES: Dict[Tuple[str, str], MetricEntry] = {
    # `total_score` is a judge score in mia_bench but the raw sum of passed rubric weights in
    # job_bench, i.e. an intermediate judge value: reassign the collision to a diagnostic.
    ('job_bench', 'judge_score'): MetricEntry(baseline='diagnostic.unspecified'),
    # $OneMillion-Bench defines pass rate as the share of tasks whose expert score reaches 0.7,
    # not as pass@k over repeated generations.
    ('one_million_bench', 'pass_rate'): MetricEntry(baseline='quality.accuracy.ratio', metric_name='Pass Rate'),
}
"""``(benchmark_name, final_metric_name)`` -> entry, only for same-name / different-meaning
collisions. Each entry carries the collision reason in a comment."""


def _validate_catalog() -> None:
    """Materialize every catalog entry at import time so illegal declarations fail fast.

    Each :class:`MetricEntry` is resolved: its ``baseline`` reference must exist in
    :data:`SEMANTIC_BASELINES` and the merged declaration must pass the contract validation.

    Raises:
        ValueError: If an entry references a baseline absent from the baseline table.
        pydantic.ValidationError: If a resolved entry violates the metric semantics contract.
    """
    for name, entry in LEGACY_METRIC_MIGRATIONS.items():
        if entry.baseline is not None and entry.baseline not in SEMANTIC_BASELINES:
            raise ValueError(
                f"metric name '{name}' in {LEGACY_MIGRATION_TABLE_LOCATION} references unknown baseline "
                f"'{entry.baseline}'; declare it at {BASELINE_TABLE_LOCATION}"
            )
        entry.resolve(name)

    for name, entry in METRIC_DEFINITIONS.items():
        entry.resolve(name)

    for (name, aggregation), entry in AGGREGATION_SEMANTICS.items():
        entry.resolve(f'{name}:{aggregation}')

    for (benchmark_name, metric_name), entry in BENCHMARK_METRIC_OVERRIDES.items():
        if entry.baseline is not None and entry.baseline not in SEMANTIC_BASELINES:
            raise ValueError(
                f"override ('{benchmark_name}', '{metric_name}') in {BENCHMARK_OVERRIDE_TABLE_LOCATION} "
                f"references unknown baseline '{entry.baseline}'; declare it at {BASELINE_TABLE_LOCATION}"
            )
        entry.resolve(metric_name)


_validate_catalog()

__all__ = [
    'AGGREGATION_SEMANTICS',
    'BENCHMARK_METRIC_OVERRIDES',
    'LEGACY_METRIC_MIGRATIONS',
    'LEGACY_MIGRATION_TABLE_LOCATION',
    'METRIC_DEFINITIONS',
    'METRIC_NAME_TABLE_LOCATION',
]
