import json
from typing import Any, Dict, Sequence

from evalscope.metrics.utils.functions import levenshtein_distance


def derive_hop_type(evidence: Sequence[Dict[str, Any]]) -> str:
    """Classify a question by the number of evidence documents and pages."""
    documents = {str(item['document']) for item in evidence if item.get('document')}
    pages = {
        (str(item['document']), item.get('page'))
        for item in evidence
        if item.get('document') and item.get('page') is not None
    }
    if len(documents) > 1:
        return 'cross_doc'
    if len(pages) > 1:
        return 'cross_page'
    return 'single'


def parse_prediction(prediction: str) -> Dict[str, Any]:
    """Parse MADQA's JSON answer contract, preserving a plain-text fallback."""
    payload = _first_json_object(prediction)
    if payload is None:
        return {'answer': [prediction.strip()] if prediction.strip() else [], 'citations': [], 'iterations': 0}

    answer = payload.get('answer', [])
    if isinstance(answer, str):
        answer = [answer]
    if not isinstance(answer, list):
        answer = []

    citations = []
    for citation in payload.get('citations', []):
        if not isinstance(citation, dict):
            continue
        document = citation.get('document') or citation.get('file')
        page = citation.get('page')
        if not isinstance(document, str) or not document.strip() or isinstance(page, bool):
            continue
        try:
            page = int(page)
        except (TypeError, ValueError):
            continue
        citations.append({'document': document.strip(), 'page': page})

    iterations = payload.get('iterations', 0)
    if isinstance(iterations, bool):
        iterations = 0
    try:
        iterations = max(0, int(iterations))
    except (TypeError, ValueError):
        iterations = 0
    if not iterations and isinstance(payload.get('search_history'), list):
        iterations = len(payload['search_history'])

    return {
        'answer': [str(item).strip() for item in answer if str(item).strip()],
        'citations': citations,
        'iterations': iterations,
    }


def anls_star(prediction: Sequence[str], answer_variants: Sequence[Sequence[str]]) -> float:
    """Calculate MADQA's maximum ANLS* score across answer variants."""
    if not answer_variants:
        return 0.0
    return max(_list_similarity(prediction, variant) for variant in answer_variants)


def citation_f1(
    predicted_citations: Sequence[Dict[str, Any]], gold_evidence: Sequence[Dict[str, Any]], level: str
) -> float:
    """Calculate MADQA citation F1 at document or page level."""
    if level == 'document':
        gold = {item.get('document') for item in gold_evidence if item.get('document')}
        predicted = {item.get('document') for item in predicted_citations if item.get('document')}
    elif level == 'page':
        gold = {
            (item.get('document'), item.get('page'))
            for item in gold_evidence
            if item.get('document') and item.get('page') is not None
        }
        predicted = {
            (item.get('document'), item.get('page'))
            for item in predicted_citations
            if item.get('document') and item.get('page') is not None
        }
    else:
        raise ValueError(f'Unsupported MADQA citation level: {level}')

    if not gold:
        return 0.0
    true_positive = len(gold & predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _first_json_object(text: str) -> Dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and ('answer' in value or 'citations' in value):
            return value
    return None


def _list_similarity(prediction: Sequence[str], reference: Sequence[str]) -> float:
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0

    similarities = [[_string_similarity(predicted, expected) for expected in reference] for predicted in prediction]
    return _maximum_assignment_sum(similarities) / max(len(prediction), len(reference))


def _string_similarity(prediction: str, reference: str) -> float:
    normalized_prediction = ' '.join(prediction.strip().lower().split())
    normalized_reference = ' '.join(reference.strip().lower().split())
    length = max(len(normalized_prediction), len(normalized_reference))
    if not length:
        return 1.0
    similarity = 1 - levenshtein_distance(normalized_prediction, normalized_reference) / length
    return similarity if similarity >= 0.5 else 0.0


def _maximum_assignment_sum(similarities: Sequence[Sequence[float]]) -> float:
    """Return the maximum-weight one-to-one assignment sum for a rectangular matrix."""
    if not similarities or not similarities[0]:
        return 0.0

    matrix = [list(row) for row in similarities]
    if len(matrix) > len(matrix[0]):
        matrix = [list(row) for row in zip(*matrix)]

    row_count = len(matrix)
    column_count = len(matrix[0])
    potentials_row = [0.0] * (row_count + 1)
    potentials_column = [0.0] * (column_count + 1)
    matching = [0] * (column_count + 1)
    previous = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        matching[0] = row
        current_column = 0
        minimum = [float('inf')] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = matching[current_column]
            delta = float('inf')
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                cost = 1 - matrix[current_row - 1][column - 1]
                reduced_cost = cost - potentials_row[current_row] - potentials_column[column]
                if reduced_cost < minimum[column]:
                    minimum[column] = reduced_cost
                    previous[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(column_count + 1):
                if used[column]:
                    potentials_row[matching[column]] += delta
                    potentials_column[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if matching[current_column] == 0:
                break
        while True:
            previous_column = previous[current_column]
            matching[current_column] = matching[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    return sum(matrix[row - 1][column - 1] for column, row in enumerate(matching) if column and row)
