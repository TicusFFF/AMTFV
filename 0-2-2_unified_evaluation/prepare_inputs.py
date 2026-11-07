"""Join native AMTFV final responses to their dataset references before scoring."""
import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare_inputs(groups):
    cases = []
    seen = set()
    for dataset_key, dataset_path, predictions_path in groups:
        dataset = read_json(dataset_path)
        if isinstance(dataset, dict):
            dataset = dataset.get('data') or dataset.get('logs')
        if not isinstance(dataset, list) or not dataset:
            raise ValueError(f'{dataset_path}: expected a nonempty list of dataset records.')

        generated = read_json(predictions_path)
        if not isinstance(generated, dict) or not isinstance(generated.get('logs'), list) or not generated['logs']:
            raise ValueError(f'{predictions_path}: expected a nonempty native logs list.')
        declared_key = generated.get('args', {}).get('dataset_key')
        if declared_key is not None and declared_key != dataset_key:
            raise ValueError(f'{predictions_path}: dataset key does not match {dataset_key!r}.')

        # Native AMTFV assigns IDs from the original dataset order before selecting a subset.
        references = {f'{dataset_key}_{index}': row for index, row in enumerate(dataset)}
        selected = {}
        for row in generated['logs']:
            if not isinstance(row, dict):
                raise ValueError(f'{predictions_path}: each log must be an object.')
            case_id = row.get('id')
            if not isinstance(case_id, str) or case_id not in references or case_id in seen:
                raise ValueError(f'{predictions_path}: unknown or duplicate case ID {case_id!r}.')
            reference = references[case_id]
            if not isinstance(reference, dict):
                raise ValueError(f'{case_id}: invalid dataset record.')
            query = next((reference.get(k) for k in ('query', 'problem', 'q', 'question', 'prompt')
                          if reference.get(k)), None)
            if not isinstance(query, str) or not query.strip() or query != row.get('query'):
                raise ValueError(f'{case_id}: generated question does not match the dataset.')
            target = next((reference[k] for k in ('target', 'answer', 'A', 'A_original')
                           if reference.get(k) is not None), None)
            if type(target) not in (str, int, float) or not str(target).strip():
                raise ValueError(f'{case_id}: missing or unsupported reference answer.')
            if not isinstance(row.get('final_response'), str):
                raise ValueError(f'{case_id}: a complete final_response field is required.')
            seen.add(case_id)
            selected[case_id] = {'id': case_id, 'query': query, 'response': row['final_response'],
                                 'target': str(target)}
        cases.extend(selected[case_id] for case_id in references if case_id in selected)
    if not cases:
        raise ValueError('At least one generated response is required.')
    return {'expected_ids': [case['id'] for case in cases], 'cases': cases}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', nargs=3, action='append', required=True,
                        metavar=('KEY', 'DATASET_JSON', 'NATIVE_OUTPUT_JSON'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    document = prepare_inputs(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        json.dump(document, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(f'Prepared {len(document["cases"])} final responses for evaluation.')


if __name__ == '__main__':
    main()
