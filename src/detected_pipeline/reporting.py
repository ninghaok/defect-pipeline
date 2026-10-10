"""Presentation records with explicit batch timing and one evaluation per model."""


def model_evaluations(test_reports, models):
    """Initial pretrained operating point, then each trained YOLO version once."""
    initial = next((r for r in sorted(test_reports, key=lambda r: r['at'])
                    if r['role'] == 'production' and r['model_version'].startswith('pretrained')), None)
    rows = []
    if initial is not None:
        rows.append({'model_version': initial['model_version'], 'kind': 'initial_pretrained',
                     'metrics': initial, 'milestone': 0})
    for model in sorted(models, key=lambda m: m['milestone']):
        rows.append({'model_version': model['model_version'], 'kind': 'yolo',
                     'metrics': model['fixed_test'], 'milestone': model['milestone']})
    return rows


def batch_model_usage(batch, used_model, next_model):
    """A decision made after a batch never changes that batch's predictions."""
    return {'used_model': used_model, 'next_model': next_model,
            'switch_effective_from_batch': batch + 1 if used_model != next_model else None}
