from detected_pipeline.reporting import batch_model_usage, model_evaluations


def test_batch_end_promotion_only_applies_to_next_batch():
    usage = batch_model_usage(3, 'pretrained', 'yolo-v80')
    assert usage['used_model'] == 'pretrained'
    assert usage['next_model'] == 'yolo-v80'
    assert usage['switch_effective_from_batch'] == 4
    assert batch_model_usage(2, 'pretrained', 'pretrained')['switch_effective_from_batch'] is None


def test_initial_pretrained_then_unique_yolo_versions_not_repeated_per_batch():
    reports = [dict(at='02', role='production', model_version='pretrained', fp=9),
               dict(at='01', role='production', model_version='pretrained', fp=3),
               dict(at='03', role='production', model_version='yolo-v40', fp=1)]
    models = [dict(model_version='yolo-v80', milestone=80, fixed_test={'fp': 4}),
              dict(model_version='yolo-v40', milestone=40, fixed_test={'fp': 1})]
    rows = model_evaluations(reports, models)
    assert [r['model_version'] for r in rows] == ['pretrained', 'yolo-v40', 'yolo-v80']
    assert rows[0]['metrics']['fp'] == 3
