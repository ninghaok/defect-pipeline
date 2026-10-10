from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from detected_pipeline.augmentation import bounded as aug
from detected_pipeline.augmentation.seas_contract import validate_generated
from detected_pipeline.roi import write_image
from detected_pipeline.util import atomic_write_json, sha256_file


def policy():
    return yaml.safe_load((Path(__file__).parents[1] / "configs/synthetic.yaml").read_text())


@pytest.mark.parametrize("real,ok", [(20,400),(30,100),(100,400),(1000,400),(100,0),(0,400)])
def test_all_quota_bounds(real, ok):
    p = policy(); n = aug.quota(real, ok, p)
    assert n <= p['max_train_synthetic']
    assert n <= real * p['max_synthetic_per_real_ng'] + 1e-9
    if real + ok:
        assert n / (real + ok + n) <= p['max_synthetic_fraction'] + 1e-9
        assert (real+n)/(real+ok+n) - real/(real+ok) <= p['max_positive_fraction_increase'] + 1e-9


def pair(root, index, area=100):
    image = root / f"image{index}.png"; mask = root / f"mask{index}.png"
    write_image(image, np.random.default_rng(index).integers(0, 256, (64,64,3), dtype=np.uint8))
    raw = np.zeros((64,64), np.uint8); raw.flat[:area] = 255
    write_image(mask, raw)
    return {"image":str(image), "mask":str(mask), "image_sha256":sha256_file(image),
            "mask_sha256":sha256_file(mask), "selection_status":"structurally_valid_unreviewed"}


def test_area_rejects_large_defects_without_filling_quota(tmp_path):
    items = [pair(tmp_path, 1, 2000), pair(tmp_path,2,100)]
    chosen = aug.select_by_area(items, [.024, .025], 4, 2, None)
    assert len(chosen) == 1 and chosen[0]['image'] == items[1]['image']


def test_roi_area_uses_effective_region(tmp_path):
    row = pair(tmp_path,1,100)
    roi = tmp_path/'roi.png'; mask = np.zeros((64,64),np.uint8); mask[32:] = 255
    write_image(roi,mask)
    assert aug.coverage(row['image'],row['mask'],roi) == 0
    assert aug.select_by_area([row], [.05], 1, 2, roi) == []


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    p = policy(); p.update(enabled=True, allow_structurally_valid_unreviewed=True)
    backend = tmp_path/'backend'; backend.mkdir()
    script=backend/'service.py'; script.write_text('# test service\n')
    p.update(command=[sys.executable,str(script)], reuse_root=str(tmp_path/'assets'),
             backend_sha256={str(script):sha256_file(script)})
    rows=[]; observations=[]
    for i in range(40):
        r=pair(tmp_path, i)
        label='NG' if i<20 else 'OK'
        rows.append(dict(copy_path=r['image'],sha256=r['image_sha256'],mask=r['mask'],sample_id=str(i),
                         label_source='folder_ground_truth' if i<20 else 'initial_calibration_yolo_train_ok',
                         camera_id='camera' if i<20 else 'initialization'))
        if i<20:
            observations.append(dict(sample_id=str(i),review='reviewed',truth=label,label_source='folder_ground_truth'))
    workspace=tmp_path/'workspace'
    atomic_write_json(workspace/'batch_reports/qiumian_xiepai/batch_0001.json',{'batch':1,'rows':observations})
    monkeypatch.setattr(aug,'feedback_identity',lambda _: 'unchanged')
    calls=[]
    def backend_run(command, **kwargs):
        calls.append(command)
        output=Path(command[command.index('--output')+1]); output.mkdir(parents=True)
        request=Path(command[command.index('--request')+1])
        items=[pair(output,100+i) for i in range(4)]
        atomic_write_json(output/'manifest.json',{'status':'completed','request_sha256':sha256_file(request),
                         'candidate_count':4,'items':items})
    monkeypatch.setattr(aug.subprocess,'run',backend_run)
    context={'initial_bank_sha':{r['sha256'] for r in rows[20:]},'forbidden_sha':set(),'release':nullcontext}
    args=[workspace,'qiumian_xiepai',40,p,rows[:20],[],rows[20:],[],None,context]
    return args, calls


def test_request_is_real_only_restart_is_idempotent_and_pool_does_not_accumulate(experiment):
    args,calls=experiment
    items,info=aug.prepare(*args)
    assert len(items)==1  # positive-class fraction increase is the binding cap here
    assert len(calls)==1
    request=json.loads(Path(info['event']['request']).read_text())
    assert len(request['support_ng'])==20 and len(request['support_ok'])==20
    assert all(r['origin']=='real' and r['split']=='train' for r in request['support_ng']+request['support_ok'])
    again,second=aug.prepare(*args)
    assert items==again and info==second and len(calls)==1
    args[2]=60
    again,_=aug.prepare(*args)
    assert again==items and len(calls)==1


def test_failed_backend_does_not_silently_train_or_retry(experiment,monkeypatch):
    args,calls=experiment
    def fail(*a,**kw): raise RuntimeError('GPU failed')
    monkeypatch.setattr(aug.subprocess,'run',fail)
    with pytest.raises(RuntimeError,match='GPU failed'): aug.prepare(*args)
    with pytest.raises(RuntimeError,match='Previous generation failed'): aug.prepare(*args)


def test_policy_change_requires_separate_run(experiment):
    args,_=experiment; aug.prepare(*args)
    args[3]=dict(args[3],max_train_synthetic=10)
    with pytest.raises(ValueError,match='Policy changed'): aug.prepare(*args)


def test_pseudo_ok_cannot_satisfy_generation_support(experiment):
    args,calls=experiment
    args[6]=[dict(r,label_source='sampling_pseudo_ok') for r in args[6]]
    items,info=aug.prepare(*args)
    assert items==[] and info['reason']=='insufficient_real_support_or_quota' and not calls


def test_heldout_overlap_rejected(experiment):
    args,_=experiment
    args[-1]['forbidden_sha']={args[4][0]['sha256']}
    with pytest.raises(ValueError,match='overlap'): aug.prepare(*args)


def test_backend_source_pin(experiment):
    args,_=experiment
    Path(args[3]['command'][1]).write_text('# modified backend')
    with pytest.raises(ValueError,match='source changed'): aug.prepare(*args)


def test_manifest_rejects_escape_and_changed_files(tmp_path):
    output=tmp_path/'output'; output.mkdir()
    row=pair(tmp_path,1)
    request=tmp_path/'request.json'; request.write_text('{}')
    manifest=output/'manifest.json'
    atomic_write_json(manifest,{'status':'completed','request_sha256':sha256_file(request),'items':[row]})
    with pytest.raises(ValueError,match='escaped'): validate_generated(manifest,request,set(),True)


def test_automatic_event_count_is_bounded(experiment):
    args,calls=experiment
    args[3]['max_events']=1
    aug.prepare(*args)
    # Additional true data must never trigger a second event at the lifetime cap.
    extra=[]
    for i in range(200,225):
        r=pair(args[0].parent,i)
        extra.append(dict(args[4][0],sample_id=str(i),copy_path=r['image'],mask=r['mask'],sha256=r['image_sha256']))
    args[4]=args[4]+extra
    aug.prepare(*args)
    assert len(calls)==1


def test_training_hook_appends_only_train_and_disabled_fingerprint_is_unchanged(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from detected_pipeline.training import seg_lifecycle as seg
    rows=[]
    for i in range(4):
        row=pair(tmp_path,i)
        rows.append(dict(copy_path=row['image'],mask=row['mask'],sha256=row['image_sha256'],sample_id=str(i)))
    ng,cal,ok,syn=rows
    monkeypatch.setattr(seg,'confirmed_rows',lambda ws,cat,label,*a: [ng,cal] if label=='NG' else [ok])
    monkeypatch.setattr(seg,'assign_ng_splits',lambda *a: ([ng],[cal]))
    monkeypatch.setattr(seg,'run_yolo_seg_training',lambda data,root,cfg: root/'last.pt')
    monkeypatch.setitem(sys.modules,'detected_pipeline.plugins.yolo_supervised',SimpleNamespace(YoloSegDetector=lambda *a: object()))
    def calibrate(detector,items,*a):
        assert all(str(p)!=syn['copy_path'] for p,m in items)
        return dict(image_threshold=.5,mask_conf_threshold=.5,records=[])
    monkeypatch.setattr(seg,'calibrate_seg',calibrate)
    config=dict(lifecycle={'min_train_ok':1},training={'seed':42,'base_checkpoint':'base.pt'})
    baseline=seg.train_candidate(tmp_path/'a','qiumian_xiepai',2,config,[],None)
    disabled=seg.train_candidate(tmp_path/'b','qiumian_xiepai',2,dict(config,synthetic={'enabled':False}),[],None)
    assert baseline['fingerprint']==disabled['fingerprint']
    def augment(*a):
        return [(Path(syn['copy_path']),Path(syn['mask']))],dict(enabled=True,selected_count=1)
    monkeypatch.setattr(aug,'prepare',augment)
    enhanced=seg.train_candidate(tmp_path/'c','qiumian_xiepai',2,dict(config,synthetic={'enabled':True}),[],None,
                                 augmentation_context={'test':True})
    assert enhanced['fingerprint']!=baseline['fingerprint']
    assert enhanced['counts']['train_ng']==baseline['counts']['train_ng']==1
    assert enhanced['dataset_stats']['train']['images']==baseline['dataset_stats']['train']['images']+1
    manifest=json.loads(Path(enhanced['training_manifest']).read_text())
    synthetic=[r for r in manifest['records'] if r['role']=='synthetic_ng']
    assert len(synthetic)==1 and synthetic[0]['split']=='train' and synthetic[0]['yolo_labels'].startswith('0 ')
    assert not (Path(enhanced['training_manifest']).parent/'dataset').exists()
