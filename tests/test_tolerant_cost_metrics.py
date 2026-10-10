"""Tolerance boundary, miss handling, pixel aggregation and cost tradeoff regression cases."""
import copy

import numpy as np
import pytest

from detected_pipeline.metric_support import segmentation_row, aggregate_segmentation
from detected_pipeline.promotion import compare_models
from detected_pipeline.roi import write_image


def row(tmp_path, name, gt, pred, detected=True):
    mask=tmp_path/f'{name}.png'
    write_image(mask,gt.astype(np.uint8)*255)
    return dict(segmentation_row('NG',mask,pred,gt.shape,external=False),
                sample_id=name,predicted_ng=detected)


def ok(name, alarm=False):
    return dict(sample_id=name,label='OK',gt_status='not_required',predicted_ng=alarm)


@pytest.mark.parametrize('offset,expected',[(0,1),(5,1),(6,0)])
def test_original_pixel_distance_boundary(tmp_path,offset,expected):
    gt=np.zeros((30,30),bool);gt[10,10]=True
    pred=np.zeros_like(gt);pred[10+offset,10+offset]=True
    r=row(tmp_path,'tiny',gt,pred)
    assert r['tolerant_agreement']==expected
    assert r['iou']==(1 if offset==0 else 0)


def test_far_false_region_penalty_caps_at_gt_area(tmp_path):
    gt=np.zeros((40,40),bool);gt[0,0]=True
    pred=gt.copy();pred[30:32,30:32]=True
    r=row(tmp_path,'extra',gt,pred)
    assert r['tolerant_matched_pixels']==1 and r['tolerant_total_pixels']==2
    assert r['tolerant_agreement']==.5
    pred[20:35,20:35]=True
    assert row(tmp_path,'more_extra',gt,pred)['tolerant_agreement']==.5


def test_cap_is_per_image_before_micro_aggregation(tmp_path):
    tiny=np.zeros((40,40),bool);tiny[0,0]=True
    extra=tiny.copy();extra[20:30,20:30]=True
    large=np.zeros_like(tiny);large[10:20,10:20]=True
    a=row(tmp_path,'tiny_extra',tiny,extra)
    b=row(tmp_path,'large_correct',large,large)
    m=aggregate_segmentation([a,b],[True,True])
    assert m['tolerant_matched_pixels']==101 and m['tolerant_total_pixels']==102
    assert m['tolerant_agreement_micro']==pytest.approx(101/102)
    assert m['tolerant_agreement_micro'] != (a['tolerant_agreement']+b['tolerant_agreement'])/2


def test_partial_coverage_with_excess_can_score_below_half(tmp_path):
    gt=np.zeros((60,60),bool);gt[0,0]=gt[30,30]=True
    pred=np.zeros_like(gt);pred[0,0]=True;pred[50:55,50:55]=True
    r=row(tmp_path,'partial',gt,pred)
    assert r['tolerant_agreement']==.25


def test_old_counts_require_recomputation(tmp_path):
    gt=np.ones((5,5),bool)
    r=row(tmp_path,'old',gt,gt)
    r['segmentation_metric_version']='obsolete'
    with pytest.raises(ValueError,match='Recompute'):
        aggregate_segmentation([r],[True])
    assert compare_models([r,ok('normal')],[r,ok('normal')])['decision']=='reject'


def test_miss_and_micro_aggregation_not_macro(tmp_path):
    small=np.zeros((30,30),bool);small[10,10]=True
    large=np.zeros_like(small);large[10:20,10:20]=True
    a=row(tmp_path,'small',small,small)
    b=row(tmp_path,'large',large,large,False)
    m=aggregate_segmentation([a,b],[True,False])
    assert m['tolerant_agreement_micro']==pytest.approx(1/101)
    assert m['tolerant_matched_pixels']==1 and m['tolerant_total_pixels']==101
    assert m['union']==101


def test_two_fewer_misses_pay_for_three_extra_false_alarms(tmp_path):
    gt=np.zeros((30,30),bool);gt[10:15,10:15]=True
    old=[row(tmp_path,str(i),gt,gt,False) for i in range(2)]+[ok(str(i+2)) for i in range(3)]
    new=copy.deepcopy(old)
    for r in new:r['predicted_ng']=True
    result=compare_models(old,new)
    assert result['decision']=='promote'
    assert result['official']['error_cost']==4 and result['candidate']['error_cost']==3


def test_lower_cost_can_accept_worse_localization(tmp_path):
    gt=np.zeros((30,30),bool);gt[10:15,10:15]=True
    old=[row(tmp_path,'defect',gt,gt),ok('normal',True)]
    new=[row(tmp_path,'defect',gt,np.zeros_like(gt)),ok('normal')]
    result=compare_models(old,new)
    assert result['decision']=='promote' and result['reason']=='lower_error_cost'
    assert result['candidate']['tolerant_agreement_micro']==0


def test_equal_cost_uses_tolerance_not_iou(tmp_path):
    gt=np.zeros((30,30),bool);gt[10,10]=True
    far=np.zeros_like(gt);far[20,20]=True
    near=np.zeros_like(gt);near[15,15]=True
    old=[row(tmp_path,'defect',gt,far),ok('normal')]
    new=[row(tmp_path,'defect',gt,near),ok('normal')]
    result=compare_models(old,new)
    assert result['official']['iou_micro']==result['candidate']['iou_micro']==0
    assert result['decision']=='promote' and result['reason']=='equal_cost_better_tolerance'
    assert compare_models(new,new)['decision']=='reject'
    legacy=copy.deepcopy(old);legacy[0]['tolerance_pixels']=2
    assert compare_models(legacy,new)['reason']=='missing_or_legacy_tolerance_counts'
