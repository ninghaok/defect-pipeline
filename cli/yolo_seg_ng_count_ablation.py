"""Fixed-split YOLO instance-segmentation NG-count experiment.

Detect first, segment second: the detection head's instance confidence decides
OK/NG (image score = max instance confidence, 0 when nothing is detected); the
mask head only shapes masks for images already classified NG.  Test data never
selects thresholds.  Splits, output layout and evaluation mirror
``yolo_ng_count_ablation.py`` so the two experiments are directly comparable.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from detected_pipeline.training.runner import run_yolo_seg_training
from detected_pipeline.calibration import auroc as score_auroc, classification_metrics

# training OK count per category; the whole val split is the calibration set and doubles as the
# training-time validation set (fixed epochs, no model selection -> no leakage). Matches the DLL engine.
SPECS = {'qiumianfupai': 400, 'qiumianxiepai': 289, 'qiusaidimian': 400, 'qiusaiwaiyuan': 400}
# ROI (255 = inspect) applied to images (outside filled white) and masks, in training and inference alike.
ROI_FILES = {'qiusaiwaiyuan': PROJECT/'roi'/'qiusaiwaiyuan.png'}
_ROI_CACHE = {}
def roi_for(category, shape):
    path = ROI_FILES.get(category)
    if path is None: return None
    key = (category, shape)
    if key not in _ROI_CACHE:
        raw = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
        if raw is None: raise FileNotFoundError(path)
        if raw.shape != shape: raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        _ROI_CACHE[key] = raw >= 128
    return _ROI_CACHE[key]
def prep(category, image):
    roi = roi_for(category, image.shape[:2])
    if roi is None: return image
    out = image.copy(); out[~roi] = 255; return out
def roi_filter(category, confs, boxes, masks):
    roi = roi_for(category, masks.shape[1:]) if len(masks) else None
    if roi is None or not len(confs): return confs, boxes, masks
    keep = np.array([(m & roi).any() for m in masks])
    return confs[keep], boxes[keep], masks[keep] & roi
EXT = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
CONF_GRID = np.unique(np.r_[0.001,0.002,0.005,np.linspace(.01,.99,99),.995,.999])

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)

def images(path):
    return sorted(str(p) for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXT)

def read(path):
    value = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if value is None: raise ValueError(f'Unreadable image: {path}')
    return value

def mask_path(root, image):
    stem = Path(image).stem
    candidates = [p for p in (root/'mask').iterdir() if p.stem in (stem, stem+'_t', stem+'_mask') and p.suffix.lower() in EXT]
    if len(candidates) != 1: raise ValueError(f'Mask match count={len(candidates)}: {image}')
    return candidates[0]

def gt(root, image):
    shape = read(image).shape[:2]
    if Path(image).parent.name == 'OK': return np.zeros(shape, bool)
    p = mask_path(root, image)
    value = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
    if value is None or value.shape != shape: raise ValueError(f'Mask/image mismatch: {p}')
    defect = value < 128  # Dataset: black defect, white background. Never infer polarity.
    if not defect.any(): raise ValueError(f'NG mask has no defect pixels: {p}')
    roi = roi_for(root.name, shape)
    return defect & roi if roi is not None else defect

def polygons(mask):
    """One normalized polygon per connected component; sub-3-point specks become their bounding box."""
    h, w = mask.shape; lines = []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        pts = contour.reshape(-1, 2).astype(np.float64)
        if len(pts) < 3:
            x, y, bw, bh = cv2.boundingRect(contour)
            pts = np.array([[x,y],[x+bw,y],[x+bw,y+bh],[x,y+bh]], np.float64)
        pts[:,0] = np.clip(pts[:,0]/(w-1), 0, 1); pts[:,1] = np.clip(pts[:,1]/(h-1), 0, 1)
        lines.append('0 ' + ' '.join(f'{v:.6f}' for v in pts.reshape(-1)))
    return lines

def dataset(root, dest, training, validation):
    stats = {'train': {'images':0,'ng':0,'polygons':0}, 'val': {'images':0,'ng':0,'polygons':0}}
    for split, items in [('train',training), ('val',validation)]:
        for i, image in enumerate(items):
            out = dest/'images'/split/f'{i:05d}{Path(image).suffix.lower()}'
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                if root.name in ROI_FILES:   # ROI category: materialise the white-filled image
                    cv2.imencode(out.suffix, prep(root.name, read(image)))[1].tofile(str(out))
                else:
                    try: out.hardlink_to(image)
                    except OSError: shutil.copy2(image,out)
            lines = polygons(gt(root,image)); lp = dest/'labels'/split/f'{i:05d}.txt'
            lp.parent.mkdir(parents=True, exist_ok=True); lp.write_text('\n'.join(lines)+('\n' if lines else ''), encoding='utf-8')
            if Path(image).parent.name=='NG' and not lines: raise ValueError(f'NG image produced no polygon: {image}')
            s=stats[split]; s['images']+=1; s['ng']+=bool(lines); s['polygons']+=len(lines)
    spec = {'path':str(dest.resolve()), 'train':'images/train','val':'images/val', 'names':{0:'defect'}}
    p = dest/'data.yaml'; p.write_text(yaml.safe_dump(spec),encoding='utf-8')
    write_json(dest/'label_stats.json', stats); return p

def metrics(labels, predictions):
    y=np.asarray(labels,bool); p=np.asarray(predictions,bool)
    tp=int((y&p).sum()); fn=int((y&~p).sum()); fp=int((~y&p).sum()); tn=int((~y&~p).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/max(1,tp+fn),
                ok_false_positive_rate=fp/max(1,fp+tn),accuracy=(tp+tn)/max(1,len(y)),
                precision=tp/max(1,tp+fp))

def sync(device):
    import torch
    if torch.cuda.is_available() and str(device) != 'cpu':
        torch.cuda.synchronize(int(device) if str(device).isdigit() else device)

def infer(model, source, a):
    """Return (score, confs[N], boxes[N,4] xyxy, masks[N,H,W] bool, ms). Score = max instance conf, 0 if none."""
    sync(a.device); start=time.perf_counter()
    r = model.predict(source=source, imgsz=a.imgsz, conf=a.conf_floor, iou=a.nms_iou, max_det=a.max_det,
                      retina_masks=True, device=a.device, verbose=False)[0]
    n = 0 if r.boxes is None else len(r.boxes)
    if n and r.masks is not None:
        confs = r.boxes.conf.detach().cpu().numpy().astype(np.float64)
        boxes = r.boxes.xyxy.detach().cpu().numpy()
        masks = r.masks.data.detach().cpu().numpy() > 0.5
        if masks.shape[1:] != source.shape[:2]:
            masks = np.stack([cv2.resize(m.astype(np.uint8), (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST) > 0 for m in masks])
    else:
        confs = np.zeros(0); boxes = np.zeros((0,4)); masks = np.zeros((0,)+source.shape[:2], bool)
    sync(a.device)
    return (float(confs.max()) if len(confs) else 0.0), confs, boxes, masks, (time.perf_counter()-start)*1000

def union(masks, keep, shape):
    return masks[keep].any(axis=0) if keep.any() else np.zeros(shape, bool)

def calibrate(model, root, items, out, a):
    records=[]; sums=np.zeros(len(CONF_GRID)); ng=0
    for i,image in enumerate(items):
        source=prep(root.name, read(image)); score,confs,boxes,masks,ms=infer(model,source,a); truth=Path(image).parent.name=='NG'
        confs,boxes,masks=roi_filter(root.name,confs,boxes,masks); score=float(confs.max()) if len(confs) else 0.0
        records.append(dict(image=image,label='NG' if truth else 'OK',score=score,n_instances=len(confs),inference_ms=ms))
        if truth:
            target=gt(root,image); ng+=1
            for j,t in enumerate(CONF_GRID):
                p=union(masks,confs>=t,target.shape); u=(p|target).sum(); sums[j]+=(p&target).sum()/u if u else 0
        if (i+1)%20==0: print(f'CALIBRATION {i+1}/{len(items)}',flush=True)
    scores=np.array([r['score'] for r in records]); labels=[r['label']=='NG' for r in records]
    # Score 0 means 'no detection'; it is never a valid NG threshold, so zero-score NG count as misses.
    search=[dict(threshold=float(t),**metrics(labels,scores>=t)) for t in np.unique(scores[scores>0])]
    search.append(dict(threshold=float(np.nextafter(scores.max(),np.inf)),**metrics(labels,np.zeros(len(scores),bool))))
    zero_ng=int(sum(s==0 for s,l in zip(scores,labels) if l))
    # Recall-first under an FPR cap: among thresholds with FPR <= max_fpr, prefer those meeting the recall
    # target (lowest FPR), else the highest recall (then lowest FPR, then highest threshold).  The
    # 'predict all OK' sentinel has FPR 0, so the capped set is never empty.
    capped=[r for r in search if r['ok_false_positive_rate']<=a.max_fpr]
    eligible=[r for r in capped if r['recall']>=a.target_recall]
    selected=min(eligible,key=lambda r:(r['ok_false_positive_rate'],-r['recall'],-r['threshold'])) if eligible else min(capped,key=lambda r:(-r['recall'],r['ok_false_positive_rate'],-r['threshold']))
    uncapped=[r for r in search if r['recall']>=a.target_recall]
    uncapped=min(uncapped,key=lambda r:(r['ok_false_positive_rate'],-r['recall'],-r['threshold'])) if uncapped else min(search,key=lambda r:(-r['recall'],r['ok_false_positive_rate'],-r['threshold']))
    ladder=[]
    for level in (1.0,0.95,0.9,0.85,0.8):
        rows=[r for r in search if r['recall']>=level]
        best=min(rows,key=lambda r:(r['ok_false_positive_rate'],-r['threshold'])) if rows else None
        ladder.append(dict(recall_level=level,threshold=best['threshold'] if best else None,recall=best['recall'] if best else None,ok_false_positive_rate=best['ok_false_positive_rate'] if best else None))
    auroc=score_auroc(scores, labels)
    conf_rows=[dict(mask_conf_threshold=float(t),mean_iou=float(v/max(1,ng))) for t,v in zip(CONF_GRID,sums)]
    seg=max(conf_rows,key=lambda r:(r['mean_iou'],r['mask_conf_threshold']))
    table(out/'calibration_scores.csv',records); table(out/'classification_threshold_search.csv',search); table(out/'mask_conf_threshold_search.csv',conf_rows)
    result={'classification':dict(selected,target_recall=a.target_recall,max_fpr=a.max_fpr,auroc=auroc,ng_without_detection=zero_ng,
                                  recall_target_reachable_under_cap=bool(eligible),fpr_cap_binding=uncapped['ok_false_positive_rate']>a.max_fpr,
                                  uncapped_choice=uncapped,recall_ladder=ladder),'segmentation':seg,
            'image_score':'max_instance_confidence','conf_floor':a.conf_floor,'mask_polarity':'black_defect','model_task':'segment'}
    write_json(out/'thresholds.json',result); return result

def evaluate(model,root,items,out,thresholds,a):
    rows=[]; t=thresholds['classification']['threshold']; mt=thresholds['segmentation']['mask_conf_threshold']
    excluded_outside_roi=0
    test_dir=out/'test'
    if test_dir.exists():
        if test_dir.resolve().parent != out.resolve(): raise ValueError('Unsafe test output path')
        shutil.rmtree(test_dir)
    for case in ('tp','fp','fn','tn'): (test_dir/case).mkdir(parents=True,exist_ok=True)
    if items: infer(model,prep(root.name, read(items[0])),a)  # warm up outside timing
    for i,image in enumerate(items):
        source=prep(root.name, read(image)); score,confs,boxes,masks,ms=infer(model,source,a)
        confs,boxes,masks=roi_filter(root.name,confs,boxes,masks); score=float(confs.max()) if len(confs) else 0.0
        actual=Path(image).parent.name=='NG'; predicted=score>=t
        keep=(confs>=mt) if predicted else np.zeros(len(confs),bool)
        mask=union(masks,keep,source.shape[:2])
        target=gt(root,image); u=(mask|target).sum(); inter=int((mask&target).sum())
        if actual and not target.any():
            excluded_outside_roi+=1
            continue
        iou=float(inter/u) if actual and u else (0.0 if actual else None)
        case='tp' if actual and predicted else 'fn' if actual else 'fp' if predicted else 'tn'
        folder=test_dir/case/f'{i:05d}'; folder.mkdir(parents=True,exist_ok=True)
        boxed=source.copy()
        for (x1,y1,x2,y2),c in zip(boxes[keep],confs[keep]):
            cv2.rectangle(boxed,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
            cv2.putText(boxed,f'{c:.2f}',(int(x1),max(12,int(y1)-4)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        confmap=np.zeros(source.shape[:2],np.float32)  # per-pixel max confidence over all detected instances
        for m,c in zip(masks,confs): confmap[m]=np.maximum(confmap[m],c)
        heat=cv2.applyColorMap(np.round(confmap*255).astype(np.uint8),cv2.COLORMAP_JET)
        for name,value in [('boxed.jpg',boxed),('heatmap.jpg',heat),('pred_mask.png',mask.astype(np.uint8)*255)]:
            cv2.imencode(Path(name).suffix,value)[1].tofile(folder/name)
        for src,name in [(Path(image),'original'+Path(image).suffix)]+([(mask_path(root,image),'original_mask'+mask_path(root,image).suffix)] if actual else []):
            try: (folder/name).hardlink_to(src)
            except OSError: shutil.copy2(src,folder/name)
        rows.append(dict(image=image,label='NG' if actual else 'OK',prediction='NG' if predicted else 'OK',case=case,score=score,
                         n_instances=len(confs),n_kept=int(keep.sum()),iou=iou,intersection=inter,union=int(u),inference_ms=ms))
        if (i+1)%20==0: print(f'TEST {i+1}/{len(items)}',flush=True)
    table(out/'test_scores.csv',rows)
    labels=np.array([r['label']=='NG' for r in rows]); scores=np.array([r['score'] for r in rows])
    result=classification_metrics(labels,[r['prediction']=='NG' for r in rows])
    ng_rows=[r for r in rows if r['label']=='NG']
    total_union=sum(r['union'] for r in ng_rows)
    result.update(test_auroc=score_auroc(scores, labels),
                  iou_micro=sum(r['intersection'] for r in ng_rows)/total_union if total_union else None,
                  primary_segmentation_metric='iou_micro',
                  excluded_outside_roi=excluded_outside_roi,
                  mean_iou_all_ng=float(np.mean([r['iou'] for r in ng_rows])) if ng_rows else None,
                  mean_inference_ms=float(np.mean([r['inference_ms'] for r in rows])) if rows else None,
                  p95_inference_ms=float(np.percentile([r['inference_ms'] for r in rows],95)) if rows else None,
                  image_threshold=t,mask_conf_threshold=mt,calibration_auroc=thresholds['classification']['auroc'],
                  calibration_recall=thresholds['classification']['recall'],calibration_fpr=thresholds['classification']['ok_false_positive_rate'],
                  max_fpr=thresholds['classification']['max_fpr'],fpr_cap_binding=thresholds['classification']['fpr_cap_binding'])
    return result

def summarize(run):
    rows=[]; bundle=run/'final_analysis_bundle'; bundle.mkdir(exist_ok=True)
    for p in sorted(run.glob('*/ng_*/result.json')):
        rows.append(json.loads(p.read_text(encoding='utf-8')))
        dest=bundle/p.parent.parent.name/p.parent.name; dest.mkdir(parents=True,exist_ok=True)
        for f in p.parent.glob('*.csv'): shutil.copy2(f,dest/f.name)
        for f in p.parent.glob('*.json'): shutil.copy2(f,dest/f.name)
    if rows: table(run/'summary.csv',rows); table(bundle/'summary.csv',rows)
    shutil.copy2(run/'split_manifest.json',bundle/'split_manifest.json')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--run-root',type=Path); parser.add_argument('--classes',nargs='+',default=list(SPECS))
    parser.add_argument('--counts',nargs='+',type=int,default=[40,60,70,80],help='NG counts; the full NG pool is always appended')
    parser.add_argument('--seed',type=int,default=42); parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--patience',type=int,default=0,help='0 = fixed epochs, use last.pt; >0 = early stopping, use best.pt')
    parser.add_argument('--imgsz',type=int,default=1024); parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--workers',type=int,default=0); parser.add_argument('--device',default='0')
    parser.add_argument('--base-checkpoint',default=str(PROJECT/'models'/'yolo26s-seg.pt'))
    parser.add_argument('--copy-paste',type=float,default=0.3); parser.add_argument('--mosaic',type=float,default=1.0)
    parser.add_argument('--conf-floor',type=float,default=0.001,help='predict conf lower bound; thresholds are searched above it')
    parser.add_argument('--nms-iou',type=float,default=0.7); parser.add_argument('--max-det',type=int,default=100)
    parser.add_argument('--target-recall',type=float,default=0.95)
    parser.add_argument('--max-fpr',type=float,default=0.2,help='OK false-positive-rate cap on the calibration set when choosing the image threshold')
    parser.add_argument('--skip-full',action='store_true',help='debug only: do not append the full NG pool to --counts')
    parser.add_argument('--prepare-only',action='store_true'); a=parser.parse_args()
    arguments={k:v for k,v in vars(a).items() if k not in ('prepare_only','run_root','dataset_root','counts','skip_full','max_fpr')}  # counts do not affect the split; allow adding runs later
    arguments.update(dataset_root=str(a.dataset_root),run_root=None)
    run=a.run_root or Path('C:/ninghao/results')/('yolo_seg_ng_count_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    run.mkdir(parents=True,exist_ok=True); mp=run/'split_manifest.json'
    if a.epochs<=0 or a.batch<=0 or a.patience<0 or not 0<a.target_recall<=1 or not 0<a.conf_floor<1 or not 0<a.max_fpr<=1: raise ValueError('Invalid training/calibration arguments')
    if not a.classes or len(set(a.classes))!=len(a.classes) or any(c not in SPECS for c in a.classes): raise ValueError('Invalid categories')
    if not Path(a.base_checkpoint).is_file(): raise FileNotFoundError(a.base_checkpoint)
    if mp.exists():
        manifest=json.loads(mp.read_text(encoding='utf-8'))
        if manifest['arguments']!=arguments: raise ValueError('Resume arguments differ from manifest')
    else:
        manifest={'arguments':arguments,'categories':{}}
        for category in a.classes:
            root=a.dataset_root/category; okn=SPECS[category]; rng=np.random.default_rng(a.seed)
            pools={f'{s}_{l}':list(rng.permutation(images(root/s/l))) for s in ['train','val','test'] for l in ['OK','NG']}
            if len(pools['train_OK'])<okn: raise ValueError(f'Insufficient train OK: {category}')
            if len(pools['train_NG'])<min(a.counts) or len(pools['val_OK'])<80 or len(pools['val_NG'])<30:   # counts above the pool are dropped later
                raise ValueError(f'Insufficient NG or calibration data: {category}')
            if not pools['test_OK'] or not pools['test_NG']: raise ValueError(f'Empty test class: {category}')
            # ROI categories: NG images whose defect lies entirely outside the ROI are not inspectable by this
            # station (another view covers that surface). Exclude them from every group and record the list.
            excluded=[]
            if category in ROI_FILES:
                for key in ('train_NG','val_NG','test_NG'):
                    keep=[]
                    for image in pools[key]:
                        if gt(root,image).any(): keep.append(image)
                        else: excluded.append({'group':key,'image':image,'reason':'defect entirely outside ROI'})
                    pools[key]=keep
                print(f'{category}: excluded {len(excluded)} NG images with defects outside the ROI',flush=True)
            calibration=pools['val_OK']+pools['val_NG']   # whole val split; also used as training-time validation
            split=dict(training_ok=pools['train_OK'][:okn],training_ng=pools['train_NG'],validation=calibration,calibration=calibration,test=pools['test_OK']+pools['test_NG'])
            for group in split.values():
                for image in group: gt(root,image) # preflight all masks before any training
            manifest['categories'][category]=split
            manifest.setdefault('excluded',{})[category]=excluded
        write_json(mp,manifest)
    duplicate_rows=[]
    for category in a.classes:
        split=manifest['categories'][category]; okn=SPECS[category]
        if len(split['training_ok'])!=okn: raise ValueError(f'Training OK count mismatch: {category}')
        if split['validation']!=split['calibration']: raise ValueError(f'Validation must equal the calibration split: {category}')
        seen={}
        for group,items in split.items():
            if not items: raise ValueError(f'Empty data group: {category}/{group}')
            for image in items:
                if not Path(image).is_file(): raise FileNotFoundError(image)
                digest=hashlib.sha256(Path(image).read_bytes()).hexdigest()
                partition='training' if group.startswith('training') else ('calibration' if group in ('validation','calibration') else group)
                if digest in seen and seen[digest][0]!=partition:
                    duplicate_rows.append(dict(category=category,first_group=seen[digest][0],first_image=seen[digest][1],second_group=partition,second_image=image))
                else: seen[digest]=(partition,image)
            print(f'CHECK {category}/{group}: {len(items)}',flush=True)
    write_json(run/'duplicate_check.json',{'cross_partition_duplicates':duplicate_rows})
    if duplicate_rows: raise ValueError(f'Cross-partition duplicates found: {len(duplicate_rows)}; see duplicate_check.json. Source data unchanged.')
    print(f'RUN ROOT: {run}',flush=True)
    if a.prepare_only:
        print('PREFLIGHT OK; no training started',flush=True); return
    from ultralytics import YOLO
    for category in a.classes:
        split=manifest['categories'][category]; root=a.dataset_root/category
        counts=list(dict.fromkeys([n for n in sorted(a.counts) if 0<n<=len(split['training_ng'])]+([] if a.skip_full else [len(split['training_ng'])])))
        for n in counts:
            out=run/category/f'ng_{n}'; out.mkdir(parents=True,exist_ok=True)
            if (out/'result.json').exists(): print(f'SKIP {category} NG={n}',flush=True); continue
            print(f'START {category} NG={n}',flush=True)
            data=out/'dataset'; training_ms=None
            if not (out/'checkpoint.json').exists():
                spec=dataset(root,data,split['training_ok']+split['training_ng'][:n],split['validation'])
                attempt=out/'model'
                if attempt.exists(): attempt=out/f'model_retry_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                start=time.perf_counter()
                checkpoint=run_yolo_seg_training(spec,attempt,dict(base_checkpoint=a.base_checkpoint,epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,workers=a.workers,seed=a.seed,amp=True,deterministic=True,patience=a.patience,device=a.device,copy_paste=a.copy_paste,mosaic=a.mosaic))
                training_ms=(time.perf_counter()-start)*1000
                shutil.copy2(data/'label_stats.json',out/'label_stats.json')
                write_json(out/'checkpoint.json',dict(path=str(checkpoint),training_ms=training_ms))
            info=json.loads((out/'checkpoint.json').read_text()); checkpoint=Path(info['path']); training_ms=info['training_ms']
            model=YOLO(str(checkpoint))
            thresholds=calibrate(model,root,split['calibration'],out,a)
            result=evaluate(model,root,split['test'],out,thresholds,a)
            result.update(category=category,training_ok=len(split['training_ok']),training_ng=n,validation_ok=sum(Path(p).parent.name=='OK' for p in split['validation']),validation_ng=sum(Path(p).parent.name=='NG' for p in split['validation']),calibration_ok=sum(Path(p).parent.name=='OK' for p in split['calibration']),calibration_ng=sum(Path(p).parent.name=='NG' for p in split['calibration']),training_ms=training_ms,checkpoint=str(checkpoint),model_task='segment',base_checkpoint=a.base_checkpoint)
            write_json(out/'result.json',result); summarize(run)
            del model
            if data.exists(): shutil.rmtree(data)  # only generated linked dataset files, never source data
    summarize(run); print(f'COPY FOR ANALYSIS: {run / "final_analysis_bundle"}',flush=True)

if __name__=='__main__': main()
