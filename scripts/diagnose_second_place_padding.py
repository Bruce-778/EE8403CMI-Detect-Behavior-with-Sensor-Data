"""Replay the pilot's original cloud validation batches on CPU to diagnose padding."""
from pathlib import Path
import argparse
import json,sys
import numpy as np
import torch
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--name',required=True,help='Fresh compact experiment record name')
args=parser.parse_args()
if Path(args.name).name != args.name:
    raise ValueError('Use a plain experiment name.')
output=root/f'experiments/results/{args.name}.json'
if output.exists():
    raise ValueError('Preserve the previous experiment record; use a fresh name.')
sys.path.insert(0,str(root/'scripts'))
from pilot_second_place_online import load_models,verified_cache
from cmi_project.second_place import load_reference
from cmi_project.second_place_evaluation import read_aligned_logits
import pandas as pd
torch.set_num_threads(2)
ref=load_reference(root/'outputs/reference_code/second_place',root/'configs/second_place_source.json')
imp=root/'outputs/kaggle_training/imported/second_place_base_five_fold_merged_v1'
models,labels,hashes=load_models(ref,imp,0)
meta,features,offsets=verified_cache(root/'outputs/second_place/online_cache_cpu_v1')
pilot=json.loads((root/'experiments/results/second_place_online_cpu_pilot_v1.json').read_text())
sample_ids=pilot['sequence_ids'][:32]
subset=meta[meta.fold==0].reset_index(names='cache_index')
cloudmeta=pd.read_csv(imp/'outputs/second_place/base_five_fold_merged/cache/metadata.csv')
cloud_ids,cloud_logits=read_aligned_logits(imp/'outputs/second_place/base_five_fold_merged/base/all/fold_0',cloudmeta,0)
rows=[]
for start in range(0,len(subset),32):
    batch=subset.iloc[start:start+32]
    if not batch.sequence_id.isin(sample_ids).any():continue
    lengths=torch.tensor([offsets[i+1]-offsets[i] for i in batch.cache_index])
    padded=torch.zeros(len(batch),int(lengths.max()),335)
    for j,i in enumerate(batch.cache_index):padded[j,:lengths[j]]=torch.from_numpy(np.array(features[offsets[i]:offsets[i+1]],copy=True))
    with torch.inference_mode(): pred=models['all'](padded,lengths,None)['gesture_logits'].numpy()
    for j,sid in enumerate(batch.sequence_id):
        if sid not in sample_ids:continue
        idx=int(batch.iloc[j].cache_index);x=np.array(features[offsets[idx]:offsets[idx+1]],copy=True)
        fixed=torch.zeros(1,200,335);fixed[0,:len(x)]=torch.from_numpy(x)
        with torch.inference_mode():fixed_logits=models['all'](fixed,torch.tensor([len(x)]),None)['gesture_logits'][0].numpy()
        original=cloud_logits[cloud_ids.get_loc(sid)]
        rows.append(dict(sequence_id=sid,length=len(x),batch_padding=int(lengths.max()),
            cpu_batch_cloud_max_diff=float(np.abs(pred[j]-original).max()),
            cpu_fixed200_cloud_max_diff=float(np.abs(fixed_logits-original).max()),
            cpu_batch_argmax_matches=bool(pred[j].argmax()==original.argmax())))
result=dict(status='completed',scope='Padding and numerical diagnosis, no scored adaptation result',
    original_inference='upstream test.py uses a single sequence with its actual length; exported validation used batch32 and max-length padding',
    rows=rows, max_cpu_batch_cloud_diff=max(r['cpu_batch_cloud_max_diff'] for r in rows),
    batch_argmax_agreement=sum(r['cpu_batch_argmax_matches'] for r in rows)/len(rows),
    variable_length_pilot_max_diff=pilot['timing_summary']['max_pre_update_logit_drift'])
output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
