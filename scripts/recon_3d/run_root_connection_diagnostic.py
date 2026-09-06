"""固定已有分区 PDE，验证短根连接及全根集合的接入状态。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from lib.recon_strategy.volume_root_connection import plan_root_connections, audit_root_connections
from lib.recon_strategy.volume_segment_guard import audit_volume_strands
from lib.pipeline_snapshot import save_stage_snapshot
from scripts.recon_3d.run_volume_partition_smoke import DiagnosticField
from scripts.recon_3d.run_pde_multiview import hair_synthesis_rk4


def run_root_diagnostic(args):
    source,seed,out=args.source_dir,args.seed_dir,args.output_dir
    if source.parent != out.parent or source==out or source.parent.name!='volume_partition_integration':
        raise ValueError('output must be a distinct run in the image volume_partition_integration directory')
    out.mkdir(parents=True,exist_ok=True)
    data=np.load(seed/'candidate_domain_seeds.npz')
    domain=np.load(source/'diagnostic_domain.npz')
    labels=domain['partition_labels'];low,high=domain['b_min'],domain['b_max']
    roots=np.load(source/'root_selection.npz')['roots_world']
    plan=plan_root_connections(roots,labels,data['solid_mask'],data['evidence_class'],data['conflict'],low,high)
    np.savez_compressed(out/'root_connection_plan.npz',**plan)
    save_stage_snapshot(out / 'stage_snapshots', 'correction', {
        'roots_world': plan['roots_world'], 'starts_world': plan['starts_world'],
        'root_labels': plan['root_labels'], 'status': plan['status'],
    }, {'source': 'root_connection_plan', 'max_distance_m': float(plan['max_distance_m'])})
    # Read the actual serialized coordinates back before checking connections.
    with np.load(out/'root_connection_plan.npz') as saved:
        plan={k:saved[k] for k in saved.files}
    connection_audit=audit_root_connections(plan,labels,data['solid_mask'],data['evidence_class'],data['conflict'],low,high)
    components,count=ndimage.label(domain['quarantine'])
    component_report=[]
    for k in range(1,count+1):
        mask=components==k;coordinates=np.argwhere(mask)
        component_report.append({'component':k,'voxels':int(mask.sum()),
            'center_world':(low+coordinates.mean(0)*(high-low)/(np.array(labels.shape)-1)).tolist(),
            'evidence_counts':{str(x):int(n) for x,n in zip(*np.unique(data['evidence_class'][mask],return_counts=True))}})
    report={'status':'diagnostic_only','full_domain_acceptance':False,'original_roots':len(roots),
        'root_status':{str(k):int(v) for k,v in zip(*np.unique(plan['status'],return_counts=True))},
        'connector_max_distance_m':float(plan['max_distance_m']),'connection_audit':connection_audit,
        'quarantined_components':component_report,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [source/'direction_field.npz',source/'diagnostic_domain.npz',source/'root_selection.npz',seed/'candidate_domain_seeds.npz']}}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if not connection_audit['passed']:
        raise ValueError('root connection audit failed')
    ids=np.flatnonzero(plan['root_labels']>0)
    spacing=(high-low)/(np.array(labels.shape)-1)
    field=np.load(source/'direction_field.npz')['field']
    strands,metrics=hair_synthesis_rk4(DiagnosticField(field),torch.device('cpu'),
        torch.tensor(plan['starts_world'][ids].T[None],dtype=torch.float32),torch.eye(4)[None],
        num_sample=100,hair_unit=float(spacing.min()*.5),
        b_min_t=torch.tensor(low,dtype=torch.float32)[:,None],b_max_t=torch.tensor(high,dtype=torch.float32)[:,None],
        partition_label_vol=torch.tensor(labels,dtype=torch.long)[None,None],
        root_partition_labels=torch.tensor(plan['root_labels'][ids]),volume_segment_guard=True,
        volume_bounds=(low,high),return_diagnostics=True)
    growth_audit=audit_volume_strands(strands,plan['root_labels'][ids],labels,low,high)
    connected=np.concatenate([roots[ids,None],strands],axis=1)
    np.savez_compressed(out/'connected_strands.npz',strands=connected,root_indices=ids,
                        root_labels=plan['root_labels'][ids],growth_start_index=np.asarray(1))
    save_stage_snapshot(out / 'stage_snapshots', 'integration', {
        'strands': connected, 'root_indices': ids, 'root_labels': plan['root_labels'][ids],
    }, {'source': 'hair_synthesis_rk4', 'num_sample': 100})
    lengths=np.linalg.norm(np.diff(connected,axis=1),axis=2).sum(1)
    report.update({'growth_audit':growth_audit,'integration':metrics,'participating_roots':len(ids),
        'length_m':{'p10':float(np.quantile(lengths,.1)),'median':float(np.median(lengths)),
                    'p90':float(np.quantile(lengths,.9)),'below_1cm':int((lengths<.01).sum())},
        'passed_with_explicit_connector_exception':connection_audit['passed'] and growth_audit['passed']})
    save_stage_snapshot(out / 'stage_snapshots', 'export', {
        'strands': connected, 'root_indices': ids, 'root_labels': plan['root_labels'][ids],
    }, {'source': 'serialized_connected_strands', 'audit_passed': bool(growth_audit['passed'])})
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['integration','quarantined_components']},ensure_ascii=False,indent=2))
    if not growth_audit['passed']:
        raise ValueError('growth audit failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--seed-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    torch.set_num_threads(1)
    run_root_diagnostic(parser.parse_args())
