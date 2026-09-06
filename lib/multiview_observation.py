"""Observation provenance and confidence gates independent of model training."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np


def file_identity(path):
    path=Path(path).resolve()
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return {'path':str(path),'sha256':digest.hexdigest(),'bytes':path.stat().st_size}


def write_observation_manifest(path, manifest):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    temporary.replace(path)


def selected_map_views(views):
    """Original front has its own branch, never enters generated-view inference."""
    if len(views)!=len(set(views)) or any(not re.fullmatch(r'[a-z][a-z0-9_]*',v) for v in views):
        raise ValueError('views must have unique safe identifiers')
    return ['front']+[v for v in views if v!='front']


def verify_observation_manifest(manifest, views):
    """Reject partial runs, missing views, mixed identities and changed assets."""
    if manifest.get('version')!=1 or manifest.get('status')!='complete':
        raise ValueError('observation manifest is incomplete or unsupported')
    records=manifest.get('observations',{})
    assets=[]
    for view in views:
        if view not in records:
            raise ValueError(f'missing observation: {view}')
        record=records[view]
        if record.get('source_kind') not in ('original','generated') or not record.get('source_group'):
            raise ValueError('missing source kind/group')
        if view=='front' and record['source_kind']!='original':
            raise ValueError('front must identify the original observation')
        assets.append(record['image'])
        for kind in ('strand_map','depth_map','seg'):
            assets.append(record['products'][kind])
    for asset in assets:
        if file_identity(asset['path'])!=asset:
            raise ValueError(f"observation asset changed: {asset['path']}")
    return {view:records[view] for view in views}


def common_observation_gate(seg, strand_rgb, depth, mesh_visible, calibration_valid):
    """Return overlapping rejection masks; rejected evidence is never air."""
    seg=np.asarray(seg,bool);strand=np.asarray(strand_rgb)
    if (strand.shape!=(*seg.shape,3) or np.shape(depth)!=seg.shape
            or np.shape(mesh_visible)!=seg.shape):
        raise ValueError('observation maps must share pixel coordinates')
    if strand.dtype!=np.uint8:
        raise ValueError('strand_rgb must be uint8 with explicit RGB encoding')
    unit=strand.astype(float)/255
    norm=np.hypot(1-2*unit[...,2],2*unit[...,1]-1)
    valid_strand=(unit[...,0]>.1)&(norm>1e-6)
    valid_depth=np.isfinite(depth)&(np.asarray(depth)>.05)
    masks={'outside_seg':~seg,'mesh_not_visible':seg&~np.asarray(mesh_visible,bool),
           'invalid_strand':seg&~valid_strand,'invalid_depth':seg&~valid_depth,
           'calibration_unverified':seg&~np.full(seg.shape,bool(calibration_valid))}
    masks['accepted']=seg&np.asarray(mesh_visible,bool)&valid_strand&valid_depth&bool(calibration_valid)
    masks['unknown_or_conflict']=seg&~masks['accepted']
    return masks


def source_budget_weights(confidences, records, generated_budget=.5, original_budget=1.):
    """Normalize observation mass with an aggregate cap for all generated groups.

    Correlated groups cannot gain weight by adding duplicate views. Different
    source kinds may not share a group. Budgets describe sampling/soft evidence
    mass, never a license to overwrite original hard anchors.
    """
    if set(confidences)!=set(records) or not confidences:
        raise ValueError('confidence and provenance views differ')
    if not 0<=generated_budget<original_budget or not np.isfinite(original_budget):
        raise ValueError('generated budget must be below finite original budget')
    groups={};out={}
    for view,confidence in confidences.items():
        value=np.asarray(confidence,float)
        if not np.isfinite(value).all() or np.any((value<0)|(value>1)):
            raise ValueError('confidence must be finite in [0,1]')
        record=records[view];kind=record['source_kind'];group=record['source_group']
        if kind not in ('original','generated') or not group:
            raise ValueError('invalid provenance')
        if group in groups and groups[group]['kind']!=kind:
            raise ValueError('source group mixes real and generated evidence')
        groups.setdefault(group,{'kind':kind,'views':[]})['views'].append(view)
        out[view]=value.copy()
    for kind,budget in [('original',original_budget),('generated',generated_budget)]:
        active=[g for g in groups.values() if g['kind']==kind and sum(out[v].sum() for v in g['views'])>0]
        for group in active:
            mass=sum(out[v].sum() for v in group['views'])
            for view in group['views']:
                out[view]*=(budget/len(active))/mass
    return out


def generation_source(render_dir, view, image_path, original_identity):
    """Link a generated image to its recorded run, never invent legacy history."""
    path=Path(render_dir)/'generation_manifest.json'
    if not path.is_file():
        return {'status':'unverified_legacy','source_group':'flux:'+original_identity['sha256']}
    manifest=json.loads(path.read_text())
    if manifest.get('version')!=1 or manifest.get('status')!='complete':
        raise ValueError('FLUX generation manifest is incomplete')
    record=manifest['observations'][view]
    if record['source_kind']!='generated' or file_identity(image_path)!=record['image']:
        raise ValueError('generated image identity disagrees with its manifest')
    if manifest['original_front']['sha256']!=original_identity['sha256']:
        raise ValueError('generation and maps refer to different original images')
    return {'status':'verified','source_group':record['source_group'],
            'manifest':file_identity(path),'record':record}
