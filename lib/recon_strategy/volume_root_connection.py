"""Explicit short root connectors through admissible evidence, separate from growth."""
import numpy as np
import torch
from scipy.spatial import cKDTree

from lib.recon_strategy.volume_segment_guard import first_invalid_segment_fraction


def plan_root_connections(roots, labels, solid, evidence, conflict, low, high,
                          max_distance=None, candidates=32):
    """Keep root identities; never change the PDE domain to accommodate roots.

    Exterior connectors may traverse only nonconflicting unknown/hair evidence.
    Their maximum length defaults to one physical voxel diagonal. Roots near two
    equally plausible labels are rejected instead of assigned by integer order.
    """
    roots = np.asarray(roots)
    low, high = np.asarray(low,float), np.asarray(high,float)
    if roots.ndim != 2 or roots.shape[1] != 3 or not np.isfinite(roots).all():
        raise ValueError('roots must be finite N-by-3')
    if labels.ndim != 3 or any(x.shape != labels.shape for x in (solid,evidence,conflict)):
        raise ValueError('all evidence must share XYZ label grid')
    if np.any(high <= low) or min(labels.shape)<2 or np.any(labels<0):
        raise ValueError('invalid bounds or labels')
    if not np.any(labels>0) or np.any((labels>0)&solid):
        raise ValueError('empty domain or domain intersects solid')
    spacing=(high-low)/(np.array(labels.shape)-1)
    limit=float(np.linalg.norm(spacing) if max_distance is None else max_distance)
    if not np.isfinite(limit) or limit<=0 or candidates<1:
        raise ValueError('positive finite connection limit and candidate count required')
    starts=roots.copy()
    assigned=np.zeros(len(roots),np.int64)
    status=np.full(len(roots),'outside_box',dtype='<U32')
    distance=np.full(len(roots),np.inf)
    inbox=((roots>=low)&(roots<=high)).all(1)
    idx=np.clip(np.rint((roots-low)/spacing).astype(int),0,np.array(labels.shape)-1)
    sampled=labels[tuple(idx.T)]
    direct=np.flatnonzero(inbox&(sampled>0))
    if len(direct):
        hits=first_invalid_segment_fraction(torch.as_tensor(roots[direct]),torch.as_tensor(roots[direct]),
            torch.as_tensor(sampled[direct]),torch.as_tensor(labels),low,high).numpy()
        valid=direct[np.isinf(hits)]
        assigned[valid]=sampled[valid];status[valid]='inside';distance[valid]=0
        status[direct[np.isfinite(hits)]]='root_boundary_contact'
    targets=np.argwhere(labels>0)
    world=low+targets*spacing
    tree=cKDTree(world)
    pending=np.flatnonzero(inbox&(sampled==0))
    if len(pending):
        distances,neighbors=tree.query(roots[pending],k=min(candidates,len(targets)))
        distances=np.asarray(distances).reshape(len(pending),-1)
        neighbors=np.asarray(neighbors).reshape(len(pending),-1)
        distance[pending]=distances[:,0]
        status[pending]='too_far'
        safe_exterior=(labels==0)&~solid&~conflict&np.isin(evidence,[0,1])
        for row, root_id in enumerate(pending):
            if distances[row,0]>limit:
                continue
            local=neighbors[row,distances[row]<=limit]
            local_distance=distances[row,distances[row]<=limit]
            local_labels=labels[tuple(targets[local].T)]
            plausible=local_labels[local_distance<=local_distance.min()+spacing.min()*.5]
            if len(np.unique(plausible))>1:
                status[root_id]='ambiguous_partition';continue
            label=int(local_labels[0])
            local=local[local_labels==label]
            allowed=(safe_exterior|(labels==label)).astype(np.int64)
            ends=world[local].astype(roots.dtype)
            begin=np.repeat(roots[root_id:root_id+1],len(ends),axis=0)
            hits=first_invalid_segment_fraction(torch.as_tensor(begin),torch.as_tensor(ends),
                torch.ones(len(ends),dtype=torch.long),torch.as_tensor(allowed),low,high).numpy()
            valid=np.flatnonzero(np.isinf(hits))
            status[root_id]='blocked_connector'
            if len(valid):
                starts[root_id]=ends[valid[0]];assigned[root_id]=label
                distance[root_id]=np.linalg.norm(starts[root_id]-roots[root_id])
                status[root_id]='short_connection'
    return {'roots_world':roots,'starts_world':starts,'root_labels':assigned,
            'status':status,'connection_distance_m':distance,'max_distance_m':limit}


def audit_root_connections(plan, labels, solid, evidence, conflict, low, high):
    """Recheck only explicitly marked connector segments, including serialized data."""
    selected=np.flatnonzero(plan['status']=='short_connection')
    failed=[]
    safe=(labels==0)&~solid&~conflict&np.isin(evidence,[0,1])
    for label in np.unique(plan['root_labels'][selected]):
        ids=selected[plan['root_labels'][selected]==label]
        allowed=(safe|(labels==label)).astype(np.int64)
        origins,ends=plan['roots_world'][ids],plan['starts_world'][ids]
        hits=first_invalid_segment_fraction(torch.as_tensor(origins),torch.as_tensor(ends),
            torch.ones(len(ids),dtype=torch.long),torch.as_tensor(allowed),low,high).numpy()
        endpoint_hits=first_invalid_segment_fraction(torch.as_tensor(ends),torch.as_tensor(ends),
            torch.full((len(ids),),int(label)),torch.as_tensor(labels),low,high).numpy()
        bad=np.isfinite(hits)|np.isfinite(endpoint_hits)|(np.linalg.norm(ends-origins,axis=1)>plan['max_distance_m'])
        failed.extend(ids[bad].tolist())
    return {'passed':not failed,'connectors':len(selected),'invalid_root_indices':failed,
            'policy':'bounded root connector exception; no growth-domain expansion'}
