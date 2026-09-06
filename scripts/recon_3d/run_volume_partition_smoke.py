"""有可见网格种子的组件诊断；明确隔离无种子组件，不作为全域验收。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.recon_strategy.volume_partition import propagate_partitions
from lib.recon_strategy.weighted_poisson import solve_weighted_screened_poisson
from lib.recon_strategy.volume_segment_guard import audit_volume_strands
from scripts.recon_3d.run_pde_multiview import hair_synthesis_rk4


class DiagnosticField:
    """Occupancy is enforced by the strict volume guard at every step."""
    def __init__(self, field):
        self._orien_vol = field.astype(np.float32)

    def query_occ(self, points, calib):
        return torch.ones((1, 1, points.shape[2]), device=points.device)


def run_diagnostic(args):
    base, source, out = args.data_dir, args.seed_dir, args.output_dir
    expected = (base/'pde_governance/volume_partition_integration').resolve()
    if expected not in out.resolve().parents:
        raise ValueError('output-dir must be in image volume_partition_integration/<run_id>')
    out.mkdir(parents=True, exist_ok=True)
    source_report = json.loads((source/'report.json').read_text())
    if source_report['config']['seed_geometry'] != 'mesh_raycast':
        raise ValueError('This diagnostic requires explicit mesh_raycast seeds')
    path = source/'candidate_domain_seeds.npz'
    data = np.load(path)
    candidate, seeds = data['domain_mask'], data['seeds']
    origin, upper = data['b_min'], data['b_max']
    spacing = (upper-origin)/(np.array(candidate.shape)-1)
    components, count = ndimage.label(candidate)
    supported = np.unique(components[seeds > 0])
    domain = candidate & np.isin(components, supported)
    quarantine = candidate & ~domain
    labels, confidence = propagate_partitions(domain, seeds, spacing, data['evidence_confidence'])
    values = np.moveaxis(data['seed_directions'], -1, 0)
    report = {'status': 'diagnostic_only', 'full_domain_acceptance': False,
              'scope': 'seeded connected components only; original mesh/map tangent signs retained',
              'candidate_voxels': int(candidate.sum()), 'solved_voxels': int(domain.sum()),
              'candidate_components': int(count), 'quarantined_voxels': int(quarantine.sum()),
              'quarantined_components': int(len(np.unique(components[quarantine]))),
              'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'config': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'partition_counts': {str(k):int(v) for k,v in zip(*np.unique(labels[domain],return_counts=True))}}
    np.savez_compressed(out/'diagnostic_domain.npz', candidate_domain=candidate,
                        solved_domain=domain, quarantine=quarantine, partition_labels=labels,
                        partition_confidence=confidence, b_min=origin, b_max=upper)
    field, metrics = solve_weighted_screened_poisson(domain, values, seeds > 0,
        partition_labels=labels, spacing=spacing, require_component_convergence=True,
        tolerance=1e-8, max_iterations=4000, device='cpu', dtype=torch.float64)
    report['pde'] = metrics.to_dict()
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    np.savez_compressed(out/'direction_field.npz', field=field)
    if not metrics.converged:
        raise ValueError('PDE component convergence gate failed; see diagnostic report')
    root_path = base/'pde_governance/strand_depth_partition/step_03_rk4_partition_guard_smoke_64/root_view_guidance.npz'
    roots = np.load(root_path)['roots_world']
    indices = np.rint((roots-origin)/spacing).astype(int)
    in_box = ((roots >= origin)&(roots <= upper)).all(1)
    clipped = np.clip(indices, 0, np.array(domain.shape)-1)
    root_labels = labels[tuple(clipped.T)]
    root_labels[~in_box] = 0
    selected = np.flatnonzero(root_labels > 0)
    report['root_source_sha256'] = hashlib.sha256(root_path.read_bytes()).hexdigest()
    report['original_roots'] = len(roots)
    report['participating_roots'] = len(selected)
    report['excluded_roots'] = int((root_labels == 0).sum())
    np.savez_compressed(out/'root_selection.npz', roots_world=roots,
                        root_labels=root_labels, selected_indices=selected)
    root_tensor = torch.from_numpy(roots[selected].astype(np.float32).T[None])
    strands, diagnostics = hair_synthesis_rk4(DiagnosticField(field), torch.device('cpu'),
        root_tensor, torch.eye(4)[None], num_sample=args.samples,
        hair_unit=float(spacing.min()*.5), b_min_t=torch.tensor(origin,dtype=torch.float32)[:,None],
        b_max_t=torch.tensor(upper,dtype=torch.float32)[:,None],
        partition_label_vol=torch.from_numpy(labels.astype(np.int64))[None,None],
        root_partition_labels=torch.from_numpy(root_labels[selected].astype(np.int64)),
        volume_segment_guard=True, return_diagnostics=True, volume_bounds=(origin, upper))
    np.savez_compressed(out/'diagnostic_strands.npz', strands=strands,
                        root_indices=selected, root_labels=root_labels[selected])
    audit = audit_volume_strands(strands, root_labels[selected], labels, origin, upper)
    lengths = np.linalg.norm(np.diff(strands,axis=1),axis=2).sum(1)
    report['integration'] = diagnostics
    report['segment_audit'] = audit
    report['length_m'] = {'p10':float(np.quantile(lengths,.1)),
                          'median':float(np.median(lengths)), 'p90':float(np.quantile(lengths,.9)),
                          'below_1cm':int((lengths<.01).sum())}
    report['per_partition'] = {str(k): {'roots':int((root_labels[selected]==k).sum()),
        'median_length_m':float(np.median(lengths[root_labels[selected]==k]))}
        for k in np.unique(root_labels[selected])}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig, axes = plt.subplots(1,2,figsize=(12,7))
    for ax, dimensions, title in zip(axes, [(0,1),(2,1)], ['Front / XY','Side / ZY']):
        display = strands[::20][:,:,dimensions]
        collection = LineCollection(display, cmap='tab10', linewidths=.45, alpha=.65)
        collection.set_array(root_labels[selected][::20])
        ax.add_collection(collection)
        ax.autoscale(); ax.set_aspect('equal'); ax.set_title(title)
        ax.set_xlabel('world metres')
    fig.suptitle('Diagnostic: seeded components only; no root attachment / postprocessing')
    fig.tight_layout(); fig.savefig(out/'diagnostic_strands.png',dpi=160); plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not audit['passed']:
        raise ValueError('Final segment audit failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--seed-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=100)
    torch.set_num_threads(1)
    run_diagnostic(parser.parse_args())
