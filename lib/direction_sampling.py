"""Deterministic uniform/salient and axial-direction sampling."""
from __future__ import annotations
import numpy as np


def sample_direction_observations(points, directions, confidence, partition_ids,
                                  salient_score=None, budget=4096,
                                  salient_fraction=.5, direction_bins=32,
                                  seed=42):
    """Select a fixed-budget observation set without changing total mass.

    Direction bins use ``abs(dot)`` so strand-map sign flips do not alter bin
    membership. Salient samples are selected per partition and score; the
    remaining budget is deterministic spatial/row-stratified random sampling.
    """
    points=np.asarray(points,float); directions=np.asarray(directions,float)
    confidence=np.asarray(confidence,float).reshape(-1)
    labels=np.asarray(partition_ids,int).reshape(-1)
    n=len(points)
    if points.shape!=(n,3) or directions.shape!=(n,3) or len(confidence)!=n or len(labels)!=n:
        raise ValueError('observation arrays must have matching N-by-3/N shapes')
    if n==0 or budget<1 or not 0<salient_fraction<1 or direction_bins<1:
        raise ValueError('invalid sampling parameters')
    norm=np.linalg.norm(directions,axis=1)
    valid=np.isfinite(points).all(1)&np.isfinite(directions).all(1)&np.isfinite(confidence)
    valid &= (norm>1e-8)&(confidence>=0)&(confidence<=1)&(labels>0)
    candidates=np.flatnonzero(valid)
    if not len(candidates): raise ValueError('no valid observations')
    budget=min(int(budget),len(candidates))
    unit=directions/np.maximum(norm[:,None],1e-12)
    # Fibonacci-like deterministic anchors on an axial sphere.
    z=1-2*(np.arange(direction_bins)+.5)/direction_bins
    theta=np.pi*(1+5**.5)*np.arange(direction_bins)
    anchors=np.c_[np.sqrt(1-z*z)*np.cos(theta),np.sqrt(1-z*z)*np.sin(theta),z]
    bins=np.argmax(np.abs(unit[candidates]@anchors.T),axis=1)
    scores=np.ones(n) if salient_score is None else np.asarray(salient_score,float).reshape(-1)
    if len(scores)!=n or not np.isfinite(scores[valid]).all(): raise ValueError('invalid salient score')
    chosen=[]; salient_target=min(int(round(budget*salient_fraction)),budget)
    for label in sorted(np.unique(labels[candidates]).tolist()):
        local=candidates[labels[candidates]==label]
        order=local[np.argsort(-scores[local],kind='stable')]
        take=max(1,int(round(salient_target*len(local)/len(candidates))))
        chosen.extend(order[:min(take,len(order))].tolist())
    chosen=np.array(sorted(set(chosen)),int)
    salient_count=min(len(chosen), salient_target)
    if len(chosen)>salient_target: chosen=chosen[np.argsort(-scores[chosen],kind='stable')[:salient_target]]
    remaining=np.setdiff1d(candidates,chosen,assume_unique=False)
    rng=np.random.default_rng(seed)
    if len(chosen)<budget and len(remaining):
        # Prefer underrepresented axial bins, then sample deterministically.
        need=budget-len(chosen)
        chosen_bins=bins[np.isin(candidates,chosen)]
        counts=np.bincount(chosen_bins,minlength=direction_bins)
        candidate_bins={int(index): int(bin_id) for index, bin_id in zip(candidates, bins)}
        remaining_counts=np.asarray([counts[candidate_bins[int(index)]] for index in remaining])
        order=remaining[np.argsort(remaining_counts,kind='stable')]
        if len(order)>need:
            order=rng.permutation(order)[:need]
        chosen=np.r_[chosen,order]
    chosen=np.unique(chosen)[:budget]
    weights=np.zeros(n,float);weights[chosen]=confidence[chosen]/max(1,len(chosen))
    return {'indices':chosen,'weights':weights,'direction_bins':bins,
            'valid_count':int(len(candidates)),'salient_count':int(salient_count),
            'budget':int(budget)}
