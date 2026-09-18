"""按实际新增目标像素验收轨迹，接受后立即更新覆盖。"""
import numpy as np


def accept_coverage_candidates(paths, lengths, charts, existing, minimum_length=.004):
    covered = {name: chart.covered(existing).ravel().copy() for name, chart in charts.items()}
    accepted = np.zeros(len(paths), bool)
    gains = np.zeros(len(paths), int)
    for i, (path, length) in enumerate(zip(paths, lengths)):
        if length < minimum_length:
            continue
        pixels = {name: chart.mask(path) for name, chart in charts.items()}
        gains[i] = sum(int(np.count_nonzero(chart.target.ravel()[pixels[name]] &
                           ~covered[name][pixels[name]])) for name, chart in charts.items())
        if gains[i] == 0:
            continue
        accepted[i] = True
        for name, ids in pixels.items():
            covered[name][ids] = True
    return accepted, gains
