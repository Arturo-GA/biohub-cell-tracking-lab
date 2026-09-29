"""Frozen, label-free E070 hypotheses shared by lab and inference builders."""
from copy import deepcopy

C3 = dict(prune='44b6:8:0.7,6bba:11:0.7', cut_nan_dist_um=8., cut_end_prob=.5, cut_end_mode='last')
F8 = dict(C3, fork_min_branch=8, fork_nan_only='1')

VARIANTS = {
    'C3_public0955': dict(xr=C3),
    'C3_fork8': dict(xr=F8),
    'F8_prune12': dict(xr=dict(F8, prune='44b6:8:0.7,6bba:12:0.7')),
    'F8_joint': dict(xr=F8, swap=.35, fork_smooth=0.),
    'F8_prune12_joint': dict(xr=dict(F8, prune='44b6:8:0.7,6bba:12:0.7'), swap=.35, fork_smooth=0.),
    'F8_p8': dict(xr=dict(F8, prune='*:8:0.7')),
    'F8_dc030': dict(xr=F8, dc_thr=.30),
    'F8_end04': dict(xr=dict(F8, cut_end_prob=.4)),
    'F8_curvature': dict(xr=F8, position_mode='curvature'),
    'F8_outlier': dict(xr=F8, position_mode='outlier'),
}

LABS = {
    'a': ['C3_public0955', 'C3_fork8', 'F8_prune12', 'F8_joint', 'F8_prune12_joint', 'F8_p8'],
    'b': ['C3_public0955', 'C3_fork8', 'F8_dc030', 'F8_end04', 'F8_curvature', 'F8_outlier'],
}


def variant(name):
    return deepcopy(VARIANTS[name])
