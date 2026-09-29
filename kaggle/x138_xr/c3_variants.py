"""CPU7 and inference share these configurations; all keep global tau at 0.6."""

C3_XR = dict(prune="44b6:8:0.7,6bba:11:0.7", cut_nan_dist_um=8.0,
             cut_end_prob=0.5, cut_end_mode="last")
FORK8_XR = dict(C3_XR, fork_min_branch=8, fork_nan_only="1")

C3_VARIANTS = {
    "C3_public0955": C3_XR,
    "C3_fork8": FORK8_XR,
    "C3_fork8_boundary": dict(FORK8_XR, fork_preserve_boundary="1"),
    "C3_fork6": dict(FORK8_XR, fork_min_branch=6),
    "C3_prune12": dict(C3_XR, prune="44b6:8:0.7,6bba:12:0.7"),
    "C3_prune14": dict(C3_XR, prune="44b6:8:0.7,6bba:14:0.7"),
    "C3_cut06": dict(C3_XR, cut_end_prob=0.6),
    "C3_cut06_fork8": dict(FORK8_XR, cut_end_prob=0.6),
    "C3_prune10_uniform": dict(C3_XR, prune="*:10:0.7"),
}


def xr_environment(variant):
    return {"BIOHUB_XR_" + key.upper(): str(value)
            for key, value in C3_VARIANTS[variant].items()}
