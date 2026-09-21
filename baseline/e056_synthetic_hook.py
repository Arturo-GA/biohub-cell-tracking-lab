# Adapted from https://www.kaggle.com/code/andnyu/biohub-947-synthetic-edge
# ================= SYNTHETIC LOW-MARGIN EDGE TIEBREAK (andnyu) =================
_synthetic_roots = [
    Path("/kaggle/input/biohub-synthetic-5fold-ensemble-v1"),
    Path("/kaggle/input/datasets/bhpepper/biohub-synthetic-5fold-ensemble-v1"),
]
_synthetic_weights = [
    p for root in _synthetic_roots if root.exists()
    for p in root.rglob("synthetic_5fold_swa.pth")
]
if len(_synthetic_weights) != 1:
    raise RuntimeError(f"Expected exactly one synthetic SWA checkpoint, found {_synthetic_weights}")
SYNTHETIC_WEIGHTS_PATH = _synthetic_weights[0]
_synthetic_sha256 = _sha256_file(SYNTHETIC_WEIGHTS_PATH)
if _synthetic_sha256 != "0eacacaf0b43bfd5a063495d6991a4911cd045c37650363d5d8826a0e7ed3dd9":
    raise RuntimeError(
        f"Synthetic SWA checksum mismatch: expected 0eacacaf0b43bfd5a063495d6991a4911cd045c37650363d5d8826a0e7ed3dd9, got {_synthetic_sha256}"
    )
os.environ["BIOHUB_SYNTHETIC_WEIGHTS"] = str(SYNTHETIC_WEIGHTS_PATH)
os.environ["BIOHUB_SYNTH_EDGE_WEIGHT"] = "0.25"
os.environ["BIOHUB_SYNTH_EDGE_MARGIN_MAX"] = "0.12"
os.environ["BIOHUB_SYNTH_EDGE_MARGIN_GAIN"] = "0.02"
print("Synthetic edge-only tiebreak checkpoint verified:", SYNTHETIC_WEIGHTS_PATH, flush=True)

_synth_s = _ps.read_text()
_synth_sig_old = '''    secondary_low_margin_max: float = 0.2,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:'''
_synth_sig_new = '''    secondary_low_margin_max: float = 0.2,
    synthetic_model: UNetNodeTransformer | None = None,
    synthetic_edge_weight: float = 0.25,
    synthetic_edge_margin_max: float = 0.12,
    synthetic_edge_margin_gain: float = 0.02,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:'''
if _synth_s.count(_synth_sig_old) != 1:
    raise RuntimeError("Synthetic signature anchor is not unique")
_synth_s = _synth_s.replace(_synth_sig_old, _synth_sig_new, 1)

_synth_encode_old = '''        del imgs

        # --- Detect cells in each frame (dedup across windows) ---'''
_synth_encode_new = '''        synthetic_unet_out = None
        if synthetic_model is not None:
            synthetic_unet_out, _synthetic_det_logits = synthetic_model.encode(imgs)
            del _synthetic_det_logits

        del imgs

        # --- Detect cells in each frame (dedup across windows) ---'''
if _synth_s.count(_synth_encode_old) != 1:
    raise RuntimeError("Synthetic encode anchor is not unique")
_synth_s = _synth_s.replace(_synth_encode_old, _synth_encode_new, 1)

_synth_raw_old = '''            raw = edge_logits_pair[0]'''
_synth_raw_new = '''            if synthetic_model is not None and n_src >= 2:
                if synthetic_unet_out is None:
                    raise RuntimeError("Synthetic model feature map missing")
                synth_feat_src = synthetic_model._index_features(
                    synthetic_unet_out[:, f_idx], p_coords_src, p_mask_src,
                )
                synth_feat_tgt = synthetic_model._index_features(
                    synthetic_unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,
                )
                synth_logits_pair = synthetic_model.predict_edges(
                    synth_feat_src, synth_feat_tgt,
                    p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                    p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt,
                )
                public_center = edge_logits_pair.mean(dim=1, keepdim=True)
                public_scale = edge_logits_pair.float().std(
                    dim=1, keepdim=True, unbiased=False
                ).clamp_min(1e-4)
                synth_center = synth_logits_pair.mean(dim=1, keepdim=True)
                synth_scale = synth_logits_pair.float().std(
                    dim=1, keepdim=True, unbiased=False
                ).clamp_min(1e-4)
                synth_aligned = (
                    (synth_logits_pair - synth_center)
                    * (public_scale / synth_scale).clamp(0.5, 2.0)
                    + public_center
                )
                public_probs = torch.softmax(edge_logits_pair[0].float(), dim=0)
                synth_probs = torch.softmax(synth_aligned[0].float(), dim=0)
                public_top2 = torch.topk(public_probs, k=2, dim=0)
                synth_top2 = torch.topk(synth_probs, k=2, dim=0)
                public_margin = public_top2.values[0] - public_top2.values[1]
                synth_margin = synth_top2.values[0] - synth_top2.values[1]
                eligible = (
                    (public_margin < synthetic_edge_margin_max)
                    & (synth_margin > public_margin + synthetic_edge_margin_gain)
                )
                uncertainty = (
                    (synthetic_edge_margin_max - public_margin)
                    / max(synthetic_edge_margin_max, 1e-6)
                ).clamp(0.0, 1.0)
                local_weight = synthetic_edge_weight * uncertainty * eligible.float()
                before_parent = public_top2.indices[0]
                edge_logits_pair = (
                    (1.0 - local_weight.view(1, 1, -1)) * edge_logits_pair
                    + local_weight.view(1, 1, -1) * synth_aligned
                )
                after_parent = torch.softmax(
                    edge_logits_pair[0].float(), dim=0
                ).argmax(dim=0)
                eligible_count = int(eligible.sum().item())
                if eligible_count:
                    print(
                        "SYNTH_EDGE", ds_path.stem, f"{t_src}->{t_tgt}",
                        "eligible=", eligible_count,
                        "parent_changes=", int((before_parent != after_parent).sum().item()),
                        flush=True,
                    )

            raw = edge_logits_pair[0]'''
if _synth_s.count(_synth_raw_old) != 1:
    raise RuntimeError("Synthetic edge anchor is not unique")
_synth_s = _synth_s.replace(_synth_raw_old, _synth_raw_new, 1)

_synth_cleanup_old = '''        if secondary_unet_out is not None:
            del secondary_unet_out
'''
_synth_cleanup_new = '''        if secondary_unet_out is not None:
            del secondary_unet_out
        if synthetic_unet_out is not None:
            del synthetic_unet_out
'''
if _synth_s.count(_synth_cleanup_old) != 1:
    raise RuntimeError("Synthetic cleanup anchor is not unique")
_synth_s = _synth_s.replace(_synth_cleanup_old, _synth_cleanup_new, 1)

_synth_load_old = '''    print(
        f"Fold {fold}: {len(test_names)} datasets | "'''
_synth_load_new = '''    synthetic_model = None
    synthetic_weights_text = os.environ.get("BIOHUB_SYNTHETIC_WEIGHTS", "").strip()
    synthetic_edge_weight = float(os.environ.get("BIOHUB_SYNTH_EDGE_WEIGHT", "0.25"))
    synthetic_edge_margin_max = float(os.environ.get("BIOHUB_SYNTH_EDGE_MARGIN_MAX", "0.12"))
    synthetic_edge_margin_gain = float(os.environ.get("BIOHUB_SYNTH_EDGE_MARGIN_GAIN", "0.02"))
    if synthetic_weights_text:
        synthetic_model, synthetic_window_size, synthetic_downsample = load_model(
            Path(synthetic_weights_text), device,
        )
        if synthetic_window_size != window_size or synthetic_downsample != downsample:
            raise ValueError(
                "Synthetic model inference grid mismatch: "
                f"public={(window_size, downsample)}, "
                f"synthetic={(synthetic_window_size, synthetic_downsample)}"
            )
        if not 0.0 < synthetic_edge_weight <= 0.5:
            raise ValueError("Synthetic edge weight must be in (0, 0.5]")
        print("Synthetic edge-only third model:", synthetic_weights_text, flush=True)

    print(
        f"Fold {fold}: {len(test_names)} datasets | "'''
if _synth_s.count(_synth_load_old) != 1:
    raise RuntimeError("Synthetic load anchor is not unique")
_synth_s = _synth_s.replace(_synth_load_old, _synth_load_new, 1)

_synth_call_old = '''                secondary_low_margin_max=secondary_low_margin_max,
            )'''
_synth_call_new = '''                secondary_low_margin_max=secondary_low_margin_max,
                synthetic_model=synthetic_model,
                synthetic_edge_weight=synthetic_edge_weight,
                synthetic_edge_margin_max=synthetic_edge_margin_max,
                synthetic_edge_margin_gain=synthetic_edge_margin_gain,
            )'''
if _synth_s.count(_synth_call_old) != 1:
    raise RuntimeError("Synthetic call anchor is not unique")
_synth_s = _synth_s.replace(_synth_call_old, _synth_call_new, 1)

compile(_synth_s, str(_ps), "exec")
_ps.write_text(_synth_s)
print("Synthetic low-margin edge-only patch applied", flush=True)
