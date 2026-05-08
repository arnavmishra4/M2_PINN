#!/usr/bin/env python3
"""
predict_glioma.py
=================
Loads trained PINN weights, runs prediction on a uniform 3D grid,
and exports everything needed for the interactive 3D viewer.

Usage
-----
    python predict_glioma.py \
        --mat   ./patient001/data.mat \
        --model ./patient001/model/finetune \
        --out   ./patient001/viz

Outputs inside --out directory:
    u_pred.npy          – 3D float32 array of predicted tumor cell density
    u_fdm.npy           – characteristic FDM solution (from .mat)
    seg_flair.npy       – FLAIR segmentation mask
    seg_t1gd.npy        – T1Gd  segmentation mask
    params.json         – learned μ_D, μ_R, thresholds, x0
    viewer_data.json    – all slices + isosurface data for the HTML viewer
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import zoom, gaussian_filter

PINNGBM_DIR = os.environ.get('PINNGBM_DIR', os.path.dirname(__file__))
sys.path.insert(0, PINNGBM_DIR)

DTYPE = np.float32


# ── Optional TF import (only needed when running against real PINN) ───────
def _try_import_tf():
    try:
        import tensorflow as tf
        from options import opts as default_opts
        from glioma  import Gmodel
        return tf, default_opts, Gmodel
    except ImportError:
        return None, None, None


# ==========================================================================
# 1.  Load the .mat file
# ==========================================================================

def load_mat(mat_file: str) -> dict:
    mat = loadmat(mat_file, mat_dtype=True)
    # Strip MATLAB meta-keys
    return {k: v for k, v in mat.items() if not k.startswith('__')}


def extract_scalar(mat: dict, key: str, default=1.0) -> float:
    v = mat.get(key, np.array([[default]]))
    if hasattr(v, 'item'):
        return float(v.item())
    return float(np.array(v).ravel()[0])


# ==========================================================================
# 2.  Run PINN prediction on a uniform grid
# ==========================================================================

def predict_pinn(mat_file: str,
                 model_dir: str,
                 resolution: int = 64,
                 batch_size: int = 8192) -> np.ndarray:
    """
    Evaluate the trained PINN at t=1 on a uniform (resolution³) grid.
    Returns u_pred of shape (resolution, resolution, resolution).
    """
    tf, default_opts, Gmodel = _try_import_tf()
    if tf is None:
        raise ImportError("TensorFlow not found. Install with: pip install tensorflow")

    import copy
    opts = copy.deepcopy(default_opts)
    opts['inv_dat_file']   = mat_file
    opts['model_dir']      = model_dir
    opts['restore']        = model_dir
    opts['N']              = 512
    opts['Ntest']          = 512
    opts['Ndat']           = 512
    opts['Ndattest']       = 512
    opts['num_init_train'] = 0
    opts['lbfgs_opts']     = None
    opts['trainD'] = opts['trainRHO'] = opts['trainx0'] = False
    opts['trainth1'] = opts['trainth2'] = False
    for k in opts['weights']:
        opts['weights'][k] = None
    opts['weights']['res'] = 1.0

    tf.random.set_seed(0)
    np.random.seed(0)

    print(f"[INFO] Loading PINN from: {model_dir}")
    g = Gmodel(opts)

    # Build uniform grid
    c    = np.linspace(0, 1, resolution, dtype=DTYPE)
    gx, gy, gz = np.meshgrid(c, c, c, indexing='ij')
    t_col = np.ones((resolution**3, 1), dtype=DTYPE)
    X     = np.hstack([t_col,
                       gx.ravel()[:, None],
                       gy.ravel()[:, None],
                       gz.ravel()[:, None]])

    print(f"[INFO] Predicting on {resolution}³ = {resolution**3:,} points ...")
    preds = []
    for i in range(0, len(X), batch_size):
        chunk = tf.constant(X[i:i+batch_size], dtype=DTYPE)
        preds.append(g.model(chunk).numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  {i:>8,} / {len(X):,}")

    u_pred = np.concatenate(preds, axis=0).reshape(resolution, resolution, resolution)
    u_pred = np.clip(u_pred, 0, 1).astype(DTYPE)
    print(f"  Done. u_max={u_pred.max():.4f}  u_mean={u_pred.mean():.6f}")
    return u_pred


# ==========================================================================
# 3.  Reconstruct 3D volumes from .mat collocation data
# ==========================================================================

def reconstruct_volume_from_mat(mat: dict,
                                 key_X:  str,
                                 key_val: str,
                                 resolution: int = 64,
                                 sigma: float = 1.0) -> np.ndarray:
    """
    Scatter sparse collocation-point data onto a uniform grid by nearest
    neighbour and optionally smooth.

    Useful for reconstructing u_fdm, y_flair, y_t1gd from .mat data points.
    """
    X   = np.array(mat[key_X],   dtype=DTYPE)    # (N, 4)  [t, x, y, z]
    val = np.array(mat[key_val], dtype=DTYPE).ravel()

    n   = resolution
    vol = np.zeros((n, n, n), dtype=DTYPE)
    cnt = np.zeros((n, n, n), dtype=DTYPE)

    # Only use the data at t≈1 if available
    if X.shape[1] == 4:
        mask = X[:, 0] > 0.9
        X    = X[mask]
        val  = val[mask]

    # Map normalised [0,1] coords → grid indices
    ix = np.round(X[:, 1] * (n-1)).astype(int).clip(0, n-1)
    iy = np.round(X[:, 2] * (n-1)).astype(int).clip(0, n-1)
    iz = np.round(X[:, 3] * (n-1)).astype(int).clip(0, n-1)

    np.add.at(vol, (ix, iy, iz), val)
    np.add.at(cnt, (ix, iy, iz), 1)
    cnt[cnt == 0] = 1
    vol /= cnt

    if sigma > 0:
        vol = gaussian_filter(vol, sigma=sigma)

    return vol.astype(DTYPE)


# ==========================================================================
# 4.  Build isosurface data (marching cubes → JSON-serialisable triangles)
# ==========================================================================

def compute_isosurface(vol: np.ndarray, level: float):
    """
    Run marching cubes and return vertices + faces as plain Python lists.
    Returns None if skimage is unavailable or the level is out of range.
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("[WARNING] scikit-image not installed. Skipping isosurface.")
        return None

    if vol.max() < level:
        return None

    verts, faces, normals, _ = marching_cubes(vol, level=level,
                                               allow_degenerate=False)
    # Normalise vertices to [0, 1]
    n = np.array(vol.shape, dtype=float) - 1
    verts_norm = verts / n

    return {
        'vertices': verts_norm.tolist(),
        'faces':    faces.tolist(),
        'normals':  normals.tolist(),
        'level':    level,
    }


# ==========================================================================
# 5.  Build 2D slice data for axial / coronal / sagittal panels
# ==========================================================================

def extract_slices(vol: np.ndarray, axis_slices: dict = None) -> dict:
    """
    Extract central slices along each axis.
    axis_slices: dict with keys 'axial', 'coronal', 'sagittal' → slice index.
                 Defaults to centre of each axis.
    """
    n = vol.shape
    if axis_slices is None:
        axis_slices = {
            'axial':    n[2] // 2,
            'coronal':  n[1] // 2,
            'sagittal': n[0] // 2,
        }

    return {
        'axial':    vol[:, :, axis_slices['axial']].tolist(),
        'coronal':  vol[:, axis_slices['coronal'], :].tolist(),
        'sagittal': vol[axis_slices['sagittal'], :, :].tolist(),
        'shape':    list(n),
        'slice_indices': axis_slices,
    }


# ==========================================================================
# 6.  Main predict + export function
# ==========================================================================

def predict_and_export(mat_file:   str,
                        model_dir:  str,
                        output_dir: str,
                        resolution: int   = 64,
                        iso_levels: list  = None,
                        use_dummy:  bool  = False):
    """
    Full pipeline: load weights → predict → export numpy + JSON for viewer.

    Parameters
    ----------
    mat_file    : preprocessed .mat file
    model_dir   : path to fine-tuned model checkpoint directory
    output_dir  : where to write all outputs
    resolution  : grid resolution per axis (64 = fast, 128 = high quality)
    iso_levels  : list of density thresholds for isosurfaces
                  default: [0.01, 0.35, 0.60]  (1%, FLAIR thresh, T1Gd thresh)
    use_dummy   : if True, use synthetic data instead of real PINN
                  (useful when TF not installed, for testing the viewer)
    """
    if iso_levels is None:
        iso_levels = [0.01, 0.35, 0.60]

    os.makedirs(output_dir, exist_ok=True)
    mat = load_mat(mat_file)

    # ── 1. Prediction ──────────────────────────────────────────────────
    if use_dummy:
        print("[INFO] Generating synthetic prediction (use_dummy=True)")
        n = resolution
        c = np.linspace(-1, 1, n, dtype=DTYPE)
        gx, gy, gz = np.meshgrid(c, c, c, indexing='ij')
        r2 = gx**2 + gy**2 + gz**2
        u_pred = np.exp(-3 * r2).astype(DTYPE)
    else:
        u_pred = predict_pinn(mat_file, model_dir, resolution=resolution)

    np.save(os.path.join(output_dir, 'u_pred.npy'), u_pred)

    # ── 2. Reconstruct reference volumes from .mat ─────────────────────
    print("[INFO] Reconstructing reference volumes from .mat ...")

    # FDM characteristic solution (if present)
    u_fdm_vol = None
    if 'uchar_dat' in mat and 'X_dat' in mat:
        u_fdm_vol = reconstruct_volume_from_mat(
            mat, 'X_dat', 'uchar_dat', resolution=resolution)
        np.save(os.path.join(output_dir, 'u_fdm.npy'), u_fdm_vol)

    # FLAIR segmentation
    seg_flair = None
    if 'u1_dat' in mat and 'X_dat' in mat:
        seg_flair = reconstruct_volume_from_mat(
            mat, 'X_dat', 'u1_dat', resolution=resolution, sigma=0.5)
        np.save(os.path.join(output_dir, 'seg_flair.npy'), seg_flair)

    # T1Gd segmentation
    seg_t1gd = None
    if 'u2_dat' in mat and 'X_dat' in mat:
        seg_t1gd = reconstruct_volume_from_mat(
            mat, 'X_dat', 'u2_dat', resolution=resolution, sigma=0.5)
        np.save(os.path.join(output_dir, 'seg_t1gd.npy'), seg_t1gd)

    # ── 3. Learned parameters ──────────────────────────────────────────
    params = {
        'mu_D':      extract_scalar(mat, 'rDe',   1.0),
        'mu_R':      extract_scalar(mat, 'rRHOe', 1.0),
        'D_nd':      extract_scalar(mat, 'DW',    0.1),
        'R_nd':      extract_scalar(mat, 'RHO',   10.0),
        'L_bar':     extract_scalar(mat, 'L',     30.0),
        'D_ratio':   extract_scalar(mat, 'D_ratio', 0.5),
        'th1_flair': extract_scalar(mat, 'th1',   0.35),
        'th2_t1gd':  extract_scalar(mat, 'th2',   0.60),
        'u_max':     float(u_pred.max()),
        'u_mean':    float(u_pred.mean()),
        'resolution': resolution,
    }
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    print(f"[INFO] Params: {params}")

    # ── 4. Isosurfaces ─────────────────────────────────────────────────
    print("[INFO] Computing isosurfaces ...")
    isosurfaces = {}
    labels = {0.01: 'infiltration_1pct',
              0.35: 'flair_threshold',
              0.60: 't1gd_threshold'}

    for level in iso_levels:
        label = labels.get(level, f'iso_{level:.2f}')
        print(f"  level={level:.2f} ({label}) ...")
        iso = compute_isosurface(u_pred, level=level)
        if iso:
            isosurfaces[label] = iso
            print(f"    → {len(iso['vertices'])} verts, {len(iso['faces'])} faces")

    # ── 5. Slice data ──────────────────────────────────────────────────
    print("[INFO] Extracting 2D slices ...")
    slices_pred  = extract_slices(u_pred)
    slices_flair = extract_slices(seg_flair)  if seg_flair  is not None else {}
    slices_t1gd  = extract_slices(seg_t1gd)   if seg_t1gd   is not None else {}
    slices_fdm   = extract_slices(u_fdm_vol)  if u_fdm_vol  is not None else {}

    # ── 6. Pack viewer_data.json ───────────────────────────────────────
    print("[INFO] Writing viewer_data.json ...")
    viewer_data = {
        'params':     params,
        'resolution': resolution,
        'isosurfaces': isosurfaces,
        'slices': {
            'u_pred':    slices_pred,
            'u_fdm':     slices_fdm,
            'seg_flair': slices_flair,
            'seg_t1gd':  slices_t1gd,
        },
    }

    vd_path = os.path.join(output_dir, 'viewer_data.json')
    with open(vd_path, 'w') as f:
        json.dump(viewer_data, f)

    size_mb = os.path.getsize(vd_path) / 1e6
    print(f"  viewer_data.json  →  {size_mb:.1f} MB")
    print(f"\n✓ All outputs saved to: {output_dir}")
    print(f"  Next step: open viewer.html in that directory\n")
    return viewer_data


# ==========================================================================
# CLI
# ==========================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Predict tumor cell density and export for 3D viewer.")
    p.add_argument('--mat',      required=True,
                   help='.mat file from preprocess_glioma.py')
    p.add_argument('--model',    default=None,
                   help='Fine-tuned model checkpoint directory')
    p.add_argument('--out',      required=True,
                   help='Output directory')
    p.add_argument('--res',      type=int, default=64,
                   help='Grid resolution (64=fast, 128=quality)')
    p.add_argument('--dummy',    action='store_true',
                   help='Use synthetic data (no PINN needed, for viewer testing)')
    args = p.parse_args()

    predict_and_export(
        mat_file   = args.mat,
        model_dir  = args.model,
        output_dir = args.out,
        resolution = args.res,
        use_dummy  = args.dummy,
    )
