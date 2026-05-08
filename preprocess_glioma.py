#!/usr/bin/env python3
"""
preprocess_glioma.py
====================
Converts raw NIfTI MRI scans into the .mat file format expected by the
pinngbm PINN codebase (github.com/Rayzhangzirui/pinngbm).

Paper reference:
    Zhang et al., "Personalized Predictions of Glioblastoma Infiltration:
    Mathematical Models, Physics-Informed Neural Networks and Multimodal Scans"
    Medical Image Analysis, 2025.

Expected inputs
---------------
    t1_path    : plain T1 scan  → used for atlas registration → geometry P(x), phi
    t1ce_path  : T1 contrast    → tumor core segmentation  y^{T1Gd}
    flair_path : FLAIR scan     → full tumor extent         y^{FLAIR}

Output
------
    A single .mat file (v5 format) containing every array the DataSet class
    and Gmodel.__init__ will look for.

Required packages
-----------------
    nibabel, numpy, scipy, scikit-image, antspy (for atlas registration)

Install:
    pip install nibabel numpy scipy scikit-image antspyx
"""

import os
import argparse
import numpy as np
import nibabel as nib
from scipy.io import savemat
from scipy.ndimage import (
    gaussian_filter, binary_fill_holes, label as nd_label,
    distance_transform_edt, zoom
)
from skimage.measure import regionprops

# --------------------------------------------------------------------------
# Optional: ANTsPy for atlas registration.  Falls back to identity if absent.
# --------------------------------------------------------------------------
try:
    import ants
    ANTS_AVAILABLE = True
except ImportError:
    ANTS_AVAILABLE = False
    print("[WARNING] antspyx not found. Brain atlas registration will be "
          "skipped and uniform tissue fractions will be used instead.\n"
          "         Install with: pip install antspyx")

DTYPE = np.float32


# ==========================================================================
# 1.  Low-level helpers
# ==========================================================================

def load_nifti(path: str):
    """Load a NIfTI file and return (data_float32, affine, header)."""
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=DTYPE)
    return data, img.affine, img.header


def normalize_intensity(vol: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Clip to [0, p-th percentile] and scale to [0, 1]."""
    p = np.percentile(vol[vol > 0], percentile)
    vol = np.clip(vol, 0, p)
    vol = vol / (p + 1e-8)
    return vol.astype(DTYPE)


def skull_strip_simple(t1: np.ndarray, threshold_frac: float = 0.15) -> np.ndarray:
    """
    Very simple skull-strip via Otsu + largest connected component.
    For production, use FSL BET or HD-BET.
    Returns a boolean brain mask.
    """
    from skimage.filters import threshold_otsu
    try:
        thresh = threshold_otsu(t1[t1 > 0])
    except Exception:
        thresh = t1.max() * threshold_frac
    binary = t1 > thresh * threshold_frac
    binary = binary_fill_holes(binary)
    labeled, num = nd_label(binary)
    if num == 0:
        return binary
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    return (labeled == largest)


def segment_tumor(t1ce: np.ndarray,
                  flair: np.ndarray,
                  brain_mask: np.ndarray,
                  t1ce_thresh_pct: float = 85.0,
                  flair_thresh_pct: float = 75.0):
    """
    Produce binary segmentations:
        y_t1gd   (tumor core   – from T1ce)
        y_flair  (full extent  – from FLAIR)

    Both restricted to the brain mask.

    Returns
    -------
    y_t1gd  : np.ndarray bool
    y_flair : np.ndarray bool
    """
    brain_vals_t1ce  = t1ce[brain_mask]
    brain_vals_flair = flair[brain_mask]

    t1ce_thresh  = np.percentile(brain_vals_t1ce,  t1ce_thresh_pct)
    flair_thresh = np.percentile(brain_vals_flair, flair_thresh_pct)

    y_t1gd  = (t1ce  > t1ce_thresh)  & brain_mask
    y_flair = (flair > flair_thresh) & brain_mask

    # FLAIR must contain T1Gd
    y_flair = y_flair | y_t1gd

    # Fill holes, keep largest CC
    y_t1gd  = _keep_largest_cc(binary_fill_holes(y_t1gd))
    y_flair = _keep_largest_cc(binary_fill_holes(y_flair))

    return y_t1gd.astype(DTYPE), y_flair.astype(DTYPE)


def _keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    labeled, num = nd_label(mask)
    if num == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax())


# ==========================================================================
# 2.  Brain atlas registration → geometry P(x) and phi
# ==========================================================================

def register_atlas_to_patient(t1_path: str,
                               atlas_wm_path: str,
                               atlas_gm_path: str,
                               brain_mask: np.ndarray):
    """
    Register a brain atlas to the patient T1 using ANTsPy rigid registration.
    Returns Pwm, Pgm (white/grey matter probability maps, same shape as T1).

    If atlas paths are None or ANTsPy is unavailable, returns uniform maps.
    """
    shape = brain_mask.shape

    if not ANTS_AVAILABLE or atlas_wm_path is None or atlas_gm_path is None:
        print("[INFO] Using uniform WM/GM fractions (no atlas registration).")
        Pwm = np.ones(shape, dtype=DTYPE) * 0.5
        Pgm = np.ones(shape, dtype=DTYPE) * 0.5
        Pwm[~brain_mask] = 0.0
        Pgm[~brain_mask] = 0.0
        return Pwm, Pgm

    print("[INFO] Registering atlas to patient T1 (rigid) ...")
    fixed  = ants.image_read(t1_path)
    moving = ants.image_read(atlas_wm_path)

    reg = ants.registration(fixed=fixed, moving=moving,
                            type_of_transform='Rigid')

    # Apply the same transform to WM and GM maps
    wm_img = ants.image_read(atlas_wm_path)
    gm_img = ants.image_read(atlas_gm_path)

    wm_reg = ants.apply_transforms(fixed=fixed, moving=wm_img,
                                   transformlist=reg['fwdtransforms'])
    gm_reg = ants.apply_transforms(fixed=fixed, moving=gm_img,
                                   transformlist=reg['fwdtransforms'])

    Pwm = wm_reg.numpy().astype(DTYPE)
    Pgm = gm_reg.numpy().astype(DTYPE)

    Pwm = np.clip(Pwm, 0, 1)
    Pgm = np.clip(Pgm, 0, 1)
    Pwm[~brain_mask] = 0.0
    Pgm[~brain_mask] = 0.0
    return Pwm, Pgm


# ==========================================================================
# 3.  Phase-field (phi) via smoothed brain mask  [Diffuse Domain Method]
# ==========================================================================

def compute_phi(brain_mask: np.ndarray, epsilon_mm: float = 3.0,
                voxel_size_mm: float = 1.0) -> np.ndarray:
    """
    Compute the diffuse-domain phase field phi from the brain mask.

    phi ≈ 1 inside the brain, transitions smoothly to 0 at the boundary.
    This approximates solving the Cahn-Hilliard equation for a short time
    (Section 2.1.2 of the paper).

    epsilon_mm : width of the diffuse interface in mm (paper uses 3 mm).
    """
    # Signed distance (in voxels, then convert to mm)
    dist_in  = distance_transform_edt( brain_mask)   * voxel_size_mm
    dist_out = distance_transform_edt(~brain_mask)   * voxel_size_mm
    signed_dist = dist_in - dist_out                  # + inside, - outside

    # Smooth tanh transition
    phi = 0.5 * (1.0 + np.tanh(signed_dist / epsilon_mm))
    phi = np.clip(phi, 0.0, 1.0).astype(DTYPE)
    return phi


# ==========================================================================
# 4.  Diffusion tensor P(x) = Pwm + Pgm / factor
#     factor = Dw / Dg = 10  (paper, Section 2.1.1)
# ==========================================================================

def compute_P_and_gradients(Pwm: np.ndarray,
                             Pgm: np.ndarray,
                             phi: np.ndarray,
                             factor: float = 10.0):
    """
    P(x) = Pwm(x) + Pgm(x) / factor

    Also compute the gradient fields needed for the non-divergence form
    of the PDE loss  (Section 2.3, Eq. 11):
        ∇(P · phi)  →  DxPphi, DyPphi, DzPphi

    Returns P, DxPphi, DyPphi, DzPphi  (all same shape as Pwm)
    """
    P    = Pwm + Pgm / factor
    Pphi = P * phi

    # Central-difference gradients (voxel units; paper assumes 1 mm/voxel)
    DxPphi = np.gradient(Pphi, axis=0).astype(DTYPE)
    DyPphi = np.gradient(Pphi, axis=1).astype(DTYPE)
    DzPphi = np.gradient(Pphi, axis=2).astype(DTYPE)

    return P.astype(DTYPE), DxPphi, DyPphi, DzPphi


# ==========================================================================
# 5.  Collocation point sampling
#     Following Appendix B of the paper:
#       - spatial coords dense at the tumor center, sparse near boundary
#       - time coords dense at early times (truncated exponential)
# ==========================================================================

def _tumor_centroid(y_t1gd: np.ndarray) -> np.ndarray:
    props = regionprops(y_t1gd.astype(np.uint8))
    if props:
        c = np.array(props[0].centroid, dtype=DTYPE)
    else:
        c = np.array(np.array(y_t1gd.shape) / 2.0, dtype=DTYPE)
    return c


def sample_residual_points(phi: np.ndarray,
                            y_t1gd: np.ndarray,
                            N: int = 50000,
                            lam: float = 0.5) -> np.ndarray:
    """
    Sample N collocation points (t, x, y, z) for the PDE residual loss.

    Spatial sampling: inverse-r² weighting from tumor centroid (denser at center).
    Temporal sampling: truncated exponential Exp(λ), t ∈ [0, 1].

    Returns X_res of shape (N, 4): columns = [t, x, y, z]  (normalised).
    """
    brain_idx = np.argwhere(phi > 0.5)            # voxels inside brain
    x0 = _tumor_centroid(y_t1gd)
    shape = np.array(phi.shape, dtype=DTYPE)

    # Compute distances from tumor center
    diff  = brain_idx - x0[None, :]
    dists = np.linalg.norm(diff, axis=1) + 1e-3   # avoid div/0

    # Inverse-r² weights → denser at centre  (3D)
    weights = 1.0 / (dists ** 2)
    weights /= weights.sum()

    chosen_idx = np.random.choice(len(brain_idx), size=N,
                                  replace=True, p=weights)
    spatial = brain_idx[chosen_idx].astype(DTYPE)

    # Normalise spatial coords to [0, 1]
    spatial_norm = spatial / (shape - 1)

    # Time: sample from truncated Exp(lam) on [0,1]
    u_unif = np.random.uniform(0, 1 - np.exp(-lam), size=N)
    t_samp = -np.log(1 - u_unif) / lam
    t_samp = np.clip(t_samp, 0, 1).astype(DTYPE)[:, None]

    X_res = np.hstack([t_samp, spatial_norm])      # (N, 4)
    return X_res


def sample_data_points(phi: np.ndarray,
                        y_t1gd: np.ndarray,
                        y_flair: np.ndarray,
                        N: int = 50000) -> tuple:
    """
    Sample N data collocation points at t = 1 (imaging time).

    Returns
    -------
    X_dat    : (N, 4)  [1.0, x_norm, y_norm, z_norm]
    u1_dat   : (N, 1)  y^FLAIR values  (0 or 1)
    u2_dat   : (N, 1)  y^T1Gd  values  (0 or 1)
    """
    brain_idx = np.argwhere(phi > 0.5)
    shape = np.array(phi.shape, dtype=DTYPE)

    chosen = np.random.choice(len(brain_idx), size=N,
                              replace=(len(brain_idx) < N))
    spatial = brain_idx[chosen].astype(DTYPE)
    spatial_norm = spatial / (shape - 1)

    t_col = np.ones((N, 1), dtype=DTYPE)
    X_dat = np.hstack([t_col, spatial_norm])

    coords = brain_idx[chosen]
    u1_dat = y_flair[coords[:, 0], coords[:, 1], coords[:, 2]].reshape(-1, 1)
    u2_dat = y_t1gd [coords[:, 0], coords[:, 1], coords[:, 2]].reshape(-1, 1)

    return X_dat, u1_dat.astype(DTYPE), u2_dat.astype(DTYPE)


def sample_bc_points(phi: np.ndarray, N: int = 5000) -> tuple:
    """
    Sample boundary collocation points (voxels where phi ≈ 0.5).
    Returns X_bc (N, 4), u_bc (N, 1) = 0, phi_bc (N, 1).
    """
    grad = np.abs(np.gradient(phi, axis=0)) + \
           np.abs(np.gradient(phi, axis=1)) + \
           np.abs(np.gradient(phi, axis=2))
    bc_idx = np.argwhere((phi > 0.1) & (phi < 0.9) & (grad > 0.01))

    shape = np.array(phi.shape, dtype=DTYPE)
    if len(bc_idx) == 0:
        bc_idx = np.argwhere(phi > 0)

    chosen  = np.random.choice(len(bc_idx), size=N,
                               replace=(len(bc_idx) < N))
    spatial = bc_idx[chosen].astype(DTYPE)
    spatial_norm = spatial / (shape - 1)
    t_col   = np.random.uniform(0, 1, size=(N, 1)).astype(DTYPE)
    X_bc    = np.hstack([t_col, spatial_norm])

    phi_bc  = phi[bc_idx[chosen, 0],
                  bc_idx[chosen, 1],
                  bc_idx[chosen, 2]].reshape(-1, 1).astype(DTYPE)
    u_bc    = np.zeros((N, 1), dtype=DTYPE)

    return X_bc, u_bc, phi_bc


# ==========================================================================
# 6.  Attach geometry arrays to residual / data collocation points
# ==========================================================================

def attach_geometry(X: np.ndarray,
                    phi_vol: np.ndarray,
                    P_vol: np.ndarray,
                    Pwm_vol: np.ndarray,
                    Pgm_vol: np.ndarray,
                    DxPphi_vol: np.ndarray,
                    DyPphi_vol: np.ndarray,
                    DzPphi_vol: np.ndarray,
                    shape: tuple) -> dict:
    """
    Lookup geometry values at the spatial coordinates in X (rows = points).
    X columns: [t, x_norm, y_norm, z_norm].

    Returns a dict of (N,1) arrays ready for the .mat file.
    """
    shape_arr = np.array(shape, dtype=DTYPE) - 1
    ix = np.round(X[:, 1] * shape_arr[0]).astype(int).clip(0, shape[0]-1)
    iy = np.round(X[:, 2] * shape_arr[1]).astype(int).clip(0, shape[1]-1)
    iz = np.round(X[:, 3] * shape_arr[2]).astype(int).clip(0, shape[2]-1)

    def lookup(vol):
        return vol[ix, iy, iz].reshape(-1, 1).astype(DTYPE)

    return {
        'phi':     lookup(phi_vol),
        'P':       lookup(P_vol),
        'Pwm':     lookup(Pwm_vol),
        'Pgm':     lookup(Pgm_vol),
        'DxPphi':  lookup(DxPphi_vol),
        'DyPphi':  lookup(DyPphi_vol),
        'DzPphi':  lookup(DzPphi_vol),
    }


# ==========================================================================
# 7.  Grid-search: estimate characteristic parameters D̄/ρ̄ and L̄
#     Following Appendix E of the paper.
# ==========================================================================

def _solve_1d_spherical(D_nd, R_nd, N_r=200, N_t=500,
                         r_max=180.0):
    """
    Solve the non-dimensional radially-symmetric Fisher-KPP PDE:

        ∂u/∂t = D * (1/r²) ∂/∂r(r² ∂u/∂r)  +  R * u(1-u)

    on r ∈ [0, r_max], t ∈ [0, 1], with u(r,0) = 0.1 exp(-0.1 r).

    Returns (r_grid, u_final).
    """
    dr = r_max / N_r
    dt = 1.0   / N_t
    r  = np.linspace(0, r_max, N_r + 1)
    r[0] = 1e-6                       # avoid div/0 at origin

    u = 0.1 * np.exp(-0.1 * r)

    for _ in range(N_t):
        # Interior: central differences
        u_rr = (u[2:] - 2*u[1:-1] + u[:-2]) / dr**2
        u_r  = (u[2:] - u[:-2]) / (2 * dr)
        dudt_int = D_nd * (u_rr + (2 / r[1:-1]) * u_r) + R_nd * u[1:-1] * (1 - u[1:-1])
        # Boundaries
        # r=0: symmetry (Neumann)  u[0] = u[1]
        # r=R: Neumann             u[N] = u[N-1]
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + dt * dudt_int
        u_new[0]    = u_new[1]
        u_new[-1]   = u_new[-2]
        u = np.clip(u_new, 0, 1)

    return r, u


def grid_search_characteristic_params(R_flair_seg: float,
                                       R_t1gd_seg: float,
                                       u_flair_c: float = 0.35,
                                       u_t1gd_c: float  = 0.60):
    """
    Find D̄/ρ̄ and L̄ by grid search in spherically symmetric geometry.

    Matches the radii at which u = u_flair_c and u = u_t1gd_c to
    R_flair_seg and R_t1gd_seg (both in mm).

    Returns: D_ratio (D̄/ρ̄), L_bar, D_nd, R_nd
    """
    D_ratio_grid = np.arange(0.1, 1.05, 0.1)   # D̄/ρ̄ ∈ [0.1, 1.0]
    L_bar_grid   = np.arange(10,  95,   5  )   # L̄   ∈ [10, 90]

    best_err  = np.inf
    best_Dratio = D_ratio_grid[0]
    best_L      = L_bar_grid[0]
    best_Dnd    = None
    best_Rnd    = None

    print(f"[INFO] Grid search: R_FLAIR={R_flair_seg:.1f} mm, "
          f"R_T1Gd={R_t1gd_seg:.1f} mm")

    for D_ratio in D_ratio_grid:
        for L_bar in L_bar_grid:
            # Non-dimensional parameters (paper Eq. 5)
            # D = sqrt(D̄/ρ̄) / L̄,   R = L̄ / sqrt(D̄/ρ̄)
            v_bar = np.sqrt(D_ratio)
            T_bar = L_bar / v_bar
            D_nd  = D_ratio * T_bar / (L_bar ** 2)   # = 1/R (by construction)
            R_nd  = T_bar                              # = L̄/v̄

            r_grid, u_final = _solve_1d_spherical(D_nd, R_nd)

            # Find radii where u crosses the threshold values
            def find_radius(threshold):
                idx = np.where(u_final >= threshold)[0]
                return r_grid[idx[-1]] if len(idx) else 0.0

            R_flair_sph = find_radius(u_flair_c)
            R_t1gd_sph  = find_radius(u_t1gd_c)

            err = (abs(R_flair_seg - R_flair_sph) / (R_flair_seg + 1e-6) +
                   abs(R_t1gd_seg  - R_t1gd_sph)  / (R_t1gd_seg  + 1e-6))

            if err < best_err:
                best_err    = err
                best_Dratio = D_ratio
                best_L      = L_bar
                best_Dnd    = D_nd
                best_Rnd    = R_nd

    print(f"[INFO] Best D̄/ρ̄={best_Dratio:.2f}, L̄={best_L:.1f}, "
          f"D_nd={best_Dnd:.4f}, R_nd={best_Rnd:.4f}, err={best_err:.4f}")
    return best_Dratio, best_L, best_Dnd, best_Rnd


def compute_segmentation_radii(y_t1gd: np.ndarray,
                                 y_flair: np.ndarray) -> tuple:
    """
    Compute R_seg^{T1Gd} and R_seg^{FLAIR} as the max distance from the
    centroid of y^{T1Gd} to any positive voxel in each mask.
    (Appendix E of the paper)
    """
    x0 = _tumor_centroid(y_t1gd)
    idx_t1gd  = np.argwhere(y_t1gd  > 0.5)
    idx_flair = np.argwhere(y_flair > 0.5)

    R_t1gd  = np.linalg.norm(idx_t1gd  - x0, axis=1).max() if len(idx_t1gd)  else 1.0
    R_flair = np.linalg.norm(idx_flair - x0, axis=1).max() if len(idx_flair) else 1.0
    return float(R_flair), float(R_t1gd)


# ==========================================================================
# 8.  FDM solver: characteristic solution ū^FDM
#     Solves the 3D diffuse-domain PDE with μ_D = μ_R = 1.
#     Used as pre-training data for the PINN.
# ==========================================================================

def solve_fdm_characteristic(phi_vol: np.ndarray,
                               P_vol:   np.ndarray,
                               D_nd:    float,
                               R_nd:    float,
                               x0_vox:  np.ndarray,
                               N_t:     int = 200,
                               tau:     float = 1e-4) -> np.ndarray:
    """
    Solve the non-dimensional PDE (paper Eq. 7 with μ_D = μ_R = 1):

        ∂(φu)/∂t = D_nd ∇·(P φ_τ ∇u)  +  R_nd φ u(1-u)

    using explicit Euler + central differences.

    Returns u_fdm of the same shape as phi_vol, representing ū^FDM at t=1.
    """
    shape = phi_vol.shape
    phi_tau = phi_vol + tau

    # Initial condition: u0(x) = 0.1 exp(-0.1 |x - x0|²)
    idx = np.indices(shape, dtype=DTYPE)
    r2  = sum((idx[i] - x0_vox[i])**2 for i in range(3))
    u   = 0.1 * np.exp(-0.1 * r2) * phi_vol

    dt = 1.0 / N_t

    Pphi_tau = P_vol * phi_tau

    print(f"[INFO] Running FDM characteristic solve ({N_t} steps) ...")
    for step in range(N_t):
        # Laplacian of u (central differences)
        lap_u = (np.roll(u,  1, axis=0) + np.roll(u, -1, axis=0) +
                 np.roll(u,  1, axis=1) + np.roll(u, -1, axis=1) +
                 np.roll(u,  1, axis=2) + np.roll(u, -1, axis=2) -
                 6 * u)

        # Gradient of Pphi_tau dotted with gradient of u
        gPx = np.gradient(Pphi_tau, axis=0)
        gPy = np.gradient(Pphi_tau, axis=1)
        gPz = np.gradient(Pphi_tau, axis=2)
        gux = np.gradient(u, axis=0)
        guy = np.gradient(u, axis=1)
        guz = np.gradient(u, axis=2)
        grad_dot = gPx*gux + gPy*guy + gPz*guz

        diffusion     = D_nd * (Pphi_tau * lap_u + grad_dot)
        proliferation = R_nd * phi_vol * u * (1.0 - u)

        phi_u_new = phi_vol * u + dt * (diffusion + proliferation)
        u_new     = phi_u_new / (phi_vol + 1e-8)
        u_new     = np.clip(u_new, 0.0, 1.0)
        u_new    *= (phi_vol > 0.01)     # zero outside brain

        u = u_new

        if (step + 1) % 50 == 0:
            print(f"  step {step+1}/{N_t}  u_max={u.max():.4f}")

    return u.astype(DTYPE)


# ==========================================================================
# 9.  Main preprocessing function
# ==========================================================================

def preprocess(t1_path:    str,
               t1ce_path:  str,
               flair_path: str,
               output_mat: str,
               atlas_wm_path: str  = None,
               atlas_gm_path: str  = None,
               N_res:      int     = 50000,
               N_dat:      int     = 50000,
               N_bc:       int     = 5000,
               run_fdm:    bool    = True,
               fdm_steps:  int     = 200,
               factor:     float   = 10.0,
               epsilon_mm: float   = 3.0,
               voxel_mm:   float   = 1.0,
               seed:       int     = 0):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    t1_path       : path to plain T1 .nii / .nii.gz
    t1ce_path     : path to T1 contrast-enhanced .nii / .nii.gz
    flair_path    : path to FLAIR .nii / .nii.gz
    output_mat    : destination .mat file path
    atlas_wm_path : (optional) WM atlas .nii for registration
    atlas_gm_path : (optional) GM atlas .nii for registration
    N_res         : number of PDE residual collocation points
    N_dat         : number of data collocation points (at t=1)
    N_bc          : number of boundary collocation points
    run_fdm       : whether to run the FDM characteristic solve
    fdm_steps     : number of time steps for FDM solver
    factor        : Dw/Dg ratio (paper uses 10)
    epsilon_mm    : diffuse interface width in mm (paper uses 3)
    voxel_mm      : voxel size in mm (BraTS data: 1 mm isotropic)
    seed          : random seed
    """
    np.random.seed(seed)
    os.makedirs(os.path.dirname(os.path.abspath(output_mat)), exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load scans
    # ------------------------------------------------------------------
    print("\n=== Step 1: Loading NIfTI scans ===")
    t1,    aff, hdr = load_nifti(t1_path)
    t1ce,  _,   _   = load_nifti(t1ce_path)
    flair, _,   _   = load_nifti(flair_path)

    print(f"  T1    shape: {t1.shape}")
    print(f"  T1ce  shape: {t1ce.shape}")
    print(f"  FLAIR shape: {flair.shape}")

    # ------------------------------------------------------------------
    # Step 2: Normalise intensities
    # ------------------------------------------------------------------
    print("\n=== Step 2: Normalising intensities ===")
    t1    = normalize_intensity(t1)
    t1ce  = normalize_intensity(t1ce)
    flair = normalize_intensity(flair)

    # ------------------------------------------------------------------
    # Step 3: Brain mask (skull strip from T1)
    # ------------------------------------------------------------------
    print("\n=== Step 3: Skull stripping (T1) ===")
    brain_mask = skull_strip_simple(t1)
    print(f"  Brain voxels: {brain_mask.sum():,}")

    # ------------------------------------------------------------------
    # Step 4: Tumor segmentations
    # ------------------------------------------------------------------
    print("\n=== Step 4: Tumor segmentation ===")
    y_t1gd, y_flair = segment_tumor(t1ce, flair, brain_mask)
    print(f"  y_T1Gd  voxels: {(y_t1gd>0.5).sum():,}")
    print(f"  y_FLAIR voxels: {(y_flair>0.5).sum():,}")

    # ------------------------------------------------------------------
    # Step 5: Atlas registration → Pwm, Pgm
    # ------------------------------------------------------------------
    print("\n=== Step 5: Atlas registration → geometry ===")
    Pwm, Pgm = register_atlas_to_patient(t1_path, atlas_wm_path,
                                          atlas_gm_path, brain_mask)

    # ------------------------------------------------------------------
    # Step 6: Phase field phi
    # ------------------------------------------------------------------
    print("\n=== Step 6: Computing phase field phi ===")
    phi = compute_phi(brain_mask, epsilon_mm=epsilon_mm,
                      voxel_size_mm=voxel_mm)

    # ------------------------------------------------------------------
    # Step 7: Geometry fields P, DxPphi, DyPphi, DzPphi
    # ------------------------------------------------------------------
    print("\n=== Step 7: Computing P(x) and gradient fields ===")
    P_vol, DxPphi_vol, DyPphi_vol, DzPphi_vol = compute_P_and_gradients(
        Pwm, Pgm, phi, factor=factor)

    # ------------------------------------------------------------------
    # Step 8: Characteristic parameters via grid search
    # ------------------------------------------------------------------
    print("\n=== Step 8: Grid search for characteristic parameters ===")
    R_flair_seg, R_t1gd_seg = compute_segmentation_radii(y_t1gd, y_flair)
    D_ratio, L_bar, D_nd, R_nd = grid_search_characteristic_params(
        R_flair_seg, R_t1gd_seg)

    # Tumor centroid (voxel coords)
    x0_vox = _tumor_centroid(y_t1gd)
    shape  = phi.shape

    # ------------------------------------------------------------------
    # Step 9: FDM characteristic solution
    # ------------------------------------------------------------------
    if run_fdm:
        print("\n=== Step 9: FDM characteristic solve ===")
        u_fdm = solve_fdm_characteristic(phi, P_vol, D_nd, R_nd,
                                          x0_vox, N_t=fdm_steps)
    else:
        print("\n=== Step 9: Skipping FDM solve (run_fdm=False) ===")
        u_fdm = np.zeros(shape, dtype=DTYPE)

    # ------------------------------------------------------------------
    # Step 10: Sample collocation points
    # ------------------------------------------------------------------
    print("\n=== Step 10: Sampling collocation points ===")
    X_res = sample_residual_points(phi, y_t1gd, N=N_res)
    X_dat, u1_dat, u2_dat = sample_data_points(phi, y_t1gd, y_flair, N=N_dat)
    X_bc,  u_bc,  phi_bc  = sample_bc_points(phi, N=N_bc)

    # Attach geometry at residual points
    geo_res = attach_geometry(X_res, phi, P_vol, Pwm, Pgm,
                               DxPphi_vol, DyPphi_vol, DzPphi_vol, shape)
    geo_dat = attach_geometry(X_dat, phi, P_vol, Pwm, Pgm,
                               DxPphi_vol, DyPphi_vol, DzPphi_vol, shape)

    # uchar: FDM values at collocation points (pre-training target)
    def lookup_fdm(X):
        s = np.array(shape, dtype=DTYPE) - 1
        ix = np.round(X[:, 1]*s[0]).astype(int).clip(0, shape[0]-1)
        iy = np.round(X[:, 2]*s[1]).astype(int).clip(0, shape[1]-1)
        iz = np.round(X[:, 3]*s[2]).astype(int).clip(0, shape[2]-1)
        return u_fdm[ix, iy, iz].reshape(-1, 1).astype(DTYPE)

    uchar_res = lookup_fdm(X_res)
    uchar_dat = lookup_fdm(X_dat)

    # ------------------------------------------------------------------
    # Step 11: Pack into .mat dict and save
    # ------------------------------------------------------------------
    print("\n=== Step 11: Saving .mat file ===")

    # Scalar values expected by Gmodel.__init__
    # rDe, rRHOe, M are initial guesses; PINN will train them
    mat_dict = {
        # ---- Residual points ----
        'X_res':       X_res,                    # (N_res, 4)  [t, x, y, z]
        'phi_res':     geo_res['phi'],            # (N_res, 1)
        'P_res':       geo_res['P'],              # (N_res, 1)
        'Pwm_res':     geo_res['Pwm'],            # (N_res, 1)
        'Pgm_res':     geo_res['Pgm'],            # (N_res, 1)
        'DxPphi_res':  geo_res['DxPphi'],         # (N_res, 1)
        'DyPphi_res':  geo_res['DyPphi'],         # (N_res, 1)
        'DzPphi_res':  geo_res['DzPphi'],         # (N_res, 1)
        'uchar_res':   uchar_res,                 # (N_res, 1)  ū^FDM at X_res

        # ---- Data points (t=1) ----
        'X_dat':       X_dat,                    # (N_dat, 4)
        'u1_dat':      u1_dat,                   # (N_dat, 1)  y^FLAIR
        'u2_dat':      u2_dat,                   # (N_dat, 1)  y^T1Gd
        'uchar_dat':   uchar_dat,                # (N_dat, 1)  ū^FDM at X_dat
        'phi_dat':     geo_dat['phi'],            # (N_dat, 1)

        # ---- Boundary points ----
        'X_bc':        X_bc,                     # (N_bc, 4)
        'u_bc':        u_bc,                     # (N_bc, 1)  = 0
        'phi_bc':      phi_bc,                   # (N_bc, 1)

        # ---- Scalars (initial parameter guesses) ----
        'rDe':         np.array([[1.0]], dtype=DTYPE),   # μ_D init
        'rRHOe':       np.array([[1.0]], dtype=DTYPE),   # μ_R init
        'M':           np.array([[1.0]], dtype=DTYPE),   # carrying capacity
        'DW':          np.array([[D_nd]], dtype=DTYPE),  # non-dim D
        'RHO':         np.array([[R_nd]], dtype=DTYPE),  # non-dim R
        'L':           np.array([[L_bar]], dtype=DTYPE), # length scale (mm)
        'factor':      np.array([[factor]], dtype=DTYPE),
        'kadc':        np.array([[1.0]], dtype=DTYPE),
        'm':           np.array([[1.0]], dtype=DTYPE),
        'A':           np.array([[0.0]], dtype=DTYPE),
        'th1':         np.array([[0.35]], dtype=DTYPE),  # u_c^FLAIR init
        'th2':         np.array([[0.60]], dtype=DTYPE),  # u_c^T1Gd  init

        # Tumor centroid in normalised coords (used as x0 init)
        'x0':          (x0_vox / (np.array(shape) - 1)).reshape(1, 3).astype(DTYPE),
        'gtx0':        (x0_vox / (np.array(shape) - 1)).reshape(1, 3).astype(DTYPE),

        # Segmentation radii (mm)
        'R_flair_seg': np.array([[R_flair_seg]], dtype=DTYPE),
        'R_t1gd_seg':  np.array([[R_t1gd_seg]],  dtype=DTYPE),
        'D_ratio':     np.array([[D_ratio]], dtype=DTYPE),
        'L_bar':       np.array([[L_bar]],   dtype=DTYPE),
    }

    savemat(output_mat, mat_dict, do_compression=True)
    size_mb = os.path.getsize(output_mat) / 1e6
    print(f"  Saved → {output_mat}  ({size_mb:.1f} MB)")
    print("\n✓ Preprocessing complete.\n")
    return output_mat


# ==========================================================================
# 10. CLI entry point
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess BraTS/patient MRI scans for pinngbm PINN.")
    p.add_argument('--t1',    required=True, help='T1 NIfTI path')
    p.add_argument('--t1ce', required=True, help='T1ce NIfTI path')
    p.add_argument('--flair', required=True, help='FLAIR NIfTI path')
    p.add_argument('--out',   required=True, help='Output .mat file path')
    p.add_argument('--atlas_wm', default=None, help='(optional) WM atlas NIfTI')
    p.add_argument('--atlas_gm', default=None, help='(optional) GM atlas NIfTI')
    p.add_argument('--N_res',    type=int, default=50000)
    p.add_argument('--N_dat',    type=int, default=50000)
    p.add_argument('--N_bc',     type=int, default=5000)
    p.add_argument('--fdm_steps',type=int, default=200)
    p.add_argument('--no_fdm',   action='store_true',
                   help='Skip FDM solve (uchar will be zeros)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    preprocess(
        t1_path    = args.t1,
        t1ce_path  = args.t1ce,
        flair_path = args.flair,
        output_mat = args.out,
        atlas_wm_path = args.atlas_wm,
        atlas_gm_path = args.atlas_gm,
        N_res      = args.N_res,
        N_dat      = args.N_dat,
        N_bc       = args.N_bc,
        run_fdm    = not args.no_fdm,
        fdm_steps  = args.fdm_steps,
        seed       = args.seed,
    )