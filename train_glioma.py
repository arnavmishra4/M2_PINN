#!/usr/bin/env python3
"""
train_glioma.py
===============
Runs the two-stage PINN training pipeline on a patient .mat file produced
by preprocess_glioma.py.

Stage 1  — Pre-training:   PINN learns ū^FDM (the characteristic solution).
Stage 2  — Fine-tuning:    PINN learns patient-specific parameters
                            (μ_D, μ_R, x0, u_c^FLAIR, u_c^T1Gd) from
                            segmentation data.

After training the script saves:
  • PINN weights (TensorFlow checkpoint)
  • Predicted tumor cell density as a .npy volume
  • options.json with all hyperparameters used
  • training_log.json with loss history

Paper reference:
    Zhang et al., Medical Image Analysis 2025.
    github.com/Rayzhangzirui/pinngbm
"""

import os
import sys
import json
import argparse
import numpy as np

# ── Make the pinngbm repo importable ──────────────────────────────────────
# Adjust this path to wherever you cloned the pinngbm repo.
PINNGBM_DIR = os.environ.get('PINNGBM_DIR', os.path.dirname(__file__))
sys.path.insert(0, PINNGBM_DIR)

# ── Core imports from the pinngbm repo ────────────────────────────────────
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    from config   import DTYPE
    from DataSet  import DataSet
    from options import Options   # str_from_dict may not exist
    from glioma   import Gmodel
except ImportError as e:
    raise ImportError(
        f"Cannot import pinngbm modules: {e}\n"
        f"Make sure PINNGBM_DIR is set correctly and the repo is installed.\n"
        f"Currently looking in: {PINNGBM_DIR}"
    )


# ==========================================================================
# Helper: save dict to JSON (handles numpy scalars)
# ==========================================================================

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def save_json(d: dict, path: str):
    with open(path, 'w') as f:
        json.dump(d, f, indent=2, cls=_NpEncoder)


# ==========================================================================
# Stage 1: Pre-training (PINN → ū^FDM)
# ==========================================================================

def build_pretrain_opts(mat_file: str,
                         model_dir: str,
                         N_res: int    = 20000,
                         N_dat: int    = 20000,
                         lr: float     = 1e-3,
                         n_adam: int   = 20000,
                         lbfgs_iter: int = 20000,
                         patience: int = 1000,
                         seed: int     = 0) -> dict:
    """
    Build the options dict for the pre-training stage.

    Mirrors the paper's Appendix A settings:
      - 4 hidden layers, 128 neurons, tanh
      - Adam (lr=0.001) then L-BFGS-B
      - Loss = L_PDE + L_char  (fitfwd / solvechar mode)
    """
    from options import opts as default_opts
    import copy
    o = copy.deepcopy(default_opts)

    o['seed']            = seed
    o['inv_dat_file']    = mat_file
    o['model_dir']       = os.path.join(model_dir, 'pretrain')
    o['tag']             = 'pretrain'

    # Network architecture (paper: 4×128, tanh)
    o['nn_opts']['num_hidden_layers']    = 4
    o['nn_opts']['num_neurons_per_layer'] = 128
    o['nn_opts']['activation']           = 'tanh'

    # Collocation sizes
    o['N']        = N_res
    o['Ntest']    = N_res
    o['Ndat']     = N_dat
    o['Ndattest'] = N_dat

    # Optimiser
    o['num_init_train'] = n_adam
    o['learning_rate_opts']['initial_learning_rate'] = lr
    o['lbfgs_opts']['maxiter'] = lbfgs_iter
    o['lbfgs_opts']['maxfun']  = lbfgs_iter

    # Loss weights: residual + char (ū^FDM) only
    for k in o['weights']:
        o['weights'][k] = None
    o['weights']['res']  = 1.0    # PDE residual loss
    o['weights']['uxr']  = 1.0    # MSE vs ū^FDM at residual pts  (L_char)
    o['weights']['udat'] = None   # no segmentation loss yet

    # Early stopping monitors training + test total loss
    o['earlystop_opts']['patience'] = patience
    o['earlystop_opts']['monitor']  = ['total', 'totaltest']
    o['earlystop_opts']['burnin']   = 1000

    # Parameters NOT trainable in pre-training
    o['trainD']   = False
    o['trainRHO'] = False
    o['trainM']   = False
    o['trainm']   = False
    o['trainA']   = False
    o['trainx0']  = False
    o['trainth1'] = False
    o['trainth2'] = False

    # Data source: use the characteristic FDM solution
    o['udatsource']   = 'char'   # uchar_dat / uchar_res from .mat
    o['initfromdata'] = True     # read rDe, rRHOe, M from .mat scalars
    o['ictransform']  = True     # output transform: u = t·uNN + u0

    o['print_res_every'] = 100
    o['ckpt_every']      = 5000
    o['saveckpt']        = True
    o['file_log']        = True

    return o


def build_finetune_opts(mat_file: str,
                         model_dir: str,
                         pretrain_dir: str,
                         N_res: int    = 5000,
                         N_dat: int    = 5000,
                         lr: float     = 1e-4,
                         n_adam: int   = 5000,
                         lbfgs_iter: int = 5000,
                         patience: int = 500,
                         seed: int     = 0) -> dict:
    """
    Build the options dict for the fine-tuning stage.

    Mirrors the paper's Appendix A fine-tuning settings:
      - Same network, restored from pre-training checkpoint
      - Adam (lr=1e-4) then L-BFGS-B
      - Loss = L_PDE + w_SEG * L_SEG   (patient mode)
      - Trainable: μ_D, μ_R, x0, u_c^FLAIR, u_c^T1Gd
    """
    from options import opts as default_opts
    import copy
    o = copy.deepcopy(default_opts)

    o['seed']         = seed
    o['inv_dat_file'] = mat_file
    o['model_dir']    = os.path.join(model_dir, 'finetune')
    o['tag']          = 'finetune'
    o['restore']      = pretrain_dir    # restore PINN weights from pre-training

    # Same network as pre-training
    o['nn_opts']['num_hidden_layers']     = 4
    o['nn_opts']['num_neurons_per_layer'] = 128
    o['nn_opts']['activation']            = 'tanh'

    # Smaller dataset for fine-tuning (paper: 5000)
    o['N']        = N_res
    o['Ntest']    = N_res
    o['Ndat']     = N_dat
    o['Ndattest'] = N_dat

    # Optimiser (smaller lr for fine-tuning)
    o['num_init_train'] = n_adam
    o['learning_rate_opts']['initial_learning_rate'] = lr
    o['lbfgs_opts']['maxiter'] = lbfgs_iter
    o['lbfgs_opts']['maxfun']  = lbfgs_iter

    # Loss weights: PDE residual + segmentation (paper Eq. 19)
    # w_SEG = 1e-3 so magnitudes are comparable
    for k in o['weights']:
        o['weights'][k] = None
    o['weights']['res']  = 1.0       # L_PDE
    o['weights']['seg1'] = 1e-3      # w_SEG * L_SEG (y^FLAIR part)
    o['weights']['seg2'] = 1e-3      # w_SEG * L_SEG (y^T1Gd part)

    # Parameter ranges (paper Section 2.3):
    # μ_D ∈ [0.75, 1.25], μ_R ∈ [0.75, 1.25]
    # u_c^FLAIR ∈ [0.2, 0.5], u_c^T1Gd ∈ [0.5, 0.8]
    o['mrange']    = [0.75, 1.25]
    o['rDrange']   = [0.75, 1.25]
    o['th2range']  = [0.5,  0.8 ]

    # Parameters trainable in fine-tuning
    o['trainD']   = True    # μ_D
    o['trainRHO'] = True    # μ_R
    o['trainM']   = False
    o['trainm']   = False
    o['trainA']   = False
    o['trainx0']  = True    # x0 (tumor initial location)
    o['trainth1'] = True    # u_c^FLAIR
    o['trainth2'] = True    # u_c^T1Gd

    # Smoothed Heaviside (paper uses sigmoid with a=20)
    o['heaviside']    = 'sigmoid'
    o['smoothwidth']  = 20

    # Monitor patient-data loss
    o['earlystop_opts']['patience'] = patience
    o['earlystop_opts']['monitor']  = ['pdattest']
    o['earlystop_opts']['burnin']   = 500

    o['initfromdata'] = True
    o['ictransform']  = True
    o['udatsource']   = 'char'    # keeps using characteristic data source

    o['print_res_every'] = 50
    o['ckpt_every']      = 2000
    o['saveckpt']        = True
    o['file_log']        = True

    return o


# ==========================================================================
# Stage execution helpers
# ==========================================================================

def run_stage(opts: dict, stage_name: str) -> str:
    """
    Instantiate Gmodel with given opts and run solve().
    Returns the model_dir (checkpoint path for next stage).
    """
    os.makedirs(opts['model_dir'], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {stage_name}")
    print(f"{'='*60}")
    print(f"  model_dir  : {opts['model_dir']}")
    print(f"  mat_file   : {opts['inv_dat_file']}")
    print(f"  adam_iters : {opts['num_init_train']}")
    print(f"  lbfgs_iter : {opts['lbfgs_opts']['maxiter']}")
    print()

    # Fix tag format string so Gmodel doesn't crash on {size}
    opts['tag'] = opts['tag'].replace('{size}', '4x128')

    # Save opts before starting (for reproducibility)
    save_json(opts, os.path.join(opts['model_dir'], 'options.json'))

    # Set seeds
    tf.random.set_seed(opts['seed'])
    np.random.seed(opts['seed'])

    g = Gmodel(opts)
    g.solve()

    return opts['model_dir']


# ==========================================================================
# Post-training: evaluate and save the predicted cell density volume
# ==========================================================================

def predict_cell_density(mat_file: str,
                          finetune_dir: str,
                          output_dir: str,
                          grid_resolution: int = 64):
    """
    After fine-tuning, evaluate the PINN on a uniform spatial grid at t=1
    and save the predicted tumor cell density as a .npy file.

    grid_resolution : number of points per spatial dimension
                      (use 64 for a quick check, 128 for higher fidelity)
    """
    from scipy.io import loadmat
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Generating cell density prediction ===")
    mat  = loadmat(mat_file)

    n    = grid_resolution
    coords = np.linspace(0, 1, n, dtype=DTYPE)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing='ij')
    t_col = np.ones((n**3, 1), dtype=DTYPE)
    X_pred = np.column_stack([
        t_col,
        gx.ravel()[:, None],
        gy.ravel()[:, None],
        gz.ravel()[:, None],
    ])

    # Build minimal opts to restore model
    from options import opts as default_opts
    import copy
    opts = copy.deepcopy(default_opts)
    opts['inv_dat_file'] = mat_file
    opts['model_dir']    = finetune_dir
    opts['restore']      = finetune_dir
    opts['N']            = 1000
    opts['Ntest']        = 1000
    opts['Ndat']         = 1000
    opts['Ndattest']     = 1000
    opts['num_init_train'] = 0
    opts['lbfgs_opts']   = None
    opts['trainD'] = opts['trainRHO'] = opts['trainx0'] = False
    opts['trainth1'] = opts['trainth2'] = False
    for k in opts['weights']:
        opts['weights'][k] = None
    opts['weights']['res'] = 1.0

    tf.random.set_seed(0)
    np.random.seed(0)
    g = Gmodel(opts)

    # Evaluate in batches to avoid OOM
    batch  = 10000
    u_pred = []
    X_tf   = tf.constant(X_pred, dtype=DTYPE)
    for i in range(0, len(X_pred), batch):
        u_batch = g.model(X_tf[i:i+batch]).numpy()
        u_pred.append(u_batch)

    u_pred = np.concatenate(u_pred, axis=0).reshape(n, n, n)
    np.save(os.path.join(output_dir, 'u_pinn_pred.npy'), u_pred)
    print(f"  Saved u_pinn_pred.npy  shape={u_pred.shape}  "
          f"max={u_pred.max():.4f}")

    # Also save a summary of the learned parameters
    params = {}
    for k, v in g.param.items():
        params[k] = float(v.numpy())
    save_json(params, os.path.join(output_dir, 'learned_params.json'))
    print(f"  Learned parameters: {params}")
    return u_pred, params


# ==========================================================================
# Main training function  (the one you call from outside)
# ==========================================================================

def train(mat_file:      str,
          output_dir:    str,
          # Pre-training
          pretrain_N_res:    int   = 20000,
          pretrain_N_dat:    int   = 20000,
          pretrain_lr:       float = 1e-3,
          pretrain_adam:     int   = 20000,
          pretrain_lbfgs:    int   = 20000,
          pretrain_patience: int   = 1000,
          # Fine-tuning
          finetune_N_res:    int   = 5000,
          finetune_N_dat:    int   = 5000,
          finetune_lr:       float = 1e-4,
          finetune_adam:     int   = 5000,
          finetune_lbfgs:    int   = 5000,
          finetune_patience: int   = 500,
          # Misc
          seed:              int   = 0,
          skip_pretrain:     bool  = False,
          skip_finetune:     bool  = False,
          predict_after:     bool  = True,
          predict_res:       int   = 64):
    """
    Run the full two-stage PINN training pipeline.

    Parameters
    ----------
    mat_file     : path to the .mat file produced by preprocess_glioma.py
    output_dir   : root directory for all outputs
    pretrain_*   : hyperparameters for Stage 1 (pre-training)
    finetune_*   : hyperparameters for Stage 2 (fine-tuning)
    seed         : random seed
    skip_pretrain: if True, skip Stage 1 (must have existing checkpoint)
    skip_finetune: if True, skip Stage 2
    predict_after: if True, run prediction after fine-tuning
    predict_res  : grid resolution for prediction volume
    """
    assert os.path.exists(mat_file), f"mat_file not found: {mat_file}"
    os.makedirs(output_dir, exist_ok=True)

    pretrain_dir = os.path.join(output_dir, 'pretrain')
    finetune_dir = os.path.join(output_dir, 'finetune')

    # ------------------------------------------------------------------ #
    # Stage 1: Pre-training                                               #
    # ------------------------------------------------------------------ #
    if not skip_pretrain:
        opts_pre = build_pretrain_opts(
            mat_file   = mat_file,
            model_dir  = output_dir,
            N_res      = pretrain_N_res,
            N_dat      = pretrain_N_dat,
            lr         = pretrain_lr,
            n_adam     = pretrain_adam,
            lbfgs_iter = pretrain_lbfgs,
            patience   = pretrain_patience,
            seed       = seed,
        )
        run_stage(opts_pre, "STAGE 1 — Pre-training (PINN → ū^FDM)")
    else:
        print(f"\n[INFO] Skipping pre-training. "
              f"Using existing checkpoint in: {pretrain_dir}")

    # ------------------------------------------------------------------ #
    # Stage 2: Fine-tuning                                                #
    # ------------------------------------------------------------------ #
    if not skip_finetune:
        opts_ft = build_finetune_opts(
            mat_file     = mat_file,
            model_dir    = output_dir,
            pretrain_dir = pretrain_dir,
            N_res        = finetune_N_res,
            N_dat        = finetune_N_dat,
            lr           = finetune_lr,
            n_adam       = finetune_adam,
            lbfgs_iter   = finetune_lbfgs,
            patience     = finetune_patience,
            seed         = seed,
        )
        run_stage(opts_ft, "STAGE 2 — Fine-tuning (patient parameters)")
    else:
        print(f"\n[INFO] Skipping fine-tuning.")

    # ------------------------------------------------------------------ #
    # Post-training: predict and save cell density volume                 #
    # ------------------------------------------------------------------ #
    if predict_after and not skip_finetune:
        u_pred, params = predict_cell_density(
            mat_file     = mat_file,
            finetune_dir = finetune_dir,
            output_dir   = output_dir,
            grid_resolution = predict_res,
        )
        print(f"\n✓ Training complete.")
        print(f"  Outputs saved in: {output_dir}")
        print(f"  Learned μ_D = {params.get('rD', '?'):.4f}, "
              f"μ_R = {params.get('rRHO', '?'):.4f}")
        return u_pred, params

    print(f"\n✓ Training complete. Outputs in: {output_dir}")
    return None, None


# ==========================================================================
# Quick smoke-test mode
# ==========================================================================

def run_smalltest(mat_file: str, output_dir: str):
    """
    Tiny run (100 Adam steps, 20 L-BFGS steps) to verify the pipeline
    end-to-end without waiting for real training to finish.
    """
    print("\n[SMALLTEST MODE] Running minimal training for pipeline check ...")
    train(
        mat_file       = mat_file,
        output_dir     = output_dir,
        pretrain_N_res = 64,
        pretrain_N_dat = 64,
        pretrain_adam  = 100,
        pretrain_lbfgs = 20,
        pretrain_patience = 50,
        finetune_N_res = 64,
        finetune_N_dat = 64,
        finetune_adam  = 100,
        finetune_lbfgs = 20,
        finetune_patience = 50,
        predict_after  = True,
        predict_res    = 16,
    )


# ==========================================================================
# CLI entry point
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-stage PINN training for GBM infiltration prediction.")
    p.add_argument('--mat',     required=True,
                   help='Path to .mat file from preprocess_glioma.py')
    p.add_argument('--out',     required=True,
                   help='Output directory for checkpoints and predictions')

    # Pre-training
    g1 = p.add_argument_group('Pre-training')
    g1.add_argument('--pre_res',     type=int,   default=20000)
    g1.add_argument('--pre_dat',     type=int,   default=20000)
    g1.add_argument('--pre_lr',      type=float, default=1e-3)
    g1.add_argument('--pre_adam',    type=int,   default=20000)
    g1.add_argument('--pre_lbfgs',   type=int,   default=20000)
    g1.add_argument('--pre_patience',type=int,   default=1000)

    # Fine-tuning
    g2 = p.add_argument_group('Fine-tuning')
    g2.add_argument('--ft_res',      type=int,   default=5000)
    g2.add_argument('--ft_dat',      type=int,   default=5000)
    g2.add_argument('--ft_lr',       type=float, default=1e-4)
    g2.add_argument('--ft_adam',     type=int,   default=5000)
    g2.add_argument('--ft_lbfgs',    type=int,   default=5000)
    g2.add_argument('--ft_patience', type=int,   default=500)

    # Misc
    p.add_argument('--seed',          type=int, default=0)
    p.add_argument('--skip_pretrain', action='store_true')
    p.add_argument('--skip_finetune', action='store_true')
    p.add_argument('--no_predict',    action='store_true')
    p.add_argument('--predict_res',   type=int, default=64)
    p.add_argument('--smalltest',     action='store_true',
                   help='Tiny run to check the pipeline works end-to-end')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.smalltest:
        run_smalltest(args.mat, args.out)
    else:
        train(
            mat_file           = args.mat,
            output_dir         = args.out,
            pretrain_N_res     = args.pre_res,
            pretrain_N_dat     = args.pre_dat,
            pretrain_lr        = args.pre_lr,
            pretrain_adam      = args.pre_adam,
            pretrain_lbfgs     = args.pre_lbfgs,
            pretrain_patience  = args.pre_patience,
            finetune_N_res     = args.ft_res,
            finetune_N_dat     = args.ft_dat,
            finetune_lr        = args.ft_lr,
            finetune_adam      = args.ft_adam,
            finetune_lbfgs     = args.ft_lbfgs,
            finetune_patience  = args.ft_patience,
            seed               = args.seed,
            skip_pretrain      = args.skip_pretrain,
            skip_finetune      = args.skip_finetune,
            predict_after      = not args.no_predict,
            predict_res        = args.predict_res,
        )