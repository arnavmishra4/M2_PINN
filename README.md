<div align="center">

# 🧠 M2 — Fisher-KPP Physics-Informed Neural Network

**Biophysical Growth Predictor · Part of [NeuroSight](../)**

*End-to-end clinical AI for GBM treatment monitoring and early detection*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange?style=flat-square)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](../LICENSE)

</div>



This repository contains the code the paper 

**Personalized Predictions of Glioblastoma Infiltration: Mathematical Models, Physics-Informed Neural Networks and Multimodal Scans**

*Ray Zirui Zhang, Ivan Ezhov, Michal Balcerak, Andy Zhu, Benedikt Wiestler, Bjoern Menze, John Lowengrub.*

Medical Image Analysis, 2025. [[Journal](https://www.sciencedirect.com/science/article/pii/S1361841524003487)] [[arXiv](https://arxiv.org/abs/2311.16536)]



![Overview](overview.png)


## Citation
If you find this code useful in your research, please consider citing:

```
Zhang, R.Z., Ezhov, I., Balcerak, M., Zhu, A., Wiestler, B., Menze, B., Lowengrub, J.S., 2025. Personalized predictions of Glioblastoma infiltration: Mathematical models, Physics-Informed Neural Networks and multimodal scans. Medical Image Analysis 101, 103423. https://doi.org/10.1016/j.media.2024.103423


@article{ZHANG2025103423,
title = {Personalized predictions of Glioblastoma infiltration: Mathematical models, Physics-Informed Neural Networks and multimodal scans},
journal = {Medical Image Analysis},
volume = {101},
pages = {103423},
year = {2025},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2024.103423},
url = {https://www.sciencedirect.com/science/article/pii/S1361841524003487},
author = {Ray Zirui Zhang and Ivan Ezhov and Michal Balcerak and Andy Zhu and Benedikt Wiestler and Bjoern Menze and John S. Lowengrub},
}
```



## Dataset and Simulations

[Dataset and example scripts](https://drive.google.com/drive/folders/1vizr-eytL2EBhO2KuQrpzsn3McnvwLWT?usp=sharing)

Patient data P1-P8 in the paper is obtained from

*Lipkova et al., Personalized Radiotherapy Design for Glioblastoma Using Mathematical Tumor Modelling, Multimodal Scans and Bayesian Inference. IEEE Transactions on Medical Imaging (2019)* [[Paper]](https://ieeexplore.ieee.org/document/8654016) [[GitHub&Data]](https://github.com/JanaLipkova/GliomaSolver).



---

## Why Physics Decodes Glioblastoma Dynamics

Structural MRI segmentation (M1) outlines visible tumor borders — but captures only a static snapshot. Glioblastoma is highly infiltrative: malignant cells reside far beyond what any standard scan can resolve.

M2 bridges that gap by fitting the **Fisher-Kolmogorov-Petrovsky-Piskunov (Fisher-KPP) partial differential equation** to each patient's imaging data. The result is a set of patient-specific biophysical parameters that reveal how aggressively a tumor is proliferating versus invading surrounding tissue.

When a follow-up scan becomes available, M2 tracks the **shifts** (Δ-deltas) in these parameters across time. This biophysical signature is the primary input used by M3 to distinguish true tumor progression from treatment-induced pseudoprogression — a distinction that standard volumetric comparison cannot reliably make.

---

## At a Glance

| Property | Detail |
|---|---|
| **Task** | Patient-specific biophysical parameter estimation and tumor cell density modeling |
| **Input** | M1 3D segmentation mask + computed R_T1Gd and R_FLAIR radii |
| **Output** | Non-dimensional parameters (μ_D, μ_R, γ), 3D cell density map u(x,t), infiltration front |
| **Architecture** | Physics-Informed Neural Network (PINN) with spatial boundary constraints |
| **Core Equation** | Fisher-KPP PDE |
| **Runtime** | ~30 min per patient volume (GPU) |
| **Framework** | TensorFlow |
| **Upstream** | Adapted from Zhang et al., *Medical Image Analysis*, 2024 · [`Rayzhangzirui/pinn`](https://github.com/Rayzhangzirui/pinn) |

---

## Architecture & Pipeline

```
M1 Segmentation Mask + Radii (R_T1Gd, R_FLAIR)
  │
  ├─ Step 1: Radii Ratio Grid Search
  │     └─ Estimates characteristic parameters (D̄/ρ̄, L̄) from spatial tumor boundaries
  │
  ├─ Step 2: FDM Pre-Training
  │     └─ PINN pre-trained on Finite Difference Method characteristic solutions (~30 min)
  │
  ├─ Step 3: Patient-Specific Fine-Tuning
  │     └─ Fine-tunes on M1 segmentation boundaries as direct supervision signal
  │
  └─ Output
        ├─ μ_D  — Normalized diffusion / invasion ratio
        ├─ μ_R  — Normalized proliferation / growth ratio
        ├─ γ    — Go-Grow Index (μ_R / μ_D)
        └─ u(x,t) — Full 3D continuous tumor cell density map
```

### Key Parameters

**μ_D (Diffusion Ratio)** — Non-dimensional index of invasion-dominant spread along brain tissue. Learned within the physics-consistent range [0.75, 1.25].

**μ_R (Proliferation Ratio)** — Non-dimensional index of cellular reproduction and growth rate. Bounded symmetrically with μ_D.

**γ (Go-Grow Index)** — Defined as μ_R / μ_D. Captures the balance between a tumor's migratory drive and its replicative drive as a single scalar.

---

## Execution Logic

M2 runs independently on each longitudinal scan. Its downstream behavior differs by timeline stage.

### Scan 1 — Baseline

- Fits μ_D, μ_R, γ from the initial imaging boundaries
- Solves the PDE forward to generate a 3D predicted density map for the next interval
- Persists parameters, timestamp, and density map into LangGraph patient state

### Scan 2 — Delta Computation

- Re-estimates parameters from the follow-up MRI volume
- Retrieves Scan 1 parameters from state and computes Δμ_D, Δμ_R, Δγ
- Passes deltas to M3 as primary classification features

> **Design note on temporal modeling:** The PDE operates in normalized, non-dimensional time. Absolute forward-time spatial predictions are not used for classification because the end-time t_end cannot be reliably anchored. The parameter deltas themselves carry the biological signal — this is an explicit architectural choice, not a limitation workaround.

---

## Biophysical Delta Signals

The parameter shifts between Scan 1 and Scan 2 carry specific clinical meaning consumed by the M3 classifier and the NeuroBio Agent.

| Pattern | Biological Interpretation | Expected Action |
|---|---|---|
| **μ_R ↓, μ_D ↓** | True treatment response — both proliferation and invasion suppressed | Continue current protocol |
| **μ_R ↓, μ_D ↑** | Dangerous infiltrative spread — cells dispersing outward rather than dying | Urgent flag; high recurrence probability |
| **μ_R ↑, μ_D stable** | Proliferation-dominant true progression; therapeutic resistance | Treatment failing — initiate reassessment |
| **μ_R stable, μ_D ↑** | Invasion-dominant progression along white matter tracts | Close monitoring of affected subregions |

---

## Integration Within NeuroSight

```
M1 (3D Res-U-Net)
  └─ Segmentation mask + radii
       │
       ▼
    M2 (Fisher-KPP PINN)  ◄── this repo
       └─ μ_D, μ_R, γ, Δ-deltas, u(x,t)
            │
            ▼
         M3 (Progression Classifier)
            └─ 4-class diagnosis
                 │
                 ▼
              M4 (Clinical RAG)
                 └─ Structured clinical report

         NeuroBio Agent reads M2 raw values + deltas
         to drive live PubMed / bioRxiv hypothesis searches
```

---

## Engineering Notes

**Convergence gate** — The LangGraph agent validates that optimized μ_D and μ_R fall within [0.75, 1.25] before passing outputs downstream. Diverged runs are caught at this node.

**Fallback routing** — If the PINN fails to converge within tolerance, the agent automatically routes to radiomics-only features for M3, injects an uncertainty flag into the clinical report, and signals the NeuroBio Agent to adjust its hypothesis strategy. The pipeline does not crash.

**Compute** — PDE grid search + iterative optimization requires ~30 minutes per patient volume on standard GPU. Plan accordingly for batch inference.

---

## Limitations

- **Simplified biology** — Fisher-KPP models diffusion and proliferation only. Mass effect, necrosis, and antiangiogenic therapeutic responses are not represented.
- **Temporal non-dimensionality** — Absolute forward-time spatial boundaries cannot be perfectly matched; this is a known property of the normalized PDE formulation, not a convergence failure.
- **GBM-specific** — Parameters and constraints are calibrated for GBM dynamics. Applying this model to other tumor types requires re-engineering the underlying PDE and re-validating bounds.

---

## Repository Structure

```
m2_pinn/
├── train_pinn.py      # Pre-training and fine-tuning execution loop
├── pinn_solver.py     # Neural network layers embedding Fisher-KPP constraints
├── grid_search.py     # Radii-ratio grid search for characteristic parameter initialization
├── data_prep.py       # Preprocessing — extracts clinical radii from M1 masks
└── README.md
```

---

## References

- Zhang, Z. et al. "Biophysical parameter estimation using physics-informed neural networks." *Medical Image Analysis*, 2024. · [GitHub](https://github.com/Rayzhangzirui/pinn)
- *NeuroSight System Master Architecture and Build Documentation*, 2025 (`NeuroSight_Pleiades.docx`)

---
<div align="center">

**NeuroSight Pipeline**

[M1 — 3D Res-U-Net](https://github.com/arnavmishra4/M1-Segmentation) · **M2 — Fisher-KPP PINN (this repo)** · [M3 — Progression Classifier](https://github.com/arnavmishra4/M3-Classifier-) · [M4 — Clinical RAG](https://github.com/arnavmishra4/M4-Report-LLM-RAG-) · [M5 — cfDNA Classifier](https://github.com/arnavmishra4/M5-CfDNA)

*Orchestrated by the NeuroBio Agent*

</div>
