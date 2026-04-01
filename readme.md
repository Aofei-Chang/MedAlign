# MedAlign: Alignment Distillation for Medical LVLMs

**Aofei Chang, Ting Wang, Fenglong Ma**

<p align="center">
  <a href="https://arxiv.org/abs/2512.18554">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg">
  </a>
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/39079">
    <img src="https://img.shields.io/badge/AAAI-2026-blue.svg">
  </a>
</p>

This repository provides the implementation of **MedAlign**, a lightweight distillation framework that transfers visual representation and attention alignment knowledge from expert medical CLIP models (e.g., UniMed-CLIP, BiomedCLIP) to existing Medical Large Vision-Language Models (Med-LVLMs), such as **HuatuoGPT-Vision** and **LLaVA-Med-1.5**.

We include all code and resources necessary for generating distillation data, training models, and running evaluation on medical VQA and report generation benchmarks.

---

## 📁 Project Structure

### 🔄 Distillation Data Generation

These folders contain the scripts to generate visual alignment and attention distillation targets from expert CLIP models:

* **`Distill_data_biomed/`**
  Distillation data generation using **BiomedCLIP** (ViT-B/16, 224px).

* **`Distill_data_unimed_B/`**
  Distillation data generation using **UniMed-CLIP** (ViT-B/16, 224px).

* **`Distill_data_unimed_L/`**
  Distillation data generation using **UniMed-CLIP** (ViT-L/14, 336px).

Each folder includes preprocessing scripts to extract image token features and attention maps, as well as interpolation utilities for aligning token resolution with Med-LVLMs.

---

### 🧠 Med-LVLM Implementations

#### `huatuoGPT_V/`

Codebase for **HuatuoGPT-Vision-7B** with alignment distillation:

* **`scripts/`**
  Example training scripts for:

  * Medical VQA
  * Report generation

* **`llava/`**
  Modified source code with alignment distillation support:

  * `model_wrapper.py`
    Wrappers implementing visual feature alignment and attention distillation modules.
  * `moe_llava.py`
    Mixture-of-Experts (MoE) implementation for adaptive Query matrix tuning in attention modules (detailed in the Appendix).

* **`eval/`**
  Evaluation and inference utilities:

  * `model_vqa.py`: inference wrapper for VQA
  * `run_eval.py`: evaluation runner for automatic metrics
  * Baseline implementations are also included for comparative analysis.

#### `llava-med-1.5/`

Implementation for **LLaVA-Med-1.5** using the same code structure as `huatuoGPT_V`. Supports both report generation and VQA tasks.

---

### 🧩 CLIP Models

* **`open_clip/`**
  CLIP model source codes, including the hooks implementation to extract prompt-aware attention maps.

---

## 📋 Training

Instructions on environment setup, dataset preparation, and training can be found in the `scripts/` folder of each model directory.

For generating distillation targets from a CLIP model, refer to the corresponding `Distill_data_*` folder and run the provided codes.

---

## 📌 Notes

* We use bilinear interpolation for resizing token-level outputs (attention maps, visual features) from expert CLIP to match the target Med-LVLM token resolution.


### 📚 Reference

If you use MedALIGN in your research, please cite:

```bibtex
@inproceedings{chang2026enhancing,
  title={Enhancing Medical Large Vision-Language Models via Alignment Distillation},
  author={Chang, Aofei and Wang, Ting and Ma, Fenglong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={24},
  pages={19952--19960},
  year={2026}
}
```

