# Vibe Spaces for Creatively Connecting and Expressing Visual Concepts

<p align="center">
  <a href="https://huzeyann.github.io/VibeSpace-webpage/">
    <img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page">
  </a>
  <a href="https://huggingface.co/spaces/huzey/VibeSpace">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" alt="Demo">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper">
  </a>
</p>

**Authors:** [Huzheng Yang](https://huzeyann.github.io)<sup>1</sup>, [Katherine Xu](https://k8xu.github.io)<sup>1</sup>, [Andrew Lu](https://scholar.google.com/citations?user=L61d7OUAAAAJ&hl=en)<sup>1</sup>, [Michael D. Grossberg](https://crest.cuny.edu/our-team/michael-grossberg)<sup>2</sup>, [Yutong Bai](https://yutongbai.com)<sup>3</sup>, [Jianbo Shi](https://www.cis.upenn.edu/~jshi)<sup>1</sup>

<sup>1</sup>UPenn, <sup>2</sup>CUNY, <sup>3</sup>UC Berkeley

---

## Overview

<p align="center">
  <img src="webpage/static/images/figures/teaser.png" width="100%">
</p>

We introduce **Vibe Space**, a hierarchical graph manifold that learns low-dimensional geodesics in feature spaces like CLIP, enabling smooth and semantically consistent transitions between concepts.

Consider blending a musician playing a violin with one playing a guitar. Different approaches identify different relevant attributes:
- **LLMs (Gemini, GPT)** might focus on object parts or style transfer
- **Musicians** would attend to the instrument and how it is played

The intuitive process of identifying and fusing meaningful attributes—the **"vibe"**—reveals creative connections between distinct concepts.

> *The term "vibe," short for "vibration," originated in 1960s jazz slang to describe the mood or feeling conveyed by music, a person, or space.*

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Gradio Demo

```bash
python -m src.app
```

Or try the online demo: [🤗 Hugging Face Space](https://huggingface.co/spaces/huzey/VibeSpace)

### Jupyter Notebook

See `demo_vibe_blending.ipynb` for interactive examples.

---

## Capabilities

### Vibe Blending
Creates coherent hybrids that merge the relevant shared attributes between images.

<p align="center">
  <img src="webpage/static/images/figures/blending_qualitative.png" width="100%">
</p>

### Vibe Analogy
With the discovered vibe, we can extrapolate to nontrivial but related concepts, enabling creative analogies that go beyond simple interpolation.

<p align="center">
  <img src="webpage/static/images/figures/figure3.jpg" width="100%">
</p>

### Negative Vibe Control
Vibe attributes are implicitly extracted by Vibe Space. The blending pair defines desired vibes, while negative pairs define vibes to suppress. By subtracting the negative vibe, we can control which attributes are blended.

<p align="center">
  <img src="webpage/static/images/figures/neg_vibe.png" width="100%">
</p>

### Extrapolation
Vibe Space can extrapolate beyond the input images to generate related concepts by extending the vibe path.

<p align="center">
  <img src="webpage/static/images/figures/extrapolate.png" width="80%">
</p>

### Training with Extra Images
Although two images suffice to train the Vibe Space, adding related exemplars can enhance the dominant attributes and suppress spurious ones.

<p align="center">
  <img src="webpage/static/images/figures_appendix/interpolation_extra_images.png" width="80%">
</p>

### N-Image Blending
Vibe Space can blend multiple images simultaneously, discovering shared attributes across multiple concepts.

<p align="center">
  <img src="webpage/static/images/figures/dog_ram_horse.png" width="95%">
</p>

---

## Citation

```bibtex
@article{yang2025vibespace,
  title={Vibe Spaces for Creatively Connecting and Expressing Visual Concepts},
  author={Yang, Huzheng and Xu, Katherine and Lu, Andrew and Grossberg, Michael D. and Bai, Yutong and Shi, Jianbo},
  year={2025}
}
```
