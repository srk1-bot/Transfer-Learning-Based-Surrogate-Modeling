# Transfer Learning-Based Surrogate Modules for Nonlinear Seismic Response Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12.9-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation and code accompanying the research paper:

> **Transfer Learning-Based Surrogate Modeling for Nonlinear Time-History Response Analysis of High-Fidelity Structural Models**  
> Keiichi ISHIKAWA, Yuma MATSUMOTO, Taro YAOYAMA, Sangwon LEE, Tatsuya ITOI 
> Preprint: arXiv:2512.14161v2

---

## 📌 Overview

This repository provides Python modules and notebooks for transfer learning-based surrogate modeling in nonlinear time-history response analysis (NLTHA) of structures. It includes model architecture definitions, data preprocessing/wave utilities, analysis scripts for structural response evaluation, and jupyter notebooks to run the programs.

### Key Features
- **Network Models (`network/`)**: PyTorch-based neural network architecture implementations for surrogate modeling.
- **Wave Utilities (`wave_util/`)**: Utility functions for processing and generating earthquake ground motion acceleration waves.
- **Analysis Modules (`analysis/`)**: Scripts and tools for nonlinear seismic response simulations and model evaluation.

---

## 🛠️ Requirements & Environment Setup

### Prerequisites
- Python 3.12.9 or higher
- CUDA-compatible GPU (Recommended for training)

### Attention
- This repository is a work in progress. Dependencies and requirements may not be fully organized or finalized yet.