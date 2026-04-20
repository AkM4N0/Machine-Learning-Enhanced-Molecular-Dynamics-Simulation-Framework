# Machine Learning-Enhanced Molecular Dynamics Simulation Framework

## Overview

This repository presents a research-oriented framework for predicting three-dimensional interaction forces between anisotropic nanoparticles by integrating molecular dynamics (MD) simulation with machine learning.

The project focuses on spherocylindrical gold nanoparticles and aims to provide an efficient and physically interpretable alternative to expensive atomistic simulations. Instead of learning full interatomic potentials, this framework directly predicts instantaneous 3D force vectors using compact geometric and orientation-based representations.

The core idea is to bridge simulation-based physics modeling and data-driven learning for scalable nanoscale interaction prediction.

---

## Authorship and Ownership

This repository was independently developed and maintained by me as part of my research work on machine learning-enhanced molecular dynamics.

All core components — including simulation workflow design, data generation, feature engineering, and model implementation — were designed and implemented by me.

This repository reflects my original research contributions and provides a complete pipeline from simulation to machine learning-based force prediction.

---

## Research Contribution

The main contributions of this work include:

- Designing and implementing a configurable MD simulation framework for controlled nanoparticle interaction, including translation, rotation, and force logging
- Building a reproducible simulation-to-data pipeline using LAMMPS with structured configuration management
- Developing a geometry analysis pipeline using clustering (DBSCAN) and PCA-based principal axis extraction
- Introducing quaternion-based orientation encoding with temporal consistency correction
- Constructing a large-scale dataset across multiple nanoparticle aspect ratios and interaction configurations
- Designing preprocessing strategies to address force sparsity, feature normalization, and temporal sequence construction
- Implementing and evaluating multiple machine learning models, including:
  - baseline MLP models
  - calibrated multi-output regression models
  - GRU-based temporal models
  - multi-task quaternion-force learning architecture (MT-QuatForceNet)
- Establishing a scalable workflow for coarse-grained force prediction in anisotropic nanoparticle systems

---

## Methodology

The framework consists of three main stages:

### 1. Molecular Dynamics Simulation
- LAMMPS-based simulation of nanoparticle interactions
- Controlled approach and rotation processes
- Force extraction and trajectory recording
- Configurable batch simulation pipeline

### 2. Geometry and Feature Extraction
- Particle identification using DBSCAN clustering
- Shape representation using PCA-based principal axes
- Quaternion-based orientation encoding
- Temporal consistency correction for orientation signals

### 3. Machine Learning Modeling
- Dataset construction across varying geometric configurations
- Feature normalization and imbalance handling
- Model training using multiple architectures
- Evaluation of force prediction accuracy and generalization

---

## Reproducibility

The repository is organized to support reproducible research.

To reproduce the workflow:

1. Run MD simulations using scripts in `6.0/`
2. Extract geometric and orientation features using `data_output/`
3. Train and evaluate models using `lammp_training/`

All modules are designed to be configurable and reusable.

---

## Repository Structure

- `6.0/`: MD simulation and analysis scripts, including configuration-driven LAMMPS workflows and nanoparticle analysis utilities
- `data_output/`: code related to nanoparticle geometry construction and pair-building utilities
- `lammp_training/`: machine learning training scripts, notebooks, and model development code

This repository currently keeps the original relative paths of those three code folders while excluding large generated outputs and heavy simulation artifacts.

---

## Related Work

This repository supports the research described in the manuscript:

**"Direct Prediction of Three-Dimensional Interaction Forces Between Anisotropic Nanoparticles Based on Quaternion Encoding and Multi-Task Deep Learning"**

(Currently under submission)

---

## Research Motivation

This work is motivated by the need to reduce the computational cost of nanoscale interaction modeling while preserving physically meaningful representations.

Compared to traditional atomistic simulations, this framework enables:

- faster force prediction
- interpretable geometric representation
- scalable modeling for larger systems

---

## Future Work

Potential extensions include:

- expanding force prediction coverage in complex interaction regions
- incorporating additional nanoparticle geometries and materials
- extending to multi-particle interaction systems
- integrating reinforcement learning for adaptive simulation control

---

## Notes

- This repository contains the code component of the research workflow
- Simulation outputs and large datasets are not included

