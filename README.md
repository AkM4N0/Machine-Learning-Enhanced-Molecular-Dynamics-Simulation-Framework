# Machine Learning-Enhanced Molecular Dynamics Simulation Framework

This repository contains a code-only snapshot of a research workflow for predicting interaction forces between anisotropic nanoparticles with a combination of molecular dynamics (MD) simulation and machine learning. The project focuses on spherocylindrical gold nanoparticles and aims to bridge the gap between atomistic simulation and efficient mesoscale force prediction.

Instead of relying on full atomic-coordinate machine learning interatomic potentials, this framework uses a compact mesoscale representation based on particle geometry, relative distance, and quaternion-based orientation. The goal is to directly predict the instantaneous 3D interaction force vector between two nanoparticles while keeping the pipeline physically interpretable and computationally efficient.

## Project Overview

The full workflow combines three tightly connected stages:

1. Molecular dynamics data generation with configurable LAMMPS-based simulation scripts.
2. Geometry and orientation extraction, including PCA-based shape analysis and quaternion encoding.
3. Machine learning training and evaluation for direct force-vector prediction.

According to the project documents, the study was designed around several core ideas:

- direct prediction of 3D force vectors rather than only energy-derived forces
- quaternion-based orientation encoding for stable anisotropic representation
- physically guided preprocessing to handle severe zero-force / non-zero-force imbalance
- progressive model development from baseline regression to temporally aware and multitask architectures

## Repository Structure

- `6.0/`: MD simulation and analysis scripts, including configuration-driven LAMMPS workflows and nanoparticle analysis utilities
- `data_output/`: code related to nanoparticle geometry construction and pair-building utilities
- `lammp_training/`: machine learning training scripts, notebooks, and model development code

This repository currently keeps the original relative paths of those three code folders while excluding large generated outputs and heavy simulation artifacts.

## Technical Highlights

Based on the accompanying documents, the framework includes:

- a YAML-configured MD pipeline with force logging, trajectory output, and timestamped run directories
- DBSCAN-based particle identification and PCA-based principal-axis extraction
- quaternion correction and temporal sign-consistency handling
- dataset generation across multiple diameter/length ratios and relative orientations
- physically aware sampling, normalization, and sequence construction
- multiple learning architectures, including baseline MLP, calibrated multi-output MLP, GRU-based residual modeling, and MT-QuatForceNet

The project is motivated by the need to model force behavior in anisotropic nanoparticle systems more efficiently than repeated atomistic simulations, while still preserving physically meaningful descriptors such as geometry and orientation.

## My Contributions

Based on the materials in the project documents, my contributions in this work include:

- designing and implementing the MD simulation framework for controlled nanoparticle approach, rotation, logging, and dataset generation
- building a reproducible configuration-driven workflow around LAMMPS for geometry setup, force extraction, and batch simulation control
- developing the geometry-analysis pipeline, including cluster identification, PCA-based descriptors, and quaternion-based orientation representation
- constructing the force-learning dataset across multiple nanoparticle aspect ratios and orientation configurations
- designing the preprocessing strategy to correct quaternion continuity, rebalance sparse force labels, normalize heterogeneous features, and construct temporal training windows
- implementing and comparing multiple machine learning models for force prediction, from baseline MLP models to GRU-based and multitask quaternion-force architectures
- organizing the project as a reusable simulation-to-learning pipeline aimed at scalable coarse-grained force prediction for nanoparticle systems

## Research Direction

The broader goal of this project is to support fast, data-driven estimation of nanoscale interaction forces without repeatedly running expensive atomistic simulations. In the longer term, the same workflow could be extended to:

- richer force maps with better coverage of asymmetric and high-gradient interaction regions
- more nanoparticle shapes, aspect ratios, and materials
- multi-particle interaction databases for larger nanoscale assemblies

## Notes

- This repository currently contains the code portion of the project only.
- Large simulation outputs, trajectories, checkpoints, and other generated artifacts are intentionally excluded from version control.

