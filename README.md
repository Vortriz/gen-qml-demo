# Generative Quantum Machine Learning via QuDDPM

[![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_yVvwf4fXrQmjRvrcSS4K7D)

This project implements and extends the research presented in the paper [**"Generative Quantum Machine Learning via Denoising Diffusion Probabilistic Models"**](https://www.alphaxiv.org/abs/2310.05866) (Zhang et al.). It was developed for the **Bring Research to Life: molab Notebook Competition**.

## Overview

The core of this project is a **Quantum Denoising Diffusion Probabilistic Model (QuDDPM)**. It is designed to learn and generate quantum state ensembles by reversing a scrambling process.

### Key Contributions
- **Ensemble Generation**: Implementation of both **Clustered** and **Circle** ensembles, the latter being a significant challenge for standard generative models due to its non-trivial topology.
- **Interactive Bloch Sphere**: Real-time visualization of quantum states on the Bloch sphere using `qutip` and `marimo`.
- **Forward/Backward Walkthrough**: Interactive UI elements that allow users to play through the diffusion (scrambling) and denoising (generation) processes.

### Extension: Direct Targeting
A novel optimization that maps denoising steps directly to the target distribution ($t=0$) rather than the previous timestep ($t-1$), resulting in faster convergence and higher-fidelity states. The notebook includes a side-by-side comparison of the standard method versus our improved approach.

## Getting Started

### Prerequisites
This project uses [Nix](https://nixos.org/) for environment management and [uv](https://github.com/astral-sh/uv) for Python dependencies.

### Installation
1.  **Enter the Nix shell**:
    ```bash
    nix develop
    ```
2.  **Install dependencies and run the notebook**:
    ```bash
    uv run marimo edit notebook.py
    ```

## Project Structure
- `notebook.py`: The interactive marimo notebook containing the narrative, UI, and visual results.
- `lib.py`: Core implementation of the forward/backward processes, quantum circuits, and loss functions.
- `assets/`: Diagrams and static visual aids used in the notebook.

## Scientific Background

### The Diffusion Process
1.  **Forward Process (Scrambling)**: An ensemble is subjected to $T$ steps of random unitaries implemented via a Hardware Efficient Ansatz (HEA), eventually reaching a Haar-random "noise" state.
2.  **Backward Process (Denoising)**: A sequence of Parameterized Quantum Circuits (PQCs) with ancilla measurements is trained to reverse the noise.
3.  **Optimal Transport**: The model uses **Wasserstein Distance** as a loss function, which is more effective than MMD, especially for capturing the geometry of complex distributions.

## Credits
- **Paper**: [arXiv:2310.05866](https://arxiv.org/abs/2310.05866)
- **Competition**: [marimo molab competition](https://marimo.io/pages/events/notebook-competition)
- **Frameworks**: [marimo](https://marimo.io/), [TensorCircuit](https://tensorcircuit.readthedocs.io/), [QuTiP](https://qutip.org/)
