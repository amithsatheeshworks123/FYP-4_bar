# Four-Bar Linkage Synthesizer

A web-based tool for designing and optimizing planar four-bar linkage mechanisms to generate specific coupler path trajectories. Combines classical kinematic analysis with multiple modern optimization algorithms.

## Overview

Given a desired path (straight line, ellipse, kidney curve, teardrop, circle, or a custom user-drawn shape), this tool finds the four-bar linkage dimensions that best reproduce that trajectory. It supports four optimization strategies:

- **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy (most robust)
- **CEM** — Cross-Entropy Method (fast, good exploration)
- **PPO (Bandit)** — Single-step policy gradient optimization
- **PPO (Sequential)** — 6-step MDP that designs one parameter per step

## Features

- Interactive web UI with real-time linkage animation
- Rocker-vs-crank angle relationship graph
- Optimization convergence chart
- Lock individual link lengths to constrain the search space
- Custom trajectory drawing mode
- Grashof criterion enforcement for crank-rocker feasibility
- Pre-trained PPO checkpoint for warm-start inference

## Project Structure

```
FYP-4_bar/
├── app.py                  # Flask server with 9 API endpoints
├── path_kinematics.py      # Forward/inverse kinematics & reward function
├── target_trajectory.py    # Trajectory shape generators (6 types)
├── optimizer.py            # CMA-ES wrapper (pycma)
├── cem.py                  # Cross-Entropy Method implementation
├── ppo.py                  # Bandit PPO (single-step optimization)
├── ppo_sequential.py       # Sequential MDP PPO (6-step design)
├── sequential_env.py       # Gym-like 6-step environment
├── path_env.py             # Stateless environment wrapper
├── ui.py                   # Legacy Matplotlib desktop UI
├── ppo_checkpoint.pt       # Pre-trained PPO model weights
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web UI template
└── static/
    ├── css/style.css
    └── js/main.js
```

## Design Parameters

Each mechanism is described by six parameters:

| Parameter | Description | Range |
|-----------|-------------|-------|
| `L1` | Ground link length | 0.05 – 0.4 m |
| `L2` | Crank link length | 0.05 – 0.4 m |
| `L3` | Coupler link length | 0.05 – 0.4 m |
| `L4` | Rocker link length | 0.05 – 0.4 m |
| `xO2` | Crank pivot X offset | -0.1 – 0.2 m |
| `yO2` | Crank pivot Y offset | -0.1 – 0.2 m |

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies:

```
numpy
matplotlib
torch
sympy>=1.12,<2
flask
cma
```

## Installation

```bash
git clone https://github.com/amithsatheeshworks123/fyp-4_bar.git
cd fyp-4_bar
pip install -r requirements.txt
```

## Running the Application

### Web UI (recommended)

```bash
python app.py
```

Open your browser at `http://localhost:5000`.

### Legacy Desktop UI

```bash
python ui.py
```

## Usage

1. **Select a trajectory type** from the dropdown (straight line, ellipse, kidney, teardrop, circle, or custom).
2. **Set trajectory parameters** (e.g., ellipse semi-axes, line position).
3. Optionally **lock specific link lengths** to hold them fixed during optimization.
4. **Choose an optimizer** and click the corresponding button.
5. Watch the **real-time animation** and **convergence chart** update as the optimizer runs.
6. Inspect the final linkage geometry and the reward metric.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the web UI |
| `GET` | `/api/trajectories` | List available trajectory types |
| `POST` | `/api/trajectory/preview` | Preview a trajectory |
| `POST` | `/api/metrics` | Compute coupler path & metrics for given params |
| `POST` | `/api/optimize/cma` | Run CMA-ES optimization |
| `POST` | `/api/optimize/cem` | Run CEM optimization |
| `POST` | `/api/optimize/ppo` | Run bandit PPO optimization |
| `POST` | `/api/optimize/ppo_extended` | Run extended PPO training |
| `POST` | `/api/optimize/ppo_sequential` | Run sequential MDP PPO |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask |
| Frontend | HTML5, CSS3, JavaScript |
| Charts | Chart.js 4.4.0 |
| Kinematics | NumPy |
| Optimization | pycma, custom CEM |
| Reinforcement Learning | PyTorch (PPO) |
| Symbolic Math | SymPy |
