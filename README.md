# Programming Assignment 1 — Reinforcement Learning

Assignment work done for the **DA6400** (Reinforcement Learning) course of my 8th Semester at IIT Madras.

Python version: **3.14.2**

## Setup

```bash
conda env create -f environment.yml
conda activate pa1_rl
```

## Running the Code

All experiments are in present within the `src/` directory - with one file for each scenario (Gridworld and Acrobot). The `__main__` block within these files is organized into clearly labeled sections. Uncomment the section for the question you want to run, and comment out the rest.

```bash
cd src/
python problem2_acrobot.py
```

Output plots are saved as `.png` files in the working directory.

## File Structure

```text
.
├── outputs/                # Generated plots
├── src/
│   ├── problem1_mdp.py
│   └── problem2_acrobot.py
├── PA1_RL.pdf              # Assignment details
├── README.md               # Overview and instructions (this file)
├── environment.yml         # Conda environment specification
├── relevant_raw_outs.txt   # Terminal outputs from experiment runs
└── report.md               # Report with results and analysis
```
