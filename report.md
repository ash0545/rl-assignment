# Report - Programming Assignment 1

## Problem 1: MDP Simulation

**1. MDP Formulation:**
The problem is modeled as a Markov Decision Process defined by the tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$.

- **State Space ($\mathcal{S}$):** The state is defined by the drone's position $(x, y)$ on the $5 \times 5$ grid and its water status $w \in \{0, 1\}$ (where $0$ is empty, $1$ is filled).
    $$\mathcal{S} = \{(x, y, w) \mid x, y \in \{0, 1, 2, 3, 4\}, w \in \{0, 1\}\}$$
    The state space is discrete and contains $|\mathcal{S}| = 5 \times 5 \times 2 = 50$ states. Terminal states are boulder cells $(2, 4, w)$ and $(3, 4, w)$, and the fire zone with water $(4, 4, 1)$.

- **Action Space ($\mathcal{A}$):** The discrete action space consists of five possible moves.
    $$\mathcal{A} = \{\text{North}, \text{South}, \text{East}, \text{West}, \text{Hover}\}$$

- **Transition Probabilities ($p(s' | s, a)$):** The probability of transitioning to state $s'$ given state $s$ and action $a$. Let $\Delta(a)$ be the intended movement direction and $\Delta(a_\perp)$ be the perpendicular directions.
  - *Normal cells:* $p(s+\Delta(a)|s,a) = 0.7$, $p(s+\Delta(a_\perp)|s,a) = 0.1$ each, $p(s|s,a) = 0.1$.
  - *Smoke cells:* $p(s+\Delta(a)|s,a) = 0.4$, $p(s+\Delta(a_\perp)|s,a) = 0.1$ each, $p(s|s,a) = 0.4$.
  - *Hover:* $p(s|s,\text{Hover}) = 1.0$.
  - *Water pickup:* Entering $(0,0)$ with $w=0$ deterministically transitions to $(0,0)$ with $w=1$.
  - *Boundaries:* Any transition leading off the grid results in remaining in the current cell.

- **Reward Function ($r(s, a, s')$):** The expected immediate reward on transition.
  - $r(s, a, s') = 100$ if $s' = (4, 4, 1)$ (Fire zone with water)
  - $r(s, a, s') = -100$ if $s' \in \{(2, 4, w), (3, 4, w)\}$ (Boulders)
  - $r(s, a, s') = -11$ if $s' \in \{(1, 2, w), (3, 2, w)\}$ (Smoke: $-1$ step penalty + $-10$ hazard)
  - $r(s, a, s') = -1$ otherwise (Standard step penalty)

- **Discount Factor ($\gamma$):** $\gamma = 0.95$

**2. Value Iteration and Optimal Policy:**
The optimal value function $v_*(s)$ is found by iteratively applying the Bellman Optimality Equation update rule:
$$V_{k+1}(s) = \max_a \sum_{s', r} p(s', r|s,a) [r + \gamma V_k(s')]$$

Once the values converge (i.e., the maximum change across all states is less than a small threshold $\theta$), the optimal deterministic policy $\pi_*(s)$ is extracted by acting greedily with respect to the optimal value function:
$$\pi_*(s) = \text{argmax}_a \sum_{s', r} p(s', r|s,a) [r + \gamma V_*(s')]$$

The required visualizations are present in the `outputs/` directory, for both phases (water_empty and water_filled).
