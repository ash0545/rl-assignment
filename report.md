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

**3. Modifying the Discount Factor ($\gamma = 0.3$)**

- **(a) What are the resulting optimal value function and policy? Explain the difference.**

The optimal value function drops drastically compared to $\gamma = 0.95$, with most states far from the goal evaluating to negative values (~ -1.4). A discount factor of $\gamma = 0.3$ makes the drone extremely myopic (short-sighted). The distant +100 reward is discounted so heavily that it cannot outweigh the immediate accumulation of -1 step penalties, causing the drone to abandon the main objective if it is too far away.

- **(b) What happens near the hazardous states when $\gamma = 0.3$? Can you explain the behavior?**

Near hazardous states, the drone adopts a purely evasive or stationary policy, such as moving away from the smoke or hovering. Because the drone is short-sighted, it prioritizes avoiding the massive immediate penalties of smoke (-11) or boulders (-100). It lacks the long-term incentive required to risk the 10% chance of the wind blowing it into a hazard while trying to navigate past it.

- **(c) Is there any case in the two MDPs where the drone prefers doing nothing (hovering) despite the per-step penalty? Explain either ways.**

Yes. In the $\gamma = 0.3$ MDP, the drone chooses to hover in cells completely surrounded by hazards (e.g., cell `(2, 2)` between two smoke cells, or `(3, 3)` near smoke and a boulder). Hovering is deterministic and unaffected by wind. The myopic drone prefers the guaranteed -1 hover penalty over taking a directional step that carries a 10% risk of being blown into the boulder. The $\gamma = 0.95$ drone never hovers (except at the lake) since doing so delays reaching the +100 goal.

- **(d) Are there any interesting states where the drone behaves differently in the two MDPs? Explain.**

Cells `(1, 3)` and `(0, 2)` during both phases are prime examples. Considering phase 1 with $\gamma = 0.95$, the drone at `(1, 3)` moves North to efficiently route around the smoke and reach the lake. With $\gamma = 0.3$, the drone at `(1, 3)` moves East—actively running away from the adjacent smoke at `(1, 2)` and burying itself in the corner, completely abandoning the lake because immediate survival outweighs the heavily discounted goal. This "avoidance" behaviour is present at `(0, 2)` as well.
