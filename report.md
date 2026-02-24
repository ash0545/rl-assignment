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

**4. Modifying Hazard Penalties and Wind Dynamics**

- **(a) What are the resulting optimal value function and policy? Explain the difference (if any) in the solution due to changing the problem in the two MDPs. Focus on the regions near hazardous cells.**
The value function drops drastically around the smoke hazards at (1, 2) and (3, 2), reflecting extreme risk. The optimal policy becomes highly risk-averse; rather than skirting the hazards as in the base case, the policy routes the drone explicitly away from them to completely avoid the 10% risk of the wind pushing it into a -91 penalty.

- **(b) Is there are scenario where hovering may be preferred? Explain either ways.**
Yes. In the Empty Water phase, cell (3, 3) prefers Hovering, with a value of exactly -20.0. Because (3, 3) is bordered by smoke at (3, 2) and a boulder at (3, 4), any directional movement carries a 10% chance of a massive penalty. The expected return of moving drops below -20, making the infinite penalty of hovering (-1 / (1 - 0.95) = -20) the optimal choice.

- **(c) Is there are scenario when a longer path to the fire zone is preferable from a particular cell? Why or why not?**
Yes. If the drone starts at (4, 4) in the Empty Water phase, the shortest path is towards the center. However, the optimal policy routes it LEFT along the bottom row. Also, at cells like (4, 2), the policy is to move DOWN. The drone intentionally drives into the bottom wall so that the intended motion keeps it safely in place, allowing the perpendicular wind to push it left or right. This longer path eliminates the risk of being blown North into the smoke.

- **(d) Given the above MDP with the −90 hazard penalty and γ = 0.95, suppose the wind becomes stronger. As a result, in all non-terminal states, the drone moves in the intended direction with a chance of 40%, in either of the perpendicular direction with a chance of 25%, and stays in same the cell rest of the times. Comment your thoughts on the solution to the resulting MDP.**
With the wind volatility increased to a 25% perpendicular push in each direction, the environment becomes too chaotic to navigate narrow safe corridors. The value function decreases grid-wide, with the drone abandoning the objective entirely from even more states (ex: cells (2, 2) and (3, 3) in both phases now default to a permanent Hover (value -20.0). The 25% risk of being blown into a hazard makes doing nothing the best choice).
