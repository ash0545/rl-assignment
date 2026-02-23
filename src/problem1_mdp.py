import numpy as np
import matplotlib.pyplot as plt
import os

# Grid world dimensions
ROWS = 5
COLS = 5
WATER_STATES = 2
NUM_STATES = ROWS * COLS * WATER_STATES

# Actions
ACTIONS = ["North", "South", "East", "West", "Hover"]
NUM_ACTIONS = len(ACTIONS)
ACTION_DELTAS = {
    "North": (-1, 0),
    "South": (1, 0),
    "East": (0, 1),
    "West": (0, -1),
    "Hover": (0, 0),
}
PERPENDICULAR_ACTIONS = {
    "North": ["East", "West"],
    "South": ["East", "West"],
    "East": ["North", "South"],
    "West": ["North", "South"],
}

# Special states
LAKE = (0, 0)
FIRE = (4, 4)
SMOKE = [(1, 2), (3, 2)]
BOULDERS = [(2, 4), (3, 4)]

# Initialize transition probabilities and rewards
P = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))
R = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))

for w in range(WATER_STATES):
    for r in range(ROWS):
        for c in range(COLS):
            s = (
                w * (ROWS * COLS) + r * COLS + c
            )  # State index, required for filling P and R

            # Terminal states => absorbing states
            if (r, c) in BOULDERS or ((r, c) == FIRE and w == 1):
                P[s, :, s] = 1.0  # ":" since all actions are to lead to the same state
                R[s, :, s] = 0.0
                continue

            for a_idx, a in enumerate(ACTIONS):
                # Probabilities are affected by wind + smoke, unless we hover
                is_smoke = (r, c) in SMOKE

                # p_intended = 0.4 if is_smoke else 0.7
                # p_perpendicular = 0.1
                # p_stay = 0.4 if is_smoke else 0.1
                # Below are the modified probabilities for Q4: Heavier Penalties, Windier MDP
                p_intended = 0.4
                p_perpendicular = 0.25
                p_stay = 0.1

                if a == "Hover":
                    p_intended = 1.0
                    p_perpendicular = 0.0
                    p_stay = 0.0

                # Generate all possible outcomes (ignoring boundaries, handled later)
                transitions = (
                    []
                )  # holds (prob, r, c), where (r,c) is where drone is trying to go

                if a == "Hover":
                    transitions.append((1.0, r, c))
                else:
                    # intending to go
                    r_int, c_int = r + ACTION_DELTAS[a][0], c + ACTION_DELTAS[a][1]
                    transitions.append((p_intended, r_int, c_int))

                    # motor break
                    transitions.append((p_stay, r, c))

                    # wind
                    for a_perp in PERPENDICULAR_ACTIONS[a]:
                        r_perp, c_perp = (
                            r + ACTION_DELTAS[a_perp][0],
                            c + ACTION_DELTAS[a_perp][1],
                        )
                        transitions.append((p_perpendicular, r_perp, c_perp))

                # Handle boundaries, water, rewards (direct assignment to matrices)
                for prob, r_new, c_new in transitions:
                    # boundaries
                    if r_new < 0 or r_new >= ROWS or c_new < 0 or c_new >= COLS:
                        r_new, c_new = r, c  # stay at the boundary

                    # fill water
                    w_new = w
                    if (r_new, c_new) == LAKE and w == 0:
                        w_new = 1

                    s_new = w_new * (ROWS * COLS) + r_new * COLS + c_new

                    P[s, a_idx, s_new] += prob

                    # rewards
                    if (r_new, c_new) == FIRE and w_new == 1:
                        R[s, a_idx, s_new] = 100
                    elif (r_new, c_new) in BOULDERS:
                        R[s, a_idx, s_new] = -100
                    elif (r_new, c_new) in SMOKE:
                        # R[s, a_idx, s_new] = -11
                        R[s, a_idx, s_new] = (
                            -91
                        )  # for Q4: Heavier Penalties, Windier MDP
                    else:
                        R[s, a_idx, s_new] = -1


# Following the algorithm given in Sutton and Barto, Pg. 83
def run_value_iteration(P, R, gamma=0.95, theta=1e-5):
    V = np.zeros(NUM_STATES)
    policy = np.zeros(NUM_STATES, dtype=int)

    print("Running Value Iteration...")
    iteration = 0
    while True:
        delta = 0.0
        # loop for each s belonging to S:
        for w in range(WATER_STATES):
            for r in range(ROWS):
                for c in range(COLS):
                    s = w * (ROWS * COLS) + r * COLS + c

                    # for handling V(terminal) = 0
                    is_terminal = False
                    if (r, c) in BOULDERS:
                        is_terminal = True
                    elif (r, c) == FIRE and w == 1:
                        is_terminal = True

                    if is_terminal:
                        continue

                    v_old = V[s]
                    action_values = np.zeros(NUM_ACTIONS)

                    # iterate over all actions to find max value
                    for a_idx in range(NUM_ACTIONS):
                        for s_new in range(NUM_STATES):
                            action_values[a_idx] += P[s, a_idx, s_new] * (
                                R[s, a_idx, s_new] + gamma * V[s_new]
                            )

                    V[s] = np.max(action_values)
                    delta = max(delta, abs(v_old - V[s]))

        iteration += 1
        if delta < theta:
            print(f"Converged in {iteration} iterations.")
            break

    # extract deterministic policy
    for w in range(WATER_STATES):
        for r in range(ROWS):
            for c in range(COLS):
                s = w * (ROWS * COLS) + r * COLS + c

                is_terminal = False
                if (r, c) in BOULDERS:
                    is_terminal = True
                elif (r, c) == FIRE and w == 1:
                    is_terminal = True

                if is_terminal:
                    continue

                action_values = np.zeros(NUM_ACTIONS)
                for a_idx in range(NUM_ACTIONS):
                    for s_new in range(NUM_STATES):
                        action_values[a_idx] += P[s, a_idx, s_new] * (
                            R[s, a_idx, s_new] + gamma * V[s_new]
                        )

                policy[s] = np.argmax(action_values)

    return V, policy


# AI generated visualization code
def visualize_phase(V, policy, w, filename):
    """Generates side-by-side plots for Value Function and Policy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    V_grid = np.zeros((ROWS, COLS))
    policy_grid = np.zeros((ROWS, COLS), dtype=int)

    for r in range(ROWS):
        for c in range(COLS):
            s = w * (ROWS * COLS) + r * COLS + c
            V_grid[r, c] = V[s]
            policy_grid[r, c] = policy[s]

    # --- Plot Value Function ---
    ax = axes[0]
    cax = ax.matshow(V_grid, cmap="coolwarm")
    fig.colorbar(cax, ax=ax)
    for r in range(ROWS):
        for c in range(COLS):
            ax.text(
                c,
                r,
                f"{V_grid[r, c]:.1f}",
                va="center",
                ha="center",
                color="black" if -50 < V_grid[r, c] < 50 else "white",
            )
    ax.set_title(f"Value Function (Water={'Filled' if w else 'Empty'})")
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Plot Policy ---
    ax = axes[1]
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_title(f"Optimal Policy (Water={'Filled' if w else 'Empty'})")
    ax.grid(color="k", linestyle="-", linewidth=1)
    ax.set_xticks(np.arange(-0.5, COLS, 1))
    ax.set_yticks(np.arange(-0.5, ROWS, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Unicode arrows: 0:North, 1:South, 2:East, 3:West, 4:Hover
    arrows = ["↑", "↓", "→", "←", "↻"]

    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in BOULDERS:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="gray"))
                ax.text(c, r, "B", va="center", ha="center", color="white")
                continue
            if (r, c) == FIRE and w == 1:
                ax.add_patch(
                    plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="orange")
                )
                ax.text(c, r, "F", va="center", ha="center", color="white")
                continue
            if (r, c) == LAKE and w == 0:
                ax.add_patch(
                    plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="lightblue")
                )
            if (r, c) in SMOKE:
                ax.add_patch(
                    plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="whitesmoke")
                )

            a = policy_grid[r, c]
            ax.text(c, r, arrows[a], va="center", ha="center", fontsize=20)

    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{filename}")
    plt.close()
    print(f"Saved visualization to outputs/{filename}")


if __name__ == "__main__":
    # print("Q1: Transitions and Rewards from (3,3), Water = 0")
    # q1_r, q1_c, q1_w = 3, 3, 0
    # s_q1 = q1_w * (ROWS * COLS) + q1_r * COLS + q1_c

    # for a_idx, a_str in enumerate(ACTIONS):
    #     print(f"Action: {a_str}")
    #     # neighbors and the cell itself
    #     for r_new in range(q1_r - 1, q1_r + 2):
    #         for c_new in range(q1_c - 1, q1_c + 2):
    #             ns = q1_w * (ROWS * COLS) + r_new * COLS + c_new
    #             prob = P[s_q1, a_idx, ns]
    #             reward = R[s_q1, a_idx, ns]
    #             print(f"  -> to ({r_new},{c_new}): P = {prob:.2f}, R = {reward}")
    #     print()

    # print("Q2: Value Iteration")
    # optimal_V, optimal_policy = run_value_iteration(P, R)

    # # The two phases constituting the MDP
    # visualize_phase(optimal_V, optimal_policy, w=0, filename="phase1_water_empty.png")
    # visualize_phase(optimal_V, optimal_policy, w=1, filename="phase2_water_filled.png")

    # print("Q3: Modified MDP setting - gamma = 0.3")
    # optimal_V, optimal_policy = run_value_iteration(P, R, gamma=0.3)
    # visualize_phase(
    #     optimal_V, optimal_policy, w=0, filename="phase1_wempty_gamma_0.3.png"
    # )
    # visualize_phase(
    #     optimal_V, optimal_policy, w=1, filename="phase2_wfilled_gamma_0.3.png"
    # )

    print("Q4: Heavier Penalties, Windier MDP")
    # hardcoded values in line 108 was changed from -11 to -91
    # another hardcoded was then applied for probabilities in lines 59-61
    optimal_V, optimal_policy = run_value_iteration(P, R)
    visualize_phase(
        optimal_V, optimal_policy, w=0, filename="phase1_water_empty_90haz_windy.png"
    )
    visualize_phase(
        optimal_V, optimal_policy, w=1, filename="phase2_water_filled_90haz_windy.png"
    )
