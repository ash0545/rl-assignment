import numpy as np

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

                p_intended = 0.4 if is_smoke else 0.7
                p_perpendicular = 0.1
                p_stay = 0.4 if is_smoke else 0.1

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
                        R[s, a_idx, s_new] = -11
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


if __name__ == "__main__":
    q1_r, q1_c, q1_w = 3, 3, 0
    s_q1 = q1_w * (ROWS * COLS) + q1_r * COLS + q1_c
    print("Q1: Transitions and Rewards from (3,3), Water = 0")

    for a_idx, a_str in enumerate(ACTIONS):
        print(f"Action: {a_str}")
        # neighbors and the cell itself
        for r_new in range(q1_r - 1, q1_r + 2):
            for c_new in range(q1_c - 1, q1_c + 2):
                ns = q1_w * (ROWS * COLS) + r_new * COLS + c_new
                prob = P[s_q1, a_idx, ns]
                reward = R[s_q1, a_idx, ns]
                print(f"  -> to ({r_new},{c_new}): P = {prob:.2f}, R = {reward}")
        print()
