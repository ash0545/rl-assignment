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
            if (r, c) in BOULDERS or (r, c) == FIRE:
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

                # Instead of updating everything simultaneously, split into possible moves followed by bounding
if __name__ == "__main__":
    pass
