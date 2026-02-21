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
