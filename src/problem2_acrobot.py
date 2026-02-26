import gymnasium as gym
import numpy as np


# creates the "bounds" of bins for the state space
def create_bins(env, num_bins=10):
    bins = []
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high

    for i in range(len(obs_low)):
        # np.linspace creates num_bins+1 edges. We take the internal num_bins-1 edges for np.digitize.
        b = np.linspace(obs_low[i], obs_high[i], num_bins + 1)[1:-1]
        bins.append(b)
    return bins


# maps continuous state to discrete state using the bins created above
def discretize_state(state, bins):
    # np.digitize returns the index of the bin to which each value belongs.
    discrete_state = tuple(np.digitize(s, b) for s, b in zip(state, bins))
    return discrete_state


# Sutton and Barto, Pg. 28: "...behave greedily most of the time, but every once in a while, say with small probability
#                           eps, instead select randomly from among all the actions with equal probability,
#                           independently of the action-value estimates."
def epsilon_greedy_policy(Q, state, epsilon, env):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        # breaking ties randomly for equal Q-values
        best_actions = np.argwhere(Q[state] == np.amax(Q[state])).flatten()
        return np.random.choice(best_actions)


# Following the algorithm given in Sutton and Barto, Pg. 130
# This is a single episode (the contents within the first loop)
def run_sarsa_episode(env, Q, bins, alpha, epsilon, gamma=0.99):
    # initialize S
    state, _ = env.reset()
    state = discretize_state(state, bins)

    # choose A from S using policy derived from Q (e.g., eps-greedy)
    action = epsilon_greedy_policy(Q, state, epsilon, env)

    total_reward = 0
    terminated = False
    truncated = False

    # loop for each step of episode:
    while not (terminated or truncated):
        # take action A, observe R, S'
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_state, bins)

        # choose A' from S' using policy derived from Q (e.g., eps-greedy)
        next_action = epsilon_greedy_policy(Q, next_state, epsilon, env)

        # SARSA update
        td_target = reward + gamma * Q[next_state][next_action] * (not terminated)
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error

        state = next_state
        action = next_action
        total_reward += reward

    return total_reward


# Following the algorithm given in Sutton and Barto, Pg. 131
# Again, a single episode (the contents within the first loop)
def run_qlearning_episode(env, Q, bins, alpha, epsilon, gamma=0.99):
    # initialize S
    state, _ = env.reset()
    state = discretize_state(state, bins)

    total_reward = 0
    terminated = False
    truncated = False

    # loop for each step of episode:
    while not (terminated or truncated):
        # choose A from S using policy derived from Q (e.g., eps-greedy)
        action = epsilon_greedy_policy(Q, state, epsilon, env)

        # take action A, observe R, S'
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_state, bins)

        # Q-learning update (max over next actions)
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state][best_next_action] * (not terminated)
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error

        state = next_state
        total_reward += reward

    return total_reward


if __name__ == "__main__":
    print("Testing Acrobot Setup...")
    env = gym.make("Acrobot-v1")
    print("Setup complete and error-free!")
    env.close()
