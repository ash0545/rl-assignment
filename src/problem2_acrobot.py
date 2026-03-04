import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


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


# train using SARSA or Q-learning for a given no of episodes, return per-episode returns
# set up to allow for hyperparameter search (step size alpha, epsilon) for 2a
# epsilon decay as well, for comparing
def train(
    env,
    algo,
    num_episodes,
    alpha,
    epsilon,
    num_bins=10,
    gamma=0.99,
    epsilon_decay=None,
    epsilon_min=0.0,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    bins = create_bins(env, num_bins)
    num_actions = env.action_space.n

    # initialize Q with zeros
    # Q = np.zeros([num_bins] * len(bins) + [num_actions])

    rng = np.random.default_rng(seed)
    Q = rng.uniform(
        low=-0.01, high=0.01, size=tuple([num_bins] * len(bins) + [num_actions])
    )

    # add this line right after Q initialization:
    env.reset(seed=seed)

    algorithm = run_sarsa_episode if algo == "sarsa" else run_qlearning_episode
    episode_returns = []
    eps = epsilon
    for episode in range(num_episodes):
        returns = algorithm(env, Q, bins, alpha, eps, gamma)
        episode_returns.append(returns)

        if epsilon_decay is not None:
            eps = max(eps * epsilon_decay, epsilon_min)

    return (
        episode_returns,
        Q,  # added since we'll need this for acting greedily for question 3
    )


# grid search over alpha and epsilon
def hyperparameter_search(algo, num_episodes=500, num_runs=3):
    alphas = [0.1, 0.3, 0.5]
    epsilons = [0.05, 0.1, 0.3]

    results = []

    env = gym.make("Acrobot-v1")

    for alpha in alphas:
        for epsilon in epsilons:
            # averaging over a few runs
            all_returns = []
            for run in range(num_runs):
                rets, _ = train(env, algo, num_episodes, alpha, epsilon, seed=run)
                all_returns.append(rets)

            mean_final = np.mean([np.mean(r[-50:]) for r in all_returns])
            results.append(
                {
                    "algo": algo,
                    "alpha": alpha,
                    "epsilon": epsilon,
                    "mean_final_return": mean_final,
                }
            )
            print(
                f"  [{algo}] alpha={alpha}, eps={epsilon} -> mean_final={mean_final:.2f}"
            )
    env.close()

    results.sort(
        key=lambda x: x["mean_final_return"], reverse=True
    )  # higher return better (closer to zero)
    return results


# mean performance with confidence intervals over 10 seeds
def plot_smoothed_returns(
    sarsa_returns,
    ql_returns,
    window=50,
    filename="plot.png",
    title="SARSA vs Q-Learning",
):

    # https://stackoverflow.com/a/14314054
    def moving_average(a):
        ret = np.cumsum(a, axis=1, dtype=float)
        ret[:, window:] = ret[:, window:] - ret[:, :-window]
        return ret[:, window - 1 :] / window

    s_smooth = moving_average(np.array(sarsa_returns))
    q_smooth = moving_average(np.array(ql_returns))

    s_mean, s_std = np.mean(s_smooth, axis=0), np.std(s_smooth, axis=0)
    q_mean, q_std = np.mean(q_smooth, axis=0), np.std(q_smooth, axis=0)

    episodes = np.arange(len(s_mean))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, s_mean, label="SARSA", color="red")
    plt.fill_between(episodes, s_mean - s_std, s_mean + s_std, color="red", alpha=0.2)

    plt.plot(episodes, q_mean, label="Q-Learning", color="blue")
    plt.fill_between(episodes, q_mean - q_std, q_mean + q_std, color="blue", alpha=0.2)

    plt.title(title)
    plt.xlabel("Episode Number")
    plt.ylabel("Episodic Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()
    print("Saved plot to q2b_plot_rs.png")


if __name__ == "__main__":
    # Q-2(a)
    # print("Hyperparameter Search: SARSA")
    # sarsa_results = hyperparameter_search("sarsa")
    # print("\nTop 3 SARSA configurations:")
    # for r in sarsa_results[:3]:
    #     print(
    #         f"  alpha={r['alpha']}, epsilon={r['epsilon']} -> {r['mean_final_return']:.2f}"
    #     )

    # print("\nHyperparameter Search: Q-Learning")
    # ql_results = hyperparameter_search("qlearning")
    # print("\nTop 3 Q-Learning configurations:")
    # for r in ql_results[:3]:
    #     print(
    #         f"  alpha={r['alpha']}, epsilon={r['epsilon']} -> {r['mean_final_return']:.2f}"
    #     )

    # env = gym.make("Acrobot-v1")
    # NUM_EPISODES = 500
    # NUM_SEEDS = 10

    # print("Epsilon Decay Check")
    # sarsa_const, _ = train(env, "sarsa", NUM_EPISODES, alpha=0.5, epsilon=0.05, seed=42)
    # sarsa_decay, _ = train(
    #     env,
    #     "sarsa",
    #     NUM_EPISODES,
    #     alpha=0.5,
    #     epsilon=1.0,
    #     epsilon_decay=0.99,
    #     epsilon_min=0.05,
    #     seed=42,
    # )

    # print(f"SARSA Constant Eps (0.05) Final Return: {np.mean(sarsa_const[-50:]):.2f}")
    # print(
    #     f"SARSA Decaying Eps (1.0->0.05) Final Return: {np.mean(sarsa_decay[-50:]):.2f}"
    # )

    # qlearn_const, _ = train(
    #     env, "qlearning", NUM_EPISODES, alpha=0.5, epsilon=0.05, seed=42
    # )
    # qlearn_decay, _ = train(
    #     env,
    #     "qlearning",
    #     NUM_EPISODES,
    #     alpha=0.5,
    #     epsilon=1.0,
    #     epsilon_decay=0.99,
    #     epsilon_min=0.05,
    #     seed=42,
    # )
    # print(
    #     f"Q-learning Constant Eps (0.05) Final Return: {np.mean(qlearn_const[-50:]):.2f}"
    # )
    # print(
    #     f"Q-learning Decaying Eps (1.0->0.05) Final Return: {np.mean(qlearn_decay[-50:]):.2f}"
    # )

    # Q-2(b)
    print("Running 10 Seeds for Plotting")
    all_sarsa = []
    all_ql = []

    for s in range(NUM_SEEDS):
        print(f"Running seed {s+1}/{NUM_SEEDS}...")
        # the best hyperparameters found:
        # SARSA: alpha=0.5, decay to 0.05
        # Q-Learning: alpha=0.3, decay to 0.1
        sarsa_ret = train(
            env,
            "sarsa",
            NUM_EPISODES,
            alpha=0.5,
            epsilon=1.0,
            epsilon_decay=0.99,
            epsilon_min=0.05,
            seed=s,
        )
        ql_ret = train(
            env,
            "qlearning",
            NUM_EPISODES,
            alpha=0.3,
            epsilon=1.0,
            epsilon_decay=0.99,
            epsilon_min=0.1,
            seed=s,
        )

        all_sarsa.append(sarsa_ret)
        all_ql.append(ql_ret)

    plot_smoothed_returns(all_sarsa, all_ql)
    env.close()
