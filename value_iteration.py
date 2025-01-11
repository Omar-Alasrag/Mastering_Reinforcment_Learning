import gym
import numpy as np

# Initialize the environment
env = gym.make("FrozenLake-v1", is_slippery=True)  # Use `is_slippery=False` for deterministic transitions
P = env.env.P  # Access transition probabilities

# print(P)

def value_iteration(P, gamma=0.5):
    """
    Perform value iteration to find the optimal value function and policy for all states.

    Parameters:
    - P: Transition probabilities for the environment (list of lists).
    - gamma: Discount factor for future rewards.
    - theta: Convergence threshold.

    Returns:
    - policy: Optimal policy for each state.
    """

    # Step 1: Initialize the value function for all states to zero
    num_states = len(P)
    num_actions = len(P[0])
    V = np.zeros((num_states))

    # Loop until convergence
    while True:
        # Step 2: Initialize a new Q-function for this iteration
        Q = np.zeros(
            (num_states, num_actions)
        )  # assuming all states have the same number of actions

        # Step 3: Update Q-values for each state-action pair
        for s in range(num_states):
            for a in range(num_actions):
                for prob, next_state, reward, done in P[s][a]:
                    # Calculate Q-value for each action
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        # Step 4: Check for convergence by comparing V with max Q-values
        
        max_Q_values = np.max(Q, axis=1)
        if np.array_equal(V, max_Q_values):
            break
        # Step 5: Update the value function with the max Q-values
        V = max_Q_values
        # pprint(Q)
    # Step 6: Extract the optimal policy for each state from the Q-values
    policy = np.argmax(
        Q, axis=1
    )  # Choose the action with the highest Q-value for each state

    return policy


policy = value_iteration(P)


print("\nOptimal Policy (action for each state):")
print(policy)

