import numpy as np

class EpsilonGreedy:
    """Epsilon Greedy Exploration Strategy."""

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        """Initialize Epsilon Greedy Exploration Strategy."""
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def _decay(self):
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

    def choose(self, q_table, state, action_space):
        """
        Classic epsilon-greedy for a Q-table:
        q_table[state] is a 1D array of action-values.
        """
        if np.random.rand() < self.epsilon:
            action = int(action_space.sample())
        else:
            action = int(np.argmax(q_table[state]))
        self._decay()
        return action

    def choose_values(self, action_values, action_space=None):
        """
        Epsilon-greedy when you already have a 1D array of action-values.
        Used by RMHLAgent.act().
        """
        if np.random.rand() < self.epsilon:
            if action_space is not None:
                action = int(action_space.sample())
            else:
                action = int(np.random.randint(len(action_values)))
        else:
            action = int(np.argmax(action_values))

        self._decay()
        return action

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon
