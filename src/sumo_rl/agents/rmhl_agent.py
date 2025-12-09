from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
import numpy as np 

class RMHLAgent:
    """Reward-Modulated Hebbian Agent class."""

    def __init__(
        self,
        starting_state,
        state_space,
        action_space,
        lr=0.01,
        gamma=0.99,
        exploration_strategy=None,
    ):
        """Initialize RMHL agent."""

        # Basic RL stuff
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.exploration = exploration_strategy or EpsilonGreedy()
        self.acc_reward = 0.0

        # Network dimensions
        self.state_dim = len(starting_state)
        self.action_dim = action_space.n

        # Hebbian weights and eligibility traces
        self.W = np.zeros((self.state_dim, self.action_dim))
        self.eligibility = np.zeros((self.state_dim, self.action_dim))

        # Reward baseline (running average)
        self.reward_baseline = 0.0
        self.baseline_alpha = 0.05  

        # Simple state scaling (to avoid huge inputs)
        self.state_scale = np.maximum(1.0, np.abs(starting_state))

    # -------------------------
    # Encoding / value function
    # -------------------------
    def encode(self, state):
        x = np.array(state, dtype=float)
        return x / self.state_scale
    
    def get_action_values(self, state_encoded):
        # Linear action-values: Q(s,a) ~ x^T W
        return state_encoded @ self.W
    
    # -------------------------
    # Acting
    # -------------------------
    def act(self):
        x = self.encode(self.state)
        action_values = self.get_action_values(x)
        # IMPORTANT: use choose_values, not choose
        self.action = self.exploration.choose_values(
            action_values,
            action_space=self.action_space
        )
        return self.action

    # -------------------------
    # Learning (Hebbian + traces)
    # -------------------------
    def learn(self, next_state, reward, done=False):
        pre = self.encode(self.state)

        # One-hot post-synaptic vector for chosen action
        post = np.zeros(self.action_dim)
        post[self.action] = 1
        
        # Reward-modulated term (you can later replace this with a true TD-error)
        next_pre = self.encode(next_state)
        next_values = self.get_action_values(next_pre)
        bootstrap = 0.0 if done else np.max(next_values)

        reward_mod = reward + self.gamma * bootstrap - self.reward_baseline
        self.reward_baseline += self.baseline_alpha * reward_mod

        # Eligibility trace
        lambda_ = 0.2  # trace decay factor
        self.eligibility = self.gamma * lambda_ * self.eligibility + np.outer(pre, post)
        
        # Hebbian update using eligibility trace
        hebbian_update = self.lr * reward_mod * self.eligibility
        self.W += hebbian_update

        # Weight clipping to prevent runaway
        self.W = np.clip(self.W, -5, 5)
        # Optional small decay:
        self.W *= 0.999

        # Move to next state
        self.state = next_state
        self.acc_reward += reward
