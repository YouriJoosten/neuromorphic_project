
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
import numpy as np 

class RMHLAgent:
    """Reward-Modulated Hebbian Agent class."""

    def __init__(self, starting_state, state_space, action_space, lr = 0.0001, gamma=0.99, exploration_strategy=EpsilonGreedy()):
        """Initialize Q-learning agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.lr = lr 
        self.gamma = gamma
        self.exploration = exploration_strategy
        self.acc_reward = 0

        self.state_dim = len(starting_state)
        self.action_dim = action_space.n
        self.W = np.zeros((self.state_dim, self.action_dim))
        self.eligibility = np.zeros((self.state_dim, self.action_dim))

        # Reward baseline (running average)
        self.reward_baseline = 0.0
        self.baseline_alpha = 0.05  

        self.state_scale = np.maximum(1.0, np.abs(starting_state))

    # Normalization addition
    def encode(self, state):
        x = np.array(state, dtype=float)
        self.state_scale = np.maximum(self.state_scale, np.abs(x))
        return x / (self.state_scale + 1e-6) 
    
    def get_action_values(self, state_encoded):
        return state_encoded @ self.W
    
    def act(self):
        x = self.encode(self.state)
        action_values = self.get_action_values(x)
        self.action = self.exploration.choose_values(action_values)
        return self.action

    def learn(self, next_state, reward, done=False):
        pre = self.encode(self.state)
        
        # One-hot post-synaptic vector for chosen action
        post = np.zeros(self.action_dim)
        post[self.action] = 1
        
        # Update reward baseline (EMA)
        reward_mod = reward + self.gamma * np.max(self.get_action_values(self.encode(next_state))) - self.reward_baseline
        self.reward_baseline += self.baseline_alpha * reward_mod

        lambda_ = 0.2  # trace decay factor for how long we keep past activity
        self.eligibility = self.gamma * lambda_ * self.eligibility + np.outer(pre, post)
        
        # Hebbian update using eligibility trace
        hebbian_update = self.lr * reward_mod * self.eligibility
        self.W += hebbian_update

        # Weight clipping to prevent runaway
        self.W = np.clip(self.W, -10, 10)
        self.W *= 0.99

        # Move to next state
        self.state = next_state
        self.acc_reward += reward

        print("||W||:", np.linalg.norm(self.W))