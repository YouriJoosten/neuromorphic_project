
import numpy as np
class PolicyGradientAgent:
    def __init__(self, obs_dim, action_space, lr=1e-3, beta_rew=0.01):
        self.obs_dim = obs_dim
        self.n_actions = action_space.n
        self.lr = lr
        self.beta_rew = beta_rew
        self.R_bar = 0.0

        rng = np.random.default_rng()
        # θ has shape (n_actions, obs_dim)
        self.theta = rng.normal(0.0, 0.01, size=(self.n_actions, self.obs_dim))

        self.last_phi = None
        self.last_probs = None
        self.last_action = None

    def _phi(self, obs):
        x = np.array(obs, dtype=np.float32).flatten()
        # optional normalisation
        return x

    def act(self, obs):
        phi = self._phi(obs)
        logits = self.theta @ phi          # shape (n_actions,)
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        action = np.random.choice(self.n_actions, p=probs)

        self.last_phi = phi
        self.last_probs = probs
        self.last_action = action
        return action

    def learn(self, next_state, reward, done=False):
        if self.last_phi is None:
            return

        # update baseline
        self.R_bar = (1 - self.beta_rew) * self.R_bar + self.beta_rew * reward
        adv = reward - self.R_bar

        # gradient of log π(a|s)
        grad = -np.outer(self.last_probs, self.last_phi)
        grad[self.last_action] += self.last_phi

        self.theta += self.lr * adv * grad
