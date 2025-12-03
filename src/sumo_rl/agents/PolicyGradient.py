import numpy as np


class PolicyGradientAgent:
    """
    Implements two modes:
      - 'reinforce' : episodic REINFORCE / policy-gradient
      - 'actor_critic' : online Actor-Critic (linear softmax actor + linear critic, TD(0))
    Use `method='actor_critic'` in the constructor to switch.
    """

    def __init__(
        self,
        obs_dim,
        action_space,
        lr=1e-3,
        beta_rew=0.1,
        gamma=0.99,
        grad_clip=5.0,
        method="actor_critic",  # "reinforce" or "actor_critic"
        lr_critic=None,  # if None, defaults to lr
    ):
        self.obs_dim = obs_dim
        self.n_actions = action_space.n
        self.lr = lr
        self.beta_rew = beta_rew
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.method = method
        self.R_bar = 0.0

        if lr_critic is None:
            lr_critic = lr
        self.lr_critic = lr_critic

        rng = np.random.default_rng()
        # theta has shape (n_actions, obs_dim) - actor parameters
        self.theta = rng.normal(0.0, 0.01, size=(self.n_actions, self.obs_dim))
        self.theta_prev = self.theta.copy()  # snapshot at episode start for change tracking

        # critic parameters (for actor-critic): linear value V(s) = v @ phi
        self.v = np.zeros(self.obs_dim, dtype=np.float32) if method == "actor_critic" else None

        # buffers used by REINFORCE mode
        self.last_phi = None
        self.last_probs = None
        self.last_action = None
        self.trajectory = []

    def start_episode(self):
        """Reset per-episode caches and snapshot theta for change tracking."""
        self.theta_prev = self.theta.copy()
        self.trajectory.clear()
        self.last_phi = None
        self.last_probs = None
        self.last_action = None

    def _phi(self, obs):
        x = np.array(obs, dtype=np.float32).flatten()
        # optional normalisation could go here
        return x

    def act(self, obs):
        phi = self._phi(obs)
        logits = self.theta @ phi  # shape (n_actions,)
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        action = np.random.choice(self.n_actions, p=probs)

        # cache for either method (reinforce uses trajectory, actor-critic uses last_phi)
        self.last_phi = phi
        self.last_probs = probs
        self.last_action = action
        return action

    def store_reward(self, reward):
        """Store transition for episodic REINFORCE updates."""
        if self.method != "reinforce":
            return
        if self.last_phi is None:
            return
        self.trajectory.append((self.last_phi.copy(), self.last_probs.copy(), self.last_action, float(reward)))

    def _reinforce_update(self):
        """Episodic REINFORCE update and training stats."""
        if not self.trajectory:
            theta_norm = float(np.linalg.norm(self.theta))
            theta_change = float(np.linalg.norm(self.theta - self.theta_prev))
            self.theta_prev = self.theta.copy()
            return {
                "episode_return": 0.0,
                "discounted_return": 0.0,
                "mean_return": 0.0,
                "R_bar": float(self.R_bar),
                "theta_norm": theta_norm,
                "theta_change": theta_change,
                "theta_update_norm": 0.0,
            }

        theta_before = self.theta.copy()
        episode_return = float(sum(step[3] for step in self.trajectory))

        # Compute discounted returns (reward-to-go)
        returns = []
        G = 0.0
        for (_, _, _, r) in reversed(self.trajectory):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns, dtype=np.float32)

        # Update running baseline using the episode mean return (before normalization)
        mean_return = float(returns.mean())
        self.R_bar = (1 - self.beta_rew) * self.R_bar + self.beta_rew * mean_return

        # Compute advantages (use baseline) and normalize advantages for stability
        advantages = returns - self.R_bar
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient update for each time step
        for (phi, probs, action, _), adv in zip(self.trajectory, advantages):
            grad_log_pi = -np.outer(probs, phi)
            grad_log_pi[action] += phi

            step_grad = adv * grad_log_pi
            grad_norm = np.linalg.norm(step_grad)
            if self.grad_clip is not None and grad_norm > self.grad_clip:
                step_grad *= self.grad_clip / (grad_norm + 1e-8)

            self.theta += self.lr * step_grad

        theta_norm = float(np.linalg.norm(self.theta))
        theta_change = float(np.linalg.norm(self.theta - self.theta_prev))
        theta_update_norm = float(np.linalg.norm(self.theta - theta_before))
        self.theta_prev = self.theta.copy()
        discounted_return = float(returns[0])

        self.trajectory.clear()
        return {
            "episode_return": episode_return,
            "discounted_return": discounted_return,
            "mean_return": mean_return,
            "R_bar": float(self.R_bar),
            "theta_norm": theta_norm,
            "theta_change": theta_change,
            "theta_update_norm": theta_update_norm,
        }

    def finish_episode(self):
        """Public API to finish an episode for REINFORCE mode."""
        if self.method != "reinforce":
            return {
                "episode_return": 0.0,
                "discounted_return": 0.0,
                "mean_return": 0.0,
                "R_bar": float(self.R_bar),
                "theta_norm": float(np.linalg.norm(self.theta)),
                "theta_change": float(np.linalg.norm(self.theta - self.theta_prev)),
                "theta_update_norm": 0.0,
            }

        stats = self._reinforce_update()
        self.last_phi = None
        self.last_probs = None
        self.last_action = None
        return stats

    def get_param_stats(self):
        """Return current param norm and change since start_episode, then refresh snapshot."""
        theta_norm = float(np.linalg.norm(self.theta))
        theta_change = float(np.linalg.norm(self.theta - self.theta_prev))
        self.theta_prev = self.theta.copy()
        return {"theta_norm": theta_norm, "theta_change": theta_change, "theta_update_norm": theta_change}

    def learn(self, reward, done=False, next_state=None):
        """
        - For 'reinforce' : accumulate (phi, probs, action, reward) and on done call _reinforce_update()
        - For 'actor_critic': perform an online TD(0) critic update and an actor update using the TD error.
          next_state must be provided (or left None when done and V(next)=0).
        """
        if self.last_phi is None:
            return

        if self.method == "reinforce":
            # Store the transition for the episode and update at episode end
            self.store_reward(reward)

            if done:
                stats = self._reinforce_update()
                self.last_phi = None
                self.last_probs = None
                self.last_action = None
                return stats

            return

        # --- actor-critic path (online) ---
        if self.method == "actor_critic":
            phi = self.last_phi
            probs = self.last_probs
            action = self.last_action

            # value estimates
            V_s = float(self.v @ phi)

            if next_state is not None:
                next_phi = self._phi(next_state)
                V_next = float(self.v @ next_phi)
            else:
                V_next = 0.0

            # TD(0) error: delta = r + gamma * V(s') - V(s)
            done_mask = 0.0 if done else 1.0
            td_target = reward + self.gamma * V_next * done_mask
            td_error = td_target - V_s  # scalar

            # critic update: v <- v + lr_critic * delta * phi
            self.v += self.lr_critic * td_error * phi

            # actor update: theta <- theta + lr * delta * grad log pi(a|s)
            grad_log_pi = -np.outer(probs, phi)
            grad_log_pi[action] += phi
            step_grad = td_error * grad_log_pi

            grad_norm = np.linalg.norm(step_grad)
            if self.grad_clip is not None and grad_norm > self.grad_clip:
                step_grad *= self.grad_clip / (grad_norm + 1e-8)

            self.theta += self.lr * step_grad

            # If episode ended, clear last cached phi/probs/action
            if done:
                self.last_phi = None
                self.last_probs = None
                self.last_action = None

            return

        # Unknown method: no-op
        return
