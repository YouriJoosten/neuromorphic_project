from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
import numpy as np


class RMHLAgent:
    """
    Reward-Modulated Hebbian Agent implementing the Exploratory Hebb (EH) rule
    from Legenstein et al. (2010).


    where:
      x_j(t)        = presynaptic activity (encoded state feature j)
      a_i(t)        = postsynaptic total input (incl. internal noise) for action unit i
      ā_i(t)       = low-pass filtered a_i
      R(t)          = instantaneous scalar reward signal
      R̄(t)         = low-pass filtered reward
      η             = learning rate

    Internal noise on a_i(t) is used for exploration (node-perturbation style).
    """

    def __init__(
        self,
        starting_state,
        state_space,
        action_space,
        lr=1e-3,
        gamma=0.99,  # kept for compatibility, not used directly in EH
        exploration_strategy=None,
        mean_alpha=0.2,      # low-pass coefficient for ā_i and R̄ (≈ 0.2 in paper)
        noise_base=0.1,      # base noise level on activations
        noise_scale=0.1,     # how much noise grows with |pre-activation|
        normalize_weights=True,
    ):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.exploration = exploration_strategy or EpsilonGreedy()
        self.acc_reward = 0.0

        # Dimensions
        self.state_dim = len(starting_state)
        self.action_dim = action_space.n

        # Synaptic weights: state_dim x action_dim
        self.W = np.zeros((self.state_dim, self.action_dim))

        # Running means for EH rule
        self.a_mean = np.zeros(self.action_dim)   # ā_i(t)
        self.R_mean = 0.0                         # R̄(t)
        self.mean_alpha = mean_alpha

        # Internal noise parameters (exploratory signal)
        self.noise_base = noise_base
        self.noise_scale = noise_scale

        # Optional weight normalization (L2 per action column, like Eq. 20)
        self.normalize_weights = normalize_weights

        # Scaling for inputs
        self.state_scale = np.maximum(1.0, np.abs(starting_state))

        # Buffers for last step (needed for credit assignment)
        self.last_pre = None   # x(t)
        self.last_a = None     # a(t)
        self.last_action_values = None
        self.action = None

    # -------------------------
    # Episode handling
    # -------------------------
    def start_episode(self, starting_state):
        """Reset agent's internal state at the beginning of each episode."""
        self.state = starting_state
        self.acc_reward = 0.0
        # keep a_mean and R_mean across episodes (like in paper: long-term averages)
        self.last_pre = None
        self.last_a = None
        self.last_action_values = None
        self.action = None

    # -------------------------
    # Encoding / activations
    # -------------------------
    def encode(self, state):
        x = np.array(state, dtype=float)
        return x / self.state_scale

    def _compute_noisy_activation(self, x):
        """
        Compute total synaptic input a_i(t) = h_i + ξ_i, and output s_i = ReLU(a_i).
        Internal noise ξ_i is drawn from uniform [-σ_i, σ_i] with
        σ_i = noise_base + noise_scale * |h_i| (cf. Eq. 13 idea).
        """
        h = x @ self.W  # shape: (action_dim,)

        # Heteroscedastic internal noise (exploration signal)
        sigma = self.noise_base + self.noise_scale * np.abs(h)
        noise = np.random.uniform(-sigma, sigma)

        a = h + noise
        s = np.maximum(0.0, a)  # threshold-linear activation (Eq. 3–4 style)

        return a, s

    def get_action_values(self, state_encoded):
        """
        For compatibility with other code: return current (noiseless) action values.
        """
        return state_encoded @ self.W

    # -------------------------
    # Acting
    # -------------------------
    def act(self):
        """
        Compute noisy activations and select an action with epsilon-greedy
        on the resulting 'action values' (s_i).
        """
        x = self.encode(self.state)

        a, s = self._compute_noisy_activation(x)

        # Store for learning
        self.last_pre = x
        self.last_a = a
        self.last_action_values = s

        # Use epsilon-greedy wrt s (can set small ε and rely on internal noise)
        self.action = self.exploration.choose_values(
            self.last_action_values,
            action_space=self.action_space
        )
        return self.action

    # -------------------------
    # Learning (EH rule)
    # -------------------------
    def learn(self, next_state, reward, done=False):
        """
        Apply EH update using *previous* state and activations and current reward.

        You may want to flip the sign of reward outside this function if your
        environment uses a cost (e.g. waiting time) instead of a reward.
        """
        if self.last_pre is None or self.last_a is None:
            # first step after reset before any action; nothing to update
            self.state = next_state
            self.acc_reward += reward
            return

        x_t = self.last_pre       # x_j(t)
        a_t = self.last_a         # a_i(t)

        # --- Update running means ā_i and R̄ (exponential low-pass) ---
        self.a_mean = (1.0 - self.mean_alpha) * self.a_mean + self.mean_alpha * a_t
        self.R_mean = (1.0 - self.mean_alpha) * self.R_mean + self.mean_alpha * reward

        delta_a = a_t - self.a_mean           # (a_i(t) - ā_i(t))
        delta_R = reward - self.R_mean        # (R(t) - R̄(t))

        # Outer product x_j * delta_a_i gives Hebbian term; scale by delta_R
        hebb = np.outer(x_t, delta_a) * delta_R

        self.W += self.lr * hebb

        # Optional L2 normalization per action column (Eq. 20 analogue)
        if self.normalize_weights:
            norms = np.linalg.norm(self.W, axis=0, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self.W /= norms

        # Move to next state
        self.state = next_state
        self.acc_reward += reward

        if done:
            # episode will be reset externally
            pass
