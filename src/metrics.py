import chex
import jax
import jax.numpy as jnp
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability

SMALL_NUMBER = 1e-20


def calculate_social_metrics(pi_s, pi_r):
    """
    Calculates the informational state of the Sender-Receiver system.
    """
    n_w = pi_s.shape[0]
    p_w = jnp.ones(n_w) / n_w

    p_wa = jnp.einsum("w,wm,ma->wa", p_w, pi_s, pi_r)
    p_w_m = p_wa.sum(axis=1)
    p_a_m = p_wa.sum(axis=0)
    
    eps = 1e-12
    
    h_wa = -jnp.sum(p_wa * jnp.log2(p_wa + eps))
    h_w  = -jnp.sum(p_w_m * jnp.log2(p_w_m + eps))
    h_a  = -jnp.sum(p_a_m * jnp.log2(p_a_m + eps))
    
    return p_wa, h_wa, h_w + h_a - h_wa
   

def calculate_coordination_success(policy: chex.Array) -> float:
  
  # Success Metric: Trace of the policy (sum of diagonal)
  # Since expected_rewards is identity, coordination is trace(policy) / num_states
  coordination_success = jnp.trace(policy) / policy.shape[0]
  
  return 1.0 - coordination_success # Minimize the failure

def calculate_mi(policy: chex.Array):
    num_states, num_messages = policy.shape
    P_m_given_w = policy
    p_w = 1.0 / num_states
    p_wm = p_w * P_m_given_w
    p_m = jnp.sum(p_wm, axis=0)
    mi = 0
    for w in range(num_states):
      for m in range(num_messages):      
        mi += p_wm[w, m] * jnp.log2(p_wm[w, m] / (p_w * p_m[m]))
    return mi

def compute_end_to_end_mi(sender_logits, receiver_logits, temp=1.0):
    """
    Computes I(W; A), the true end-to-end channel capacity of the signaling game.
    Maximizing this indicates a perfect separating convention.
    """
    num_w = sender_logits.shape[0]
    
    # 1. Convert logits to policies
    pi_s = jax.nn.softmax(sender_logits / temp, axis=-1)  # p(m|w)
    pi_r = jax.nn.softmax(receiver_logits / temp, axis=-1)  # p(a|m)
    
    # 2. Assume uniform prior over world states
    p_w = jnp.ones(num_w) / num_w
    
    # 3. Compute the End-to-End transition matrix p(a|w)
    # This marginalizes out the message 'm': p(a|w) = sum_m p(m|w) * p(a|m)
    p_a_given_w = jnp.dot(pi_s, pi_r)
    
    # 4. Compute the Joint Distribution p(w, a)
    # p(w, a) = p(w) * p(a|w)
    p_wa = p_w[:, None] * p_a_given_w
    
    # 5. Compute the Marginal Distribution p(a)
    p_a = jnp.sum(p_wa, axis=0)
    
    # 6. Calculate Shannon Entropies
    def entropy(p):
        return -jnp.sum(p * jnp.log(p + 1e-12))
        
    h_a = entropy(p_a)
    
    # Conditional entropy H(A|W) = sum_w p(w) * H(A|w)
    h_a_given_w_elements = -jnp.sum(p_a_given_w * jnp.log(p_a_given_w + 1e-12), axis=-1)
    h_a_given_w = jnp.sum(p_w * h_a_given_w_elements)
    
    # 7. True Mutual Information I(W; A)
    i_wa = h_a - h_a_given_w
    
    return i_wa

def calculate_free_energy_(policy: chex.Array) -> chex.Array:
    num_states, num_messages = policy.shape
    P_m_given_w = policy
    
    p_w = 1.0 / num_states
    p_wm = p_w * P_m_given_w
    p_m = jnp.sum(p_wm, axis=0)
    
    # Conditional Entropy H(W|M) (Residual Uncertainty)
    h_w_given_m = 0
    
    for w in range(num_states):
      for m in range(num_messages):
        # Uncertainty calculation: -sum p(w,m) log p(w|m)
        p_w_given_m = p_wm[w, m] / p_m[m] if p_m[m] > SMALL_NUMBER else 0
        h_w_given_m -= p_wm[w, m] * jnp.log2(p_w_given_m + SMALL_NUMBER)
                    
    # 3. Complexity (KL Divergence from uniform prior)
    # For a uniform prior, D_KL is just log(n) - H(Z)
    h_current = -jnp.sum(P_m_given_w * jnp.log2(P_m_given_w + SMALL_NUMBER)) / num_states
    complexity = jnp.log2(num_states) - h_current
    
    # Free Energy = Uncertainty + Complexity
    vfe = h_w_given_m + complexity
    
    return vfe

def calculate_free_energy(
    sender_policy: chex.Array,      # shape: (|W|, |M|)
    receiver_policy: chex.Array,    # shape: (|M|, |A|)
    utility: chex.Array,            # shape: (|W|, |A|)
    kappa: float = 1.0,
) -> chex.Array:
    """
    Joint VFE = I(W;M) + I(M;A) - 2·κ·E[U]  (before equilibrium tightness)
    At separating equilibrium this reduces to I(W;A) - κ·E[U] (tight bound).
    """
    num_states, num_messages = sender_policy.shape
    p_w = 1.0 / num_states
    
    # --- Sender complexity: I(W;M) ---
    p_m = jnp.sum(p_w * sender_policy, axis=0)          # marginal P(m)
    log_ratio = jnp.log2(sender_policy + 1e-10) - jnp.log2(p_m + 1e-10)
    kl_per_state = jnp.sum(sender_policy * log_ratio, axis=1)
    sender_vfe = jnp.sum(p_w * kl_per_state)             # = I(W;M)
    
    # --- Receiver complexity: I(M;A) ---
    p_a = jnp.sum(p_m[:, None] * receiver_policy, axis=0)  # marginal P(a)
    log_ratio_r = jnp.log2(receiver_policy + 1e-10) - jnp.log2(p_a + 1e-10)
    kl_per_message = jnp.sum(receiver_policy * log_ratio_r, axis=1)
    receiver_vfe = jnp.sum(p_m * kl_per_message)         # = I(M;A)
    
    # --- Expected utility (accuracy) ---
    # p(w,a) = p(w) Σ_m π^s(m|w) π^r(a|m)
    joint = p_w * sender_policy @ receiver_policy          # shape: (|W|, |A|)
    expected_utility = jnp.sum(joint * utility)
    
    # --- Total VFE ---
    # F_total = I(W;M) + I(M;A) - 2·κ·E[U]
    # Note: the paper's joint VFE bound (Lemma 1) has a residual L ≥ 0.
    # At separating equilibrium L = 0 and I(W;M) = I(M;A) = I(W;A).
    vfe = sender_vfe + receiver_vfe - 2.0 * kappa * expected_utility
    
    return vfe

def expected_utility(
	Z_s: chex.Array, 
	Z_r: chex.Array, 
	U: chex.Array, 
	p_w: chex.Array, 
	temp: float = 1.0
) -> chex.Array:
	"""Computing expected utility"""
	pi_s = jax.nn.softmax(Z_s * temp)	
	pi_r = jax.nn.softmax(Z_r * temp)
	return jnp.einsum("w,wm,ma,wa->", p_w, pi_s, pi_r, U)


def corrected_phi(logits: chex.Array, temp: float = 1.0) -> chex.Array:
    """
    Competition modulator: active near origin, vanishes at permutation matrices.
    
    Uses normalized Simpson index (Gini heterogeneity) for each row and column.
    
    Properties:
    - phi(0) = 1 (full competition at uniform state)
    - phi(P) = 0 for any permutation matrix P (no competition at fixed point)
    - 0 <= phi_ij <= 1 everywhere
    """
    sigma = jax.nn.softmax(logits * temp, axis=-1)
    n_rows, n_cols = sigma.shape
    
    # Row heterogeneity: 0 at one-hot, (n-1)/n at uniform
    row_gini = 1.0 - jnp.sum(sigma ** 2, axis=-1, keepdims=True)
    # Normalize to [0,1]: divide by max possible value (n-1)/n
    H_row = row_gini / ((n_cols - 1.0) / n_cols)
    H_row = jnp.clip(H_row, 0.0, 1.0)
    
    # Column heterogeneity
    col_gini = 1.0 - jnp.sum(sigma ** 2, axis=0, keepdims=True)
    H_col = col_gini / ((n_rows - 1.0) / n_rows)
    H_col = jnp.clip(H_col, 0.0, 1.0)
    
    # Competition active when BOTH row and column are undecided
    return H_row * H_col

def calculate_laplacian_inhibition(logits: chex.Array):
    """Constructs the bipartite competition term L*z."""

    row_sum = jnp.sum(logits, axis=1, keepdims=True) - logits
    col_sum = jnp.sum(logits, axis=0, keepdims=True) - logits
    Lz = row_sum + col_sum

    # Project out uniform (gauge) mode: subtract mean
    Lz = Lz - jnp.mean(Lz)
    return Lz

def make_laplacian(logits: chex.Array):
    """Laplacian for an n_row x n_col matrix.
    L = L_row + L_col  where each is a complete-graph clique."""
    n_row, n_col = logits.shape
    I_r, I_c = jnp.eye(n_row), jnp.eye(n_col)
    J_r, J_c = jnp.ones((n_row, n_row)), jnp.ones((n_col, n_col))
    # row clique: within each row, all entries connected
    L_row = jnp.kron(I_r, n_col * I_c - J_c)
    # column clique: within each col, all entries connected
    L_col = jnp.kron(n_row * I_r - J_r, I_c)
    return (L_row + L_col)


def corrected_laplacian_inhibition(logits: chex.Array) -> chex.Array:
    """
    Graph Laplacian for the bipartite state-message graph.
    L = L_row + L_col where:
    - L_row: for each state i, logits[i,:] form a clique (competition among messages)
    - L_col: for each message j, logits[:,j] form a clique (competition among states)
    
    This is the operator L such that:
    - L @ 1 = 0 (consensus mode is nullspace)
    - All other eigenvalues are positive
    """
    n_states, n_messages = logits.shape
    
    # L_row: n_messages * logits - sum(logits, axis=1, keepdims=True)
    # This is the Laplacian of a complete graph on n_messages nodes
    row_sum = jnp.sum(logits, axis=1, keepdims=True)
    L_row = n_messages * logits - row_sum
    
    # L_col: n_states * logits - sum(logits, axis=0, keepdims=True)
    # This is the Laplacian of a complete graph on n_states nodes
    col_sum = jnp.sum(logits, axis=0, keepdims=True)
    L_col = n_states * logits - col_sum
    
    return L_row + L_col

def balanced_laplacian_inhibition(logits: chex.Array):
    n_rows, n_cols = logits.shape
    row_mean = jnp.mean(logits, axis=1, keepdims=True)
    col_mean = jnp.mean(logits, axis=0, keepdims=True)
    return (logits - row_mean) / n_cols + (logits - col_mean) / n_rows

def make_laplacian2(logits: chex.Array) -> chex.Array:
    """
    Bizyaeva-style Laplacian for opinion dynamics with competition.
    L Z = (J-I)Z + Z(J-I) = JZ + ZJ - 2Z
    
    This is NOT the standard graph Laplacian. It has negative eigenvalues
    for competitive modes, essential for enforcing one-to-one mappings.
    
    Spectrum for n×n: {2(n-1), n-2, -2} with multiplicities.
    """
    n_row, n_col = logits.shape
    
    # JZ: row sums broadcast across columns
    row_sums = jnp.sum(logits, axis=1, keepdims=True)
    JZ = jnp.broadcast_to(row_sums, logits.shape)
    
    # ZJ: column sums broadcast across rows
    col_sums = jnp.sum(logits, axis=0, keepdims=True)
    ZJ = jnp.broadcast_to(col_sums, logits.shape)
    
    # Bizyaeva Laplacian
    return JZ + ZJ - 2 * logits

def dynamical_vfe(Z_s: chex.Array, Z_r: chex.Array, U, p_w: chex.Array, flags):
    """
    F(Z) = γ/2 ||Z||² - Σ 1/β ln(cosh(βZ+ε)) - η/2 Z^T L Z - κ E[U]
    """

    # 1. Dissipation (complexity)
    diss = 0.5 * flags.damping * (jnp.sum(Z_s**2) + jnp.sum(Z_r**2))
    
    # 2. Commitment (symmetry-breaking)
    commit = -(flags.force / flags.self_attention) * (
        jnp.sum(
            jnp.log(
                jnp.cosh(
                    flags.self_attention * Z_s 
                    + flags.eps
                )
            )
        ) +
        jnp.sum(
            jnp.log(
                jnp.cosh(
                    flags.self_attention * Z_r
                    + flags.eps 
                )
            )
        )
    ) if flags.self_attention > 0 else 0.0

    phi_s = corrected_phi(Z_s, flags.temperature)
    phi_r = corrected_phi(Z_r, flags.temperature)

    L_s = calculate_laplacian_inhibition(Z_s) 
    L_r = calculate_laplacian_inhibition(Z_r) 
    
    # 3. Laplacian competition
    lap_s = -0.5 * flags.coupling * jnp.sum(Z_s * L_s * phi_s)
    lap_r = -0.5 * flags.coupling * jnp.sum(Z_r * L_r * phi_r)
    
    # 4. Expected utility (accuracy, negative because F = Complexity - Accuracy)
    acc = -2 * flags.learning_rate * expected_utility(Z_s, Z_r, U, p_w, flags.temperature)
    
    total = diss + commit + acc + lap_s + lap_r
    return total
def nonlinear_inhibition(logits: chex.Array, temp: float = 1.0) -> chex.Array:
    """
    Nonlinear competition that:
      - is 0 at uniform logits (preserves the pitchfork bifurcation point)
      - is 0 at one-hot rows/columns (preserves permutation fixed points)
      - amplifies differences during the transient
    """
    pi = jax.nn.softmax(logits * temp, axis=-1)

    # Row spread: 0 at one-hot, (1 - 1/n) at uniform
    row_spread = 1.0 - jnp.sum(pi ** 2, axis=-1, keepdims=True)

    # Column spread (measure how "peaked" each column is)
    col_sums = jnp.sum(pi, axis=0, keepdims=True)
    pi_col_norm = pi / (col_sums + 1e-10)
    col_spread = 1.0 - jnp.sum(pi_col_norm ** 2, axis=0, keepdims=True)

    # Centered logits
    row_mean = jnp.mean(logits, axis=-1, keepdims=True)
    col_mean = jnp.mean(logits, axis=0, keepdims=True)

    # Anti-diffusion: pushes entries away from their row/column means
    # Strength is proportional to how far the row/column is from one-hot
    return row_spread * (logits - row_mean) + col_spread * (logits - col_mean)

@jax.jit
def calculate_cic(speaker_probs, listener_probs):
    """
    A version of Causal Influence of Communication (CIC).
    Args:
        speaker_probs: p(m | w), shape [num_states, num_messages]
        listener_probs: p(a | m), shape [num_messages, num_actions]
    Returns:
        cic: The scalar CIC value.
    """

    # 1. Compute p(a | w) = sum_m [ p(a | m) * p(m | w) ]
    # speaker_probs: [W, M, 1]
    # listener_probs: [1, M, A]
    # p(m, a | s) = p(a | m) * p(m | w)
    p_m_a_given_w = listener_probs[jnp.newaxis] * speaker_probs[..., jnp.newaxis]
    p_a_given_w = jnp.sum(p_m_a_given_w, axis=1)

    # 2. Compute Conditional Mutual Information I(M; A | S)
    # I(M; A | S) = E_s [ sum_{m,a} p(m,a|s) * log( p(a|s,m) / p(a|s) ) ]

    # Expand p_a_given_s for division: [S, A] -> [S, 1, A]
    p_a_given_s_expanded = p_a_given_w[:, jnp.newaxis]

    # Compute the log ratio
    log_ratio = jnp.log((listener_probs + SMALL_NUMBER) / (p_a_given_s_expanded + SMALL_NUMBER))

    # Sum over messages and actions, then average over states
    inner_sum = jnp.sum(p_m_a_given_w * log_ratio, axis=(1, 2))
    cic = jnp.mean(inner_sum)

    return cic


def compute_coordination_success_analytical(pi_s, pi_r, U):
    """
    Greedy coordination: sender picks argmax message, receiver picks argmax action.
    """
    best_actions = U.argmax(axis=1)
    # Greedy sender: m*(w) = argmax_m Z_s[w,m]
    msg_greedy = pi_s.argmax(axis=1)
    # Greedy receiver: a*(m) = argmax_a Z_r[m,a]
    act_greedy = pi_r[msg_greedy].argmax(axis=1)
    
    return jnp.mean((act_greedy == best_actions).astype(jnp.float32))


def compute_expected_reward(pi_s, pi_r, p_w, U):
    """Exact E[U] from policies."""
    return jnp.einsum("w,wm,ma,wa->", p_w, pi_s, pi_r, U)

def get_observation_index(state, player_id: int) -> int:
  """
  Extracts the observation tensor from OpenSpiel and converts it 
  to an integer index for the Z matrix.
  """
  obs_tensor = state.observation_tensor(player_id)
  
  # Convert to index using argmax
  # If the world is State 1, the tensor might be [0, 1, 0]. Argmax returns 1.
  obs_index = obs_tensor[3:].index(1)
  
  return int(obs_index)

class MatrixSignalingPolicy(policy.Policy):
	def __init__(self, game, sender_matrix, receiver_matrix):
		"""
		Args:
				game: The OpenSpiel game instance.
				sender_matrix: 2D array of shape (num_states, num_messages). 
												Rows sum to 1.
				receiver_matrix: 2D array of shape (num_messages, num_actions). 
													Rows sum to 1.
		"""
		super().__init__(game, player_ids=[0, 1])
		self.matrices = {
			0: sender_matrix,
			1: receiver_matrix
		}

		num_states, num_messages = sender_matrix.shape
		assert jnp.allclose(
				sender_matrix.sum(1), jnp.ones(num_states), atol=1e-1
		), sender_matrix.sum(1)
		num_messages, num_actions = receiver_matrix.shape
		assert jnp.allclose(
				receiver_matrix.sum(1), jnp.ones(num_messages), atol=1e-1
		), receiver_matrix.sum(1)

		self.legal_actions = {
			0: num_messages,
			1: num_actions
		}

	def action_probabilities(self, state, player_id=None):
		if player_id is None:
			player_id = state.current_player()

		obs_idx = get_observation_index(state, player_id)

		probs = self.matrices[player_id][obs_idx]
		legal_actions = range(self.legal_actions[player_id])
		
		# Dictionary comprehension is faster and cleaner
		action_probs = {action: float(probs[action]) for action in legal_actions}
		
		total_prob = sum(action_probs.values())
		if total_prob > 0:
				return {a: p for a, p in action_probs.items()}
		
		# Fallback to uniform if something is broken
		return {a: 1.0 / len(legal_actions) for a in legal_actions}

def calculate_exploitability(game, sender_beliefs, receiver_beliefs) -> float:

    wrapped_policy = MatrixSignalingPolicy(game, sender_beliefs, receiver_beliefs)
    
    # 2. Calculate NashConv
    nash_conv_value = exploitability.nash_conv(game, wrapped_policy)
	
    s_br_utility = exploitability.best_response(game, wrapped_policy, 0)
    r_br_utility = exploitability.best_response(game, wrapped_policy, 1)
    
    return nash_conv_value, (s_br_utility, r_br_utility)
