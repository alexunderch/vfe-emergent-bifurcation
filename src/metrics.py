import jax
import jax.numpy as jnp
import chex

from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy

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
    
    return h_wa, h_w + h_a - h_wa
   



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

def calculate_free_energy(policy: chex.Array) -> chex.Array:
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
