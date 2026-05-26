from typing import NamedTuple
import enum
import jax
import jax.numpy as jnp
import diffrax

import pyspiel
import copy

import numpy as np
import chex
from tqdm.auto import tqdm
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy
from functools import fsolve

from metrics import (
	calculate_coordination_success, 
	calculate_social_metrics, 
	calculate_free_energy, 
	calculate_cic,
)

import matplotlib.pyplot as plt

class Flags(NamedTuple):
	num_states: int = 3 
	num_messages: int = 3 
	payoffs: str = "classic"
	num_iterations: int = 100
	num_runs: int = 5
	log_interval: int = 10
	intergration_step: float = 0.01
	end_time: float = 0.5
	num_saves: int = 100
	baseline_alpha: float = 0.01

	temperature: float = 1.0
	learning_rate: float = 1.0
	force: float = 1.5
	force_eps: float = 0.025
	damping: float = 1.0
	coupling: float = 1.0
	
	seed: int = 42

class PlayerID(enum.IntEnum):
	SENDER = 0 
	RECEIVER = 1

def biffurcation(
	game,
	flags: Flags,
	parameter_config: dict, 
	role: PlayerID = PlayerID.SENDER
):
	rng, rng_sender, rng_receiver = jax.random.split(jax.random.key(flags.seed), 3)
	# Sender: State -> Message
	sender_beliefs = jax.random.uniform(rng_sender, (flags.num_states, flags.num_messages)) * 0.1
	# Receiver: Message -> Action
	receiver_beliefs = jax.random.uniform(rng_receiver, (flags.num_messages, game.num_distinct_actions())) * 0.1

	rng, _, reward, actions = game_iter(
		game, flags, rng, (sender_beliefs, receiver_beliefs)
	)
	
	baseline = flags.baseline_alpha * reward
	grad_sender = compute_extrinsic_reward_grad(
		sender_beliefs, 
		actions[int(PlayerID.SENDER)], 
		reward[int(PlayerID.SENDER)],
		baseline[int(PlayerID.SENDER)],
		flags.force,
		flags.learning_rate
	)

	grad_receiver = compute_extrinsic_reward_grad(
		receiver_beliefs, 
		actions[int(PlayerID.RECEIVER)], 
		reward[int(PlayerID.RECEIVER)],
		baseline[int(PlayerID.RECEIVER)],
		flags.force,
		flags.learning_rate
	)
	
	def find_equilibria():
		""" Finds the roots of dot_z = -gamma*z + tanh(beta*z) """
		base_args = (flags.force, flags.force_eps, flags.damping, flags.coupling)
		args = base_args + (
			(sender_beliefs.shape, grad_sender) 
			if role == PlayerID.SENDER else (receiver_beliefs.shape, grad_receiver)
		)
		def func(x):
			return opinion_dynamics(0.0, x, args)
		
		# We test different starting points to find all branches
		b = sender_beliefs if role == PlayerID.SENDER else receiver_beliefs
		root, info, ier, msg = fsolve(func, b, full_output=True)
		jacobian_matrix = jax.jacfwd(func)(root)
	
		# 3. Calculate Eigenvalues
		eigenvals = jnp.linalg.eigvals(jacobian_matrix)
		
		seeds = [-b, b * 0, b]
		roots = set()
		for s in seeds:
			root, info, ier, msg = fsolve(func, s, full_output=True)
			if ier == 1:
				roots.add(round(root[0], 4))
		return sorted(list(roots)), eigenvals

	assert "name" in parameter_config, parameter_config
	assert "min" in parameter_config, parameter_config
	assert "max" in parameter_config, parameter_config
	assert "num_points" in parameter_config, parameter_config

	param_name = parameter_config["name"]
	params = np.linspace(parameter_config["min"], parameter_config["max"], parameter_config["num_points"])

	stable_branches = []
	unstable_branches = []
	leading_eigenvalues = []

	for p in params:
			flags = flags._replace(**{param_name: p})
			eqs, eigs = find_equilibria()
			print(eqs)
			# max_real_eig = jnp.max(jnp.real(eigenvals))

			leading_eigenvalues.append(jnp.max(jnp.real(eigs)))
			print(leading_eigenvalues[-1])
			# if len(eqs) == 0:
			# 	continue
			if len(eqs) == 1:
					# Before bifurcation: Only the origin is stable
					stable_branches.append((p, eqs[0]))
			elif len(eqs) == 2:
					# After bifurcation: Origin is unstable, outer branches are stable
					# eqs[0] is negative branch, eqs[1] is origin, eqs[2] is positive branch
					stable_branches.append((p, eqs[0]))
					# stable_branches.append((p, eqs[2]))
					unstable_branches.append((p, eqs[1]))
			else:
					# After bifurcation: Origin is unstable, outer branches are stable
					# eqs[0] is negative branch, eqs[1] is origin, eqs[2] is positive branch
					stable_branches.append((p, eqs[0]))
					stable_branches.append((p, eqs[2]))
					unstable_branches.append((p, eqs[1]))

	# Convert to arrays for plotting
	sb = np.array(stable_branches)
	ub = np.array(unstable_branches)

	# Plotting
	plt.figure(figsize=(10, 6))

	# Plot Stable Branches
	plt.scatter(sb[:, 0], sb[:, 1], color='blue', s=10, label='Stable Equilibrium (Signaling)')

	# Plot Unstable Branch (The Babbling state after bifurcation)
	if len(ub) > 0:
			plt.plot(ub[:, 0], ub[:, 1], '--', color='red', label='Unstable Equilibrium (Babbling)')

	# Plotting the Babbling state before it becomes unstable
	before_bif = sb[sb[:, 0] < (flags.force + 0.05)]
	plt.plot(before_bif[:, 0], before_bif[:, 1], color='blue')

	# Formatting
	# plt.axvline(x=flags.gamma, color='black', linestyle=':', label=f'Critical Beta (bc={gamma})')
	plt.title("Pitchfork Bifurcation: The Birth of Meaning")
	plt.xlabel("Sensitivity (Beta)")
	plt.ylabel("Internal Belief State (z*)")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.show()

	plt.figure(figsize=(10, 6))
	plt.plot(params, leading_eigenvalues, lw=2.5, color='darkorange', label=r"Leading Eigenvalue $\lambda_{max}$")
	plt.axhline(0, color='black', linestyle='--', alpha=0.6) # Stability Boundary
	plt.axvline(1.0, color='red', linestyle=':', label="Bifurcation Point")

	plt.fill_between(params, leading_eigenvalues, 0, 
									where=(jnp.array(leading_eigenvalues) < 0), color='green', alpha=0.1, label="Stable Region")
	plt.fill_between(params, leading_eigenvalues, 0, 
									where=(jnp.array(leading_eigenvalues) > 0), color='red', alpha=0.1, label="Unstable Region")

	plt.title("Spectral Analysis: Stability of the 'Babbling' Equilibrium", fontsize=14)
	plt.xlabel(r"Sensitivity Parameter ($\beta$)")
	plt.ylabel(r"Real Part of Leading Eigenvalue $Re(\lambda_{max})$")
	plt.legend()
	plt.grid(True, which='both', linestyle='--', alpha=0.5)

	plt.show()

def plot_biffurcation(flags, parameter_config):
	game = "lewis_signaling"
	
	#game parameters
	num_players = 2
	num_states = flags.num_states
	num_messages = flags.num_messages
    

	if flags.payoffs == "random":
		payoffs = np.random.random((num_states, num_states))
		payoffs_str = ",".join([str(x) for x in payoffs.flatten()])		
	elif flags.payoffs == "classic":
		payoffs_list = [1, 0, 0, 0, 1, 0, 0, 0, 1]
		payoffs = np.array(payoffs_list).reshape((num_states, num_states))
		payoffs_str = ",".join([str(x) for x in payoffs.flatten()])
	elif flags.payoffs == "penalty":
		payoffs = np.array([[10, -10, 1], [-10, 10, 1], [1, 1, 2]])
		payoffs_str = ",".join([str(x) for x in payoffs.flatten()])
	elif flags.payoffs == "bottleneck":
		payoffs = np.array([[11, 0, 0], [0, 7, 6], [0, 6, 7]])
		payoffs_str = ",".join([str(x) for x in payoffs.flatten()])
	elif flags.payoffs == "climbing":
    # This is a particular payoff matrix that is hard for decentralized
    # algorithms. Introduced in C. Claus and C. Boutilier, "The dynamics of
    # reinforcement learning in cooperative multiagent systems", 1998, for
    # simultaneous action games, but it is difficult even in the case of
    # signaling games.
		payoffs = np.array([[11, -30, 0], [-30, 7, 6], [0, 0, 5]]) / 30
		payoffs_str = ",".join([str(x) for x in payoffs.flatten()])
	else:
		payoffs_str = flags.payoffs
		try:
			payoffs_list = [float(x) for x in payoffs_str.split(",")]
			payoffs = np.array(payoffs_list).reshape((num_states, num_states))
		except ValueError:
			raise ValueError(
					"There should be {} (states * actions) elements in payoff. "
					"Found {} elements".format(num_states * num_states, len(payoffs_list))
			) from None

	env_config = {
			"num_states": num_states,
			"num_messages": num_messages,
			"payoffs": payoffs_str,
	}

	game = pyspiel.load_game("lewis_signaling", env_config)

	biffurcation(game, flags, parameter_config)


def get_policy_from_logits(logits: chex.Array, temperature: float) -> chex.Array:
  """Converts internal beliefs into a valid probability distribution."""
  return jax.nn.softmax(logits * temperature)


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


def play_step(
		belief_matrix: chex.Array, 
		rng: chex.PRNGKey, 
		temperature: float, 
		obs_idx: int | None
	) -> int:
	# Get the index (e.g., which World State or which Message)
	logits = belief_matrix[obs_idx] if obs_idx else belief_matrix
	policy = get_policy_from_logits(logits, temperature)
	
	# Sample action (Message for Sender, Action for Receiver)
	action = jax.random.choice(rng, jnp.arange(len(policy)), p=policy)
	
	return action


def play_step_greedy(
		belief_matrix: chex.Array, 
		rng: chex.PRNGKey, 
		temperature: float, 
		obs_idx: int | None
	) -> int:
	# Get the index (e.g., which World State or which Message)
	logits = belief_matrix[obs_idx] if obs_idx is not None else belief_matrix
	policy = get_policy_from_logits(logits, temperature)
	
	# Sample action (Message for Sender, Action for Receiver)
	action = jnp.argmax(policy, axis=-1)
	
	return action


def play_epsilon_greedy(
	  belief_matrix: chex.Array, 
	  rng: chex.PRNGKey, 
	  epsilon: float, 
	  temperature: float, 
	  obs_idx: int | None
) -> int:
	# Get the index (e.g., which World State or which Message)
	
	logits = belief_matrix[obs_idx] if obs_idx else belief_matrix
	soft_policy = get_policy_from_logits(logits, temperature) 
	hard_policy = jnp.argmax(soft_policy, -1, keepdims=True)

	policy = epsilon * soft_policy + (1. - epsilon) * hard_policy
	
	# Sample action (Message for Sender, Action for Receiver)
	action = jax.random.choice(rng, jnp.arange(len(policy)), p=policy)
	
	return action


def soft_signaling_map(logits, temperature):
	"""Converts the belief matrix Z into a joint probability distribution P(W, M)."""
	# We treat Z as logits for signals given world states
	# pi(m|w) = softmax(Z)
	pi_m_given_w = get_policy_from_logits(logits, temperature)
	
	# Assume a uniform prior over world states P(w) = 1/N
	n_w = logits.shape[0]
	p_w = 1.0 / n_w
	
	# Joint distribution P(w, m)
	p_wm = p_w * pi_m_given_w
	return p_wm

def compute_extrinsic_reward_grad(logits, action_taken, payoffs, baselines, beta, kappa):
  """
  Computes the gradient of the expected reward potential.
  
  Args:
      logits: Current belief state (z).
      action_taken: The discrete index sampled from the policy.
      rewards: Scalar rewards for both agents from OpenSpiel.
  baselines: ...
      beta: Inverse temperature (Precision).
      kappa: Reward sensitivity (Environmental Coupling).
  """
  # 1. Compute policy (internal precision scaling)
  pi = get_policy_from_logits(logits, beta)
    
	# 2. Create the Indicator vector (One-hot encoding of the action taken)
  indicator = jax.nn.one_hot(action_taken, pi.shape[-1])
    
  # 3. Compute Sampled Advantage
  # How much better was this specific reward than what we usually get?
  advantage = payoffs - baselines
    
  # 4. The REINFORCE Gradient
  # Form: kappa * beta * (R - b) * (Indicator - pi)
  grad = kappa * beta * advantage * (indicator - pi)
  
  return grad

def extrinsic_analytical_reward_grad(z_s, z_r, U_mat, p_w, temperature):
    """
    z_s : (W, M) sender logits
    z_r : (M, A) receiver logits  
    U_mat : (W, A) shared payoff matrix from OpenSpiel
    p_w : (W,) prior over world states
    """
    pi_s = get_policy_from_logits(z_s, temperature)   # (W, M)
    pi_r =  get_policy_from_logits(z_r, temperature)  # (M, A)
    
    # --- Sender: for each (w,m), expected U over receiver's response ---
    # u_s[w,m] = sum_a pi_r[m,a] * U[w,a]
    u_s = jnp.einsum("ma,wa->wm", pi_r, U_mat)   # (W, M)
    u_mean_s = jnp.sum(pi_s * u_s, axis=1, keepdims=True)  # (W, 1)
    grad_s = pi_s * (u_s - u_mean_s)   # (W, M)
    
    # --- Receiver: for each (m,a), expected U over sender's state ---
    # p(w|m) = p(w) * pi_s[w,m] / p(m)
    p_m = jnp.einsum("w,wm->m", p_w, pi_s)   # (M,)
    p_w_given_m = (p_w[:, None] * pi_s) / (p_m[None, :] + 1e-12)  # (W, M)
    
    # u_r[m,a] = sum_w p(w|m) * U[w,a]
    u_r = jnp.einsum("wm,wa->ma", p_w_given_m, U_mat)   # (M, A)
    u_mean_r = jnp.sum(pi_r * u_r, axis=1, keepdims=True)  # (M, 1)
    grad_r = pi_r * (u_r - u_mean_r)   # (M, A)
    
    return grad_s, grad_r

def make_laplacian(n_row, n_col):
	"""Laplacian for an n_row x n_col matrix.
	L = L_row + L_col  where each is a complete-graph clique."""
	I_r, I_c = jnp.eye(n_row), jnp.eye(n_col)
	J_r, J_c = jnp.ones((n_row, n_row)), jnp.ones((n_col, n_col))
	# row clique: within each row, all entries connected
	L_row = jnp.kron(I_r, n_col * I_c - J_c)
	# column clique: within each col, all entries connected
	L_col = jnp.kron(n_row * I_r - J_r, I_c)
	return L_row + L_col


def get_beta_schedule(step, total_steps, beta_init=0.5, beta_final=5.0):
  """
  Computes the precision (beta) at a given training step.
  We use a cosine schedule for a smooth, quasi-static thermodynamic cooling.
  """
  fraction = jnp.clip(step / total_steps, 0.0, 1.0)
  
  # Cosine annealing from beta_init to beta_final
  cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * fraction))
  beta_t = beta_final + (beta_init - beta_final) * cosine_decay
  
  return beta_t

def calculate_laplacian_inhibition(logits: chex.Array):
	"""Constructs the bipartite competition term L*z."""
	# Row-wise competition: Signals competing for the same world state
	# We use a simple All-to-All inhibition minus self-feedback
	row_sum = jnp.sum(logits, axis=1, keepdims=True) - logits
	
	# Column-wise competition: World states competing for the same signal
	col_sum = jnp.sum(logits, axis=0, keepdims=True) - logits
	
	return row_sum + col_sum

def opinion_dynamics(t: chex.Array, x: chex.Array, args: tuple) -> chex.Array:
	r""" 
	\dot{z} = -\gamma z + \beta \tanh(z) + \kappa \nabla_{z} \mathbb{E}_{z}R - \eta \mathcal{L}z
	"""
	beta, eps, gamma, eta, shape, reward_grad = args
	logits = x.reshape(shape)

	# 1. Decay Term
	decay = -gamma * logits
	
	# 2. Nonlinear Commitment (The Bifurcation Driver)
	commitment = jnp.tanh(beta * logits + eps)
	
	# 3. REINFORCE-like Reward Drive
	reinforce = reward_grad
	
	# 4. Competitive Laplacian (Enforces Uniqueness)
	competition = -eta * calculate_laplacian_inhibition(logits)
	
	dLogits =  decay + commitment + competition + reinforce
	return dLogits.flatten()


def corrected_opinion_dynamics(t: chex.Array, x: chex.Array, args: tuple) -> chex.Array:
	r""" 
	\dot{z} = -\gamma z + \beta \tanh(z) + \kappa \nabla_{z} \mathbb{E}_{z}R - \eta \mathcal{L}z
	"""
	beta, temp, eps, gamma, kappa, eta, shape_s, shape_r, payoffs = args
	
	thr=shape_s[0]*shape_s[1]
	logits_s = x[:thr].reshape(shape_s)
	logits_r = x[thr:].reshape(shape_r)

	def compute_common_terms(logits):
		# 1. Decay Term
		decay = -gamma * logits
		# 2. Nonlinear Commitment (The Bifurcation Driver)
		commitment = jnp.tanh(beta * logits + eps)
		return decay + commitment
	
	n_w = logits_s.shape[0]
	p_w = jnp.ones(n_w) / n_w

	# 3. REINFORCE-like Reward Drive
	grads_s, grads_r = extrinsic_analytical_reward_grad(logits_s, logits_r, payoffs, p_w, temp)
	# 4. Competitive Laplacian (Enforces Uniqueness)
	L_s = make_laplacian(*grads_s.shape)    # sender Laplacian  (N_WM x N_WM)
	L_r = make_laplacian(*grads_r.shape)    # receiver Laplacian (N_MA x N_MA)
	
	dLogits_s = compute_common_terms(logits_s) + kappa * grads_s - eta * (L_s @ logits_s.flatten()).reshape(shape_s)
	dLogits_r = compute_common_terms(logits_r) + kappa * grads_r - eta * (L_r @ logits_r.flatten()).reshape(shape_r)

	return jnp.concatenate([dLogits_s.flatten(), dLogits_r.flatten()])

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

def game_iter(game: pyspiel.Game, flags: Flags, rng: chex.PRNGKey, beliefs: tuple) -> tuple:
	rng, rng_c, rng_s, rng_r = jax.random.split(rng, 4)

	assert len(beliefs) == 2
	sender_beliefs, receiver_beliefs = beliefs

	# Lewis Signalling state consists of (3 + num_states) bits:
	# 2 bits to indicate whose turn it is.
	# 1 bit to indicate whether it's terminal
	# one-hot vector for the state/message

	state = game.new_initial_state()
	
	# --- Episode Playthrough ---
	# 1. Nature chooses State
	if state.is_chance_node():
		outcomes = state.chance_outcomes()
		action_list, prob_list = zip(*outcomes)
		chance_action = jax.random.choice(rng_c, jnp.asarray(action_list), p=jnp.asarray(prob_list))
		state.apply_action(chance_action)
	
	if not state.is_terminal():
		# 2. Sender picks Message m
		w_idx = get_observation_index(state, PlayerID.SENDER)
		m = play_step(sender_beliefs, rng_s,  flags.force, w_idx)
		state.apply_action(m)
	
	if not state.is_terminal():
		# 3. Receiver picks Action a
		m_idx = get_observation_index(state, PlayerID.RECEIVER)
		a = play_step(receiver_beliefs, rng_r,  flags.force, m_idx)
		state.apply_action(a)
	
	# --- Reward and Gradient Calculation ---
	reward = state.returns()

	return rng, state, jnp.array(reward), jnp.array([m, a])

def run_simulation(flags: Flags, beliefs: tuple = None, return_logs: bool = True):
	game = "lewis_signaling"
	
	#game parameters
	num_players = 2
	num_states = flags.num_states
	num_messages = flags.num_messages
    

	if flags.payoffs == "random":
		payoffs = jnp.asarray(np.random.random((num_states, num_states)))
		payoffs_str = ",".join([str(x) for x in payoffs.flatten()])		
	elif flags.payoffs == "classic":
		payoffs_list = [1, 0, 0, 0, 1, 0, 0, 0, 1]
		payoffs = jnp.array(payoffs_list).reshape((num_states, num_states))
		payoffs_str = ",".join([str(x) for x in payoffs.flatten()])
	elif flags.payoffs == "prisoner_dilemma":
		payoffs = jnp.array([[2, 0], [3, 1]])
		num_states = num_messages = 2
		payoffs = payoffs.reshape((num_states, num_states))
		payoffs_str = ",".join([str(x) for x in payoffs.flatten()])
	elif flags.payoffs == "hawk_and_dove":
		payoffs = jnp.array([[1, 7], [2, 3]])
		num_states = num_messages = 2
		payoffs = payoffs.reshape((num_states, num_states))
		payoffs_str = ",".join([str(x) for x in payoffs.flatten()])
	elif flags.payoffs == "penalty":
		payoffs = jnp.array([[10, -10, 1], [-10, 10, 1], [1, 1, 2]])
		payoffs_str = ",".join([str(x) for x in payoffs.flatten()])
	elif flags.payoffs == "bottleneck":
		payoffs = jnp.array([[11, 0, 0], [0, 7, 6], [0, 6, 7]])
		payoffs_str = ",".join([str(x) for x in payoffs.flatten()])
	elif flags.payoffs == "climbing":
    # This is a particular payoff matrix that is hard for decentralized
    # algorithms. Introduced in C. Claus and C. Boutilier, "The dynamics of
    # reinforcement learning in cooperative multiagent systems", 1998, for
    # simultaneous action games, but it is difficult even in the case of
    # signaling games.
		payoffs = jnp.array([[11, -30, 0], [-30, 7, 6], [0, 0, 5]]) / 30
		payoffs_str = ",".join([str(x) for x in payoffs.flatten()])
	else:
		payoffs_str = flags.payoffs
		try:
			payoffs_list = [float(x) for x in payoffs_str.split(",")]
			payoffs = jnp.array(payoffs_list).reshape((num_states, num_states))
		except ValueError:
			raise ValueError(
					"There should be {} (states * actions) elements in payoff. "
					"Found {} elements".format(num_states * num_states, len(payoffs_list))
			) from None

	env_config = {
		"num_states": num_states,
		"num_messages": num_messages,
		"payoffs": payoffs_str,
	}

	game = pyspiel.load_game("lewis_signaling", env_config)
	num_actions = game.num_distinct_actions()
     

	delta_t = flags.intergration_step

	logs = dict(
		# rewards_s = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		# rewards_r = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		# leading_eigenvalue_s = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		# leading_eigenvalue_r = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		# opts_s = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		# opts_r = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		rewards = np.zeros((flags.num_runs, flags.num_iterations * flags.log_interval)),
		leading_eigenvalue = np.zeros((flags.num_runs, flags.num_iterations * flags.log_interval)),
		coordination_success_s = np.zeros((flags.num_runs, flags.num_iterations * flags.log_interval)),
		coordination_success_r = np.zeros((flags.num_runs, flags.num_iterations * flags.log_interval)),
		free_energy_s = np.zeros((flags.num_runs, flags.num_iterations * flags.log_interval)),
		free_energy_r = np.zeros((flags.num_runs, flags.num_iterations * flags.log_interval)),
		social_entropy = np.zeros((flags.num_runs, flags.num_iterations * flags.log_interval)),
		joint_mi = np.zeros((flags.num_runs, flags.num_iterations * flags.log_interval)),
		cic = np.zeros((flags.num_runs, flags.num_iterations * flags.log_interval)),
		expl = np.zeros((flags.num_runs, flags.num_iterations * flags.log_interval)),
		convergence_point = np.zeros((num_states, num_states)),
		percent_opt = 0
	)

	sender_beliefs_= np.zeros((flags.num_runs, num_states, num_messages))
	receiver_beliefs_ = np.zeros((flags.num_runs, num_messages, num_actions))

	for i in tqdm(range(flags.num_runs), total=flags.num_runs):
		rng = jax.random.key(flags.seed + i)
		# baseline = jnp.zeros(num_players)
		
		# for every run
		if beliefs is not None:
			assert len(beliefs) == 2
			sender_beliefs, receiver_beliefs = beliefs
			assert sender_beliefs.shape == (num_states, num_messages)
			assert receiver_beliefs.shape == (num_messages, num_actions)

		else:
			rng, rng_sender, rng_receiver = jax.random.split(rng, 3)
			# Sender: State -> Message
			sender_beliefs = jax.random.normal(rng_sender, (num_states, num_messages)) * 0.01
			# Receiver: Message -> Action
			receiver_beliefs = jax.random.normal(rng_receiver, (num_messages, num_actions)) * 0.01

		
		for ep in range(flags.num_iterations):
			
			## beta_t = get_beta_schedule(ep, flags.num_iterations, beta_init=0.5, beta_final=1.0)
			## flags = flags.replace(force=beta_t)
			# base_args = (flags.force, flags.force_eps, flags.damping, flags.coupling)

			# rng, state, reward, actions = game_iter(
			# 	game, flags, rng, (sender_beliefs, receiver_beliefs)
			# )
			
			# baseline = (1.0 - flags.baseline_alpha) * baseline + flags.baseline_alpha * reward
			# grad_sender = compute_extrinsic_reward_grad(
			# 	sender_beliefs, 
			# 	actions[int(PlayerID.SENDER)], 
			# 	reward[int(PlayerID.SENDER)],
			# 	baseline[int(PlayerID.SENDER)],
			# 	flags.force,
			# 	flags.learning_rate
			# )

			# grad_receiver = compute_extrinsic_reward_grad(
			# 	receiver_beliefs, 
			# 	actions[int(PlayerID.RECEIVER)], 
			# 	reward[int(PlayerID.RECEIVER)],
			# 	baseline[int(PlayerID.RECEIVER)],
			# 	flags.force,
			# 	flags.learning_rate
			# )

			sender_shape = sender_beliefs.shape
			receiver_shape = receiver_beliefs.shape
	
			base_args = (
				flags.force, 
				flags.temperature, 
				flags.force_eps, 
				flags.damping, 
				flags.learning_rate,
				flags.coupling,
				sender_shape,
				receiver_shape,
				payoffs
			)

			# --- ODE Integration Step (The Learning) ---
			
			# term = diffrax.ODETerm(opinion_dynamics)
			term = diffrax.ODETerm(corrected_opinion_dynamics) 
			solver = diffrax.Dopri5()

			# def make_func(t, args):
			# 	def func(x):
			# 		return opinion_dynamics(t, x, args)
			# 	return func
			
			# # Evolve Sender
			# sender_shape = sender_beliefs.shape
			# sol_s = diffrax.diffeqsolve(term, solver, t0=0, t1=flags.end_time, dt0=delta_t, y0=sender_beliefs.flatten(), 
			# 								args=base_args+(sender_shape, grad_sender))
			# sender_func = make_func(0, base_args+(sender_shape, grad_sender))
			# sender_beliefs = jnp.clip(sol_s.ys[-1].reshape(sender_shape), -100, 100)
			# if collect_trajectories:
			# 	sender_trajectories[i].append(sol_s.ys)
			
			# # Evolve Receiver
			# receiver_shape = receiver_beliefs.shape
			# sol_r = diffrax.diffeqsolve(term, solver, t0=0, t1=flags.end_time, dt0=delta_t, y0=receiver_beliefs.flatten(), 
			# 								args=base_args+(receiver_shape, grad_receiver))
			# receiver_func = make_func(0, base_args+(receiver_shape, grad_receiver))
			
			# receiver_beliefs = jnp.clip(sol_r.ys[-1].reshape(receiver_shape), -100, 100)
			# if collect_trajectories:
			# 	receiver_trajectories[i].append(sol_s.ys)

			
			def leading_eig(z):
				# differentiate vector field w.r.t. z at a dummy time (autonomous)
				def func(x):
					return corrected_opinion_dynamics(0.0, x, base_args)
				jac_fn = jax.jacfwd(lambda z_: func(z_))
				J = jac_fn(z)
				eigs = jnp.linalg.eigvals(J)
				return jnp.max(eigs.real)
			
			y0 = jnp.concatenate([sender_beliefs.flatten(), receiver_beliefs.flatten()])
			saveat = diffrax.SaveAt(ts=jnp.linspace(0, flags.end_time, flags.log_interval))
			sol = diffrax.diffeqsolve(
				term, 
				solver, 
				t0=0,
				t1=flags.end_time, 
				dt0=delta_t, 
				y0=y0, 
				args=base_args, 
				saveat=saveat, 
				max_steps=100000
			)
			thr = num_states*num_messages
			sender_beliefs, receiver_beliefs = sol.ys[-1][:thr], sol.ys[-1][thr:]

			sender_beliefs = jnp.clip(sender_beliefs.reshape(sender_shape), -100, 100)
			receiver_beliefs = jnp.clip(receiver_beliefs.reshape(receiver_shape), -100, 100)

			if return_logs:
				sender_policy, receiver_policy = jax.tree.map(
					lambda x: get_policy_from_logits(x, flags.temperature), (sender_beliefs, receiver_beliefs)
				)
				
				# max_reward_s = payoffs[get_observation_index(state, PlayerID.SENDER)].max()
				# max_reward_r = payoffs[get_observation_index(state, PlayerID.RECEIVER)].max()
				
				cur_idx = (i, ep // flags.log_interval)
				expl, _ = calculate_exploitability(game, sender_policy, receiver_policy)  

				# logs["rewards_s"][cur_idx] += reward[PlayerID.SENDER] / flags.log_interval
				# logs["rewards_r"][cur_idx] += reward[PlayerID.RECEIVER] / flags.log_interval

				# logs["leading_eigenvalue_s"][cur_idx] += jnp.max(
				# 	jnp.real(
				# 		jnp.linalg.eigvals(
				# 			jax.jacfwd(sender_func)(sender_beliefs.flatten())
				# 		)
				# 	)
				# ) / flags.log_interval

				# logs["leading_eigenvalue_r"][cur_idx] += jnp.max(
				# 	jnp.real(
				# 		jnp.linalg.eigvals(
				# 			jax.jacfwd(receiver_func)(receiver_beliefs.flatten())
				# 		)
				# 	)
				# )  / flags.log_interval

				# logs["opts_s"][cur_idx] += np.isclose(reward[PlayerID.SENDER], max_reward_s) / flags.log_interval * 100
				# logs["opts_r"][cur_idx] += np.isclose(reward[PlayerID.RECEIVER], max_reward_r) / flags.log_interval * 100

				logs["leading_eigenvalue"][cur_idx] = leading_eig(jnp.concatenate([sender_beliefs.flatten(), receiver_beliefs.flatten()]))
				logs["coordination_success_s"][cur_idx] += calculate_coordination_success(sender_policy) / flags.log_interval * 100
				logs["coordination_success_r"][cur_idx] += calculate_coordination_success(receiver_policy) / flags.log_interval * 100

				logs["free_energy_s"][cur_idx] += calculate_free_energy(sender_policy) / flags.log_interval
				logs["free_energy_r"][cur_idx] += calculate_free_energy(receiver_policy) / flags.log_interval


				soc_ent, joint_mi = calculate_social_metrics(sender_policy, receiver_policy) 
				logs["social_entropy"][cur_idx] += soc_ent / flags.log_interval
				logs["joint_mi"][cur_idx] += joint_mi / flags.log_interval
				logs["cic"][cur_idx] += calculate_cic(sender_policy, receiver_policy) / flags.log_interval
				logs["expl"][cur_idx] += expl / flags.log_interval

		base_info_state0 = [1.0, 0.0, 0.0] + [0.0] * num_states
		base_info_state1 = [0.0, 1.0, 0.0] + [0.0] * num_states 
		rng_s, rng_r, rng = jax.random.split(rng, 3)
		if return_logs:
			for s in range(num_states):
				info_state0 = copy.deepcopy(base_info_state0)
				a_idx = s
				info_state0[a_idx + 3] = 1.0
				m = play_step_greedy(sender_beliefs, rng_s, flags.temperature, a_idx)

				info_state1 = copy.deepcopy(base_info_state1)
				m_idx = m
				info_state1[m_idx + 3] = 1.0
				a = play_step_greedy(receiver_beliefs, rng_r, flags.temperature, m_idx)

				logs["convergence_point"][s, a] += 1
				best_act = payoffs[s].argmax()
				logs["percent_opt"] += int(a == best_act) / flags.num_runs / num_states
				logs["rewards"][cur_idx] += payoffs[s, a] / flags.log_interval

		sender_beliefs_[i] = sender_beliefs
		receiver_beliefs_[i] = receiver_beliefs

	# return_value = jax.tree.map(lambda x: x.mean(0), (sender_beliefs_, receiver_beliefs_)), logs

	return (sender_beliefs_, receiver_beliefs_), logs

		
if __name__ == "__main__":
  run_simulation(Flags())