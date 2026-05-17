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


from metrics import (
	calculate_coordination_success, 
	calculate_social_metrics, 
	calculate_free_energy, 
	calculate_cic,
)

class Flags(NamedTuple):
	num_states: int = 3 
	num_messages: int = 3 
	payoffs: str = "classic"
	num_iterations: int = 100
	num_runs: int = 5
	log_interval: int = 10
	intergration_step: float = 0.01
	end_time: float = 0.5
	baseline_alpha: float = 0.01


	# learning_rate: float = 0.05
	# force: float = 4.5
	# force_eps: float = 0.01
	# damping: float = 1.5
	# coupling: float = 0.5

	learning_rate: float = 0.01
	force: float = 5.0
	force_eps: float = 0.01
	damping: float = 1.
	coupling: float = 0.67
	
	## Classic coordination
	# learning_rate: float = 0.05
	# force: float = 2.0
	# force_eps: float = 0.025
	# damping: float = 1.35
	# coupling: float = 0.55

	## Stag Hunt
	# learning_rate: float = 0.1
	# force: float = 3.5
	# force_eps: float = 0.01
	# damping: float = 1.25
	# coupling: float = 0.55
	
	seed: int = 42

class PlayerID(enum.IntEnum):
	SENDER = 0 
	RECEIVER = 1


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


def play_step(belief_matrix: chex.Array, rng: chex.PRNGKey, temperature: float, obs_idx: int) -> int:
	# Get the index (e.g., which World State or which Message)
	logits = belief_matrix[obs_idx]
	policy = get_policy_from_logits(logits, temperature)
	
	# Sample action (Message for Sender, Action for Receiver)
	action = jax.random.choice(rng, jnp.arange(len(policy)), p=policy)
	
	return action


def play_step_g(belief_matrix: chex.Array, rng: chex.PRNGKey, temperature: float, obs_idx: int) -> int:
	# Get the index (e.g., which World State or which Message)
	logits = belief_matrix[obs_idx]
	policy = get_policy_from_logits(logits, temperature)
	
	# Sample action (Message for Sender, Action for Receiver)
	action = jnp.argmax(policy, axis=-1)
	
	return action


def play_epsilon_greedy(
	  belief_matrix: chex.Array, 
	  rng: chex.PRNGKey, 
	  epsilon: float, 
	  temperature: float, 
	  obs_idx: int
) -> int:
	# Get the index (e.g., which World State or which Message)
	
	logits = belief_matrix[obs_idx]
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

def run_simulation(flags: Flags, beliefs: tuple = None, return_logs: bool = True, collect_trajectories: bool = False):
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
	elif flags.payoffs == "prisoner_dilemma":
		payoffs = np.array([[2, 0], [3, 1]])
		num_states = num_messages = 2
		payoffs = payoffs.reshape((num_states, num_states))
		payoffs_str = ",".join([str(x) for x in payoffs.flatten()])
	elif flags.payoffs == "hawk_and_dove":
		payoffs = np.array([[1, 7], [2, 3]])
		num_states = num_messages = 2
		payoffs = payoffs.reshape((num_states, num_states))
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
	num_actions = game.num_distinct_actions()
     

	delta_t = flags.intergration_step

	logs = dict(
		# br_s = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		# br_r = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		rewards_s = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		rewards_r = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		leading_eigenvalue_s = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		leading_eigenvalue_r = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		opts_s = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		opts_r = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		coordination_success_s = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		coordination_success_r = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		free_energy_s = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		free_energy_r = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		social_entropy = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		joint_mi = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		cic = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		expl = np.zeros((flags.num_runs, flags.num_iterations // flags.log_interval)),
		convergence_point = np.zeros((num_states, num_states)),
		percent_opt = 0
	)

	sender_beliefs_= np.zeros((flags.num_runs, num_states, num_messages))
	receiver_beliefs_ = np.zeros((flags.num_runs, num_messages, num_actions))
    
	if collect_trajectories:
		sender_trajectories = [[] for _ in range(flags.num_runs)]
		receiver_trajectories = [[] for _ in range(flags.num_runs)]

	for i in tqdm(range(flags.num_runs), total=flags.num_runs):
		rng = jax.random.key(flags.seed + i)
		baseline = jnp.zeros(num_players)
		
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
			# beta_t = get_beta_schedule(ep, flags.num_iterations, beta_init=0.5, beta_final=1.0)
			# flags = flags.replace(force=beta_t)
			base_args = (flags.force, flags.force_eps, flags.damping, flags.coupling)

			rng, state, reward, actions = game_iter(
				game, flags, rng, (sender_beliefs, receiver_beliefs)
			)
			
			baseline = (1.0 - flags.baseline_alpha) * baseline + flags.baseline_alpha * reward
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

			# --- ODE Integration Step (The Learning) ---
			
			term = diffrax.ODETerm(opinion_dynamics)
			solver = diffrax.Dopri5()

			def make_func(t, args):
				def func(x):
					return opinion_dynamics(t, x, args)
				return func
			
			# Evolve Sender
			sender_shape = sender_beliefs.shape
			sol_s = diffrax.diffeqsolve(term, solver, t0=0, t1=flags.end_time, dt0=delta_t, y0=sender_beliefs.flatten(), 
											args=base_args+(sender_shape, grad_sender))
			sender_func = make_func(0, base_args+(sender_shape, grad_sender))
			sender_beliefs = jnp.clip(sol_s.ys[-1].reshape(sender_shape), -100, 100)
			if collect_trajectories:
				sender_trajectories[i].append(sol_s.ys)
			
			# Evolve Receiver
			receiver_shape = receiver_beliefs.shape
			sol_r = diffrax.diffeqsolve(term, solver, t0=0, t1=flags.end_time, dt0=delta_t, y0=receiver_beliefs.flatten(), 
											args=base_args+(receiver_shape, grad_receiver))
			receiver_func = make_func(0, base_args+(receiver_shape, grad_receiver))
			
			receiver_beliefs = jnp.clip(sol_r.ys[-1].reshape(receiver_shape), -100, 100)
			if collect_trajectories:
				receiver_trajectories[i].append(sol_s.ys)

			# print(jax.tree.map(jnp.linalg.norm, (sender_beliefs, receiver_beliefs)))
			if return_logs:
				sender_policy, receiver_policy = jax.tree.map(
					lambda x: get_policy_from_logits(x, flags.force), (sender_beliefs, receiver_beliefs)
				)
				
				max_reward_s = payoffs[get_observation_index(state, PlayerID.SENDER)].max()
				max_reward_r = payoffs[get_observation_index(state, PlayerID.RECEIVER)].max()
				
				cur_idx = (i, ep // flags.log_interval)
				expl, _ = calculate_exploitability(game, sender_policy, receiver_policy)  

				logs["rewards_s"][cur_idx] += reward[PlayerID.SENDER] / flags.log_interval
				logs["rewards_r"][cur_idx] += reward[PlayerID.RECEIVER] / flags.log_interval
				# logs["br_s"][cur_idx] += br_s / flags.log_interval
				# logs["br_r"][cur_idx] += br_r / flags.log_interval

				logs["leading_eigenvalue_s"][cur_idx] += jnp.max(
					jnp.real(
						jnp.linalg.eigvals(
							jax.jacfwd(sender_func)(sender_beliefs.flatten())
						)
					)
				) / flags.log_interval

				logs["leading_eigenvalue_r"][cur_idx] += jnp.max(
					jnp.real(
						jnp.linalg.eigvals(
							jax.jacfwd(receiver_func)(receiver_beliefs.flatten())
						)
					)
				)  / flags.log_interval

				logs["opts_s"][cur_idx] += np.isclose(reward[PlayerID.SENDER], max_reward_s) / flags.log_interval * 100
				logs["opts_r"][cur_idx] += np.isclose(reward[PlayerID.RECEIVER], max_reward_r) / flags.log_interval * 100
			
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
				m = play_step_g(sender_beliefs, rng_s, flags.force, a_idx)

				info_state1 = copy.deepcopy(base_info_state1)
				m_idx = m
				info_state1[m_idx + 3] = 1.0
				a = play_step_g(receiver_beliefs, rng_r, flags.force, m_idx)

				logs["convergence_point"][s, a] += 1
				best_act = payoffs[s].argmax()
				logs["percent_opt"] += int(a == best_act) / flags.num_runs / num_states
			
		sender_beliefs_[i] = sender_beliefs
		receiver_beliefs_[i] = receiver_beliefs

	return_value = jax.tree.map(lambda x: x.mean(0), (sender_beliefs_, receiver_beliefs_)), logs

	if collect_trajectories:
		return return_value + (sender_trajectories, receiver_trajectories)
	return return_value

		
if __name__ == "__main__":
  run_simulation(Flags())