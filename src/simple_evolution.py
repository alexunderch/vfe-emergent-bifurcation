# import copy
import enum
from typing import NamedTuple

import chex
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pyspiel
from tqdm.auto import tqdm

from metrics import (
	calculate_cic,
	calculate_exploitability,
	calculate_free_energy,
	dynamical_vfe,
	calculate_social_metrics,
	compute_coordination_success_analytical,
	compute_expected_reward,
	calculate_laplacian_inhibition
)


class Flags(NamedTuple):
	num_states: int = 3 
	num_messages: int = 3 
	payoffs: str = "classic"
	num_iterations: int = 1
	num_runs: int = 20
	intergration_step: float = 0.01
	end_time: float = 0.5
	num_saves: int = 100
	baseline_alpha: float = 0.01

	temperature: float = 1.0
	learning_rate: float = 3.0
	force: float = 2.5
	force_eps: float = 0.025
	damping: float = 1.35
	coupling: float = 0.5
	
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

def corrected_opinion_dynamics(t: chex.Array, x: chex.Array, args: tuple) -> chex.Array:
	r""" 
	\dot{z} = -\gamma z + \beta \tanh(z) + \kappa \nabla_{z} \mathbb{E}_{z}R - \eta \mathcal{L}z
	"""
	beta, temp, eps, gamma, kappa, eta, shape_s, shape_r, payoffs = args
	
	thr=shape_s[0]*shape_s[1]
	logits_s = x[:thr].reshape(shape_s)
	logits_r = x[thr:].reshape(shape_r)

	decay_s = -(gamma) * logits_s
	decay_r = -gamma * logits_r
    
	tanh_s = jnp.tanh(beta * logits_s + eps)
	tanh_r = jnp.tanh(beta * logits_r + eps)
	
	n_w = logits_s.shape[0]
	p_w = jnp.ones(n_w) / n_w

	# 3. REINFORCE-like Reward Drive
	grads_s, grads_r = extrinsic_analytical_reward_grad(logits_s, logits_r, payoffs, p_w, temp)
	# 4. Competitive Laplacian (Enforces Uniqueness)
	dLogits_s = decay_s \
				+ tanh_s \
				+ kappa * grads_s \
				- eta * calculate_laplacian_inhibition(logits_s)
	 
	dLogits_r = decay_r \
				+ tanh_r \
				+ kappa * grads_r \
				- eta * calculate_laplacian_inhibition(logits_r) 

	return jnp.concatenate([dLogits_s.flatten(), dLogits_r.flatten()])

def run_simulation(flags: Flags, beliefs: tuple = None):

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
		rewards = np.zeros((flags.num_runs, flags.num_iterations * flags.num_saves)),
		leading_eigenvalue = np.zeros((flags.num_runs, flags.num_iterations * flags.num_saves)),
		coordination_success = np.zeros((flags.num_runs, flags.num_iterations * flags.num_saves)),
		free_energy_dyn = np.zeros((flags.num_runs, flags.num_iterations * flags.num_saves)),
		free_energy_mi = np.zeros((flags.num_runs, flags.num_iterations * flags.num_saves)),
		social_entropy = np.zeros((flags.num_runs, flags.num_iterations * flags.num_saves)),
		joint_mi = np.zeros((flags.num_runs, flags.num_iterations * flags.num_saves)),
		cic = np.zeros((flags.num_runs, flags.num_iterations * flags.num_saves)),
		expl = np.zeros((flags.num_runs, flags.num_iterations * flags.num_saves)),
		convergence_point = np.zeros((num_states, num_states)),
		percent_opt = 0
	)

	sender_beliefs_= np.zeros((flags.num_runs, num_states, num_messages))
	receiver_beliefs_ = np.zeros((flags.num_runs, num_messages, num_actions))
	p_w = jnp.ones(num_states) / num_states

	for i in tqdm(range(flags.num_runs), total=flags.num_runs):
		rng = jax.random.key(flags.seed + i)
		
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
				payoffs,
			)

			# --- ODE Integration Step (The Learning) ---
			
			# term = diffrax.ODETerm(opinion_dynamics)
			term = diffrax.ODETerm(corrected_opinion_dynamics) 
			solver = diffrax.Dopri5()

			def leading_eig(z):
				# differentiate vector field w.r.t. z at a dummy time (autonomous)
				def func(x):
					return corrected_opinion_dynamics(0.0, x, base_args)
				jac_fn = jax.jacfwd(lambda z_: func(z_))
				J = jac_fn(z)
				eigs = jnp.linalg.eigvals(J)
				return jnp.max(eigs.real)
			
			y0 = jnp.concatenate([sender_beliefs.flatten(), receiver_beliefs.flatten()])
			saveat = diffrax.SaveAt(ts=jnp.linspace(0, flags.end_time, flags.num_saves))
			sol = diffrax.diffeqsolve(
				term, 
				solver, 
				t0=0,
				t1=flags.end_time, 
				dt0=delta_t, 
				y0=y0, 
				args=base_args, 
				saveat=saveat, 
				max_steps=int(1e6),
				stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
			)
			thr = num_states*num_messages
			for t in range(flags.num_saves):

				sender_beliefs, receiver_beliefs = sol.ys[t][:thr], sol.ys[t][thr:]
				# print(jax.tree.map(jnp.linalg.norm, (sender_beliefs, receiver_beliefs)))
				# sender_beliefs = jnp.clip(sender_beliefs.reshape(sender_shape), -100, 100)
				# receiver_beliefs = jnp.clip(receiver_beliefs.reshape(receiver_shape), -100, 100)
				
				sender_beliefs = sender_beliefs.reshape(sender_shape)
				receiver_beliefs = receiver_beliefs.reshape(receiver_shape)
				
				sender_policy, receiver_policy = jax.tree.map(
					lambda x: get_policy_from_logits(x, flags.temperature), 
					(sender_beliefs, receiver_beliefs)
				)
				
				cur_idx = (i, ep * flags.num_saves + t)
				expl, _ = calculate_exploitability(game, sender_policy, receiver_policy)  

				logs["leading_eigenvalue"][cur_idx] = leading_eig( 
					jnp.concatenate([sender_beliefs.flatten(), receiver_beliefs.flatten()])
				) 

				p_wa, soc_ent, joint_mi = calculate_social_metrics(sender_policy, receiver_policy) 

				logs["free_energy_dyn"][cur_idx] = dynamical_vfe(
					sender_beliefs, receiver_beliefs, payoffs, p_w,
					flags
				) 

				logs["free_energy_mi"][cur_idx] = calculate_free_energy(
					sender_policy, receiver_policy, payoffs,
					flags.learning_rate
				) 

				logs["social_entropy"][cur_idx] = soc_ent 
				logs["joint_mi"][cur_idx] = joint_mi 
				logs["cic"][cur_idx] = calculate_cic(sender_policy, receiver_policy) 
				logs["expl"][cur_idx] = expl 
					
				logs["rewards"][cur_idx] = compute_expected_reward(
					sender_policy, receiver_policy, p_w, payoffs
				)
				logs["coordination_success"][cur_idx] = compute_coordination_success_analytical(
					sender_policy, receiver_policy, payoffs
				) * 100


		# base_info_state0 = [1.0, 0.0, 0.0] + [0.0] * num_states
		# base_info_state1 = [0.0, 1.0, 0.0] + [0.0] * num_states 
		# rng_s, rng_r, rng = jax.random.split(rng, 3)

		# for s in range(num_states):
		# 	info_state0 = copy.deepcopy(base_info_state0)
		# 	a_idx = s
		# 	info_state0[a_idx + 3] = 1.0
		# 	m = play_step_greedy(sender_beliefs, rng_s, flags.temperature, a_idx)

		# 	info_state1 = copy.deepcopy(base_info_state1)
		# 	m_idx = m
		# 	info_state1[m_idx + 3] = 1.0
		# 	a = play_step_greedy(receiver_beliefs, rng_r, flags.temperature, m_idx)

		# 	logs["convergence_point"][s, a] += 1
		# 	best_act = payoffs[s].argmax()
		# 	logs["percent_opt"] = int(a == best_act) / flags.num_runs / num_states
		# 	logs["rewards"][cur_idx] += payoffs[s, a] 

		sender_beliefs_[i] = sender_beliefs
		receiver_beliefs_[i] = receiver_beliefs

	return (sender_beliefs_, receiver_beliefs_), logs

		
if __name__ == "__main__":
  run_simulation(Flags())