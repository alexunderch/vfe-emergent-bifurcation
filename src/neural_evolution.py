from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
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
	calculate_laplacian_inhibition,
	corrected_phi
)

"""
neural_vfe_baseline.py
Neural VFE Dynamics baseline for Lewis signalling game.

Key difference from REINFORCE:
- Loss = VFE potential F(Z) from Eq. 12 (NOT just expected utility)
- Network trained by VFE descent = discretized ODE flow
- Spectral diagnostic (Jacobian of **ODE** field at network outputs) tracks emergence
"""

class Flags(NamedTuple):
	num_states: int = 3 
	num_messages: int = 3 
	num_actions: int = 3
	hidden: int = 64

	payoffs: str = "classic"
	num_runs: int = 20
	log_every: int  = 1
	num_iterations: int = 100

	lr: float 				 = 0.01
	temperature: float    	 = 1.0
	learning_rate: float  	 = 30.0
	force: float          	 = 1.0
	self_attention: float 	 = 2.0
	eps: float     		  	 = 0.025
	damping: float        	 = 1.15
	coupling: float       	 = 0.55

	seed: int = 42


class SenderNet(NamedTuple):
    W1: chex.Array
    b1: chex.Array
    W2: chex.Array
    b2: chex.Array

class ReceiverNet(NamedTuple):
    W1: chex.Array
    b1: chex.Array
    W2: chex.Array
    b2: chex.Array

def init_sender(key, flags: Flags):
    k1, k2 = jax.random.split(key)
    return SenderNet(
        W1=jax.random.normal(k1, (flags.num_states, flags.hidden)) * jnp.sqrt(2.0 / flags.num_states),
        b1=jnp.zeros(flags.hidden),
        W2=jax.random.normal(k2, (flags.hidden, flags.num_messages)) * 0.01,
        b2=jnp.zeros(flags.num_messages),
    )

def init_receiver(key, flags: Flags):
    k1, k2 = jax.random.split(key)
    return ReceiverNet(
        W1=jax.random.normal(k1, (flags.num_messages, flags.hidden)) * jnp.sqrt(2.0 / flags.num_messages),
        b1=jnp.zeros(flags.hidden),
        W2=jax.random.normal(k2, (flags.hidden, flags.num_actions)) * 0.01,
        b2=jnp.zeros(flags.num_actions),
    )

def forward(net, x):
    h = jax.nn.relu(x @ net.W1 + net.b1)
    return h @ net.W2 + net.b2

def get_implicit_Z(sender_net, receiver_net, flags: Flags):
    """
    Evaluate network on all possible inputs to recover the full Z matrices.
    Z_s[i,j] = sender_logits for state i, message j
    Z_r[i,j] = receiver_logits for message i, action j
    """
    states = jnp.eye(flags.num_states)
    msgs = jnp.eye(flags.num_messages)
    
    Z_s = jax.vmap(lambda s: forward(sender_net, s))(states)      # (W, M)
    Z_r = jax.vmap(lambda m: forward(receiver_net, m))(msgs)      # (M, A)
    
    return Z_s, Z_r

def get_policy_from_logits(logits: chex.Array, temperature: float) -> chex.Array:
  """Converts internal beliefs into a valid probability distribution."""
  return jax.nn.softmax(logits * temperature)


def expected_utility(
	Z_s: chex.Array, 
	Z_r: chex.Array, 
	U: chex.Array, 
	p_w: chex.Array, 
	temp: float = 1.0
) -> chex.Array:
	"""Computing expected utility"""
	pi_s = get_policy_from_logits(Z_s, temp)	
	pi_r = get_policy_from_logits(Z_r, temp)
	return jnp.einsum("w,wm,ma,wa->", p_w, pi_s, pi_r, U)

def extrinsic_analytical_reward_grad(
	Z_s: chex.Array, 
	Z_r: chex.Array, 
	U_mat: chex.Array, 
	p_w: chex.Array, 
	temperature: float):
    """
    Z_s : (W, M) sender logits
    Z_r : (M, A) receiver logits  
    U_mat : (W, A) shared payoff matrix from OpenSpiel
    p_w : (W,) prior over world states
    """
    pi_s = get_policy_from_logits(Z_s, temperature)   # (W, M)
    pi_r =  get_policy_from_logits(Z_r, temperature)  # (M, A)
    
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


@partial(jax.jit, static_argnames = ("flags", "opts"))
def train_step(sender_net, receiver_net, state_s, state_r, 
               U, p_w, flags: Flags, opts: tuple):
	"""
	One gradient descent step on VFE potential F(Z(θ)).
	By chain rule: ∇_θ F = J_θ^T ∇_Z F = -J_θ^T f(Z).
	So this is discretized VFE descent = neuralized ODE flow.
	"""

	def loss_fn(sender_net, receiver_net):
		Z_s, Z_r = get_implicit_Z(sender_net, receiver_net, flags)
		return dynamical_vfe(Z_s, Z_r, U, p_w, flags)

    # Sender gradient: dF/dθ_s = dF/dZ_s * dZ_s/dθ_s
	loss_s, grads_s = jax.value_and_grad(
        lambda s: loss_fn(s, receiver_net)
    )(sender_net)
    
    # Receiver gradient: dF/dθ_r = dF/dZ_r * dZ_r/dθ_r  
	loss_r, grads_r = jax.value_and_grad(
		lambda r: loss_fn(sender_net, r)
	)(receiver_net)
    
	# Total loss (should match, verify with assert in debug)
	loss = (loss_s + loss_r) / 2  # They are identical by construction

	opt_s, opt_r = opts

	# Apply updates
	updates_s, state_s = opt_s.update(grads_s, state_s, sender_net)
	new_sender = optax.apply_updates(sender_net, updates_s)

	updates_r, state_r = opt_r.update(grads_r, state_r, receiver_net)
	new_receiver = optax.apply_updates(receiver_net, updates_r)

	return new_sender, new_receiver, state_s, state_r, loss


def compute_mi(sender_net, receiver_net, flags: Flags, key, n_samples=2000):
    """Empirical I(W;A) via Monte Carlo."""
    keys = jax.random.split(key, n_samples)
    states = jax.random.randint(keys[0], (n_samples,), 0, flags.num_states)
    
    Z_s, Z_r = get_implicit_Z(sender_net, receiver_net, flags)
    pi_s = get_policy_from_logits(Z_s, flags.temperature)
    pi_r = get_policy_from_logits(Z_r, flags.temperature)
    
    # Sample messages and actions from policies
    def sample_episode(k, world):
        msg_logits = pi_s[world]
        msg = jax.random.categorical(k, msg_logits * flags.temperature)
        act_logits = pi_r[msg]
        act = jax.random.categorical(k, act_logits * flags.temperature)
        return world, act
    
    keys = jax.random.split(keys[0], n_samples)
    worlds, actions = jax.vmap(sample_episode)(keys, states)
    
    # Empirical joint
    joint = jnp.zeros((flags.num_states, flags.num_actions))
    for i in range(n_samples):
        joint = joint.at[worlds[i], actions[i]].add(1.0 / n_samples)
    
    p_w = joint.sum(axis=1, keepdims=True)
    p_a = joint.sum(axis=0, keepdims=True)
    
    mask = joint > 1e-10
    mi = jnp.sum(mask * joint * jnp.log2(joint / (p_w * p_a + 1e-10) + 1e-10))
    return float(mi)


def ode_vector_field(Z_s, Z_r, U, p_w, flags: Flags):
	"""
	f(Z) = -∇F(Z) = -γZ + tanh(βZ+ε) + κ∇U - ηLZ
	Returns flattened vector [f_s_flat; f_r_flat].
	"""

	u, temp, alpha, eps, gamma, kappa, eta = (
		flags.force, 
		flags.temperature, 
		flags.self_attention, 
		flags.eps, 
		flags.damping, 
		flags.learning_rate, 
		flags.coupling
	)
	logits_s = Z_s
	logits_r = Z_r
	payoffs = U

	decay_s = -gamma * logits_s
	decay_r = -gamma * logits_r
	
	tanh_s = jnp.tanh(alpha * logits_s + eps)
	tanh_r = jnp.tanh(alpha * logits_r + eps)

	n_w = logits_s.shape[0]
	p_w = jnp.ones(n_w) / n_w

	# Phi-modulated Laplacian competition
	phi_s = corrected_phi(logits_r, temp)
	phi_r = corrected_phi(logits_r, temp)

	L_s = calculate_laplacian_inhibition(logits_s) * phi_s
	L_r = calculate_laplacian_inhibition(logits_r) * phi_r

	# 3. REINFORCE-like Reward Drive
	grads_s, grads_r = extrinsic_analytical_reward_grad(logits_s, logits_r, payoffs, p_w, temp)
	# 4. Competitive Laplacian (Enforces Uniqueness)
	dLogits_s = decay_s \
				+ tanh_s \
				+ kappa * grads_s + eta * L_s
	
	dLogits_r = decay_r \
				+ tanh_r \
				+ kappa * grads_r + eta * L_r
	

	return jnp.concatenate([dLogits_s.flatten(), dLogits_r.flatten()])

def spectral_diagnostic(Z_s, Z_r, U, p_w, flags):
    """
    Compute leading eigenvalue of Jacobian of ODE field at current Z.
    This is the EXACT same diagnostic from the paper, applied to the
    neural network's implicit Z.
    """
    z_vec = jnp.concatenate([Z_s.flatten(), Z_r.flatten()])
    
    def field_fn(z):
        thr = flags.num_states * flags.num_messages
        zs = z[:thr].reshape(flags.num_states, flags.num_messages)
        zr = z[thr:].reshape(flags.num_messages, flags.num_actions)
        return ode_vector_field(zs, zr, U, p_w, flags)
    
    jac_fn = jax.jacfwd(field_fn)
    J = jac_fn(z_vec)
    eigs = jnp.linalg.eigvals(J)
    return jnp.max(eigs.real)


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
     

	logs = dict(
		rewards = np.zeros((flags.num_runs, flags.num_iterations//flags.log_every)),
		loss = np.zeros((flags.num_runs, flags.num_iterations//flags.log_every)),    
		leading_eigenvalue = np.zeros((flags.num_runs, flags.num_iterations//flags.log_every)),
		coordination_success = np.zeros((flags.num_runs, flags.num_iterations//flags.log_every)),
		free_energy_dyn = np.zeros((flags.num_runs, flags.num_iterations//flags.log_every)),
		free_energy_mi = np.zeros((flags.num_runs, flags.num_iterations//flags.log_every)),
		social_entropy = np.zeros((flags.num_runs, flags.num_iterations//flags.log_every)),
		joint_mi = np.zeros((flags.num_runs, flags.num_iterations//flags.log_every)),
		cic = np.zeros((flags.num_runs, flags.num_iterations//flags.log_every)),
		expl = np.zeros((flags.num_runs, flags.num_iterations//flags.log_every)),
	)

	sender_beliefs_= np.zeros((flags.num_runs, num_states, num_messages))
	receiver_beliefs_ = np.zeros((flags.num_runs, num_messages, num_actions))
	p_w = jnp.ones(num_states) / num_states
      

	for i in tqdm(range(flags.num_runs), total=flags.num_runs):
		rng = jax.random.key(flags.seed + i)
		
		# for every run
		if beliefs is not None:
			assert len(beliefs) == 2
			sender_params, receiver_params = beliefs
		else:
			rng, rng_sender, rng_receiver = jax.random.split(rng, 3)
			# Sender: State -> Message
			sender_params = init_sender(rng_sender, flags)
			# Receiver: Message -> Action
			receiver_params = init_receiver(rng_receiver, flags)

		# Optimizers (Adam approximates continuous flow for small lr)
		opt_s = optax.adam(flags.lr)
		opt_r = optax.adam(flags.lr)

		state_s = opt_s.init(sender_params)
		state_r = opt_r.init(receiver_params)

		for ep in range(flags.num_iterations):

			sender_params, receiver_params, state_s, state_r, loss = train_step(
            	sender_params, receiver_params, state_s, state_r, payoffs, p_w, flags, (opt_s, opt_r)
        	)
			sender_beliefs, receiver_beliefs = get_implicit_Z(sender_params, receiver_params, flags)
                  
			sender_policy, receiver_policy = jax.tree.map(
				lambda x: get_policy_from_logits(x, flags.temperature), 
				(sender_beliefs, receiver_beliefs)
			)

			cur_idx = (i, ep)
			expl, _ = calculate_exploitability(game, sender_policy, receiver_policy)  

			logs["loss"][cur_idx] = jnp.mean(loss)
			logs["leading_eigenvalue"][cur_idx] = spectral_diagnostic(
				sender_beliefs, receiver_beliefs, payoffs, p_w, flags
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
				
			logs["coordination_success"][cur_idx] = compute_coordination_success_analytical(
				sender_policy, receiver_policy, payoffs
			) * 100
				
			logs["rewards"][cur_idx] = compute_expected_reward(
				sender_policy, receiver_policy, p_w, payoffs
			)

		sender_beliefs_[i] = sender_beliefs
		receiver_beliefs_[i] = receiver_beliefs

	return (sender_beliefs_, receiver_beliefs_), logs

if __name__ == "__main__":
    run_simulation(Flags())