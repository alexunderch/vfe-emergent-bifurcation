import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
import enum
import pyspiel

from simple_evoultion import (
	Flags as SFlags, 
	run_simulation as simple_simulation,
	game_iter, 
	get_policy_from_logits,
	opinion_dynamics,
	compute_extrinsic_reward_grad, 
	calculate_exploitability,
	PlayerID
)
from metrics import calculate_mi, calculate_social_metrics, calculate_cic, calculate_free_energy
from scipy import stats 
from scipy.optimize import fsolve
import jax
import jax.numpy as jnp

class Role(enum.StrEnum):
	SENDER = "sender"
	RECEIVER = "receiver"

params = {
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
}
mpl.rcParams.update(params)

def biffurcation(
	game,
	flags: SFlags,
	parameter_config: dict, 
	role: Role = Role.SENDER
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
			if role == Role.SENDER else (receiver_beliefs.shape, grad_receiver)
		)
		def func(x):
			return opinion_dynamics(0.0, x, args)
		
		# We test different starting points to find all branches
		b = sender_beliefs if role == Role.SENDER else receiver_beliefs
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


	# Simulation Parameters

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

def get_leading_eigenvalue(z, args):
    """
    Computes the exact leading eigenvalue of the Jacobian.
    z: current flat belief state (logits)
    """
    # 1. Define the drift function (the right-hand side of your ODE)
    # We use a lambda to ensure we only differentiate w.r.t. z
    def drift(z_val):
        # This should be your opinion_dynamics function
        return opinion_dynamics(0, z_val, args)

    # 2. Compute the Jacobian matrix: J = d(dot_z) / dz
    # Since our system is symmetric, we can use hessian if we have the VFE
    # but jacfwd is the direct way for any drift function.
    jacobian = jax.jacfwd(drift)(z)

    # 3. Get eigenvalues. eigh is for symmetric/hermitian matrices.
    # It is faster and more stable than eig().
    eigenvals = jnp.linalg.eigvalsh(jacobian)

    # 4. The leading eigenvalue is the maximum
    return jnp.max(eigenvals)

def one_parameter_sweep(
	flags: SFlags, 
	sim: Callable,
	parameter_config: dict
):
	# --- SWEEP : Varying a parameter ---

	assert "name" in parameter_config, parameter_config
	assert "min" in parameter_config, parameter_config
	assert "max" in parameter_config, parameter_config
	assert "num_points" in parameter_config, parameter_config
	

	param_name = parameter_config["name"]
	params = np.linspace(parameter_config["min"], parameter_config["max"], parameter_config["num_points"])
	joint_mi, cic, mi_sender, mi_receiver, vfe, expl = [], [], [], [], [], []

	for p in params:
		flags = flags._replace(**{param_name: p})

		(sender_beliefs, receiver_beliefs), _ = sim(flags, return_logs=False)
		
		sender_policy, receiver_policy = jax.tree.map(
			lambda x: get_policy_from_logits(x, flags.force), (sender_beliefs, receiver_beliefs)
		)
		mi_sender.append(calculate_mi(sender_policy))
		mi_receiver.append(calculate_mi(receiver_policy))
		joint_mi.append(calculate_social_metrics(sender_policy, receiver_policy)[1])
		cic.append(calculate_cic(sender_policy, receiver_policy))
		vfe.append(calculate_free_energy(sender_policy)+calculate_free_energy(receiver_policy))
		expl.append(calculate_exploitability(game, sender_policy, receiver_policy)[0])


	# --- Plotting a 1D Bifurcation Diagram ---
	fig, ax = plt.subplots(figsize=(14, 5))

	# Plot MI vs Beta
	ax.plot(params, mi_sender, 'o-', color='firebrick', markersize=4, label = "Sender MI")
	ax.plot(params, mi_receiver, 'x-', color='blue', markersize=4, label = "Receiver MI")
	ax.plot(params, joint_mi, '--', color='green', markersize=4, label = r"$I(\mathcal{W};\mathcal{A})$")
	ax.plot(params, expl, '.--', color='brown', markersize=4, label = r"NashConv")
	ax.plot(params, cic, '.-', color='black', markersize=4, label = "CIC")
	ax.plot(params, vfe, 'o--', color='indigo', markersize=4, label = "VFE")


	ax.legend(loc="best")
	# ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Theoretical Bifurcation')
	ax.set_title(r"Bifurcation: MI vs. Dissipation $\gamma$")
	ax.set_xlabel(r"Dissipation rate $\gamma$")
	ax.set_ylabel("Mutual Information (bits)")
	ax.grid(True, alpha=0.3)

	plt.savefig(f'onep_sweep_{param_name}_{flags.payoffs}.pdf', format = 'pdf', bbox_inches='tight')
	plt.tight_layout()
	plt.show()

def two_parameter_sweep(
	game,
	flags:  SFlags, 
	sim: Callable,
	parameter_config_lhs: dict,
	parameter_config_rhs: dict,
	metric: Callable
):
  
	assert "name" in parameter_config_lhs and "name" in parameter_config_rhs
	assert "min" in parameter_config_lhs and "min" in parameter_config_rhs
	assert "max" in parameter_config_lhs and "max" in parameter_config_lhs
	assert "num_points" in parameter_config_lhs and "num_points" in parameter_config_rhs

	param_name_lhs, param_name_rhs = parameter_config_lhs["name"], parameter_config_rhs["name"]
	params_lhs = np.linspace(parameter_config_lhs["min"], parameter_config_lhs["max"], parameter_config_lhs["num_points"])
	params_rhs = np.linspace(parameter_config_rhs["min"], parameter_config_rhs["max"], parameter_config_rhs["num_points"])
	results = np.zeros((len(params_lhs), len(params_rhs)))
	results_reward = np.zeros((len(params_lhs), len(params_rhs)))

	rng = jax.random.key(flags.seed)
	param_names = dict(
	learning_rate = r"$\kappa$",
	force = r"$\beta$",
	force_eps = r"$\epsilon$",
	damping = r"$\gamma$",
	coupling = r"$\eta$",
	)

	for i, alpha in enumerate(params_lhs):
		for j, beta in enumerate(params_rhs):
			flags = flags._replace(**{param_name_lhs: alpha, param_name_rhs: beta})
			(sender_beliefs, receiver_beliefs), _ = sim(flags, return_logs=False)
			sender_policy, receiver_policy = jax.tree.map(
				lambda x: get_policy_from_logits(x, flags.force), (sender_beliefs, receiver_beliefs)
			)
			results[i, j] = calculate_social_metrics(sender_policy, receiver_policy)[1]
			rng, _, reward, _ = game_iter(
				game, flags, rng, (sender_beliefs, receiver_beliefs)
			)
			results_reward[i, j] = reward[0]



    # Plotting
	plt.figure(figsize=(10, 8))
	plt.imshow(results, extent=[params_lhs[0], params_lhs[-1], params_rhs[0], params_rhs[-1]], 
							origin='lower', aspect='auto', cmap='magma')
	plt.colorbar(label=r'Mutual Information $I(\mathcal{W}; \mathcal{A})$ (bits)')
	plt.xlabel(r'Sensitifity rate $\beta$')
	plt.ylabel(r'Damping rate $\gamma$')
	plt.title('Phase Diagram: Climbing game')

	# Annotate Regimes
	# plt.text(0.2, 1.0, 'I. Babbling', color='white', fontweight='bold', ha='center')
	# plt.text(0.3, 4.0, 'II. Pooling', color='white', fontweight='bold', ha='center')
	# plt.text(1.5, 4.0, 'III. Signaling', color='white', fontweight='bold', ha='center')
	plt.show()
	plt.tight_layout()

	# plt.imshow(results_reward, extent=[params_lhs[0], params_lhs[-1], params_rhs[0], params_rhs[-1]], 
	# 						origin='lower', aspect='auto', cmap='magma')
	# plt.colorbar(label=r'Eval payoff $U$')
	# plt.xlabel(r'Sensitifity rate $\beta$')
	# plt.ylabel(r'Damping rate $\gamma$')
	# plt.title('Phase Diagram: Emergence of Communication')

	# # Annotate Regimes
	# # plt.text(0.2, 1.0, 'I. Babbling', color='white', fontweight='bold', ha='center')
	# # plt.text(0.3, 4.0, 'II. Pooling', color='white', fontweight='bold', ha='center')
	# # plt.text(1.5, 4.0, 'III. Signaling', color='white', fontweight='bold', ha='center')
	# plt.show()
	# plt.tight_layout()



def hysteresis(
	flags:  SFlags, 
	sim: Callable, 
	parameter_config: dict,
	role: Role | str = Role.SENDER
):
  
	assert "name" in parameter_config, parameter_config
	assert "min" in parameter_config, parameter_config
	assert "max" in parameter_config, parameter_config
	assert "num_points" in parameter_config, parameter_config

	param_name = parameter_config["name"]
	params = np.linspace(parameter_config["min"], parameter_config["max"], parameter_config["num_points"])

  # 1. FORWARD SWEEP (Ramp Up)
	mi_forward = []
	for p in params:
		# mi_forward.append(calculate_mi(sender_beliefs if role == Role.SENDER else receiver_beliefs))
		(sender_beliefs, receiver_beliefs), _ =sim(flags._replace(**{param_name: p}), return_logs=False)
		sender_policy, receiver_policy = jax.tree.map(
				lambda x: get_policy_from_logits(x, flags.force), (sender_beliefs, receiver_beliefs)
		)

		mi_forward.append(calculate_social_metrics(sender_policy, receiver_policy)[1])


  	# 2. BACKWARD SWEEP (Ramp Down)
	mi_backward = []
  	# Start the backward sweep using a "perfected" state from the highest beta
	# (sender_beliefs, receiver_beliefs), _ = sim(flags._replace(**{param_name: params[-1]}))
	for b in reversed(params):

		(sender_beliefs, receiver_beliefs), _ =sim(flags._replace(**{param_name: b}), (sender_beliefs, receiver_beliefs), return_logs=False)
		sender_policy, receiver_policy = jax.tree.map(
			lambda x: get_policy_from_logits(x, flags.force), (sender_beliefs, receiver_beliefs)
		)

		mi_backward.append(calculate_social_metrics(sender_policy, receiver_policy)[1])

	# Reverse the backward results so they align with the 'betas' array
	mi_backward = mi_backward[::-1]

  # --- Plotting Hysteresis ---
	plt.figure(figsize=(10, 6))
	plt.plot(params, mi_forward, 'o-', label='Forward (Ramp Up)', color='gray', alpha=0.5)
	plt.plot(params, mi_backward, 'o-', label='Backward (Ramp Down)', color='crimson', linewidth=2)
	plt.fill_between(params, mi_forward, mi_backward, color='crimson', alpha=0.1, label='Hysteresis Region')

	plt.title("Hysteresis Loop: The Robustness of Emergent Meaning")
	plt.xlabel("Sensitivity (Beta)")
	plt.ylabel(r"Mutual Information $I(\mathcal{W}, \mathcal{A})$ (bits)")
	plt.legend()
	plt.grid(True)
	plt.savefig(f'hysteresis_{param_name}_{flags.payoffs}.pdf', format = 'pdf', bbox_inches='tight')
	plt.show()

# def policy_evolution(ts, policy_history, num_states, num_messages):
# 	"""
# 	Plots the probability of each message over time for each observation.
# 	policy_history: shape [time, observation, message]
# 	"""
# 	fig, axes = plt.subplots(1, num_states, figsize=(15, 4), sharey=True)
# 	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 	for obs_idx in range(num_states):
# 		ax = axes[obs_idx]
# 		for msg_idx in range(num_messages):
# 			# Extract the probability of message 'msg_idx' given observation 'obs_idx'
# 			probs = policy_history[:, obs_idx, msg_idx]
# 			ax.plot(ts, probs, label=f"Signal {msg_idx}", color=colors[msg_idx % len(colors)], lw=2)
					
# 		ax.set_title(f"Observation {obs_idx} Policy")
# 		ax.set_xlabel("Time")
# 		if obs_idx == 0:
# 			ax.set_ylabel("Probability $P(m|o)$")
# 		ax.grid(True, alpha=0.3)

# 	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# 	plt.tight_layout()
# 	plt.show()
  
def evaluate_policies(
	flags:  SFlags, 
	sim: Callable, 
	role = Role.SENDER, 
	parameter_config: dict = None
):
	if parameter_config is not None:
		assert "name" in parameter_config, parameter_config
		assert "min" in parameter_config, parameter_config
		assert "max" in parameter_config, parameter_config
		assert "num_points" in parameter_config, parameter_config

		param_name = parameter_config["name"]
		params = np.linspace(parameter_config["min"], parameter_config["max"], parameter_config["num_points"])
	else: 
		params = range(1)

  
	def init_fig():
		fig, ax = plt.subplots(1, 1)
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)
		return fig, ax

	def plot_scalars(
		scalars,
		repetition_axis=0,
		scalar_labels=None,
		title=None,
		ax_labels=None,
	):
		"""Plots scalar on ax by filling 1 standard error.

		Args:
				scalars: List of scalars to plot (mean taken over repetition axis)
				repetition_axis: Axis to take the mean over
				scalar_labels: Labels for the scalars (for legend)
				title: Figure title
				ax_labels: Labels for x and y axis (list of 2 strings)
		"""
		if not all([len(s.shape) == 2 for s in scalars]):
			raise ValueError("Only 2D arrays supported for plotting")

		if scalar_labels is None:
			scalar_labels = [None] * len(scalars)

		if len(scalars) != len(scalar_labels):
			raise ValueError(
					"Wrong number of scalar labels, expected {} but received {}".format(
							len(scalars), len(scalar_labels)
					)
			)
		
		_, plot_axis = init_fig()
		for i, scalar in enumerate(scalars):
			xs = np.arange(scalar.shape[1 - repetition_axis]) * flags.log_interval
			mean = scalar.mean(axis=repetition_axis)
			sem = stats.sem(scalar, axis=repetition_axis)
			plot_axis.plot(xs, mean, label=scalar_labels[i])
			plot_axis.fill_between(xs, mean - sem, mean + sem, alpha=0.5)

		plot_axis.grid(axis="both")
		if title is not None:
			plot_axis.set_title(title)
		if ax_labels is not None:
			plot_axis.set_xlabel(ax_labels[0])
			plot_axis.set_ylabel(ax_labels[1])

		if scalar_labels is not None:
			plot_axis.legend()

	def plot_confusion_matrix(cm, cmap=plt.cm.Blues, title=None):
		"""Plot the confusion matrix.

		Args:
				cm (np.ndarray): Confusion matrix to plot
				cmap: Color map to be used in matplotlib's imshow
				title: Figure title

		Returns:
				Figure and axis on which the confusion matrix is plotted.
		"""
		fig, ax = plt.subplots()
		ax.imshow(cm, interpolation="nearest", cmap=cmap)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlabel("Receiver's action", fontsize=14)
		ax.set_ylabel("Sender's state", fontsize=14)
		# Loop over data dimensions and create text annotations.
		fmt = "d"
		thresh = cm.max() / 2.0
		for i in range(cm.shape[0]):
			for j in range(cm.shape[1]):
				ax.text(
						j,
						i,
						format(cm[i, j], fmt),
						ha="center",
						va="center",
						color="white" if cm[i, j] > thresh else "black",
				)
		fig.tight_layout()
		if title is not None:
			ax.set_title(title)
		return fig, ax
	
	logs = []
	for p in params:
		
		if parameter_config is None:
			_, log = sim(flags)
		else:
			_, log = sim(flags._replace(**{param_name: p}))
		logs.append(log)

	param_names = dict(
		learning_rate = r"$\kappa$",
		force = r"$\beta$",
		force_eps = r"$\epsilon$",
		damping = r"$\gamma$",
		coupling = r"$\eta$",
	)

	plot_scalars(
		[l["leading_eigenvalue" + ("_r" if role == Role.RECEIVER else "_s")] for l in logs],
		title=r"Real Part of Leading Eigenvalue $Re(\lambda_{max})$",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", r"$Re(\lambda_{max})$"],
	)
	plot_scalars(
		[l["opts" + ("_r" if role == Role.RECEIVER else "_s")] for l in logs],
		title="Percentage of optimal actions",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", "% optimal actions"],
	)

	plot_scalars(
		[l["rewards" + ("_r" if role == Role.RECEIVER else "_s")] for l in logs],
		title="Reward graph",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", "Reward per episode"],
	)
	plot_scalars(
		[l["opts" + ("_r" if role == Role.RECEIVER else "_s")] for l in logs],
		title="Percentage of optimal actions",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", "% optimal actions"],
	)

	plot_scalars(
		[l["coordination_success" + ("_r" if role == Role.RECEIVER else "_s")] for l in logs],
		title="Coordination success of the system",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", "% optimal actions"],
	)

	plot_scalars(
		[l["free_energy" + ("_r" if role == Role.RECEIVER else "_s")] for l in logs],
		title="Free energy of the system",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", "VFE (bits)"],
	)

	plot_scalars(
		[l["cic"] for l in logs],
		title="CIC of the joint policy",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", " Entropy (bits)"],
	)

	plot_scalars(
		[l["social_entropy"] for l in logs],
		title="Entropy of the joint policy",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", " Entropy (bits)"],
	)

	plot_scalars(
		[l["joint_mi"] for l in logs],
		title="Joint Mutual information of the system",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", "MI (bits)"],
	)

	plot_scalars(
		[l["expl"] for l in logs],
		title="Exploitability of the system",
		scalar_labels=None if parameter_config is  None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", "NashConv"],
	)

	plot_confusion_matrix(
		logs[0]["convergence_point"].astype(int), title="Final policy"
	)

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

def plot_coordination_snap(	
	flags:  SFlags, 
	sim: Callable, 
	parameter_config: dict
):
	# --- SWEEP : Varying a parameter ---

	assert "name" in parameter_config, parameter_config
	assert "min" in parameter_config, parameter_config
	assert "max" in parameter_config, parameter_config
	assert "num_points" in parameter_config, parameter_config
	

	param_name = parameter_config["name"]
	params = np.linspace(parameter_config["min"], parameter_config["max"], parameter_config["num_points"])
	joint_mi, eig_s, eig_r = [], [], []

	for p in params:
		flags = flags._replace(**{param_name: p})
		_, logs = sim(flags)
		joint_mi.append(logs["joint_mi"][:, -1])
		eig_s.append(logs["leading_eigenvalue_s"][:, -1])
		eig_r.append(logs["leading_eigenvalue_r"][:, -1])

	
	fig, ax1 = plt.subplots(figsize=(10, 6))

	# Axis 1: Leading Eigenvalue
	color = 'tab:red'
	scalar = np.array(joint_mi).transpose()
	xs = np.arange(scalar.shape[1]) * flags.log_interval
	mean1 = scalar.mean(axis=0)
	sem1 = stats.sem(scalar, axis=0)
	ax1.plot(params, mean1, color=color, label="Coordination")
	ax1.fill_between(params, mean1 - sem1, mean1 + sem1, alpha=0.5,  color=color)
	ax1.set_ylabel(r"$I(\mathcal{W}; \mathcal{A})$ bits")
	ax1.set_ylim(0, 1.6) # Max for 3 states is log2(3) ~ 1.58
	ax1.tick_params(axis='y', labelcolor=color)
	ax1.set_xlabel(r"Sensitivity $\beta$")

	# Axis 2: Mutual Information
	ax2 = ax1.twinx()
	color = 'tab:blue'
	scalar = np.array(eig_s).transpose()
	xs = np.arange(scalar.shape[1]) * flags.log_interval
	mean1 = scalar.mean(axis=0)
	sem1 = stats.sem(scalar, axis=0)
	ax2.plot(params, mean1, color=color, label=r"Stability ($Z^s$)")
	ax2.axhline(0, color='black', linestyle='--', alpha=0.5) # The Zero Crossing
	ax2.fill_between(params, mean1 - sem1, mean1 + sem1, alpha=0.5, color=color)
	ax2.set_ylabel(r"$Re(\lambda_{\max})$", color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	color = 'tab:green'
	scalar = np.array(eig_r).transpose()
	xs = np.arange(scalar.shape[1]) * flags.log_interval
	mean1 = scalar.mean(axis=0)
	sem1 = stats.sem(scalar, axis=0)
	ax2.plot(params, mean1, color=color, label=r"Stability ($Z^r$)")
	ax2.fill_between(params, mean1 - sem1, mean1 + sem1, alpha=0.5, color=color)

	plt.legend()
	plt.grid()
	plt.title('The Coordination Snap: Bifurcation vs. Information')

	plt.savefig(f'snap_{flags.payoffs}.pdf', format = 'pdf', bbox_inches='tight')

	fig.tight_layout()
	plt.show()

if __name__ == "__main__":

	# learning_rate: float = 0.01
	# temperature: float = 1.0
	# decay: float = 4.0
	# force: float = 0.1
	# damping: float = 1.8
	# coupling: float = 1.2
	# symm_break: float = 1.0

	parameter_config_beta = {
		"name": "force",
		"min": 1.0,
		"max": 7,
		"num_points": 10
	}

	parameter_config_gamma = {
		"name": "damping",
		"min": 0.0,
		"max": 6,
		"num_points": 10
	}

	parameter_config_eta = {
		"name": "coupling",
		"min": 0.0,
		"max": 1.0,
		"num_points": 10
	}

	parameter_config_kappa = {
		"name": "learning_rate",
		"min": 0.0,
		"max": 0.5,
		"num_points": 10
	}
	# for cfg in [parameter_config_beta, parameter_config_gamma, parameter_config_kappa]:
	# one_parameter_sweep(SFlags(), simple_simulation, parameter_config_beta)
	game = "lewis_signaling"
	flags = SFlags(payoffs="climbing")
	
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
	# two_parameter_sweep(game, SFlags(
	# 	payoffs="climbing", end_time=0.15, num_iterations=100,
	# 	log_interval=1, 
	# 	learning_rate = 0.2,
	# 	force = 2.5,
	# 	# force_eps = 0.025,
	# 	# damping = 1.35,
	# 	force_eps = 0.01,
	# 	damping = 1.35,
	# 	coupling = 0.75, 
	# ), simple_simulation, parameter_config_beta, parameter_config_gamma, None)
	# one_parameter_sweep(SFlags(
	# 	learning_rate = 0.08,
	# 	force = 2.5,
	# 	force_eps = 0.025,
	# 	damping = 1.35,
	# 	coupling = 0.55,
	# 	payoffs="classic", end_time=0.15, num_iterations=100, log_interval=1), simple_simulation, parameter_config_beta)
	evaluate_policies(
		SFlags(
			payoffs="climbing", 
			end_time=0.15, 
			num_iterations=80, 
			log_interval=1, 
			learning_rate = 0.08,
			force = 2.5,
			# force_eps = 0.025,
			# damping = 1.35,
			force_eps = 0.01,
			damping = 1.25,
			coupling = 0.55,
		), 
		simple_simulation,
		role=Role.RECEIVER,
		parameter_config=parameter_config_beta
	)
	# plot_coordination_snap(SFlags(payoffs="prisoner_dilemma", end_time=0.15, num_iterations=30, log_interval=1), simple_simulation, parameter_config_beta)
	# plot_coordination_snap(SFlags(
	# 		payoffs="classic", 
	# 		end_time=0.15, 
	# 		num_iterations=100, 
	# 		log_interval=1, 
	# 		learning_rate = 0.1,
	# 		force_eps = 0.02,
	# 		damping = 1.3,
	# 		coupling = 0.45), simple_simulation, parameter_config_beta)

	
	# hysteresis(SFlags(payoffs="classic", end_time=0.25, num_iterations=50, log_interval=1), simple_simulation, parameter_config_beta)
	# hysteresis(SFlags(), simple_simulation, parameter_config1, Role.RECEIVER)

	# plot_biffurcation(SFlags(payoffs="classic", end_time=0.15, num_iterations=10, log_interval=1), parameter_config_beta)
	

