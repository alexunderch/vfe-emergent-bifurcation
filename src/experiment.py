import enum
from typing import Callable

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyspiel
from scipy import stats
import pandas as pd

from metrics import (
	calculate_cic,
	calculate_exploitability,
	calculate_free_energy,
	calculate_mi,
	calculate_social_metrics,
)
from neural_evolution import Flags as NFlags
from neural_evolution import run_simulation as neural_simulation
from simple_evolution import (
	Flags as SFlags,
)
from simple_evolution import (
	get_policy_from_logits,
)
from simple_evolution import (
	run_simulation as simple_simulation,
)

class Role(enum.StrEnum):
	SENDER = "sender"
	RECEIVER = "receiver"

params = {
    "font.size": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
	"lines.linewidth": 2
}
mpl.rcParams.update(params)


def one_parameter_sweep(
	game: pyspiel.Game,
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
		joint_mi.append(calculate_social_metrics(sender_policy, receiver_policy)[2])
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
	flags:  SFlags, 
	sim: Callable,
	parameter_config_lhs: dict,
	parameter_config_rhs: dict,
):
  
	assert "name" in parameter_config_lhs and "name" in parameter_config_rhs
	assert "min" in parameter_config_lhs and "min" in parameter_config_rhs
	assert "max" in parameter_config_lhs and "max" in parameter_config_lhs
	assert "num_points" in parameter_config_lhs and "num_points" in parameter_config_rhs

	param_name_lhs, param_name_rhs = parameter_config_lhs["name"], parameter_config_rhs["name"]
	params_lhs = np.linspace(parameter_config_lhs["min"], parameter_config_lhs["max"], parameter_config_lhs["num_points"])
	params_rhs = np.linspace(parameter_config_rhs["min"], parameter_config_rhs["max"], parameter_config_rhs["num_points"])
	results = np.zeros((len(params_lhs), len(params_rhs)))
	# results_reward = np.zeros((len(params_lhs), len(params_rhs)))

	# rng = jax.random.key(flags.seed)
	# param_names = dict(
	# learning_rate = r"$\kappa$",
	# force = r"$\beta$",
	# force_eps = r"$\epsilon$",
	# damping = r"$\gamma$",
	# coupling = r"$\eta$",
	# )

	for i, alpha in enumerate(params_lhs):
		for j, beta in enumerate(params_rhs):
			flags = flags._replace(**{param_name_lhs: alpha, param_name_rhs: beta})
			_, logs = sim(flags)
			results[i, j] = logs["joint_mi"].mean(0)[-1]

    # Plotting
	plt.figure(figsize=(10, 8))
	plt.imshow(results, extent=[params_lhs[0], params_lhs[-1], params_rhs[0], params_rhs[-1]], 
							origin='lower', aspect='auto', cmap='magma')
	plt.colorbar(label=r'Mutual Information $I(\mathcal{W}; \mathcal{A})$ (bits)')
	plt.xlabel(r'Sensitifity rate $\beta$')
	plt.ylabel(r'Damping rate $\gamma$')
	plt.title('Phase Diagram: Classic Lewis game')

	plt.show()
	plt.tight_layout()


def plot_comparison(ode_logs, neural_logs, save_path):
	"""
	Generate Figure X: ODE vs Neural VFE Dynamics.
	Shows that the same bifurcation phenomenon occurs in both.
	"""
	fig, axes = plt.subplots(2, 2, figsize=(10, 8))

	ode_mi_all = np.array(ode_logs["joint_mi"])
	ode_mi_mean = ode_mi_all.mean(axis=0)
	ode_mi_std = stats.sem(ode_mi_all, axis=0)

	ode_eig_all = np.array(ode_logs["leading_eigenvalue"])
	ode_eig_mean = ode_eig_all.mean(axis=0)
	ode_eig_std = stats.sem(ode_eig_all, axis=0)
	xs = np.arange(ode_mi_all.shape[1])

	# --- Neural VFE panels (middle & right: mean ± std over seeds) ---

	mi_all = np.array(neural_logs["joint_mi"])
	mi_mean = mi_all.mean(axis=0)
	mi_std = stats.sem(mi_all, axis=0)

	steps = np.arange(mi_all.shape[1])

	eig_all = np.array(neural_logs["leading_eigenvalue"])
	eig_mean = eig_all.mean(axis=0)
	eig_std = stats.sem(mi_all, axis=0)

	coord_all = np.array(neural_logs["coordination_success"])
	coord_mean = coord_all.mean(axis=0)
	coord_std = stats.sem(coord_all, axis=0)

	ode_coord_all = np.array(ode_logs["coordination_success"])
	ode_coord_mean = ode_coord_all.mean(axis=0)
	ode_coord_std = stats.sem(ode_coord_all, axis=0)


	axes[0, 0].plot(steps, mi_mean, 'purple', lw=1.5, label="Neural VFE")
	axes[0, 0].fill_between(steps, mi_mean - mi_std, mi_mean + mi_std, 
							alpha=0.2, color='purple')
	
	axes[0, 0].plot(xs, ode_mi_mean, 'orange', lw=1.5, label="ODE VFE")
	axes[0, 0].fill_between(xs, ode_mi_mean - ode_mi_std, ode_mi_mean + ode_mi_std,  alpha=0.2, color='orange')
	axes[0, 0].set_title(r"$I(\mathcal{W};\mathcal{A})$")
	axes[0, 0].set_ylabel("Bits")
	axes[0, 0].set_xlabel("Episodes")

	axes[0, 0].axhline(1.585, color='gray', ls='--', alpha=0.5, label = "Max MI")
	axes[0, 0].set_ylim(0, 1.8)
	axes[0, 0].legend()

	axes[1, 0].plot(steps, eig_mean, 'purple', lw=1.5, label="Neural VFE")
	axes[1, 0].fill_between(steps, eig_mean - eig_std, eig_mean + eig_std,
							alpha=0.2, color='purple')
	
	axes[1, 0].plot(xs, ode_eig_mean, 'orange', lw=1.5, label="ODE VFE")
	axes[1, 0].fill_between(xs, ode_eig_mean - ode_eig_std, ode_eig_mean + ode_eig_std,  alpha=0.2, color='orange')

	axes[1, 0].axhline(0, color='r', ls='--', alpha=0.5, label=r"Re$(\lambda_{\max})=0$")
	axes[1, 0].set_title(r"Re($\lambda_{\max}$)")
	axes[1, 0].set_xlabel("Episodes")
	axes[1, 0].set_ylabel("Eigenvalue")
	axes[1, 0].legend()

	# --- Coordination comparison ---
	axes[0, 1].plot(steps, coord_mean, 'purple', lw=1.5, label="Neural VFE")
	axes[0, 1].fill_between(steps, coord_mean - coord_std, coord_mean + coord_std,  alpha=0.2, color='purple')

	axes[0, 1].plot(xs, ode_coord_mean, 'orange', lw=1.5, label="ODE")
	axes[0, 1].fill_between(xs, ode_coord_mean - ode_coord_std, ode_coord_mean + ode_coord_std,  alpha=0.2, color='orange')
	
	axes[0, 1].set_title("Coordination Success (%)")
	axes[0, 1].set_ylim(0, 105)
	axes[0, 1].set_xlabel("Episodes")
	axes[0, 1].legend()

	# --- VFE descent curve ---
	vfe_all = np.array(neural_logs["free_energy_dyn"])
	vfe_mean = vfe_all.mean(axis=0)
	vfe_std = stats.sem(vfe_all, axis=0)

	ode_vfe_all = np.array(ode_logs["free_energy_dyn"])
	ode_vfe_mean = ode_vfe_all.mean(axis=0)
	ode_vfe_std = stats.sem(ode_vfe_all, axis=0)

	axes[1, 1].plot(steps, vfe_mean, 'purple', lw=1.5, label="Neural VFE")
	axes[1, 1].fill_between(steps, vfe_mean - vfe_std, vfe_mean + vfe_std,  alpha=0.2, color='purple')

	axes[1, 1].plot(xs, ode_vfe_mean, 'orange', lw=1.5, label="ODE VFE")
	axes[1, 1].fill_between(xs, ode_vfe_mean - ode_vfe_std, ode_vfe_mean + ode_vfe_std,  alpha=0.2, color='orange')
	axes[1, 1].set_title(r"$\mathcal{F}(Z)$ descent")
	axes[1, 1].set_xlabel("Episodes")
	axes[1, 1].set_ylabel("VFE potential")

	for ax in axes.flat:
		ax.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.show()

	return fig


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
		(sender_beliefs, receiver_beliefs), _ =sim(flags._replace(**{param_name: p}))
		sender_policy, receiver_policy = jax.tree.map(
				lambda x: get_policy_from_logits(x, flags.temperature), (sender_beliefs, receiver_beliefs)
		)

		mi_forward.append(calculate_social_metrics(sender_policy, receiver_policy)[2])


  	# 2. BACKWARD SWEEP (Ramp Down)
	mi_backward = []
  	# Start the backward sweep using a "perfected" state from the highest beta
	# (sender_beliefs, receiver_beliefs), _ = sim(flags._replace(**{param_name: params[-1]}))
	for b in reversed(params):

		(sender_beliefs, receiver_beliefs), _ =sim(flags._replace(**{param_name: b}), (sender_beliefs, receiver_beliefs))
		sender_policy, receiver_policy = jax.tree.map(
			lambda x: get_policy_from_logits(x, flags.temperature), (sender_beliefs, receiver_beliefs)
		)

		mi_backward.append(calculate_social_metrics(sender_policy, receiver_policy)[2])

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


def evaluate_policies(
	flags:  SFlags, 
	sim: Callable, 
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
		fig, ax = plt.subplots(1, 1, figsize=(10, 7))
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
			xs = np.arange(scalar.shape[1 - repetition_axis])
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
		self_attention = r"$\alpha$",
	)

	plot_scalars(
		[l["leading_eigenvalue"] for l in logs],
		title=r"Real Part of Leading Eigenvalue $Re(\lambda_{max})$",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", r"$Re(\lambda_{max})$"],
	)

	plot_scalars(
		[l["rewards"] for l in logs],
		title="Reward graph",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", "Reward per episode"],
	)
	# plot_scalars(
	# 	[l["opts" + ("_r" if role == Role.RECEIVER else "_s")] for l in logs],
	# 	title="Percentage of optimal actions",
	# 	scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
	# 	ax_labels=["Episodes", "% optimal actions"],
	# )

	# plot_scalars(
	# 	[l["coordination_success" + ("_r" if role == Role.RECEIVER else "_s")] for l in logs],
	# 	title="Coordination success of the system",
	# 	scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
	# 	ax_labels=["Episodes", "% optimal actions"],
	# )

	plot_scalars(
		[l["coordination_success"] for l in logs],
		title="Coordination success of the system",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", "% optimal actions"],
	)


	plot_scalars(
		[l["free_energy_dyn"] for l in logs],
		title="Potential Free energy of the system",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", "VFE"],
	)

	plot_scalars(
		[l["free_energy_mi"] for l in logs],
		title="Variational Free energy of the system",
		scalar_labels=None if parameter_config is None else [f"{param_names[param_name]}={p:.1f}" for p in params],
		ax_labels=["Episodes", "VFE"],
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

	# plot_confusion_matrix(
	# 	logs[0]["convergence_point"].astype(int), title="Final policy"
	# )
	
	if parameter_config is not None:
		print_data = []
		for log, p in zip(logs, params):
			zc = []
			for seed in log["leading_eigenvalue"]:
				val = np.where(np.diff(np.sign(seed)))[0]
				if len(val) == 0:
					val = [1000]
				zc.append(val[0])
			zc = np.array(zc)

			print_data.append({
				f"Parameter {param_names[param_name]}": f"{p:.1f}",
				"NashConv": f"{log["expl"][:, -1].mean(0):.2f}±{stats.sem(log["expl"][:, -1], axis=0):.2f}",
				r"Re(\lambda_{\max})": f"{log["leading_eigenvalue"][:, -1].mean(0):.2f}±{stats.sem(log["leading_eigenvalue"][:, -1], axis=0):.2f}",
				r"MI": f"{log["joint_mi"][:, -1].mean(0):.2f}±{stats.sem(log["joint_mi"][:, -1], axis=0):.2f}",
				r"Coord. Success": f"{log["coordination_success"][:, -1].mean(0):.2f}±{stats.sem(log["coordination_success"][:, -1], axis=0):.2f}",
				"Emergence time": f"{zc.mean():.2f}±{stats.sem(zc, axis=0) if zc.mean() != np.inf else np.inf:.2f}"
			})
		print(pd.DataFrame(print_data))

	plt.show()

def plot_coordination_snap(	
	flags:  SFlags, 
	sim: Callable, 
	parameter_config: dict,
	origin: bool = False,
):
	# --- SWEEP : Varying a parameter ---

	assert "name" in parameter_config, parameter_config
	assert "min" in parameter_config, parameter_config
	assert "max" in parameter_config, parameter_config
	assert "num_points" in parameter_config, parameter_config
	

	param_name = parameter_config["name"]
	params = np.arange(parameter_config["min"], parameter_config["max"], parameter_config["num_points"])
	joint_mi, eigs = [], []

	for p in params:
		flags = flags._replace(**{param_name: p})
		_, logs = sim(flags)
		joint_mi.append(logs["joint_mi"].max(-1))
		eigs.append(logs["leading_eigenvalue"][:, 0 if origin else -1]) #at the origin
		print(p, np.mean(eigs[-1]), np.mean(joint_mi[-1]))

	fig, ax1 = plt.subplots(figsize=(10, 7))

	# Axis 1: Mutual Information
	color = 'tab:red'
	scalar = np.array(joint_mi).transpose()
	mean1 = scalar.mean(axis=0)
	sem1 = stats.sem(scalar, axis=0)
	ln1 = ax1.plot(params, mean1, color=color, label="Coordination")
	ax1.fill_between(params, mean1 - sem1, mean1 + sem1, alpha=0.5,  color=color)
	ax1.set_ylabel(r"joint $I(\mathcal{W}; \mathcal{A})$ bits")
	ln2 = ax1.axhline(1.58, color='black', linestyle='--', alpha=0.5, label=r"Max MI $=\log_2(3)\approx 1.58$") # The Zero Crossing
	ax1.set_ylim(0, 1.6) # Max for 3 states is log2(3) ~ 1.58
	ax1.tick_params(axis='y', labelcolor=color)
	ax1.set_xlabel(r"Sensitivity $\beta$")


	# Axis 2: Leading eigenvalues
	ax2 = ax1.twinx()
	color = 'tab:blue'
	scalar = np.array(eigs).transpose()
	mean1 = scalar.mean(axis=0)
	sem1 = stats.sem(scalar, axis=0)
	ln3 = ax2.plot(params, mean1, color=color, label=r"Cons. lead. eigenval. of Z=$(Z^s, Z^r)$")
	ln4 = ax2.axvline(flags.damping-flags.coupling, color="black", linestyle='-.', alpha=0.5, label=r"$\beta_c=\gamma-\eta$") #
	ln5 = ax2.axhline(0, color='black', linestyle='--', alpha=0.5, label=r"Re$(\lambda_{\max})=0$") # The Zero Crossing
	ln6 = ax2.axhline(-flags.damping, color='black', linestyle='--', alpha=0.5, label=r"Cons. Re$(\lambda_{\max})=\gamma-\eta$") # The Zero Crossing
	ax2.fill_between(params, mean1 - sem1, mean1 + sem1, alpha=0.5, color=color)
	ax2.set_ylabel(r"Re$(\lambda_{\max})$", color=color)
	ax2.tick_params(axis='y', labelcolor=color)
	
	# lns = ln1+ln3+[ln4]+[ln5]
	lns = ln1+[ln2]+ln3+[ln4, ln5, ln6]
	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs, loc=0)

	plt.grid(axis="both")
	plt.title('The Coordination Snap: Bifurcation vs. Information')

	plt.savefig(f'snap_{flags.payoffs}.pdf', format = 'pdf', bbox_inches='tight')

	fig.tight_layout()
	plt.show()

if __name__ == "__main__":

	parameter_config_beta = {
		"name": "self_attention",
		"min": 0.15,
		"max": 5.0,
		"num_points": 0.5
	}

	plot_coordination_snap(
		SFlags(payoffs="classic", end_time=0.5, num_iterations=1, eps=0.0), #approx. analytic Jacobian
		simple_simulation, 
		parameter_config_beta,
		origin=True
	) 

	parameter_config_beta = {
		"name": "self_attention",
		"min": 0.1,
		"max": 7.0,
		"num_points": 1.0
	}

	plot_coordination_snap(
		SFlags(payoffs="classic", end_time=100.0, num_iterations=1, eps=0.0), 
		simple_simulation, 
		parameter_config_beta
	)
	
	parameter_config_beta = {
		"name": "self_attention",
		"min": 0.05,
		"max": 7.0,
		"num_points": 4
	}

	parameter_config_gamma = {
		"name": "damping",
		"min": 0.0,
		"max": 2.5,
		"num_points": 4
	}

	parameter_config_eta = {
		"name": "coupling",
		"min": 0.0,
		"max": 1.2,
		"num_points": 4
	}
	
	parameter_config_kappa = {
		"name": "learning_rate",
		"min": 0.0,
		"max": 30.5,
		"num_points": 4
	}
	
	evaluate_policies(
		SFlags(
			payoffs="classic", 
			end_time=100.0, 
			num_iterations=1, 
		), 
		simple_simulation,
		parameter_config=parameter_config_kappa
	)

	evaluate_policies(
		NFlags(
			payoffs="classic", 
			num_iterations=100, 
		), 
		neural_simulation,
		parameter_config=parameter_config_beta
	)


	parameter_config_beta = {
		"name": "force",
		"min": 0.0,
		"max": 6.5,
		"num_points": 100
	}
	parameter_config_gamma = {
		"name": "damping",
		"min": 0.0,
		"max": 6.5,
		"num_points": 100
	}

	two_parameter_sweep(SFlags(
		payoffs="climbing", end_time=100.0, num_iterations=1,
	), simple_simulation, parameter_config_beta, parameter_config_gamma, None)

	evaluate_policies(
		NFlags(
			payoffs="classic", 
			num_iterations=10000, 
			num_runs=20
		), 
		neural_simulation,
		parameter_config=parameter_config_beta
	)

	_, ode_logs = simple_simulation(SFlags(payoffs="classic", end_time=100.0, num_iterations=1))
	_, neural_logs = neural_simulation(NFlags(payoffs="classic", num_iterations=int(1e4), lr=1e-2))
	plot_comparison(ode_logs, neural_logs, "./fig_neural_vfe.pdf")

	# Correlation analysis
	mi_all = np.array(neural_logs["joint_mi"]).mean(axis=0)
	eig_all = np.array(neural_logs["leading_eigenvalue"]).mean(axis=0)
	from scipy.stats import pearsonr
	corr, pval = pearsonr(eig_all, mi_all)
	print(f"\nCorrelation(λ_max, MI): r={corr:.3f}, p={pval:.4f}")