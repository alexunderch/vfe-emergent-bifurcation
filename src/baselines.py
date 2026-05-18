"""Analytical reference states requiring no training."""

import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict

from src.metrics import calculate_social_metrics, calculate_free_energy, calculate_exploitability


def baseline_no_communication(game_config: Dict) -> Dict:
    """
    Uniform random policies. Information-theoretic floor.
    I(W;A) = 0, VFE = max, NashConv = max exploitable.
    """
    n_states = game_config['num_states']
    n_messages = game_config['num_messages']
    n_actions = game_config['num_actions']
    
    pi_s = jnp.ones((n_states, n_messages)) / n_messages
    pi_r = jnp.ones((n_messages, n_actions)) / n_actions
    
    _, mi = calculate_social_metrics(pi_s, pi_r)
    vfe = calculate_free_energy(pi_s) + calculate_free_energy(pi_r)
    
    # NashConv requires OpenSpiel game object; return placeholder
    return {
        'pi_s': pi_s,
        'pi_r': pi_r,
        'mi': float(mi),           # ≈ 0
        'vfe': float(vfe),         # maximum
        'nash_conv': None,         # compute externally if needed
        'label': 'No-communication'
    }


def baseline_oracle(game_config: Dict) -> Dict:
    """
    Fixed one-hot identity mapping. Information-theoretic ceiling.
    I(W;A) = log2(|W|), VFE = min, NashConv = 0.
    """
    n = min(game_config['num_states'], game_config['num_messages'], game_config['num_actions'])
    
    pi_s = jnp.eye(game_config['num_states'], game_config['num_messages'])
    pi_r = jnp.eye(game_config['num_messages'], game_config['num_actions'])
    
    _, mi = calculate_social_metrics(pi_s, pi_r)
    vfe = calculate_free_energy(pi_s) + calculate_free_energy(pi_r)
    
    return {
        'pi_s': pi_s,
        'pi_r': pi_r,
        'mi': float(mi),           # = log2(n)
        'vfe': float(vfe),         # minimum
        'nash_conv': 0.0,
        'label': 'Oracle convention'
    }

