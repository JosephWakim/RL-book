"""
Implement policy and value iteration.

The purpose of this code is to make sure I understand the RL-book codebase.
This code does not constitute original work.

Joseph Wakim
February 13, 2021
"""

import numpy as np
import operator
from typing import Mapping, Iterator, TypeVar, Tuple, Dict

from rl.iterate import converged, iterate
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        FiniteMarkovRewardProcess,
                                        FinitePolicy)
from rl.distribution import FiniteDistribution, Categorical, Constant, Choose
import rl.dynamic_programming as dp
from rl.dynamic_programming import V, S, A


def initialize(
    mdp: FiniteMarkovDecisionProcess
) -> Tuple[V[S], FinitePolicy]:
    """Initialize value function and policy.

    Initialize the value function to zeros at each state, and initialize the
    policy to a random choice of the action space at each non-terminal state.

    :param mdp: Object representation of a finite Markov decision process
    :returns: Value function initialized at zeros for each state
    :returns: Random Initial policy
    """
    # Set value function at each state equal to zero
    v_0: V[S] = {s: 0 for s in mdp.states()}
    # Set the policy to be a random choice of the action space at each state
    pi_0: FinitePolicy[S, A] = FinitePolicy(
        {s: Choose(set(mdp.actions(s))) for s in mdp.non_terminal_states}
    )
    return v_0, pi_0


def policy_iteration(
    mdp: FiniteMarkovDecisionProcess,
    gamma: float,
    tolerance: float,
    max_iters: int
) -> Tuple[V[S], FinitePolicy]:
    """Implement policy iteration on a finite MDP.

    :param mdp: Object representation of a finite Markov decision process
    :param gamma: Discount factor
    :param tolerance: Difference in maximum value functions between iterations
        for convergence
    :param max_iters: Maximum number of iterations to allow
    :returns: Optimal policy
    """
    vf, pi = initialize(mdp)
    n_iter = 0

    while True:

        n_iter += 1
        delta = 0
        v = vf.copy()
        mrp: FiniteMarkovRewardProcess[S] = mdp.apply_finite_policy(pi)

        # Policy evaluation
        vf: V[S] = {mrp.non_terminal_states[i]: v for i, v in enumerate(
            mrp.get_value_function_vec(gamma)
        )}
        diffs = np.absolute(np.subtract(list(vf.values()), list(v.values())))
        diffs = np.append(diffs, delta)
        delta = np.max(diffs)

        # Policy improvement
        pi: FinitePolicy[S, A] = dp.greedy_policy_from_vf(
            mdp, vf, gamma
        )

        if n_iter == max_iters:
            print("Maximum iterations reached.")
            return vf, pi
        if delta < tolerance:
            return vf, pi
