"""Batch methods.
"""

from typing import (Iterable, TypeVar, Mapping, Callable, Sequence, Tuple,
                    Dict, List)

import numpy as np
import rl.markov_process as mp
import rl.markov_decision_process as mdp
import rl.distribution as dist
import rl.dynamic_programming as dp
import _Wakim.Extra_Practice.Vampire as vampire
from _Wakim.Assignment11.Q1_Q2_Q3_tabular_MC_TD import get_traces


S = TypeVar('S')
A = TypeVar('A')
R = TypeVar('R')
V = Mapping[S, float]


def get_trivial_policy(
    mdp_obj: mdp.FiniteMarkovDecisionProcess[S, A]
) -> mdp.FinitePolicy[S, A]:
    """Generate a policy which randomly selects actions for each state.

    :param mdp_obj: Markov decision process for which to get uniform policy.
    :returns: Policy which assigns a uniform distribution to each action for
        each state.
    """
    state_action_dict: Dict[S, A] = {}

    for state in mdp_obj.states():
        actions = list(mdp_obj.actions(state))

        if len(actions) > 0:
            num_actions = len(actions)
            uniform_prob = 1/num_actions
            uniform_actions = dist.Categorical(
                {action: uniform_prob for action in actions}
            )
            state_action_dict[state] = uniform_actions
        else:
            state_action_dict[state] = None

    return mdp.FinitePolicy(state_action_dict)


def get_traces_over_random_actions(
    mdp_obj: mdp.FiniteMarkovDecisionProcess[S, A],
    start_state_dist: dist.FiniteDistribution[S],
    num_traces: int
) -> Iterable[mdp.TransitionStep[S, A]]:
    """Generate random atomic experiences.

    :param mdp_obj: MDP object from which we are sampling atomic experiences
    :param start_state_dist: Distribution of starting states
    :param num_traces: Number of traces to sample from the mdp
    :returns: Database of atomic experiences
    """
    pi: mdp.FinitePolicy[S, A] = get_trivial_policy(mdp_obj)

    traces = []
    for i, trace in enumerate(mdp_obj.action_traces(start_state_dist, pi)):
        if i == num_traces:
            break
        traces.append(list(trace))

    return [trace[i] for trace in traces for i in range(len(trace))]


def f1(vampire_state: vampire.State) -> float:
    """Add value to feature vector for number of villagers.

    :param vampire_state: Current state of the MDP representing the vampire
        problem
    :returns: Float representing feature value at current state and
        and specifically the number of villagers present.
    """
    return vampire_state.n


def f2(vampire_state: vampire.State):
    """Add value to feature vector for the presence of the vampire.

    :param vampire_state: Current state of the MDP representing the vampire
        problem
    :returns: Float representing feature value at current state and
        and specifically whether the vampire is present (1.0 for present,
        0.0 otherwise).
    """
    return int(vampire_state.v)


def g1(vampire_state: vampire.State, action: A) -> float:
    """Add value to feature vector for number of villagers.

    :param vampire_state: Current state of the MDP representing the vampire
        problem
    :param action: Action being made in the vampire problem
    :returns: Float representing feature value at current state and
        and specifically the number of villagers present.
    """
    if action.p is not None:
        return vampire_state.n - action.p
    else:
        return vampire_state.n


def g2(vampire_state: vampire.State, action: A):
    """Add value to feature vector for the presence of the vampire.

    :param vampire_state: Current state of the MDP representing the vampire
        problem
    :param action: Action being made in the vampire problem
    :returns: Float representing feature value at current state and
        and specifically whether the vampire is present (1.0 for present,
        0.0 otherwise).
    """
    if action.p is not None:
        return -int(vampire_state.v) * action.p
    else:
        return 0


X = TypeVar('X')
feature_functions: Sequence[Callable[[X], float]] = [f1, f2]
action_feature_funcs: Sequence[Callable[[X, A], float]] = [g1, g2]


def get_q_feature_vec(
    action_feature_funcs: Sequence[Callable[[X, A], float]],
    x: X,
    action: A
) -> np.ndarray:
    """Get the feature vector for approximating the action value function.

    :param feature_function: Functions which convert the current state to
        feature values
    :param x: current state of the MDP
    :return: Feature vector based on state and action
    """
    return np.array(
        [f(x, action) for f in action_feature_funcs]
    )


def get_feature_vec(
    feature_functions: Sequence[Callable[[X], float]],
    x: X
) -> np.ndarray:
    """
    Get the feature vector corresponding to the current state of the MDP.

    Adapted from `function_approx.py`

    :param feature_function: Functions which convert the current state to
        feature values
    :param x: current state of the MDP
    :return: Feature vector
    """
    return np.array(
        [f(x) for f in feature_functions]
    )


def LSTD(
    feature_functions: Sequence[Callable[[X], float]],
    experiences: Iterable[mp.TransitionStep[S]],
    gamma: float
) -> Sequence[float]:
    """Least squares temporal difference algorithm.

    :param feature_functions: Functions which convert the current state to
        feature values
    :param experiences: Atomic experiences from batch
    :param gamma: Discount factor
    :returns: Weights associated with feature functions
    """
    m = len(feature_functions)

    # Initialize A, b
    A = np.zeros((m, m))
    b = np.zeros((m, 1))

    # Iterate through atomic experiences and update A, b
    for i, step in enumerate(experiences):
        phi_1 = np.atleast_2d(get_feature_vec(feature_functions, step.state)).T
        phi_2 = np.atleast_2d(get_feature_vec(
            feature_functions, step.next_state)
        ).T

        A += np.outer(phi_1, (phi_1 - gamma * phi_2))
        b += (phi_1 * step.reward)

    return np.dot(np.linalg.inv(A), b)


def linear_q_approx(
    state: S,
    action: A,
    feature_functions: Sequence[Callable[[S, A], float]],
    weights: np.ndarray
) -> float:
    """Using linear function approximation given state and action to get Q.

    :param state: State at which action value function is being approximated
    :param action: Action at which action value function is being approximated
    :param feature_functions: Functions which calculate feature vectors from
        state action pairs
    :param weights: Weights for linear function approximator
    :returns: Linear approximated action value function
    """
    return np.matmul(
        get_q_feature_vec(feature_functions, state, action), weights
    )[0]


def vampire_actions_from_state(
    state: vampire.State
) -> List[vampire.Action]:
    """Get all actions from a given state of the vampire problem.

    :param state: State of the vampire problem
    :returns: All possible actions from the state of the vampire problem
    """
    if state.n == 0:
        return [vampire.Action(None)]
    return [vampire.Action(p) for p in range(0, state.n)]


def LSTDQ(
    feature_functions: Sequence[Callable[[X], float]],
    experiences: Iterable[mp.TransitionStep[S]],
    gamma: float
) -> Sequence[float]:
    """Least squares temporal difference algorithm with Q-learning update.

    :param feature_functions: Functions which convert the current state to
        feature values
    :param experiences: Atomic experiences from batch or mini-batch
    :param policy: Policy for selecting action based on next state
    :param gamma: Discount factor
    :returns: Weights associated with feature functions
    """
    m = len(feature_functions)

    # Initialize A, b
    A = np.zeros((m, m))
    b = np.zeros((m, 1))
    weights = np.zeros((m, 1))

    # Iterate through atomic experiences and update A, b
    for i, step in enumerate(experiences):

        # Get value function approximation of current state and action
        phi_1 = np.atleast_2d(get_q_feature_vec(
            feature_functions, step.state, step.action
        )).T

        # Get the next action by selecting the action that maximizes Q
        actions = vampire_actions_from_state(step.next_state)
        Qs = [linear_q_approx(
            step.next_state, action, feature_functions, weights
        ) for action in actions]
        action = actions[np.argmax(Qs)]
        phi_2 = np.atleast_2d(get_q_feature_vec(
            feature_functions, step.next_state, action
        )).T

        # Update A and b
        A += np.outer(phi_1, (phi_1 - gamma * phi_2))
        b += (phi_1 * step.reward)

    return np.dot(np.linalg.inv(A), b)


def LSPI(
    feature_functions: Sequence[Callable[[X], float]],
    experiences: Iterable[mp.TransitionStep[S]],
    gamma: float
) -> Tuple[V[S], mdp.FinitePolicy[S, A]]:
    """Predict the optimal policy and VF using the LSPI algorithm.

    Simulate a bunch of random actions from each state to generate a database
    of experiences

    :param feature_functions: Functions which convert the current state to
        feature values
    :param experiences: Atomic experiences from batch
    :param gamma: Discount factor
    :returns: Optimal value function as predicted by LSPI
    :returns: Optimal policy as predicted by LSPI
    """
    pass


def main():
    """Test the LSTD algorithm on the Vampire problem MDP.
    """
    from pprint import pprint

    # Specify a starting state distribution for the number of villagers
    num_villagers: int = 10
    start_state_dist: dist.Categorical[S] = dist.Categorical(
        {
            vampire.State(
                i, True
            ): 1 / num_villagers for i in range(1, num_villagers+1)
        }
    )

    # Represent the problem as an MDP
    vampire_mdp: mdp.FiniteMarkovDecisionProcess[S, A] =\
        vampire.VampireMDP(num_villagers)

    # Use dynamic programming to obtain the optimal value function and policy
    true_val, pi = dp.policy_iteration_result(vampire_mdp, 1)
    print("True optimal value function: ")
    pprint(true_val)

    # Express the vampire problem as an MRP and sample experiences
    vampire_mrp: mp.FiniteMarkovRewardProcess[S] =\
        vampire_mdp.apply_finite_policy(pi)
    num_traces = 10000
    traces = get_traces(vampire_mrp, start_state_dist, num_traces)
    experiences = [trace[i] for trace in traces for i in range(len(trace))]

    # Generate feature vector, weights, and approx VF for non-terminal states
    vf = {}
    weights = LSTD(feature_functions, experiences, 1)

    for i in range(1, num_villagers+1):
        vampire_state = vampire.State(n=i, v=True)
        vf[vampire_state] = np.matmul(
            get_feature_vec(feature_functions, vampire_state), weights
        )[0]
    print("Predicted optimal value function: ")
    pprint(vf)

    # Generate a random set of atomic experiences from random policies
    random_experiences = get_traces_over_random_actions(
        vampire_mdp,
        start_state_dist,
        10000
    )
    lstdq_weights = LSTDQ(
        action_feature_funcs, random_experiences, 1
    )
    print(lstdq_weights)


if __name__ == "__main__":
    main()
