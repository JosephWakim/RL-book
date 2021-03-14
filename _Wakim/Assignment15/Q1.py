"""Implementing Tabular Algorithms for Assignment 15, Q1.

This code adds to starter code provided by Dr. Rau at...
https://github.com/coverdrive/MDP-DP-RL/blob/master/src/examples/exam_problems/mrp_tdmc_outline.py

Joseph Wakim
CME 241 - Assignment 15, Problems 1, 2
March 11, 2021
"""


from typing import Sequence, Tuple, Mapping, Dict
from collections import defaultdict

import numpy as np


S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]


def def_value() -> float:
    """Return a default value for a dictionary being filled

    :returns: Default value of zero
    """
    return 0


def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    n_visit: Dict[S, int] = defaultdict(def_value)
    s_visit: Dict[S, float] = defaultdict(def_value)

    for i, step in enumerate(state_return_samples):
        n_visit[step[0]] += 1
        s_visit[step[0]] += step[1]

    return {s: s_visit[s] / n_visit[s] for s in n_visit.keys()}


def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    prob_func: ProbFunc = {}
    s1_visit: Dict[S, int] = defaultdict(def_value)
    s2_visit: Dict[S, int] = defaultdict(def_value)
    s_to_s_counter: Dict[S, S] = defaultdict(def_value)
    s_reward_tot: Dict[S, float] = defaultdict(def_value)

    for step in srs_samples:
        s1 = step[0]
        s2 = step[2]
        s_to_s_counter[(s1, s2)] += 1
        s1_visit[s1] += 1
        s2_visit[s2] += 1
        s_reward_tot[s2] += step[1]

    for start_state in s1_visit.keys():
        prob_func[start_state] = {
            step[1]: s_to_s_counter[step] / s1_visit[start_state]
            for step in s_to_s_counter.keys()
            if step[0] == start_state
        }

    reward_func: RewardFunc = {
        s: s_reward_tot[s] / s2_visit[s] for s in s_reward_tot.keys()
    }

    return prob_func, reward_func


def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """
    states = list(reward_func.keys())
    non_term_states = list(prob_func.keys())

    n_states = len(states)
    P = np.zeros((n_states, n_states))
    R = np.zeros((n_states, 1))

    for i, s1 in enumerate(non_term_states):
        for j, s2 in enumerate(states):
            if s2 in prob_func[s1].keys():
                P[i, j] = prob_func[s1][s2]

    for j, s2 in enumerate(states):
        R[j] = reward_func[s2]

    gamma = 1   # Assume a discount factor of 1
    return np.matmul(np.linalg.inv((np.identity(n_states) - gamma * P)), R)


def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 300000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """
    np.random.seed(1)
    val_funct = defaultdict(def_value)
    gamma = 1   # Assume a discount factor of 1

    for i in range(num_updates):
        rand_ind = np.random.randint(len(srs_samples))
        sample = srs_samples[rand_ind]
        alpha = learning_rate * (i / learning_rate_decay + 1) ** (-0.5)
        val_funct[sample[0]] += alpha * (
            sample[1] + gamma * val_funct[sample[2]] - val_funct[sample[0]]
        )

    return val_funct


def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """
    def get_feature_vec(state: S) -> np.ndarray:
        """Define feature function.

        :param state: State for which feature is being defined
        :return: feature vector
        """
        if state == "A":
            return np.array([1, 0, 0])
        elif state == "B":
            return np.array([0, 1, 0])
        else:
            return np.array([0, 0, 1])

    # Assume a discount factor of 1
    gamma = 1

    # Initialize A, b
    A = np.ones((3, 3))
    b = np.ones((3, 1))

    # Iteratively construct A, b from experiences
    for step in srs_samples:
        phi_1 = np.atleast_2d(get_feature_vec(step[0])).T
        phi_2 = np.atleast_2d(get_feature_vec(step[2])).T

        A += np.outer(phi_1, (phi_1 - gamma * phi_2))
        b += (phi_1 * step[1])

    weights = np.dot(np.linalg.inv(A), b)

    states = ["A", "B", "T"]
    val_funct = {}
    for state in states:
        feature_vec = get_feature_vec(state)
        val_funct[state] = np.matmul(feature_vec, weights)[0]

    return val_funct


if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)

    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)

    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))