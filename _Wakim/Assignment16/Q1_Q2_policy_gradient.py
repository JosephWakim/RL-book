"""
Policy Gradient Algorithms.

TODO: Look into initialization of VF approximation. The function approximation is stuck at zero.

Joseph Wakim
CME 241 - Assignment 11, Problems 1-3
March 12, 2021
"""

from typing import TypeVar, Dict, Mapping, Sequence, Callable, Tuple, Optional

import numpy as np
import rl.markov_process as mp
import rl.markov_decision_process as mdp
import rl.distribution as dist
import rl.function_approx as fa
from rl.returns import returns
import rl.chapter7.asset_alloc_discrete as alloc


S = TypeVar('S')
A = TypeVar('A')
R = TypeVar('R')
V = Mapping[S, float]


class AssetAllocPolicy(mdp.Policy[S, A]):
    """Class representation of the asset allocation policy.
    """

    def __init__(
        self,
        action_func: Callable[[S], Sequence[A], float],
        feature_funcs: Sequence[Callable[[S, A], float]],
        weights: np.ndarray
    ):
        """Initialize the AssetAllocPolicy object.

        :param action_func: Function for generating action space from state
        :param feature_funcs: Functions for generating feature vector from
            current state and proposed action
        :param weights: Weights of linear function approximation for features
        """
        self.action_func = action_func
        self.feature_funcs = feature_funcs
        self.weights = weights

    def act(self, state: S) -> Optional[dist.Distribution[A]]:
        """Generate a distribution of actions from a state

        :param state: Current state for which to generate action distribution
        :returns: Distribution of actions from state.
        """
        return self.softmax(state)

    def softmax(
        self,
        state: S
    ) -> Optional[dist.Distribution[A]]:
        """Generate an action distribution for a state using softmax algorithm.

        :param state: State from which to generate policy
        :param action_func: Function for generating action space from state
        :param feature_funcs: Functions for generating feature vector from
            current state and proposed action
        :param weights: Weights of linear function approximation for features
        :returns: Distribution of action probabilities from state
        """
        actions = self.action_func(state)
        if actions is None:
            return
        tot_prob: float = 0
        act_prob_dict = {}
        for a in actions:
            prob: float = np.exp(
                np.dot(get_feature_vec(
                    self.feature_funcs, state, a), self.weights
                )
            )[0]
            act_prob_dict[a]: Dict[A, float] = prob
            tot_prob += prob
        return dist.Categorical(
            {a: act_prob_dict[a] / tot_prob for a in actions}
        )


def get_feature_vec(
    feature_functions: Sequence[Callable[[S], float]],
    s: S,
    a: A
) -> np.ndarray:
    """Get the feature vector corresponding to a state of the Vampire problem.

    :param feature_functions: Functions that generate feature values from state
    :param s: Current state of the vampire problem
    :param a: Proposed action performed by the vampire problem
    :returns: Feature vector for approximation of policy
    """
    return np.array(
        [f((s, a)) for f in feature_functions]
    )


def get_vf_feature_vec(
    vf_feature_functions: Sequence[Callable[[S], float]],
    s: S
) -> np.ndarray:
    """Get the feature vector corresponding to a state of the Vampire problem.

    :param feature_functions: Functions that generate feature values from state
    :param s: Current state of the vampire problem
    :returns: Feature vector for approximation of value function
    """
    return np.array(
        [f((s)) for f in vf_feature_functions]
    )


def get_episode(
    mdp_obj: mdp.FiniteMarkovDecisionProcess[S, A],
    start_state_dist: dist.Categorical[S],
    policy: AssetAllocPolicy[S, A],
    gamma: float,
    tolerance: float
) -> Sequence[mdp.TransitionStep[S, A]]:
    """Generate an episode from the asset allocation MDP.

    :param mdp_obj: MDP representation of the asset allocation problem
    :param start_state_dist: Starting state distribution
    :param policy: The policy with which to simulate the episode
    :param gamma: Discount factor
    :param tolerance: Accumulated discount factor below which simulation
        terminates
    :returns: Sequence of transition steps from the episode
    """
    episode_iterator = mdp_obj.simulate_actions(start_state_dist, policy)
    return list(returns(episode_iterator, gamma, tolerance))


def get_returns(
    episode: Sequence[mdp.TransitionStep[S, A]],
    gamma: float
) -> Sequence[float]:
    """Get returns corresponding to each step in the episode.

    :param episode: The episode for which returns are being calculated
    :param gamma: Discount factor
    :returns: Returns for each step of the episode
    """
    rewards = [s.reward for s in episode]
    rewards_reverse = rewards.copy()
    rewards_reverse.reverse()
    reverse_returns_from_state = rewards_reverse
    for i in range(1, len(reverse_returns_from_state)):
        reverse_returns_from_state[i] = gamma *\
            reverse_returns_from_state[i-1] + reverse_returns_from_state[i]
    returns_from_state = reverse_returns_from_state
    returns_from_state.reverse()
    return returns_from_state


def REINFOCE(
    mdp_obj: mdp.FiniteMarkovDecisionProcess[S, A],
    start_state_dist: dist.Categorical[S],
    gamma: float,
    feature_funcs: Sequence[Callable[[S, A], float]],
    num_iters: int
) -> np.ndarray:
    """Apply Monte Carlo policy gradient algorithm to determine optimal policy.

    :param mdp_obj: MDP representation of the problem to simulate experiences
    :param start_state_dist: Starting state distribution for the MDP
    :param gamma: Discount factor
    :param feature_funcs: Functions for generating feature vector from current
        state and proposed action
    :param num_iters: Number of iterations to run in updating policy parameters
    :returns: vector of weight parameters for policy function approximation
    """
    theta = np.zeros((len(feature_funcs), 1))
    for i in range(num_iters):
        policy = AssetAllocPolicy(mdp_obj.actions, feature_funcs, theta)
        episode: Sequence[mp.TransitionStep[S, A]] = get_episode(
            mdp_obj, start_state_dist, policy, gamma, gamma**100
        )
        for transition in episode:
            s = transition.state
            expected_phi = np.atleast_2d(np.average(np.array([
                get_feature_vec(
                    feature_funcs, s, a
                ) for a in mdp_obj.actions(s)
            ]), axis=0))
            alpha = 1 / ((i+1) ** 0.2)
            theta += alpha * (
                np.atleast_2d(get_feature_vec(
                    feature_funcs, s, transition.action
                )).T - expected_phi.T
            )
    return theta


def actor_critic(
    mdp_obj: mdp.FiniteMarkovDecisionProcess[S, A],
    start_state_dist: dist.Categorical[S],
    gamma: float,
    feature_funcs: Sequence[Callable[[S, A], float]],
    vf_feature_funcs: Sequence[Callable[[S], float]],
    num_iters: int,
    time_horizon: int,
    decay_theta: float,
    decay_w: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the Actor-Critic-TD-Error algorithm to determine optimal policy.

    :param mdp_obj: MDP representation of the problem to simulate experiences
    :param start_state_dist: Starting state distribution for the MDP
    :param gamma: Discount factor
    :param feature_funcs: Functions for generating policy feature vector from
        current state and proposed action
    :param vf_feature_funcs: Functions for generating value function feature
        vector from current state
    :param num_iters: Number of iterations to run in updating policy parameters
    :param time_horizon: Number of time steps in each episode
    :param decay_theta: Factor by which to decay eligibility trace of theta
    :param decay_w: Factor by which to decay eligibility trace of w
    :returns: vector of weight parameters for policy function approximation
    :returns: vector of weight parameters for value function approximation
    """
    theta = np.zeros((len(feature_funcs), 1))
    w = np.zeros((len(vf_feature_funcs), 1))

    for i in range(num_iters):
        eligibility_theta = 0
        eligibility_w = 0
        P = 1
        state = start_state_dist.sample()
        for t in range(time_horizon):

            # ACTOR Generate step
            policy = AssetAllocPolicy(mdp_obj.actions, feature_funcs, theta)
            episode: Sequence[mp.TransitionStep[S, A]] = get_episode(
                mdp_obj, dist.Constant(state), policy, gamma, gamma**2
            )

            # CRITIC approximates value function
            vf_0 = np.matmul(
                get_vf_feature_vec(vf_feature_funcs, episode[0].state), w
            )
            vf_1 = np.matmul(
                get_vf_feature_vec(vf_feature_funcs, episode[1].state), w
            )
            delta = episode[0].reward + gamma * vf_1 -\
                vf_0
            dVdw = vf_1 - vf_0

            # ACTOR and CRITIC update weights
            eligibility_w = gamma * decay_w * eligibility_w + dVdw
            expected_phi = np.atleast_2d(np.average(np.array([
                get_feature_vec(
                    feature_funcs, episode[0].state, a
                ) for a in mdp_obj.actions(episode[0].state)
            ]), axis=0))
            eligibility_theta = gamma * decay_theta * eligibility_theta + P *\
                np.atleast_2d(get_feature_vec(
                    feature_funcs, episode[0].state, episode[0].action
                )).T - expected_phi.T
            w += 1/(i+1) * delta * eligibility_w
            theta += 1/(i+1) * delta * eligibility_theta
            P = gamma * P

            # ACTOR updates state for next iteration
            state = episode[1].state

    return theta, w


def main():
    """Run the policy gradient algorithms.
    """
    steps: int = 4
    μ: float = 0.13
    σ: float = 0.2
    r: float = 0.07
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_var: float = 0.1

    excess: float = μ - r
    var: float = σ * σ
    base_alloc: float = excess / (a * var)

    risky_ret: Sequence[dist.Gaussian] = [
        dist.Gaussian(μ=μ, σ=σ) for _ in range(steps)
    ]
    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
    alloc_choices: Sequence[float] = np.linspace(
        2 / 3 * base_alloc,
        4 / 3 * base_alloc,
        11
    )
    feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = \
        [
            lambda _: 1.,
            lambda w_x: w_x[0],
            lambda w_x: w_x[1],
            lambda w_x: w_x[1] * w_x[1]
        ]
    dnn: fa.DNNSpec = fa.DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    init_wealth_distr: dist.Gaussian = dist.Gaussian(
        μ=init_wealth, σ=init_wealth_var
    )

    aad: alloc.AssetAllocDiscrete = alloc.AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )

    mdp_obj = aad.get_mdp(t=len(riskless_ret)-1)
    theta = REINFOCE(
        mdp_obj,
        init_wealth_distr,
        0.95,
        feature_funcs,
        100
    )
    print("Theta Estimate by MC: ")
    print(theta)

    vf_feature_funcs: Sequence[Callable[[Tuple[float]], float]] = \
        [
            lambda _: 1.,
            lambda w_x: w_x,
            lambda w_x: w_x ** 2,
        ]
    theta_ac, w_ac = actor_critic(
        mdp_obj,
        init_wealth_distr,
        0.95,
        feature_funcs,
        vf_feature_funcs,
        10,
        100,
        0.95,
        0.95
    )
    print("Theta Estimate by Action-Critic: ")
    print(theta_ac)
    print("VF function approx. weights by Action-Critic: ")
    print(w_ac)


if __name__ == "__main__":
    main()
