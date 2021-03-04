"""
Implementation of Tabular Monte Carlo and Temporal Difference Prediction.

Joseph Wakim
CME 241 - Assignment 11, Problems 1-3
March 1, 2021
"""

from typing import Iterable, TypeVar, Dict, Mapping

import rl.markov_process as mp
import rl.markov_decision_process as mdp
import rl.distribution as dist
import rl.dynamic_programming as dp
import _Wakim.Extra_Practice.Vampire as vampire


S = TypeVar('S')
A = TypeVar('A')
R = TypeVar('R')
V = Mapping[S, float]


def get_traces(
    vampire_mrp: mp.FiniteMarkovRewardProcess[S],
    start_state_dist: dist.FiniteDistribution[S],
    num_traces: int
) -> Iterable[Iterable[mp.TransitionStep[S]]]:
    """Generate a finite number of traces from the vampire problem MDP.

    :param vampire_mrp: Markov reward process representing vampire problem with
        known policy
    :param start_state_dist: Distribution of starting states when evaluating
        the vampire problem
    :param num_traces: Number of traces to generate
    :returns: Traces from simulations of the vampire problem
    """
    return [
        list(vampire_mrp.simulate_reward(start_state_dist)) for i in range(
            num_traces
        )
    ]


def tabular_mc_prediction(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    gamma: float
) -> V[S]:
    """Predict value function using every pass tabular Monte Carlo.

    :param traces: MRP traces
    :param gamma: Discount factor
    :returns: Prediction of value function at each state, by tabular MC
        prediction
    """
    n_visits: Dict[S, int] = {}
    s_visits: Dict[S, float] = {}

    for episode in traces:
        rewards = [s.reward for s in episode]
        rewards_reverse = rewards.copy()
        rewards_reverse.reverse()
        reverse_returns_from_state = rewards_reverse
        for i in range(1, len(reverse_returns_from_state)):
            reverse_returns_from_state[i] = gamma *\
                reverse_returns_from_state[i-1] + reverse_returns_from_state[i]
        returns_from_state = reverse_returns_from_state
        returns_from_state.reverse()
        returns_from_state_dict = {
            episode[i].state: returns_from_state[i]
            for i in range(len(episode))
        }

        for i, step in enumerate(episode):
            if step.state in n_visits:
                n_visits[step.state] += 1
            else:
                n_visits[step.state] = 1
            if step.state in s_visits:
                s_visits[step.state] += returns_from_state_dict[step.state]
            else:
                s_visits[step.state] = returns_from_state_dict[step.state]

    return {s: s_visits[s] / n_visits[s] for s in n_visits.keys()}


def tabular_td_prediction(
    transitions: Iterable[mp.TransitionStep[S]],
    learning_rate: float,
    gamma: float
) -> V[S]:
    """Predict the value function using tabular temporal difference.

    :param transitions: Transition steps reporting state, next state, and
        reward obtained from atomic experiences
    :param learning_rate: Factor of TD error by which to update value function
        in each iteration
    :param gamma: Discount factor
    """
    state_returns: V[S] = {}
    for step in transitions:
        if step.state not in state_returns:
            state_returns[step.state] = 0

        # Update TD(0) prediction of VF for `step.state`
        if step.next_state not in state_returns:
            state_returns[step.state] += learning_rate * (
                step.reward + gamma * 0 -
                state_returns[step.state]
            )
        else:
            state_returns[step.state] += learning_rate * (
                step.reward + gamma * state_returns[step.next_state] -
                state_returns[step.state]
            )
    return state_returns


def main():
    """Run the prediction algorithms on the vampire problem.
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

    # Apply Tabular MC prediction to approximate optimal value function
    vampire_mrp: mp.FiniteMarkovRewardProcess[S] =\
        vampire_mdp.apply_finite_policy(pi)
    num_traces = 10000
    traces = get_traces(vampire_mrp, start_state_dist, num_traces)
    pred_val_mc = tabular_mc_prediction(traces, 1)
    print("Predicted optimal value function by MC prediction: ")
    pprint(pred_val_mc)

    # Apply Tabular TD prediction to approximate optimal value function
    atomic_experiences = [step for trace in traces for step in trace]
    pred_val_td = tabular_td_prediction(atomic_experiences, 0.05, 1)
    print("Predicted optimal value function by TD prediction: ")
    pprint(pred_val_td)


if __name__ == "__main__":
    main()
