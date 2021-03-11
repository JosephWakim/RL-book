"""
Implementation of Control algorithms.

    - Implement Tabular MC Control - GLIE
    - Implement Tabular SARSA
    - Implement Tabular Q-learning

Joseph Wakim
CME 241 - Assignment 13, Problems 1-4
March 6, 2021
"""

from typing import Iterable, TypeVar, Dict, Mapping, Tuple
from collections import defaultdict
from pprint import pprint

import numpy as np
import rl.markov_decision_process as mdp
import rl.distribution as dist
import rl.dynamic_programming as dp
import rl.function_approx as fa
import _Wakim.Extra_Practice.Vampire as vampire


S = TypeVar('S')
A = TypeVar('A')
R = TypeVar('R')
V = Mapping[S, float]
Q = Mapping[Tuple[S, A], float]


def get_random_policy(
    mdp_obj: mdp.FiniteMarkovDecisionProcess[S, A]
) -> mdp.FinitePolicy[S, A]:
    """Generate a random policy for an MDP by uniform sampling of action space.

    This function is used to initialize the policy during MC Control.

    :param mdp_obj: MDP object for which random policy is being generated
    :returns: Random deterministic policy for MDP
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
            random_action = uniform_actions.sample()
            state_action_dict[state] = dist.Constant(random_action)
        else:
            state_action_dict[state] = None

    return mdp.FinitePolicy(state_action_dict)


def def_value() -> float:
    """Return a default value for a dictionary being filled

    :returns: Default value of zero
    """
    return 0


def get_episode(
    mdp: mdp.FiniteMarkovDecisionProcess[S, A],
    policy: mdp.FinitePolicy[S, A],
    start_state_dist: dist.FiniteDistribution[S]
) -> Iterable[mdp.TransitionStep[S, A]]:
    """Generate a single episode from the MDP and the policy.

    Beginning at a random starting state, generate a complete episode.

    :param mdp: Markov decision process from which to generate an episode
    :param policy: Policy applied to the mdp when generating policy
    :param start_state_dist: Distribution of starting states for episode
    :returns: Transition steps forming the episode
    """
    return list(mdp.simulate_actions(start_state_dist, policy))


def tabular_mc_control(
    mdp_obj: mdp.FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    start_state_dist: dist.FiniteDistribution,
    max_iters: int = 1000
) -> Tuple[Q[S, A], mdp.FinitePolicy[S, A]]:
    """Run Tabular Monte Carlo control to obtain optimal policy and VF.

    NOTE: Even with 300,000 iterations, this selects a suboptimal (but close)
    action 30% of the time.

    :param mdp_obj: Markov decision process which we are simulating
    :param gamma: Discount factor
    :param start_state_dist: Distribution of starting states from MDP
    :param max_iters: Maximum iterations to run when finding optimal VF
        (default 1000)
    :returns: Optimal value function associated with the optimal policy
    :returns: Optimal policy obtained by tabular MC control
    """
    policy: mdp.FinitePolicy[S, A] = get_random_policy(mdp_obj)
    num_visits: Dict[Tuple[S, A], int] = defaultdict(def_value)
    action_val: Dict[Tuple[S, A], int] = defaultdict(def_value)

    for k_iter in range(1, max_iters + 1):
        # Generate an episode
        episode = get_episode(mdp_obj, policy, start_state_dist)
        rewards = [s.reward for s in episode]
        rewards_reverse = rewards.copy()
        rewards_reverse.reverse()
        reverse_returns_from_state = rewards_reverse.copy()
        for i in range(1, len(reverse_returns_from_state)):
            reverse_returns_from_state[i] = gamma *\
                reverse_returns_from_state[i-1] + reverse_returns_from_state[i]
        returns_from_state = reverse_returns_from_state.copy()
        returns_from_state.reverse()
        # Evaluation
        for i, step in enumerate(episode):
            num_visits[(step.state, step.action)] += 1
            action_val[(step.state, step.action)] +=\
                1 / num_visits[(step.state, step.action)] *\
                (returns_from_state[i] - action_val[(step.state, step.action)])
        # Improvement
        epsilon = 1/(k_iter**0.1)
        policy = mdp.policy_from_q(
            q=fa.Tabular(action_val),
            mdp=mdp_obj,
            ϵ=epsilon
        )
    return (
        action_val,
        mdp.policy_from_q(
            q=fa.Tabular(action_val),
            mdp=mdp_obj,
            ϵ=0
        )
    )


def tabular_sarsa(
    mdp_obj: mdp.FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    start_state_dist: dist.FiniteDistribution,
    max_iters: int = 1000
) -> Tuple[Q[S, A], mdp.FinitePolicy[S, A]]:
    """Run Tabular SARSA control to obtain optimal policy and VF.

    :param mdp_obj: Markov decision process which we are simulating
    :param gamma: Discount factor
    :param start_state_dist: Distribution of starting states from MDP
    :param max_iters: Maximum iterations to run when finding optimal VF
        (default 1000)
    :returns: Optimal value function associated with the optimal policy
    :returns: Optimal policy obtained by tabular SARSA
    """
    policy: mdp.FinitePolicy[S, A] = get_random_policy(mdp_obj)
    num_visits: Dict[Tuple[S, A], int] = defaultdict(def_value)
    action_val: Dict[Tuple[S, A], int] = defaultdict(def_value)

    for k_iter in range(1, max_iters + 1):
        # Pick a random starting state
        state = start_state_dist.sample()
        # Generate the episode, with eps-greedy policy updates in each step
        while True:
            episode = get_episode(mdp_obj, policy, dist.Constant(state))
            s1 = episode[0].state
            a1 = episode[0].action
            r = episode[0].reward
            s2 = episode[0].next_state
            if policy.act(s2) is not None:
                a2 = policy.act(s2).sample()
            else:
                a2 = None
            # Update action-value function
            num_visits[(s1, a1)] += 1
            learning_rate = 1 / num_visits[(s1, a1)]
            action_val[(s1, a1)] += learning_rate * (
                r + gamma * action_val[(s2, a2)] - action_val[(s1, a1)]
            )
            # Check if at a terminal state
            if a2 is None:
                break
            # Update policy
            epsilon = 1/(k_iter**0.1)
            policy = mdp.policy_from_q(
                q=fa.Tabular(action_val),
                mdp=mdp_obj,
                ϵ=epsilon
            )
            state = s2
    return (
        action_val,
        mdp.policy_from_q(
            q=fa.Tabular(action_val),
            mdp=mdp_obj,
            ϵ=0
        )
    )


def tabular_qlearning(
    mdp_obj: mdp.FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    start_state_dist: dist.FiniteDistribution,
    max_iters: int = 1000
) -> Tuple[Q[S, A], mdp.FinitePolicy[S, A]]:
    """Run Tabular q-learning to obtain optimal policy and VF.

    Tabular q-learning is very similar to tabular SARSA. The difference between
    the two algorithms lies in the update rule for the action-value function.
    For tabular SARSA, the TD-target is selected based on the action at the
    next state as specified by the current policy. By q-learning, the TD-target
    is selected based on the action at the next state that maximizes the reward
    from that state.

    :param mdp_obj: Markov decision process which we are simulating
    :param gamma: Discount factor
    :param start_state_dist: Distribution of starting states from MDP
    :param max_iters: Maximum iterations to run when finding optimal VF
        (default 1000)
    :returns: Optimal value function associated with the optimal policy
    :returns: Optimal policy obtained by tabular q-learning
    """
    policy: mdp.FinitePolicy[S, A] = get_random_policy(mdp_obj)
    num_visits: Dict[Tuple[S, A], int] = defaultdict(def_value)
    action_val: Dict[Tuple[S, A], int] = defaultdict(def_value)

    for k_iter in range(1, max_iters + 1):
        # Pick a random starting state
        state = start_state_dist.sample()
        # Generate the episode, with eps-greedy policy updates in each step
        while True:
            episode = get_episode(mdp_obj, policy, dist.Constant(state))
            s1 = episode[0].state
            a1 = episode[0].action
            r = episode[0].reward
            s2 = episode[0].next_state
            if policy.act(s2) is not None and s2 in [
                key[0] for key in action_val
            ]:
                vals = np.array(
                    [action_val[key] for key in action_val if key[0] == s2]
                )
                next_action_val = np.max(vals)
            else:
                a2 = None
                next_action_val = 0
            # Update action-value function
            num_visits[(s1, a1)] += 1
            learning_rate = 1 / num_visits[(s1, a1)]
            action_val[(s1, a1)] += learning_rate * (
                r + gamma * next_action_val - action_val[(s1, a1)]
            )
            # Check if at a terminal state
            if a2 is None:
                break
            # Update policy
            epsilon = 1/(k_iter**0.1)
            policy = mdp.policy_from_q(
                q=fa.Tabular(action_val),
                mdp=mdp_obj,
                ϵ=epsilon
            )
            state = s2
    return (
        action_val,
        mdp.policy_from_q(
            q=fa.Tabular(action_val),
            mdp=mdp_obj,
            ϵ=0
        )
    )


def print_if_optimal(q: Q[S, A], opt_pi: mdp.FinitePolicy[S, A]):
    """Print entries of the action value function for optimal policy.

    :param q: Action-value function
    :param opt_pi: Optimal policy
    """
    for entry in q:
        if opt_pi.act(entry[0]) is not None:
            if entry[1] == opt_pi.act(entry[0]).sample():
                print("State: ")
                print(entry[0])
                print("Action: ")
                print(entry[1])
                print("Action-Value: ")
                print(q[entry])
                print()


def main():
    """Run the control algorithms.

    Test the control algorithms using the `Vampire Problem` MDP.
    """

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
    print("True optimal policy: ")
    print(pi)
    print()
    print("True optimal value function: ")
    pprint(true_val)

    # Apply tabular MC control to obtain the optimal policy and value function
    pred_action_val, pred_pi = tabular_mc_control(
        vampire_mdp,
        1,
        start_state_dist,
        10000
    )
    print("Predicted optimal policy: ")
    for i in range(1, num_villagers+1):
        print("Num Villagers: " + str(i) + "; Vampire Alive: True")
        print(pred_pi.act((vampire.State(i, True))))
    print()
    print("Predicted optimal action-value function: ")
    print_if_optimal(pred_action_val, pred_pi)

    # Apply tabular SARSA to obtain the optimal policy and value function
    pred2_action_val, pred2_pi = tabular_sarsa(
        vampire_mdp,
        1,
        start_state_dist,
        10000
    )
    print("Predicted optimal policy: ")
    for i in range(1, num_villagers+1):
        print("Num Villagers: " + str(i) + "; Vampire Alive: True")
        print(pred2_pi.act((vampire.State(i, True))))
    print()
    print("Predicted optimal action-value function: ")
    print_if_optimal(pred2_action_val, pred2_pi)

    # Apply tabular Q-learning to obtain the optimal policy and value function
    pred3_action_val, pred3_pi = tabular_qlearning(
        vampire_mdp,
        1,
        start_state_dist,
        100000
    )
    print("Predicted optimal policy: ")
    for i in range(1, num_villagers+1):
        print("Num Villagers: " + str(i) + "; Vampire Alive: True")
        print(pred3_pi.act((vampire.State(i, True))))
    print()
    print("Predicted optimal action-value function: ")
    print_if_optimal(pred3_action_val, pred3_pi)


if __name__ == "__main__":
    main()
