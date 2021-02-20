"""
Select croak actions to maximize probability for successful river cross.

This example builds on the original 'Frog Escape' problem from assignment 3. In
assignment 3, we generated value functions from all combinations of croak
actions at each state, and we selected the action which maximized the value
function at each state. This approach becomes computationally intractible for
problems with large state or action spaces. In this example, we instead
implement policy and value iteration to determine the policy which optimizes
the probability of successfully escaping the river.

Joseph Wakim
CME 241
08-Feb-2021
"""

from typing import (Tuple, Iterator, TypeVar)

import numpy as np
import matplotlib.pyplot as plt
import rl.markov_decision_process as mdp
import rl.dynamic_programming as dp
import Assignment3.Frog_Escape as fe


I = TypeVar('I')


def calculate_error(
    v1: dp.V[dp.S],
    v2: dp.V[dp.S]
) -> float:
    """Quantify the error between value functions of successive iterations.

    :param v1: Value function vector obtained during first iteration
    :param v2: Value function vector obtained during second iteration
    :returns: Maximum difference between value function vectors
    """
    return max(abs(v1[s] - v2[s]) for s in v1)


def convert_to_tuple(output: I) -> Tuple[I]:
    """Convert value to tuple, if not already tuple.

    :param output: Entity which must be cast as a tuple.
    """
    return tuple([output]) if not isinstance(output, tuple) else output


def iterate(
    iterator: Iterator[I],
    max_iters: int = 1000,
    tolerance: float = dp.DEFAULT_TOLERANCE
) -> Tuple[I, np.ndarray]:
    """Run iterator to convergence or iteration limit.

    :param iterator: Either policy or value iterator
    :param max_iters: maximum number of iterations to execute
    :param tolerance: Tolerance to identify convergence.
    :returns: Optimal output of iterator, with vector of errors.
    """
    output: I = iterator.__next__()
    output: Tuple[I] = convert_to_tuple(output)
    n_iter: int = 1
    errors = []

    while True:
        n_iter += 1
        old_output: Tuple[I] = output
        output = iterator.__next__()
        output = convert_to_tuple(output)
        error = calculate_error(output[0], old_output[0])
        errors.append(error)
        if error < tolerance:
            return output, np.array(errors)
        if n_iter == max_iters:
            print(
                "Policy iteration failed to converge after "+str(max_iters)+\
                    " iterations."
            )
            return output, np.array(errors)


def frog_escape_policy_iteration(
    frog_MDP: fe.FrogProblemMDP,
    gamma: float = 1,
    max_iters: int = 1000
) -> Tuple[Tuple[dp.V[dp.S], mdp.FinitePolicy], np.ndarray]:
    """Get optimal policy for 'Frog Escape' problem by policy iteration.

    :param frog_MDP: MDP representation of the 'Frog Escape' problem.
    :param gamma: Discount factor
    :param max_iters: Maximum number of policy iterations to execute
    :returns: Optimal value function and policy, with vector of errors obtained
        during convergence. Returned in form ((val_func, policy), errors).
    """
    policy_iterator: Iterator[Tuple[dp.V[dp.S], mdp.FinitePolicy[dp.S, dp.A]]] =\
        dp.policy_iteration(
            mdp=frog_MDP,
            gamma=gamma
        )
    return iterate(policy_iterator, max_iters)


def frog_escape_value_iteration(
    frog_MDP: fe.FrogProblemMDP,
    gamma: float = 1,
    max_iters: int = 1000
) -> mdp.FinitePolicy:
    """Get optimal policy for 'Frog Escape' problem by value iteration.

    :param frog_MDP: MDP representation of the 'Frog Escape' problem.
    :param gamma: Discount factor (default 1)
    :param max_iters: Maximum number of value iterations to execute (default
        1000)
    """
    value_iterator: Iterator[dp.V[dp.S]] = dp.value_iteration(
        mdp=frog_MDP,
        gamma=gamma
    )
    vf_errors: Tuple[dp.V[dp.S], np.ndarray] = iterate(
        iterator=value_iterator,
        max_iters=max_iters
    )
    pi: mdp.FinitePolicy[dp.S, dp.A] = dp.greedy_policy_from_vf(
        mdp=frog_MDP,
        vf=vf_errors[0][0],
        gamma=gamma
    )
    return (vf_errors[0][0], pi), vf_errors[1]


def plot_error(
    error: np.ndarray,
    file_name: str
):
    """Plot error progression during convergence.

    :param error: Vector of error values at each iteration.
    :param filename: Path at which to save convergence plot
    """
    plt.clf()
    plt.plot(error)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.savefig(
        fname=file_name,
        dpi=600
    )


def compare_convergence_methods(n_lily):
    """Compare convergence to optimal policy by policy and value iteration.

    :param n_lily: Number of lily pads in 'Frog Escape' problem.
    """
    # Specify the river
    river: fe.River = fe.River(n_lily)
    # Instantiate the MDP representation of the `Frog Escape` problem
    frog_MDP: mdp.FiniteMarkovDecisionProcess[fe.FrogState, fe.Croak] =\
        fe.FrogProblemMDP(river)

    # Run policy iteration
    print("POLICY ITERATION")
    print()
    policy_iter_vf_pi_errors: Tuple[Tuple[dp.V[dp.S], mdp.FinitePolicy], np.ndarray] =\
        frog_escape_policy_iteration(frog_MDP)
    print("Optimal Value function: ")
    print(policy_iter_vf_pi_errors[0][0])
    print()
    print("Optimal Policy: ")
    print(policy_iter_vf_pi_errors[0][1])
    print()
    plot_error(
        policy_iter_vf_pi_errors[1],
        "policy_iteration_convergence_plot_n_"+str(n_lily)+".png"
    )

    # Run value iteration
    print("VALUE ITERATION")
    print()
    value_iter_vf_pi_errors: Tuple[dp.V[dp.S], mdp.FinitePolicy] =\
        frog_escape_value_iteration(frog_MDP)
    print("Optimal Value function: ")
    print(value_iter_vf_pi_errors[0][0])
    print()
    print("Optimal Policy: ")
    print(value_iter_vf_pi_errors[0][1])
    print()
    plot_error(
        value_iter_vf_pi_errors[1],
        "value_iteration_convergence_plot_n_"+str(n_lily)+".png"
    )


def main():
    """Evaluate policies to determine optimal policy.
    """
    for n in [3, 6, 9, 12, 15]:
        compare_convergence_methods(n)


if __name__ == "__main__":
    main()
