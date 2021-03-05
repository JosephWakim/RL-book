"""
Implementation of n-step Bootstrapping Prediction, TD Lambda.

Joseph Wakim
CME 241 - Assignment 12, Problems 1, 2
March 4, 2021
"""

from typing import Iterable, TypeVar, Dict, Mapping, Callable, List
from collections import defaultdict

import matplotlib.pyplot as plt
import rl.markov_process as mp
import rl.markov_decision_process as mdp
import rl.distribution as dist
import rl.dynamic_programming as dp
import _Wakim.Extra_Practice.Vampire as vampire
from _Wakim.Assignment11.Q123_Tabular_MC_TD import get_traces


S = TypeVar('S')
A = TypeVar('A')
R = TypeVar('R')
V = Mapping[S, float]


def downweight_eligibility(
    eligibility: Dict[S, float],
    gamma: float,
    lambda_param: float
) -> Dict[S, float]:
    """Downweight the eligibility trace for a single transition.

    :param eligibility: Eligibility trace of each state
    :param gamma: Discount rate
    :param lambda_param: Factor to down-weight historical data
    :returns: Updated eligibility trace reflecting new step
    """
    for key in eligibility:
        eligibility[key] *= lambda_param * gamma
    return eligibility


def increment_eligibility(
    eligibility: Dict[S, float],
    lambda_param: float,
    state: S
) -> Dict[S, float]:
    """Increment the eligibility trace for a new state.

    :param eligibility: Eligibility trace of each state
    :param lambda_param: Factor to down-weight historical data
    :param state: State encountered at current step
    :returns: Updated eligibility trace reflecting new step
    """
    eligibility[state] += 1
    return eligibility


def def_value() -> float:
    """Return a default value for a dictionary being filled

    :returns: Default value of zero
    """
    return 0


def get_learning_rate(iter_count: int) -> float:
    """Learning rate function of iteration count.

    :param iter_count: Count of current iteration
    :returns: Learning rate for current iteration count
    """
    return 1 / (iter_count ** 0.66)


def remove_terminal_vampire_states(val_func: V[S]) -> V[S]:
    """Remove value function of terminal vampire states from dict.

    :param val_func: Mapping of all states to predicted value function
    :returns: Value function mapping with terminal states removed
    """
    return {state: val_func[state] for state in val_func if state.v}


def print_non_terminal_vampire_states(val_func: V[S]):
    """Print value function of non-terminal vampire states.

    :param val_func: Mapping of all states to predicted value function
    """
    filtered_val_func = remove_terminal_vampire_states(val_func)
    for state in filtered_val_func:
        print(
            "State: ",
            state,
            " VF: ",
            val_func[state]
        )


def tabular_n_step_bootstrap(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    learning_rate: Callable[[int], float],
    n_step: int,
    gamma: float
) -> V[S]:
    """Predict the value function using tabular TD-lambda.

    :param traces: MRP traces obtained from episodes
    :param learning_rate: Factor of TD error by which to udpdate value function
        in each iteration; some function of the iteration count
    :param n_step: Number of forward-looking steps in bootstrap
    :param gamma: Discount factor
    :returns: Predicted value function associated with each state
    :returns: Convergence of VF after each iteration
    """
    val_func: V[S] = defaultdict(def_value)
    vf_convergence = []
    num_iter = 0

    for episode in traces:
        for i, step in enumerate(episode):
            return_n = 0
            num_iter += 1
            for j, future in enumerate(
                episode[i: min(len(episode), i+n_step+1)]
            ):
                return_n += gamma**j * future.reward
                if j + i == len(episode)-1 or j == n_step:
                    return_n += gamma**j * val_func[future.next_state]
            td_error = return_n - val_func[step.state]
            val_func[step.state] += learning_rate(num_iter) * td_error
            vf_convergence.append(val_func.copy())

    return val_func, vf_convergence


def tabular_TD_lambda(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    learning_rate: Callable[[int], float],
    lambda_param: float,
    gamma: float
) -> V[S]:
    """Predict the value function using tabular TD-lambda.

    :param traces: MRP traces obtained from episodes.
    :param learning_rate: Factor of TD error by which to udpdate value function
        in each iteration; some function of the iteration count
    :param lambda_param: Factor to down-weight historical data
    :param gamma: Discount factor
    :returns: Predicted value associated with each state
    :returns: Convergence of VF after each iteration
    """
    # Initialize eligibility trace and value function at each position
    eligibility: Dict[S, float] = defaultdict(def_value)
    val_func: V[S] = defaultdict(def_value)
    vf_convergence = []
    num_iter: int = 0

    # Loop through each atomic experience in the trace
    for episode in traces:
        for step in episode:
            num_iter += 1
            alpha = learning_rate(num_iter)
            td_error = step.reward + gamma * val_func[step.next_state] -\
                val_func[step.state]
            eligibility = increment_eligibility(
                eligibility=eligibility,
                lambda_param=lambda_param,
                state=step.state
            )
            for state in val_func:
                val_func[state] += alpha * td_error *\
                    eligibility[state]
            eligibility = downweight_eligibility(
                eligibility=eligibility,
                gamma=gamma,
                lambda_param=lambda_param
            )
            vf_convergence.append(val_func.copy())

    return val_func, vf_convergence


def plot_convergence(vf_convergence: List[V[S]], save_path: str):
    """Plot convergence of value function at each state.

    :param vf_convergence: VFs at each step of convergence
    :param save_path: path at which to save convergence plots
    """
    convergence_dict: Mapping[S, List[float]] = {
        state: [
            VF_iter[state] for VF_iter in vf_convergence
        ] for state in vf_convergence[-1]
    }

    plt.figure()
    convergence_dict = remove_terminal_vampire_states(convergence_dict)
    for state in convergence_dict:
        plt.plot(convergence_dict[state], label=state.n)

    plt.legend(loc=(1.04, 0))
    plt.xlabel("Number of iterations")
    plt.ylabel("Predicted VF")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)


def run_tabular_td_lambda(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    learning_rate: Callable[[int], float],
    lambda_param: List[float],
    gamma: float
):
    """Plot convergence of predicted VF from each state using TD-lambda.

    :param traces: MRP traces obtained from episodes.
    :param learning_rate: Factor of TD error by which to udpdate value function
        in each iteration; some function of the iteration count
    :param lambda_param: Factors by which to down-weight historical data
    :param gamma: Discount factor
    """

    for l in lambda_param:

        vf, vf_conv = tabular_TD_lambda(
            traces=traces,
            learning_rate=get_learning_rate,
            lambda_param=l,
            gamma=1
        )
        file_name =\
            "plots/tabular_TD_convergence_lambda_"+str(l).replace('.', 'p')
        plot_convergence(
            vf_conv,
            file_name + '.png'
        )


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

    # Express the vampire problem as an MRP and sample traces
    vampire_mrp: mp.FiniteMarkovRewardProcess[S] =\
        vampire_mdp.apply_finite_policy(pi)
    num_traces = 100000
    traces = get_traces(vampire_mrp, start_state_dist, num_traces)

    # Apply tabular TD-lambda to approximate optimal value function
    pred_val_td_lambda, _ = tabular_TD_lambda(
        traces=traces,
        learning_rate=get_learning_rate,
        lambda_param=0.5,
        gamma=1
    )
    print("Predicted value function by TD-lambda prediction: ")
    print_non_terminal_vampire_states(pred_val_td_lambda)

    # Apply tabular n-step boostrap to predict optimal value function
    pred_val_n_step, _ = tabular_n_step_bootstrap(
        traces=traces,
        learning_rate=get_learning_rate,
        n_step=3,
        gamma=1
    )
    print("Predicted value function by tabular n-step prediction: ")
    print_non_terminal_vampire_states(pred_val_n_step)

    # Plot Convergence of VF prediction by TD-lambda at various lambdas
    run_tabular_td_lambda(
        traces=traces,
        learning_rate=get_learning_rate,
        lambda_param=[0, 0.25, 0.5, 0.75, 0.99],
        gamma=1
    )


if __name__ == "__main__":
    main()
