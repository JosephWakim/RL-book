"""
Select croak actions to maximize probability for successful river cross.

A frog is crossing a river containing n lily pads. At each step of the voyage,
the frog selects one of two croaking actions (A and B) which dictate where the
frog will move next based on its current position i. Croak A moves the frog
back one space with probability i/n and forward one space with probability
(n-i)/n. Croak B moves the frog to any position (except its current position)
with uniform probability. The journey is successful if the frog reaches
position n, and unsuccessful if the frog lands on position 0, where a snake
is waiting to eat it.

This problem uses a Markov decision process to select the croaking action at
each state which maximizes the probability that the frog will successfully
cross the river.

Joseph Wakim
CME 241
29-Jan-2021
"""

import sys
from dataclasses import dataclass
from typing import (Mapping, Dict, Optional, Tuple, List)
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from rl.distribution import (Constant, Categorical, FiniteDistribution)
import rl.markov_process as mp
import rl.markov_decision_process as mdp
import rl.markov_process as mp


@dataclass(frozen=True)
class FrogState:
    """Class representation of the frog's position in the river.

    The problem statement is framed in such a way that using current position
    is convenient for calculating transition probabilities. This is because all
    transition probabilities are based on current position and/or the total
    number of lily pads.

    :param position: Location of the frog in the river.
    """
    position: int


@dataclass(frozen=True)
class Croak:
    """Class representation of the croaks that the frog can take.

    The frog can take one of two actions: croak A or croak B. The action
    influences the probabilities of transitioning from current state S_t
    to any different next state S_{t+1}

    :param croak_A: Flag indicating if the croak was A (indicated by True) or B
        (indicated by False).
    """
    croak_A: bool


FrogReward = FiniteDistribution[Tuple[mdp.S, float]]
CroakMapping = Mapping[mdp.A, FrogReward[mdp.S]]
FrogCroakMapping = mdp.StateActionMapping[FrogState, Optional[CroakMapping]]


@dataclass
class River:
    """Transition probabilities from positions in river.

    All transition probabilities are dependent on the number of lily pads in
    the river.

    :param n_lily: Number of lily pads between riverbanks.
    """
    n_lily: int


class FrogProblemMDP(mdp.FiniteMarkovDecisionProcess[mdp.S, mdp.A]):
    """Markov decision process for croak selection while crossing river.
    """

    def __init__(self, river: River):
        """Initialize the MDP representation of the Frog problem.

        :param river: Object representation of the river the frog is crossing.
        """
        self.river = river
        self.deterministic_policies:\
            List[mdp.FinitePolicy[FrogState, Croak]] =\
                self.get_determ_policies()
        
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> FrogCroakMapping:
        """Get the mapping from states and actions to rewards and new states.
        """
        d: Dict[
            FrogState, Dict[Croak, Categorical[Tuple[FrogState, float]]]
        ] = {
            FrogState(self.river.n_lily) : None,
            FrogState(0) : None
        }

        for position in range(1, self.river.n_lily):
            state: FrogState = FrogState(position)
            d1: Dict[Croak, Categorical[Tuple[FrogState, float]]] = {}
            
            for croak_A in (True, False):
                sr_probs_dict: Dict[Tuple[FrogState, float], float] =\
                    {
                        (FrogState(next_p), self.reward_at_position(next_p)):
                        self.get_transit_prob_from_croak(
                            state, FrogState(next_p), croak_A
                        ) for next_p in range(self.river.n_lily+1)
                    }
                d1[croak_A] = Categorical(sr_probs_dict)
            
            d[state] = d1
        return d

    def get_transit_prob_from_croak(
        self,
        start_state: FrogState,
        next_state: FrogState,
        croak_A: bool
    ) -> Optional[float]:
        """Get transition probability from one state to another, given croak.

        Encode the rules for transition probabilities given in the problem
        statement.

        :param start_state: Initial position of the frog
        :param next_state: Proposed next position of the frog
        :param croak_A: Indicator of the croak type, where `True` indicates A
            and `False` indicates B
        :returns: Probability of transition or `None` if at terminal state
        """
        if croak_A:
            return self.croak_A_probs(start_state, next_state)
        else:
            return self.croak_B_probs(start_state, next_state)
    
    def croak_A_probs(self, s: FrogState, next_s: FrogState) -> float:
        """Return probabilities of state transition given croak A.

        :param s: Initial position of the frog
        :param next_s: Proposed next position of the frog
        """
        if next_s.position == s.position + 1:
            return 1 - (s.position / self.river.n_lily)
        elif next_s.position == s.position - 1:
            return s.position / self.river.n_lily
        else:
            return 0

    def croak_B_probs(self, s: FrogState, next_s: FrogState) -> float:
        """Return probabilities of state transition given croak B.

        :param s: Initial position of the frog
        :param next_s: Proposed next position of the frog
        """
        if next_s.position == s.position:
            return 0
        else:
            return 1 / self.river.n_lily

    def reward_at_position(self, next_position: int) -> float:
        """Get the reward for each position.

        We are not concerned with how many hops it takes to savely cross the
        river. We are only concerned with maximizing the probability of a
        successful passage. Therefore, assign a reward of 1 for transitions
        to the winning position and a reward of 0 at all other transitions.

        :param next_position: Lily pad position in river after hop.
        :returns: Reward associated with transition to `next_position`.
        """
        if next_position == self.river.n_lily:
            return 1
        else:
            return 0

    def get_determ_policies(self) ->\
        List[mdp.FinitePolicy[FrogState, Croak]]:
        """Get all deterministic policies associated with the MDP.

        Recursively generate all deterministic policies for the `Frog Escape`
        MDP, and return these finite deterministic policies in a list. Because
        we are assuming deterministic policies, actions are selected with
        constant probabilities at each state for each policy.

        :returns: List of all possible mappings of states to deterministic
            policies
        """
        policy_combos: List[List[bool]] = [None] * (2 ** self.river.n_lily)
        n_combos = [0]
        actions = [None] * self.river.n_lily

        def add_to_policy(actions: List[Optional[bool]], position: int=0):
            """Get combinations of croak_A settings in deterministic policies.

            :param actions: List of actions in current deterministic policy
            :param position: Current lily pad position when recursively forming
                deterministic policies (default = 0)
            """
            if position == self.river.n_lily:
                policy_combos[n_combos[0]] = actions.copy()
                n_combos[0] += 1
                return
            else:
                for action in (True, False):
                    actions[position] = action
                    add_to_policy(actions, position+1)

        add_to_policy(actions)
        return [
            mdp.FinitePolicy(
                {FrogState(i) : Constant(policy[i]) \
                    for i in range(1, self.river.n_lily)}
            ) for policy in policy_combos
        ]


def get_policy_val_funct(
    frog_MDP: mdp.FiniteMarkovDecisionProcess[FrogState, Croak],
    policy: mdp.FinitePolicy,
    gamma: float = 1
) -> np.ndarray:
    """Get the value function assocaited with a policy.

    Use `mdp.apply_finite_policy` to generate a policy-implied finite MRP, then
    apply `mp.get_value_function_vec` to get the value function associated with
    the MRP. Note that we neglect discounting by setting `gamma` to 1 because
    we do not care how long it takes for the frog to cross the river; we only
    care that the frog does so successfully.

    :param frog_MDP: Markov Decision Process representing the `Frog Escape 
        Problem`
    :param policy: Deterministic policy of croak actions
    :param gamma: Discount factor
    :returns: Evaluations of the value function associated with policy implied
        Markov reward process at each state
    """
    finite_MRP: mp.FiniteMarkovRewardProcess[FrogState] =\
        frog_MDP.apply_finite_policy(policy)
    return finite_MRP.get_value_function_vec(gamma)


def get_optimal_policy(
    policies: List[mdp.FinitePolicy],
    val_functs: np.ndarray
) -> mdp.FinitePolicy:
    """Identify optimal policy from value functions of policy-implied MRPs.
    
    The optimal policy is defined as the policy which maximizes the value
    function for all states. Filter policies whose value functions are less
    than the maximum at each state.

    :param policies: N Deterministic policies of croak actions
    :param val_functs: (N, M) array containing evaluations of N value functions
        associated with N policy implied Markov reward processes at each of M 
        state.
    :returns: Optimal policy
    """
    ind_max = []
    for i in range(val_functs.shape[1]):
        for j in range(len(val_functs[:, i])):
            if val_functs[j,i] == max(val_functs[:,i]):
                ind_max.append(j)
    ind_opt = max(set(ind_max), key=ind_max.count)
    return policies[ind_opt]


def get_croak_seq_from_policy(policy: mdp.FinitePolicy) -> Dict[int, bool]:
    """Obtain deterministic croak types at each state given a policy.

    When representing croak types, `True` indicates croak type A and `False`
    indicates type B. This function parses a mapping of `FrogStates` to
    deterministic croak types and returns a mapping of croak positions to
    the boolean croak A type indicators.

    Use this function when color-coding plots based on actions encoded in
    a policy.

    :param policy: Deterministic policy of croak actions
    :returns: Mapping of frog positions to boolean croak A indicators
    """
    states = list(policy.policy_map.keys())
    positions = [state.position for state in states]
    actions = [
        croak_A for dist in [
            dists.table().keys() for dists in list(policy.policy_map.values())
        ] for croak_A in dist
    ]
    positions.insert(0, 0)
    positions.append(len(positions))
    actions.insert(0, None)
    actions.append(None)
    return {positions[i] : actions[i] for i in range(len(positions))}


def plot_escape_probability(
    frog_MDP: mdp.FiniteMarkovDecisionProcess[FrogState, Croak],
    policy: mdp.FinitePolicy,
    n: int
):
    """Plot the escape probabilities at each state of the MDP.

    A reward of 1 was specified for the winning transition, and all other
    transitions were assigned a reward of 0. With a discount factor of 1, the
    value function at each position gives the probability of successfully
    escaping the river. Therefore, to plot escape probabilities, plot the value
    function at each position.

    :param frog_MDP: Markov Decision Process representing the `Frog Escape 
        Problem`
    :param policy: Deterministic policy of croak actions
    :param n: Number of lily pads in the river
    """
    # Load Data
    croak_actions = get_croak_seq_from_policy(policy)
    escape_probabilities = get_policy_val_funct(frog_MDP, policy).tolist()
    # Add trivial escape probabilities at starting and ending positions
    escape_probabilities.insert(0, 0)
    escape_probabilities.append(1)
    df = pd.DataFrame(
        dict(
            position=list(croak_actions.keys()),
            esc_prob=escape_probabilities,
            croak=list(croak_actions.values())
        )
    )

    # Generate plot
    colors = {True: 'blue', False: 'red', None: 'black'}
    fig, ax = plt.subplots()
    ax.scatter(
        x=df["position"],
        y=df["esc_prob"], 
        c=df["croak"].map(colors)
    )
    croak_A = mpatches.Patch(color='blue', label='croak A')
    croak_B = mpatches.Patch(color='red', label='croak B')
    no_croak = mpatches.Patch(color='black', label='no croak')
    ax.legend(handles=[croak_A, croak_B, no_croak], loc="lower right")
    ax.set_xticks(np.arange(0, n+1, 1))
    ax.set_ylim((0, 1))
    ax.set_yticks(np.arange(0, 1, 0.25))
    ax.set_ylabel("Escape probability")
    ax.set_xlabel("Lily pad position")
    ax.set_title("Frog Escape Outcomes, n = " + str(n))
    plt.savefig(
        "Frog_Escape_Outcomes_n_"+str(n)+".png",
        dpi=600
    )


def solve_frog_escape_problem(n: int):
    """Solve the 'frog escape problem' for a given river size

    Find the optimal deterministic policy of A/B croaks based on position in
    the river, and plot the escape probability associated with each position
    using the policy.

    :param n: Number of lily pads in the river, where lily pad `n` is the
    winning position.
    """
    # Define the size of the river in the problem
    n_lily: int = n
    # Specify the river
    river: River = River(n_lily)
    # Instantiate the MDP representation of the `Frog Escape` problem
    frog_MDP: mdp.FiniteMarkovDecisionProcess[FrogState, Croak] =\
        FrogProblemMDP(river)
    # Determine all possible policies
    policies: List[mdp.FinitePolicy] = frog_MDP.get_determ_policies()
    # Get value functions associated with all possible policies
    val_functs: np.ndarray = np.array(
        [get_policy_val_funct(frog_MDP, policy) for policy in policies]
    )
    # Get the optimal policy
    opt_policy: mdp.FinitePolicy = get_optimal_policy(policies, val_functs)
    # Plot the escape probabilities
    plot_escape_probability(frog_MDP, opt_policy, n_lily)


def main():
    """Evaluate policies to determine optimal policy.
    """
    river_sizes = [3, 6, 9]
    for river_size in river_sizes:
        solve_frog_escape_problem(river_size)


if __name__ == "__main__":
    main()