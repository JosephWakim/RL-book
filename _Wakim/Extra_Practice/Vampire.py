"""
Vampire Problem.

Joseph Wakim
February 14, 2021
CME 241
"""

from dataclasses import dataclass
from typing import (Mapping, Dict, Optional, Tuple)

import rl.dynamic_programming as dp
from rl.distribution import (Categorical, FiniteDistribution)
import rl.markov_decision_process as mdp


@dataclass(frozen=True)
class State:
    """Class representation of the state of the village.

    :param n: Number of villagers in the village
    :param v: Indicator for whether the vampire is alive (True) or not (False)
    """
    n: int
    v: bool


VillageReward = FiniteDistribution[Tuple[mdp.S, float]]
ActionMapping = Mapping[mdp.A, VillageReward[mdp.S]]
VillageActionMapping = mdp.StateActionMapping[State, Optional[ActionMapping]]


@dataclass(frozen=True)
class Action:
    """Class representation of the action space.

    :param p: Number of villagers to sacrifice
    """
    p: int


class VampireMDP(mdp.FiniteMarkovDecisionProcess[mdp.S, mdp.A]):
    """Class representatation of the Vampire Problem as an MDP.
    """

    def __init__(self, init_pop: int):
        """Initialize the MDP representation of the Vampire problem.

        :param init_pop: Initial population of the village.
        """
        self.init_pop = init_pop
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> VillageActionMapping:
        """Get the mapping from states and actions to rewards and new states.
        """
        d: Dict[
            State, Dict[Action, Categorical[Tuple[State, float]]]
        ] = {}

        # Specify terminal state transition probabilies
        for i in range(self.init_pop+1):
            d[State(i, False)] = None
        d[State(0, True)] = None

        # Specify non-terminal state transition probabilities
        for i in range(1, self.init_pop+1):
            state: State = State(i, True)
            d1: Dict[Action, Categorical[Tuple[State, float]]] = {}

            for p in range(i):
                sr_probs_dict: Dict[Tuple[State, float], float] = {}
                sr_probs_dict[(State(i-p, False), i-p)] = p/i
                sr_probs_dict[(State(i-p-1, True), 0)] = 1 - p/i
                d1[Action(p)] = Categorical(sr_probs_dict)

            d[state] = d1

        return d


if __name__ == "__main__":
    from pprint import pprint
    vampire_mdp = VampireMDP(100)
    val, pi = dp.policy_iteration_result(vampire_mdp, 1)
    pprint(pi)
    pprint(val)
