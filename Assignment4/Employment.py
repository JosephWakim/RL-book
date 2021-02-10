"""
Employment problem.

Each day, a person enters the day employed or unemployed. If employed, the
person earns their wage for the day, but faces some constant non-zero chance
of losing their job that day. If unemployed, the person is offered a new job
from a set of possible jobs, each carrying a fixed probability. The unemployed
individual may choose to accept the new job and earn that job's salary for the
day, or reject the offer, earn unemployment, and begin the next day with a new
offer.

The 'Employment Problem' poses the following question: what policy for
accepting or rejecting job offers yields the greatest expected utility of
accumulated wages?

Joseph Wakim
CME241
08-Feb-2021

TODO:
-   Implement a numerical iteration algorithm which uses the Bellman Optimality
    Equation to obtain the optimal policy.
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
class JobState:
    """Class representation of employment states for MDP representation.

    Job position 0 represents unemployment, while positions 1, 2, ... N
    represent employment at companies 1, 2, ... N.

    :param current_job: Index of `Market.job_market` representing current place
        of employment (or unemployment)
    :param offer_job: Index of `Market.job_market` representing offer of
        employment. If `current_job` does not represent unemployed, then
        `offer_job` will be equal to current job.
    """
    current_job: int
    offer_job: int


@dataclass(frozen=True)
class Accept:
    """Class representationn of job acceptance.

    :param accept: True if accepting offer of employment, else false
    """
    accept: bool


@dataclass(frozen=True)
class Market:
    """Class representation of the jobs available for employment.

    :param job_salaries: Mapping of jobs to salaries
    :param job_prob: Mapping of jobs to employment offer probabilities
    :param alpha: Probability of losing a job at any given day
    """
    job_salaries: Dict[int, float]
    job_prob: Dict[int, float]
    alpha: float


JobReward = FiniteDistribution[Tuple[mdp.S, float]]
AcceptMapping = Mapping[mdp.A, JobReward[mdp.S]]
JobAcceptMapping = mdp.StateActionMapping[JobState, AcceptMapping]


class EmploymentMDP(mdp.FiniteMarkovDecisionProcess[mdp.S, mdp.A]):
    """Markov decision process for employment acceptance or rejection.
    """

    def __init__(self, market: Market):
        """Initialize the MDP representation of the employment problem.

        :param market: Market of all jobs available for employment, with
            associated wages and employment probabilities
        """
        self.market = market
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> JobAcceptMapping:
        """Get the mapping from states and actions to rewards and new states.
        """
        d: Dict[JobState, Dict[Accept, Categorical[Tuple[JobState, float]]]] = {}

        for p1 in self.market.job_salaries.keys():
            for p2 in list(self.market.job_salaries.keys())[1:]:
                state = JobState(p1, p2)
                if p1 not in (0, p2):
                    continue

                d1: Dict[Accept, Categorical[Tuple[JobState, float]]] = {}

                for accept in (True, False):
                    sr_probs_dict: Dict[Tuple[JobState, float], float] = {}
                    if p1 == 0:
                        if accept:
                            sr_probs_dict[
                                (JobState(p2, p2), self.market.job_salaries[p2])
                            ] = 1
                        else:
                            for job in list(self.market.job_salaries.keys())[1:]:
                                sr_probs_dict[
                                    (
                                        JobState(p1, job),
                                        self.market.job_salaries[p1]
                                    )
                                ] = self.market.job_prob[job]

                    else:
                        sr_probs_dict[state, self.market.job_salaries[p1]] =\
                            1-self.market.alpha
                        for job in list(self.market.job_salaries.keys())[1:]:
                            sr_probs_dict[
                                (
                                    JobState(0, job),
                                    self.market.job_salaries[p1]
                                )
                            ] = self.market.alpha * self.market.job_prob[job]

                    d1[accept] = Categorical(sr_probs_dict)

                d[state] = d1

        for d1 in d:
            print(d1)
            print(d[d1])
            print()
        return d


def main():
    """Evaluate the employment scenario.
    """
    # Specify salaries
    job_salaries: Dict[int, float] = {
        0: 5,   # Unemployment
        1: 10,
        2: 20,
        3: 40,
    }
    # Specify hiring probabilities
    job_prob: Dict[int, float] = {
        1: 0.6,
        2: 0.3,
        3: 0.1
    }
    # Set unemployment probability
    alpha: float = 0.01
    # generate MDP
    market: Market = Market(job_salaries, job_prob, alpha)
    employment_mdp: EmploymentMDP[JobState, Accept] = EmploymentMDP(market)


if __name__ == "__main__":
    main()