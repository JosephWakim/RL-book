"""
Calculate expected number of hops required to cross the river.

By:                     Joseph Wakim
Collaborated with:      Michael Bechinhausen
Course:                 CME 241
Date:                   January 16, 2021
"""

from dataclasses import dataclass
from typing import (Mapping, Dict, Optional)
import itertools

import numpy as np

from rl.distribution import (Constant, Categorical)
import rl.markov_process as mp
from Snakes_and_Ladders import generate_histogram


@dataclass(frozen=True)
class FrogState:
    """Class representation of remaining number of hops to cross river.

    By representing the state as the number of lilypads required to cross the
    river, rather than the position of the frog, we reduce the number of times
    that the number of remaining positions has to be calculated.

    :param remaining: Number of remaining lilypads before end of river
    """
    remaining: int

class FrogProblemMPFinite(mp.FiniteMarkovProcess[FrogState]):
    """
    Finite Markov process simulating the 'Frog Problem'.

    The 'Frog Problem' says taht there is a frog placed at one end of a
    river, separated from the opposing end with a finite number of lilypads.
    The frog has a uniform likelihood of jumping to any of the lilypads ahead
    of it. The question posed by the problem is: what is the expected number of
    hops required to cross the river. This problem can be posed as a finite
    Markov process since possible hops depend only on current positions and are
    agnostic to the history of previous hops.
    """

    def __init__(self, n_lilypads: int):
        """Initialize finite Markov process representation of 'Frog problem'.

        :param n_lilypads: Number of lilypads between riverbanks.
        """
        self.n_lilypads = n_lilypads
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> mp.Transition[mp.S]:
        """
        Get the transition map associated with the 'Frog problem'.

        The frog transitions with uniform probability to any of the remaining
        lilypads ahead of it. Due to the uniform probability distribution, the
        probability of hopping to any remaining lilypad is given by the
        reciprocal of the number of remaining lilypads.

        :returns: Transition map representing the 'Frog problem'.
        """
        d: Dict[FrogState, Optional[Categorical[FrogState]]] = {
            FrogState(0) : None
        }
        for remaining in range(1, self.n_lilypads+1):
            frog_state = FrogState(remaining)
            frog_state_prob_map: Mapping[FrogState: float] = {
                FrogState(i) : (1 / remaining) for i in range(0, remaining)
            }
            d[frog_state] = Categorical(frog_state_prob_map)
        return d


def frog_problem_traces(num_lilypads: int, n_traces: int) -> np.ndarray:
    """Simulate frog problem to predict expected hops required to cross river.

    In each simulation, the frog starts on a riverbank, so the starting state
    will always be equal to the total number of lilypads in the simulation.

    :param num_lilypads: Number of lilypads between riverbanks
    :param n_traces: Number of traces to generate
    :return: Hopping counts required to cross the river obtained from traces
    """
    frog_problem_sim = FrogProblemMPFinite(
        n_lilypads=num_lilypads
    )
    start_state = Constant(FrogState(num_lilypads))
    return np.fromiter(
        (len(list(trace)) for trace in itertools.islice(
            frog_problem_sim.traces(start_state),
            n_traces+1
        )), int
    )


def main():
    """Run simulations of 'Frog Problem' outcomes to approximate expected val.
    """
    # Generate 1000 Traces
    num_sims = 1000
    # Specify 10 lilypads between riverbanks
    num_lilypads = 1000
    # Simulate Outcomes
    num_hops = frog_problem_traces(num_lilypads, num_sims)
    # Calculate expected number of hops required to cross the river
    expected_hops = np.average(num_hops)
    # Plot distribution of the number of hops required to cross the river
    generate_histogram(
        data=num_hops,
        file_name="Frog_Hops_to_Cross_River.png",
        x_axis_label="Num. Hops",
        bins=np.arange(0, 20, 1)
    )
    print("Expected Hops: " + str(expected_hops))


if __name__ == "__main__":
    main()
