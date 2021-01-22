"""
Practice implementing a concrete class from abstract `FiniteDistribution`.

Implement a  subclass of `FiniteDistribution` representing a uniform
distribution.

Joseph Wakim
CME 241
January 19, 2021
"""

from typing import (Mapping, List)

import numpy as np
from rl.distribution import (A, FiniteDistribution)


class UniformDistribution(FiniteDistribution[A]):
    """A distribution with a finite number of values of uniform probability."""
    values: List[A]

    def __init__(self, values: List[A]):
        """
        Initialize the UniformDistribution object.

        :param values: Values occuring with uniform probability.
        """
        self.values = values
        self.num_vals = len(values)

    def sample(self) -> A:
        """
        Sample a random value from the uniform distribution.

        :returns: Random, uniformally selected value
        """
        ind = np.random.randint(0, self.num_vals+1)
        return self.values[ind]

    def table(self) -> Mapping[A, float]:
        """
        Tabular representation of uniform distribution PDF.

        :returns: Probability of selecting each unique value
        """
        unique, freq = np.unique(self.values, return_counts=True)
        uniform_prob = 1 / self.num_vals
        return {unique[i]: uniform_prob * freq[i] for i in range(len(unique))}
