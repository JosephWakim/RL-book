"""
Extension of chapter 1 stock price example using Markov Reward process.

This module offers an extension of the stock price process 3 example from
chapter 2 of 'Foundations of Reinforced Learning with Applications in Finance'
by Prof. Ashwin Rao and Tikhon Jelvis.

The stock price model will be represented as a Markov Reward Process, where
the reward at each time is given by a function of the stock price.

Joseph Wakim
CME 241
January 21, 2021
"""

from typing import (Tuple, Callable, List)
import itertools

import matplotlib.pyplot as plt
from rl.distribution import (Constant, SampledDistribution)
import rl.markov_process as mp
from rl.chapter2.stock_price_mp import StateMP3, StockPriceMP3


def get_stock_price(state: StateMP3, init_price: float) -> float:
    """Get the current stock price.

    :param StateMP3: Current state of the stock in process 3 (indicating
        the number of previous upward and downward movements)
    :param init_price: Initial stock price
    :returns: The current price of the stock
    """
    return init_price + state.num_up_moves - state.num_down_moves


class StockPriceMRP3(mp.MarkovRewardProcess[StateMP3]):
    """Markov Reward Process representing the third stock price example.

    In this example, stock movements are "reverse-pulling", bias against the
    historically predominant direction of change.
    """

    def __init__(self,
        alpha: float,
        stock_price_MP3: StockPriceMP3,
        init_price: float,
        f: Callable[[float], float]
    ):
        """Initialize `StockPriceMRP3` object.

        :param alpha3: Reverse-pull strength (>= 0)
        :param stock_price_MP3: Object representation of stock price model cast
            as a Markov process
        :param init_price: Initial stock price
        :param f: Function computing reward at time t from state at time t
        """
        self.alpha = alpha
        self.init_price = init_price
        self.MP_model = stock_price_MP3
        self.f = f

    def transition_reward(
        self, state: StateMP3
    ) -> SampledDistribution[Tuple[StateMP3, float]]:
        """Map each state to next states with reward of the transition.

        :param StateMP3: Current state of the stock in process 3 (indicating
            the number of previous upward and downward movements)
        :returns: Distribution of states and associated rewareds
        """

        def sample_next_state_reward(state=state) -> Tuple[StateMP3, float]:
            """Sample distribution of next state/rewards given current state.

            :param state: Stock state at time t-1
            :returns: Stock state and reward at time t
            """
            next_state: StateMP3 = self.MP_model.transition(state).sample()
            next_stock_price: float = get_stock_price(
                next_state, self.init_price
            )
            reward: float = self.f(next_stock_price)
            return next_state, reward

        return SampledDistribution(sample_next_state_reward)


def reward_MRP_simulation(
    alpha: float,
    gamma: float,
    num_time: int,
    stock_MP3: StockPriceMP3,
    init_price: float,
    f: Callable[[float], float]
) -> List[Tuple[StateMP3, float]]:
    """
    Simulate reward from MRP stock price model for fixed time interval.

    :param alpha: Reverse-pull strength in stock model
    :param gamma: discount factor
    :param num_time: Number of time steps during which to record simulation
    :param stock_MP3: Markov process representation of the stock price model
    :param init_price: Initial stock price
    :param f: Function computing reward at time t from state at time t
    :returns: List of (state, reward) tuples obtained by MRP simulation
    """
    stock_MRP3: StockPriceMRP3 = StockPriceMRP3(
        alpha, stock_MP3, init_price, f
    )
    start: Constant = Constant(StateMP3(0, 0))
    return [
        (step.next_state, step.reward * gamma ** (t+1)) \
            for t, step in enumerate(
                itertools.islice(
                    stock_MRP3.simulate_reward(start),
                    num_time+1
                )
            )
    ]


def plot_trace(
    t: List[float],
    y: List[float],
    x_axis_label: str,
    y_axis_label: str,
    title: str,
    file_name: str
):
    """Plot a single-trace simulation of a stock price model.

    :param t: Simulation time points in trace
    :param y: Simulated output from trace (typically stock price or reward)
    :param x_axis_label: Label for x axis of plot
    :param y_axis_label: Label for x axis of plot
    :param title: Title of plot
    :param file_name: Path at which to save plot
    """
    plt.figure()
    plt.plot(t, y)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.savefig(file_name, dpi=600)


def example_reward_func(price: float) -> float:
    """Model of reward as a function of the stock price.

    :param price: stock price
    :returns: value of reward
    """
    return price * 0.98


def main():
    """Simulate stock prices by MRP model for fixed time interval.
    """

    init_price: float = 100     # Initial stock price
    gamma: float = 0.95         # Discount rate
    alpha: float = 0.75         # Reverse-pull strength
    num_time: int = 100         # Number of time steps

    stock_MP3: StockPriceMP3 = StockPriceMP3(alpha)
    state_rewards: List[Tuple[StateMP3, float]] = reward_MRP_simulation(
        alpha, gamma, num_time, stock_MP3, init_price, example_reward_func
    )

    times, prices, rewards = [], [], []
    for t, state_reward in enumerate(state_rewards):
        times.append(t)
        prices.append(get_stock_price(state_reward[0], init_price))
        rewards.append(state_reward[1])

    plot_trace(
        times,
        rewards,
        "Time Steps",
        "Reward",
        "Reward Trace for MRP Stock Example",
        "extended_stock_example_reward_trace.png"
    )
    plot_trace(
        times,
        prices,
        "Time Steps",
        "Prices",
        "Price Trace for MRP Stock Example",
        "extended_stock_example_price_trace.png"
    )


if __name__ == "__main__":
    main()
