"""
Two store inventory problem.

This code is adapted from `rl/chapter3/simple_inventory_mdp_cap.py`.

Joseph Wakim
CME 241 - Assignment 4 - Problem 4
February 13, 2021
"""

import sys
from dataclasses import dataclass
from typing import Tuple, Dict
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_decision_process import FinitePolicy, StateActionMapping
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical, Constant
from scipy.stats import poisson


@dataclass(frozen=True)
class InventoryState:
    on_hand_store_1: int
    on_order_store_1: int
    on_hand_store_2: int
    on_order_store_2: int

    def inventory_position(self) -> Tuple[int, int]:
        return (self.on_hand_store_1 + self.on_order_store_1,
                self.on_hand_store_2 + self.on_order_store_2)


InvOrderMapping = StateActionMapping[InventoryState, int]


class SimpleInventoryMDPCap(FiniteMarkovDecisionProcess[InventoryState, int]):

    def __init__(
        self,
        capacity_1: int,
        capacity_2: int,
        poisson_lambda_1: float,
        poisson_lambda_2: float,
        holding_cost_1: float,
        holding_cost_2: float,
        stockout_cost_1: float,
        stockout_cost_2: float,
        order_cost: float,
        transfer_cost: float
    ):
        self.capacity_1: int = capacity_1
        self.capacity_2: int = capacity_2
        self.poisson_lambda_1: float = poisson_lambda_1
        self.poisson_lambda_2: float = poisson_lambda_2
        self.holding_cost_1: float = holding_cost_1
        self.holding_cost_2: float = holding_cost_2
        self.stockout_cost_1: float = stockout_cost_1
        self.stockout_cost_2: float = stockout_cost_2
        self.order_cost = order_cost
        self.transfer_cost = transfer_cost

        self.poisson_distr_1 = poisson(poisson_lambda_1)
        self.poisson_distr_2 = poisson(poisson_lambda_2)
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[int, Categorical[Tuple[InventoryState,
                                                            float]]]] = {}
        for alpha_1 in range(self.capacity_1 + 1):
            for beta_1 in range(self.capacity_1 + 1 - alpha_1):
                for alpha_2 in range(self.capacity_2 + 1):
                    for beta_2 in range(self.capacity_2 + 1 - alpha_2):
                        state: InventoryState = InventoryState(
                            alpha_1, beta_1, alpha_2, beta_2
                        )
                        ip_1: int = state.inventory_position()[0]
                        ip_2: int = state.inventory_position()[1]
                        d1: Dict[int, Categorical[Tuple[InventoryState, float]]] = {}

                        for transfer in range(
                            -min(
                                self.capacity_2 - alpha_2 - beta_2, alpha_1 + beta_1
                            ), min(
                                self.capacity_1 - alpha_1 - beta_1, alpha_2 + beta_2
                            )
                        ):
                            for order_1 in range(self.capacity_1 - ip_1 + transfer + 1):
                                for order_2 in range(self.capacity_2 - ip_2 - transfer + 1):
                                    sr_probs_dict: Dict[Tuple[InventoryState, float], float] = {}
                                    morning_alpha_1 = alpha_1 - transfer
                                    morning_alpha_2 = alpha_2 + transfer
                                    base_reward_1: float = -self.holding_cost_1 * morning_alpha_1
                                    base_reward_2: float = -self.holding_cost_2 * morning_alpha_2
                                    if order_1 != 0:
                                        base_reward_1 -= self.order_cost
                                    if order_2 != 0:
                                        base_reward_2 -= self.order_cost
                                    if transfer != 0:
                                        base_reward_1 -= self.transfer_cost
                                    
                                    for demand_1 in range(ip_1):
                                        for demand_2 in range(ip_2):
                                            prob = self.poisson_distr_1.pmf(demand_1) * self.poisson_distr_2.pmf(demand_2)
                                            next_state = InventoryState(morning_alpha_1 - demand_1, order_1, morning_alpha_2 - demand_2, order_2)
                                            reward = base_reward_1 + base_reward_2
                                            sr_probs_dict[(next_state, reward)] = prob

                                    for demand_1 in range(ip_1):
                                        prob = self.poisson_distr_1.pmf(demand_1) * (1 - self.poisson_distr_2.cdf(ip_2-1))
                                        next_state = InventoryState(morning_alpha_1 - demand_1, order_1, 0, order_2)
                                        reward_2 = base_reward_2 - self.stockout_cost_2 * (prob * (self.poisson_lambda_2 - ip_2) + ip_2 * self.poisson_distr_2.pmf(ip_2))
                                        reward = base_reward_1 + reward_2
                                        sr_probs_dict[(next_state, reward)] = prob

                                    for demand_2 in range(ip_2):
                                        prob = self.poisson_distr_2.pmf(demand_2) * (1 - self.poisson_distr_1.cdf(ip_1-1))
                                        next_state = InventoryState(0, order_1, morning_alpha_2 - demand_2, order_2)
                                        reward_1 = base_reward_1 - self.stockout_cost_1 * (prob * (self.poisson_lambda_1 - ip_1) + ip_1 * self.poisson_distr_1.pmf(ip_1))
                                        reward = base_reward_2 + reward_1
                                        sr_probs_dict[(next_state, reward)] = prob

                                    d1[(order_1, order_2, transfer)] = Categorical(sr_probs_dict)

                        if len(d1.keys()) > 0:
                            d[state] = d1

        return d


if __name__ == '__main__':
    from pprint import pprint

    user_capacity_1 = 3
    user_capacity_2 = 5
    user_poisson_lambda_1 = 1
    user_poisson_lambda_2 = 2
    user_holding_cost_1 = 1.0
    user_holding_cost_2 = 1.2
    user_stockout_cost_1 = 10.0
    user_stockout_cost_2 = 8.0
    order_cost = 2
    transfer_cost = 1.1

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
        SimpleInventoryMDPCap(
            capacity_1=user_capacity_1,
            capacity_2=user_capacity_2,
            poisson_lambda_1=user_poisson_lambda_1,
            poisson_lambda_2=user_poisson_lambda_2,
            holding_cost_1=user_holding_cost_1,
            holding_cost_2=user_holding_cost_2,
            stockout_cost_1=user_stockout_cost_1,
            stockout_cost_2=user_stockout_cost_2,
            order_cost=order_cost,
            transfer_cost=transfer_cost
        )

    print("MDP Transition Map")
    print("------------------")
    print(si_mdp)
