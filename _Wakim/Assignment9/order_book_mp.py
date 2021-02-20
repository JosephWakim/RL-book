"""
Simulate order book dynamics with a Markov process.

In our representation of order book dynamics as a Markov process, OrderBook
objects constitute states. Transition probabilities are defined by random
arrivals of market and limit buy and sell orders.

There is a tendancy for markets to fill gaps in the order book. Therefore,
assume that the number of buy and sell orders is randomly selected based
on the bid-ask spread. The number of buy and sell orders is also affected by
the current size of the order book and the ratio of bids to asks. 

With these factors in mind, assume that the number of buy orders is normally
distributed, centered around (num bids - num asks + spread) and that the number
of sell  orders is normally distributed centered around (num bids - num asks +
spread), both with a floor of zero. Of these orders, assume uniform probability
of market vs limit orders. For buy limit orders, assume bids at discrete
distributions centered around the (bid price +  1/4 spread). For sell limit
orders, assume asks at discrete distributions centered around (ask price - 
1/4 spread).

Joseph Wakim
CME 241, Winter 2021, Assignment 9, Q1
February 12, 2021
"""

import numpy as np
import rl.chapter9.order_book as ob
import rl.markov_process as mp
