"""
'Snakes and Ladders' as a Markov process.

Create a data structure to represent state transition probabilities in 'Snakes
and Ladders,' and generate sample traces using the `traces` method of the
`FiniteMarkovProcess` object.

Joseph Wakim
CME 241
January 16, 2021

TODO:
    - Adjust so you must land *EXACTLY* on end piece or just reach end.
    - Change states from integers to GameState instances
    - Implement using `traces` instead of `simulate`
"""

from typing import (Mapping, Dict, Optional)

import numpy as np
import matplotlib.pyplot as plt
from rl.distribution import (Constant, Categorical)
import rl.markov_process as mp


class GameState:
    """Position on the board."""

    def __init__(
        self,
        proposed_position: int,
        board_size: int,
        snakes_and_ladders: Mapping[int, int]
    ):
        """
        Initialize GameState object.

        :param proposed_position: Proposed position on playing board, prior to
            snake and ladder adjustments
        :param board_size: Number of grid spaces on the board, excluding starting
            position at tile zero
        :param snakes_and_ladders: Mapping of starting to ending positions
            associated with snakes and ladders
        """
        position = self.adjust_for_board_end(
            proposed_position,
            board_size
        )
        self.position = self.adjust_for_snakes_and_ladders(
            position,
            snakes_and_ladders
        )

    def adjust_for_snakes_and_ladders(
        self,
        position: int,
        snakes_and_ladders: Mapping[int, int]
    ) -> int:
        """
        Adjust `GameState.positon` depending on location of snakes and ladders.

        :param position: Proposed position on playing board, prior to
            snake and ladder adjustments
        :param snakes_and_ladders: Mapping of starting to ending positions
            associated with snakes and ladders
        :return: Position on the board after adjusting for snakes and ladders
        """
        if position in snakes_and_ladders.keys():
            position = snakes_and_ladders[position]
        return position

    def adjust_for_board_end(self, position: int, board_size: int):
        """
        Return `None` if position lies beyond final tile.

        If returning a value of `None`, the `GameState` object flags for the
        termination of the finite Markov process simulation.

        :param position: Proposed position on playing board, prior to snake and
            ladder adjustments
        :param board_size: Number of grid spaces on the board, excluding starting
            position at tile zero
        :returns: position after adjusting for the end of the board
        """
        if position > board_size:
            return None
        return position


class SnakesAndLaddersMPFinite(mp.FiniteMarkovProcess[GameState]):
    """
    Finite Markov process simulating 'Snakes and Ladders'.

    'Snakes and Ladders' is a classic game where moves are made on a tiled
    playing board based on values rolled with a die. Scattered along the board
    are ladders, which advance the player to a further tile, and snakes, which
    move the player to the previous tile. Since each turn is dependent only on
    the position of the player on the board, agnostic to the history of
    positions, the game can be represented as a Markov process.
    """

    def __init__(
        self,
        snakes_and_ladders: Mapping[int, int],
        num_tiles: int,
        dice_sides: int
    ):
        """
        Initialize `SnakesAndLaddersMPFinite` object.

        Specify the gameboard, including the number of tiles and the starting/
        ending positions of each snake and ladder. Specify the number of sides
        on the die governing gameplay, and calculate the uniform probability of
        rolling each value of the die. Obtain the transition probability map
        representing the game board.

        :param snakes_and_ladders: Mapping of starting to ending positions
            associated with snakes and ladders
        :param num_tiles: Number of tiles on the board
        :param dice_sides: Number of sides on a fair die
        """
        self.num_tiles = num_tiles
        self.snakes_and_ladders = snakes_and_ladders
        self.dice_sides = dice_sides
        self.uniform_distr = 1 / dice_sides
        super().__init__(self.get_transition_map())

    def get_standard_tiles(self):
        """Get tile positions not resulting in a snake or ladder move."""
        return [
            i for i in range(
                0, self.num_tiles
            ) if i not in self.snakes_and_ladders.keys()
        ]

    def get_transition_map(self) -> mp.Transition[mp.S]:
        """
        Get the transition map associated with the playing board.

        The transition map associated with each move is dictated by the uniform
        probability of rolling each value on the die, the landing position
        associated with each possible roll, the presence of a snake or ladder
        at the landing position, and proximity to the end of the board.

        A `GameState` object is instantiated for each tile position on the
        board. That object is mapped to N subsequent positions with uniform
        probability, where N is the number of sides on the playing die.
        Corrections are made to the position attribute of each `GameState` 
        object if the position lies beyond the end of the board or corresponds
        to snakes or ladders.

        :returns: Transition map representing "Snakes and Ladders" board
        """
        d: Dict[GameState, Categorical[GameState]] = {}
        d[None] = None
        for position in range(self.num_tiles+1):
            game_state_prob_map: Mapping[Optional[int], float] = {
                GameState(
                    i,
                    self.num_tiles,
                    self.snakes_and_ladders
                ).position : self.uniform_distr for i in range(
                    position+1,
                    position+self.dice_sides+1
                )
            }
            d[position] = Categorical(game_state_prob_map)
        return d


def snakes_and_ladders_trace(
    SL_positions: Mapping[int, int],
    n_tiles: int,
    n_dice_sides: int,
    start: int,
    n_traces: int
) -> np.ndarray:
    """
    Generate traces of "Snakes and Ladders" gameplay outcomes.

    Start by instantiating a `SnakesAndLaddersMPFinite` object with the
    specified gameboard. Define a constant starting position, where players are
    placed at the start of the game (this is typically zero). Then create
    traces of the gameplay and store the number of moves in each trace in a
    vector to be returned by this function.

    :param SL_positions: Mapping of starting to ending positions
        associated with snakes and ladders on the board
    :param n_tiles: Number of tiles on the board
    :param n_dice_sides: Number of sides on a fair die
    :param start: starting position on the board
    :param n_traces: Number of traces to generate
    :returns: array of Markov process traces
    """
    snakes_and_ladders_sim = SnakesAndLaddersMPFinite(
        snakes_and_ladders=SL_positions,
        num_tiles=n_tiles,
        dice_sides=n_dice_sides
    )
    start_state = Constant(start)
    return np.array([
        len(
            list(snakes_and_ladders_sim.simulate(start_state))
        ) for _ in range(n_traces)
    ])


def generate_histogram(
    data: np.ndarray,
    file_name: str,
    x_axis_label: str,
    y_axis_label: Optional[str] = 'Frequency',
    density: Optional[bool] = False,
    nbins: Optional[int] = 20
):
    """
    Plot a histogram of the trace_lengths.

    :param data: Array of trace lengths being plotted
    :param file_name: Path to file at which histogram will save
    :param x_axis_label: independent axis label
    :param y_axis_label: dependent axis label (default "frequency")
    :param density: If True, plots histogram as densities, else plots frequency
        (default = False)
    :param nbins: Number of bins to include in the histogram (default 20)
    """
    plt.figure()
    plt.hist(data, density=density, bins=nbins, edgecolor='black', linewidth=1)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.savefig(file_name, dpi=600)
    print("Histogram Saved!")


def main():
    """Run the simulation."""
    # Generate 100 Traces
    n_sims = 1000
    # Represent gameplay with a board of 100 tiles and a 6-sided die
    n_board_tiles = 100
    n_sides_on_die = 6
    starting_position = 0
    # Model simulation of Snakes and Ladders board referenced by assignment
    snakes_and_ladders_map = {
        1:38, 4:14, 9:31, 16:6, 21:42, 28:84, 36:44, 47:26, 49:11, 51:67,
        56:53, 62:19, 64:60, 71:91, 80:100, 87:24, 93:73, 95:75, 98:78
    }
    # Generate traces
    traces_lengths = snakes_and_ladders_trace(
        snakes_and_ladders_map,
        n_board_tiles,
        n_sides_on_die,
        starting_position,
        n_sims
    )
    # Generate histogram
    generate_histogram(
        data=traces_lengths,
        file_name="Snakes_and_Ladders_Trace_Lengths.png",
        x_axis_label="Num. Moves"
    )


if __name__ == "__main__":
    main()
