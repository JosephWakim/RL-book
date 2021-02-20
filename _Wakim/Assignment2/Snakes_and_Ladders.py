"""
'Snakes and Ladders' as a Markov process.

Create a data structure to represent state transition probabilities in 'Snakes
and Ladders,' and generate sample traces using the `traces` method of the
`FiniteMarkovProcess` object.

Joseph Wakim
CME 241
January 16, 2021
"""

from typing import (Mapping, Dict, Optional, Tuple)
from dataclasses import dataclass
import itertools

import numpy as np
import matplotlib.pyplot as plt
from rl.distribution import (Constant, Categorical, SampledDistribution)
import rl.markov_process as mp


@dataclass(frozen=True)
class GameState:
    """Position on the board.

    :param position: Position on the game board
    """
    position: int

    def change_position(self, new_positon: int) -> mp.S:
        """Change the position of the current GameState.

        This change occurs due to a snake or ladder.

        :param new_position: Updated position of the GameState object
        :return: The GameState object with the updated position
        """
        return GameState(new_positon)

    def check_SL(
        self,
        SL_map: Mapping[int, int]
    ) -> mp.S:
        """Adjust position on the board to account for snakes and ladders.

        :param SL_map: Mapping of starting to ending positions associated with
            snakes and ladders
        :returns: New GameState object with position adjusted for snakes and
            ladders.
        """
        if self.position in SL_map.keys():
            return self.change_position(SL_map[self.position])
        return self

    def check_board_end(self, board_size: int, roll: int) -> Optional[mp.S]:
        """Reject moves which fall past the end of the board.

        If a move falls beyond the end of the board, correct the state position
        to be equal to the proposed move position less the previous roll.

        :param board_size: Number of grid spaces on the board, excluding the
            starting position at tile zero
        :param roll: Previous value rolled on the die
        :returns: GameState object if not on the final tile, else None.
        """
        if self.position > board_size:    # Reject moves beyond board
            return self.change_position(self.position - roll)
        return self


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
        d: Dict[GameState, Optional[Categorical[GameState]]] = {
            GameState(self.num_tiles) : None
        }
        for position in range(self.num_tiles):
            game_state = GameState(position)
            game_state_prob_map: Mapping[GameState, float] = {
                GameState(i).check_SL(
                    self.snakes_and_ladders
                ).check_board_end(
                    self.num_tiles, i-position
                ) : self.uniform_distr for i in range(
                    position+1,
                    position+self.dice_sides+1
                )
            }
            d[game_state] = Categorical(game_state_prob_map)
        return d


class SnakesAndLaddersMRPFinite(mp.FiniteMarkovRewardProcess[GameState]):
    """Representation 'Snakes and Ladders' as a Markov reward process.

    Represent the 'Snakes and Ladders' board game as a finite Markov
    rewared process to determine the expected number of moves to
    reach the end of the board.
    """
    
    def __init__(
        self,
        snakes_and_ladders: Mapping[int, int],
        num_tiles: int,
        dice_sides: int
    ):
        """
        Initialize `SnakesAndLaddersMRPFinite` object.

        :param snakes_and_ladders: Mapping of starting to ending positions
            associated with snakes and ladders
        :param num_tiles: Number of tiles on the board
        :param dice_sides: Number of sides on a fair die
        """
        self.num_tiles = num_tiles
        self.snakes_and_ladders = snakes_and_ladders
        self.dice_sides = dice_sides
        self.uniform_distr = 1 / dice_sides
        super().__init__(self.get_transition_reward_map())

    def get_transition_reward_map(self) -> mp.RewardTransition[GameState]:
        """Generate transition-reward map associated with 'Snakes and Ladders'.
        
        :return: Transition-rewared map for MRP representation of the 'Snakes
            and Ladders' boardgame.
        """
        d: Dict[GameState, Optional[Categorical[Tuple[GameState, int]]]] = {
            GameState(self.num_tiles) : None
        }
        for position in range(self.num_tiles):
            reward = 1  # To enable counting of moves
            game_state = GameState(position)
            game_state_prob_map: Dict[Tuple[GameState, float], int] = {
                (GameState(i).check_SL(
                    self.snakes_and_ladders
                ).check_board_end(
                    self.num_tiles, i-position
                ), reward) : self.uniform_distr for i in range(
                    position+1,
                    position+self.dice_sides+1
                )
            }
            d[game_state] = Categorical(game_state_prob_map)
        return d


def snakes_and_ladders_traces(
    SL_positions: Mapping[int, int],
    n_tiles: int,
    n_dice_sides: int,
    start: int,
    n_traces: int
) -> np.ndarray:
    """Generate traces of "Snakes and Ladders" gameplay outcomes.

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
    :returns: Move counts from Markov process traces
    """
    snakes_and_ladders_sim: SnakesAndLaddersMPFinite =\
        SnakesAndLaddersMPFinite(
            snakes_and_ladders=SL_positions,
            num_tiles=n_tiles,
            dice_sides=n_dice_sides
        )
    start_state: Constant = Constant(start)
    return np.fromiter(
        (len(list(trace)) for trace in itertools.islice(
            snakes_and_ladders_sim.traces(start_state),
            n_traces+1
        )), int
    )


def get_expected_counts_by_MRP(
    SL_positions: Mapping[int, int],
    n_tiles: int,
    n_dice_sides: int
) -> float:
    """Generate an expected number of turns required to finish the game.

    By representing the 'Snakes and Ladders' game as a Markov Reward Process
    where each transition carries a uniform reward of 1, we can calculate
    the expected number of turns required to finish the game by the expected
    accumulated reward for the process, starting from position 0. We assume a
    discount factor of 1 because we are only concerned with how many moves are
    made, not when moves are made.

    :param SL_positions: Mapping of starting to ending positions
        associated with snakes and ladders on the board
    :param n_tiles: Number of tiles on the board
    :param n_dice_sides: Number of sides on a fair die
    :param start: starting position on the board
    :returns: Expected number of moves required to complete the game
    """
    snakes_and_ladders_MRP_finite: SnakesAndLaddersMRPFinite =\
        SnakesAndLaddersMRPFinite(
            snakes_and_ladders=SL_positions,
            num_tiles=n_tiles,
            dice_sides=n_dice_sides
        )
    gamma = 1
    V: np.ndarray = snakes_and_ladders_MRP_finite.get_value_function_vec(gamma)
    return V[0]


def generate_histogram(
    data: np.ndarray,
    file_name: str,
    x_axis_label: str,
    y_axis_label: Optional[str] = 'Frequency',
    density: Optional[bool] = False,
    bins: Optional[int] = 20
):
    """Plot a histogram of the trace_lengths.

    :param data: Array of trace lengths being plotted
    :param file_name: Path to file at which histogram will save
    :param x_axis_label: independent axis label
    :param y_axis_label: dependent axis label (default "frequency")
    :param density: If True, plots histogram as densities, else plots frequency
        (default = False)
    :param nbins: Number of bins or bin edges to include in the histogram
        (default 20)
    """
    plt.figure()
    plt.hist(data, density=density, bins=bins, edgecolor='black', linewidth=1)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.savefig(file_name, dpi=600)


def main():
    """Run the simulations of 'Snakes and Ladders' gameplay outcomes.
    """
    # Generate 100 Traces
    n_sims: int = 1000
    # Represent gameplay with a board of 100 tiles and a 6-sided die
    n_board_tiles: int = 100
    n_sides_on_die: int = 6
    starting_position: GameState = GameState(0)
    # Model simulation of Snakes and Ladders board referenced by assignment
    snakes_and_ladders_map: Mapping[int, int] = {
        1:38, 4:14, 9:31, 16:6, 21:42, 28:84, 36:44, 47:26, 49:11, 51:67,
        56:53, 62:19, 64:60, 71:91, 80:100, 87:24, 93:73, 95:75, 98:78
    }
    # Generate traces
    traces_lengths = snakes_and_ladders_traces(
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
    # Get the expected number of moves required to finish the game
    expected_moves = get_expected_counts_by_MRP(
        snakes_and_ladders_map,
        n_board_tiles,
        n_sides_on_die
    )
    # Compare the average number of moves from each method
    print("Expected Number of Moves: " + str(expected_moves))
    print(
        "Average Number of Moves in traces: " + str(np.average(traces_lengths))
        )


if __name__ == "__main__":
    main()
