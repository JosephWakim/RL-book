"""
Implement Univariate B-Spline.

Joseph Wakim
CME 241 - Assignment 5 - Problem 1
Winter 2021
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Iterable, Tuple, Sequence, TypeVar
import rl.function_approx as fa


X = TypeVar('X')
Y = float


def ground_truth_func(x: X) -> Y:
    """True function being approximated with BSpline.

    :param x: input of true function
    :returns: output of true function
    """
    return x * np.cos(x / (2 * np.pi))


def univar_func_approx(x: X) -> Y:
    """
    Univariate function applied during function approximation.

    :param x: Input variable
    :returns: Transformation of input
    """
    return 1 + x + x**2


def univar_BSpline(
    training: Iterable[Tuple[X, Y]],
    inputs: Sequence[X],
    feature_func: Callable[[X], float],
    degree: int,
    error_tolerance: Optional[float]
) -> Sequence[Y]:
    """Approximate a univariate function using BSpline.

    BSpline is implemented as a class in `rl.function_approx.py` and is
    imported for use to approximate a univariate function.

    :param training: Inputs and corresponding outputs at which to fit
        approximator
    :param inputs: Values at which to evaluate function approximation
    :feature_func: Function approximation of inputs
    :param degree: Degree of the BSpline function approximator
    :param error_tolerance: Tolerance when solving BSpline approximator
    :returns: Predicted outputs from BSpline approximation at inputs
    """
    # Create instance of BSpline function approximator
    b_spline: fa.BSplineApprox = fa.BSplineApprox(feature_func, degree)
    # Fit the function approximator to provided training data
    b_spline = b_spline.solve(training, error_tolerance)
    # Evalate the function approximator on the inputs
    return b_spline.evaluate(inputs)


def main():
    """Generate synthetic function and approximate function using BSpline.
    """

    # Generate synthetic training data
    train_x: np.ndarray = np.arange(0, 101, 10)
    train_y: np.ndarray = np.array(
        [ground_truth_func(x) for x in train_x]
    )
    train_xy: Iterable[Tuple[X, Y]] = [
        (train_x[i], train_y[i]) for i in range(len(train_x))
    ]

    # Generate inputs at which to approximate true function
    eval_x = np.arange(0, 101, 0.01)
    true_y = np.array(
        [ground_truth_func(x) for x in eval_x]
    )

    plt.figure()
    plt.scatter(train_x, train_y, label="Training Points")
    plt.plot(eval_x, true_y, label="True Function")

    for degree in range(1, 6):
        # Generate predictions of true function using function approximation
        eval_y = univar_BSpline(
            training=train_xy,
            inputs=eval_x,
            feature_func=univar_func_approx,
            degree=degree,
            error_tolerance=0.001
        )
        plt.plot(eval_x, eval_y, label="BSpline, deg. "+str(degree))

    plt.legend()
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Univariate B-Spline Function Approx.")
    plt.savefig("univariate_bspline.png", dpi=600)


if __name__ == "__main__":
    main()
