
__all__ = [
    "crosscheck",
    "benchmark",
    "strong_scaling",
]


def crosscheck(field, tol=None, correct=None, tunable_options=None, samples=None):
    """
    Checks that the result of the calculation matches changing the tunable parameters.

    Parameters
    ----------
    tol: float
        Tolerance to use in the crosscheck.
    correct: Field
        The correct result. If none then the results are checked one with the other.
    tunable_options: list
        List of tunable options to crosscheck.
        If none all of them are used on the crosscheck.
    samples: int
        Number of samples to use in the crosscheck.
        The space of the tunable options is sample randomly.
        If None all the combinations are checked.
    """
    pass


def benchmark(tunable, tunable_options=None, **tune_kwargs):
    """
    Benchmarks the calculation with respect to a set of parameters.
    I.e. a set of parameters is scanned, the remaining is tuned.
    The best time per set of parameters is returned.

    Parameters
    ----------
    tunable_options: list
        List of tunable options to benchmark.
        If none, no option is scanned and the best time is returned.
    tune_kwargs: dict
        Parameters to use in the tuning.
    """
    
    pass


def strong_scaling(tunable, **kwargs):
    """
    Runs the strong scaling of the calculation.
    I.e. same as benchmarking the number of workers.
    
    Parameters
    ----------
    kwargs: dict
        Paramters passed to benchmark.
    """
    
    pass

