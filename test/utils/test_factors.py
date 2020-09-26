import numpy
from lyncs.utils import prime_factors, factors


def test_factors():
    nums = [1, 20, 30, 123, 211]
    for num in nums:
        assert numpy.prod(list(prime_factors(num))) == num
        assert numpy.prod(list(factors(num))) % num == 0
