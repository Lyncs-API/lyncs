def _test_read_config():
    import lyncs
    import numpy

    conf_path = lyncs.__path__[0] + "/../tests/conf.1000"
    conf = lyncs.load(conf_path)
    assert conf.size == 4 ** 4 * 4 * 3 * 3

    # without distribution
    conf.chunks = conf.dims
    reference = conf.result()
    assert reference.shape == conf.field_shape

    from itertools import product

    chunks = product(*[[1, 2, 4]] * 4)
    for chunk in chunks:
        chunk = dict(zip(["t", "z", "y", "x"], chunk))
        conf = lyncs.load(conf_path)
        conf.chunks = chunk
        read = conf.result()
        assert numpy.all(read == reference)
