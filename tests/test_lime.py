def test_read_config():
    import lyncs
    conf_path = lyncs.__path__[0]+"/../tests/conf.1000"
    conf = lyncs.load(conf_path)
    assert conf.size == 4**4*4*3*3
