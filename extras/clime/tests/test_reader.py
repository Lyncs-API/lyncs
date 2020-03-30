def test_reader():
    from lyncs_clime import Reader
    import os
    print(os.getcwd())
    reader = Reader("tests/conf.1000")
    
    assert len(reader) == len(list(reader))
    assert "precision" in str(reader)
    
