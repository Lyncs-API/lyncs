def test_client():
    from lyncs.mpi import Client
    client = Client(num_workers=1)
    assert client._server is not None
    assert len(client.workers) == 1
    
    client.close_server()
    assert client._server is None
    
    
