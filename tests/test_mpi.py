def test_client():
    from lyncs.mpi import Client
    client = Client(num_workers=1)
    assert client._server is not None
    assert len(client.workers) == 1

    comms = client.create_comm(actor = True)
    comms = [comm.result() for comm in comms]
    assert [comm.rank for comm in comms] == [0]
    
    client.close_server()
    assert client._server is None
    
    
