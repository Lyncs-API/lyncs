"Utils for calls using MPI via dask"

def get_n_workers(
        client,
        num_workers = None,
        workers = None,
        resources = None,
    ):
    workers = workers or list(client.scheduler_info()["workers"].keys())
    num_workers = num_workers or len(workers)

    assert num_workers <= len(workers)

    # TODO select accordingly requested resources

    # TODO implement some rules to choose wisely n workers
    # e.g. workers not busy

    return workers[:num_workers]
        
def create_comm(
        client,
        actor = True,
        **kwargs
    ):
    workers = get_n_workers(client, **kwargs)
    ranks = [client.scheduler_info()["workers"][w]["id"] for w in workers]
    group = client.scatter([ranks]*len(ranks), workers=workers, broadcast=True)

    def _create_comm(group):
        from mpi4py import MPI
        return MPI.COMM_WORLD.Create_group(MPI.COMM_WORLD.group.Incl(group))
    
    return client.map(_create_comm, group, actor=actor)

