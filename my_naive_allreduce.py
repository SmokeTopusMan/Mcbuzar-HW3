import numpy as np


def allreduce(send, recv, comm, op):
    """ Naive all reduce implementation

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    op : associative commutative binary operator
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    recv[:] = send[:]

    reqs = []
    for i in range(size):
        if i != rank:
            req = comm.Isend(send, i)
            reqs.append(req)

    buffer = np.empty_like(recv)
    for _ in range(size-1):
        comm.Recv(buffer)
        recv[:] = op(recv, buffer)

    for req in reqs:
        req.Wait()
