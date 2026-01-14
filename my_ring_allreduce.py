import numpy as np


def ringallreduce(send, recv, comm, op):
    """ ring all reduce implementation
    You need to use the algorithm shown in the lecture.

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

    dest = (rank + 1) % size
    source = (rank - 1) % size

    chunk_len = len(send) // size
    for i in range(size - 1):
        send_chunk_idx = (rank - i) % size
        recv_chunk_idx = (rank - i - 1) % size
        send_chunk_start = send_chunk_idx * chunk_len
        send_chunk_end = send_chunk_start + chunk_len if send_chunk_idx != size -1 else len(recv)
        recv_chunk_start = recv_chunk_idx * chunk_len
        recv_chunk_end = recv_chunk_start + chunk_len if recv_chunk_idx != size -1 else len(recv)

        req = comm.Isend(recv[send_chunk_start: send_chunk_end], dest)
        buffer = np.empty_like(recv[recv_chunk_start:recv_chunk_end])
        comm.Recv(buffer, source)
        recv[recv_chunk_start:recv_chunk_end] = op(recv[recv_chunk_start:recv_chunk_end], buffer)
        req.Wait()

    for i in range(size -1):
        send_chunk_idx = (rank - i + 1) % size
        recv_chunk_idx = (rank - i) % size
        send_chunk_start = send_chunk_idx * chunk_len
        send_chunk_end = send_chunk_start + chunk_len if send_chunk_idx != size -1 else len(recv)
        recv_chunk_start = recv_chunk_idx * chunk_len
        recv_chunk_end = recv_chunk_start + chunk_len if recv_chunk_idx != size -1 else len(recv)

        req = comm.Isend(recv[send_chunk_start: send_chunk_end], dest)
        buffer = np.empty_like(recv[recv_chunk_start:recv_chunk_end])
        comm.Recv(buffer, source)
        recv[recv_chunk_start:recv_chunk_end] = buffer
        req.Wait()

