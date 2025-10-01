import torch
from torch import Tensor
from torch_geometric.nn.pool import knn


def knn_search(db, query, N, engine="torch", return_edge_index=False):

    assert engine in ["torch"], f"Unknown engine {engine}"

    match engine:
        case "torch":
            idx = knn(db, query, N) # return index with shape 2 * (n_query * N), [[query_idx], [dst_idx]]
            if return_edge_index:
                return idx
            return idx[1].reshape(query.shape[0], N)
        
        case _:
            idx = knn(db, query, N) # return index with shape 2 * (n_query * N), [[query_idx], [dst_idx]]
            if return_edge_index: 
                return idx
            return idx[1].reshape(query.shape[0], N)


def gather_neighbor(pc: Tensor, neighbor_idx: Tensor):
    # pc : batch * channels * npoints
    # neighbor_idx : batch * npoints * num_neighbor
    batch_size, d, _ = pc.shape[:3]

    return torch.gather(input=pc, dim=2, index=neighbor_idx.view(batch_size, -1).unsqueeze(1).repeat(1, d, 1)) # batch * (npoints * num_neighbor) * channels