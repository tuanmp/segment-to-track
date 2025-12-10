import torch
from torch import Tensor
from torch_geometric.nn.pool import knn


def knn_search(db, query, N, engine="torch", return_edge_index=False):

    assert engine in ["torch", "fastgraph"], f"Unknown engine {engine}"

    def torch_knn(db, query, N, return_edge_index):
        idx = knn(db, query, N)
        if return_edge_index:
            return idx
        return idx[1].reshape(query.shape[0], N)

    match engine:
        case "torch":
            return torch_knn(db, query, N, return_edge_index)

        case _:
            return torch_knn(db, query, N, return_edge_index)


def gather_neighbor(pc: Tensor, neighbor_idx: Tensor):
    # pc : batch * channels * npoints
    # neighbor_idx : batch * npoints * num_neighbor
    batch_size, d, _ = pc.shape[:3]

    return torch.gather(input=pc, dim=2, index=neighbor_idx.view(batch_size, -1).unsqueeze(1).repeat(1, d, 1)) # batch * (npoints * num_neighbor) * channels
