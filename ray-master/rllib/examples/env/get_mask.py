import numpy as np
# import networkx as nx
from numba import jit

@jit(nopython=True, cache=True)
def create_logits_mask_by_first_edge_graph(edge_indexes, num_edge, nvec):
    """
       create action mask for selecting the first edge

    """
    # find first edge
    # adj matrix of graphs
    # adj_mat = nx.to_numpy_matrix(graph, nodelist=range(nvec))[None]

    # bs = adj_mats.shape[0]
    # total_mask = []
    max_edge = edge_indexes.shape[0]
    total_mask = np.zeros(shape=(1, max_edge), dtype=np.int8)


    # edges = edge_indexes[:num_edge]
    # edges = np.where(adj_mats[i, :, :] > 0)

    # max_edge = max(max_edge, num_edges)
    # mask = np.zeros(shape=(1, max_edge), dtype=np.int8)
    all_half_edges = edge_indexes[:num_edge // 2]  # only use the directed edge (a->b), not (b->a)  # location[np.where(location[:, 0] < location[:, 1])[0]]
    all_valid_edges = edge_indexes[:num_edge] # only use the directed edge (a->b), not (b->a)  # location[np.where(location[:, 0] < location[:, 1])[0]]

    # restore adj matrix
    rawobs = np.zeros(shape=(nvec, nvec), dtype=np.int8)
    for edge in all_valid_edges:
        rawobs[edge[0], edge[1]] = 1  # edge is an ndarray (2,), we cannot index using rawobs[edge] (is array with shape (2, 15))
#
    for idx_1, edge_1 in enumerate(all_half_edges):
        encoded_edge_1 = idx_1
        # check if they are one-hop connected

        mask = np.zeros(shape=(max_edge,), dtype=np.int8)
        for idx_2, edge_2 in enumerate(all_valid_edges):

            fail_cond = edge_2[0] in edge_1 or edge_2[1] in edge_1 or\
                        int(rawobs[edge_2[0], edge_1[0]]) + int(rawobs[edge_2[0], edge_1[1]]) + \
                        int(rawobs[edge_2[1], edge_1[0]]) + int(rawobs[edge_2[1], edge_1[1]]) > 0

            mask[idx_2] = not fail_cond

        total_mask[0, encoded_edge_1] = mask.any()


    return total_mask

@jit(nopython=True, cache=True)
def create_logits_mask_by_second_edge_graph(edge_indexes, num_edges, nvec, last_actions):
    """
    create action mask for selecting the second edge
    """
    bs = edge_indexes.shape[0]

    max_edge = edge_indexes.shape[1]


    total_mask = np.zeros(shape=(bs, max_edge), dtype=np.int8)

    for i in range(bs):
        num_edge = num_edges[i]
        edges = edge_indexes[i]
        all_valid_edges = edges[:num_edge]

        adj_mat = np.zeros(shape=(nvec, nvec), dtype=np.int8)
        for edge in all_valid_edges:
            adj_mat[edge[0], edge[1]] = 1

        mask = np.zeros(shape=(max_edge,), dtype=np.int8)

        last_action = last_actions[i]
        edge_1 = edges[last_action]

        for idx_2, edge_2 in enumerate(all_valid_edges):
            fail_cond = edge_2[0] in edge_1 or edge_2[1] in edge_1 or \
                        int(adj_mat[edge_2[0], edge_1[0]]) + int(adj_mat[edge_2[0], edge_1[1]]) + \
                        int(adj_mat[edge_2[1], edge_1[0]]) + int(adj_mat[edge_2[1], edge_1[1]]) > 0

            mask[idx_2] = not fail_cond
        total_mask[i, :] = mask

    return total_mask


