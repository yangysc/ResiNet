import gym
from gym.spaces import Box, Dict
import random
import networkx as nx
import numpy as np
from copy import deepcopy
from ray.rllib.examples.env.graphenv import GraphEdgeArgumentEnv, edge_to_index
from ray.rllib.examples.env.get_mask import create_logits_mask_by_first_edge_graph



class ParametricActionsEdgeGraph(gym.Env):
    """Edge rewiring environment
    """

    def __init__(self, config):

        try:
            verbose = config.worker_index == 1 and config.vector_index == 0
        except:
            verbose = False
        self.wrapped = GraphEdgeArgumentEnv(max_action=config["max_action"],
                                            filtration_order=config["filtration_order"],
                                            use_stop=config["with_stop_action"],
                                            use_SimulatedAnnealing=config["with_SimulatedAnnealing"],
                                            is_train=config["is_train"],
                                            dataset_type=config["dataset_type"],
                                            max_num_node=config["max_num_node"],
                                            max_num_edge=config["max_num_edge"],
                                            alpha=config["alpha"],
                                            robust_measure=config["robust_measure"],
                                            single_obj=config["single_obj"],
                                            second_obj_func=config["second_obj_func"],
                                            reward_scale=config["reward_scale"],
                                            sequentialMode=config["sequentialMode"],
                                            add_penality=config["add_penality"],
                                            attack_strategy=config["attack_strategy"],
                                            break_tie=config["break_tie"],
                                            verbose=verbose  # worker_index is not stored as str...
                                            )

        self.ndim = 8 + 6 + 1  # node feature dim, including, 1) filtrates weight, the first one to be removed, the larger
        self.sp_length_ndim = 5
        # 2) shortest path length, including degree (the 2-dim),  3) 0-1 label indicating this node is removed by
        # attack or just lost all connection since its neighbors are attacked.
        self.max_sp = 1
        self.use_filtration = self.wrapped.use_filtration
        self.filtration_order = config["max_num_node"] //2 if config["dataset_type"] == "ba_large" or\
        config["dataset_type"] == "ba_medium" or config["dataset_type"] ==  'ba_small' or config["dataset_type"] == "ba_small_30"  else  self.wrapped.filtration_order# 0== original graph; 1 == original graph + only one subgraph
        # self.wrapped.reset()
        self.action_space = self.wrapped.action_space
        self.max_node = self.wrapped.max_node if config["dataset_type"].startswith('example_') else config["max_num_node"]
        self.max_edge = self.wrapped.max_edge if config["dataset_type"].startswith('example_') else config["max_num_edge"]


        self.ob_filtration_size = self.filtration_order + 1  # how many sub filtrated graphs are stored as obs


        # position embedding of the node removal order
        self.node_attack_order_embedding_dim = 8
      

        if self.use_filtration:

            self.observation_space = Dict({
                'node_feats': Box(-np.inf, np.inf, shape=(self.ob_filtration_size, self.wrapped.max_node, self.ndim)),
                'num_edges': Box(0, self.max_node ** 2, shape=(self.ob_filtration_size, 1), dtype=int),
                'num_nodes': Box(0, self.max_node, shape=(self.ob_filtration_size, 1), dtype=int),
                'max_node_id': Box(0, self.max_node, shape=(self.ob_filtration_size, 1), dtype=int),
                'node_id_index': Box(0, self.max_node, shape=(self.ob_filtration_size, self.max_node), dtype=int), # node id in each filtrated graph
                'edge_index': Box(-1, self.max_node ** 2,
                                  shape=(self.ob_filtration_size, self.max_edge, 2), dtype=int),
                'edge_index2pos': Box(-1, self.max_edge,shape=(self.ob_filtration_size, self.max_edge), dtype=int),

                "action_mask": Box(-1, 2, shape=(self.max_edge,), dtype=int),
                'graph_property': Box(0, np.inf, shape=(1,), dtype=float),
            })
        else:
            self.observation_space = Dict({
                'node_feats': Box(-np.inf, np.inf, shape=(self.wrapped.max_node, self.ndim)),
                'num_edges': Box(0, self.max_node ** 2, shape=(1,), dtype=int),
                'num_nodes': Box(0, self.max_node, shape=(1, ), dtype=int),
                'graph_property': Box(0, np.inf, shape=(1, ), dtype=float),
                'max_node_id': Box(0, self.max_node, shape=(1,), dtype=int),
                'node_id_index': Box(0, self.max_node, shape=(self.max_node,), dtype=int),
                'edge_index': Box(-1, self.max_node ** 2,
                                  shape=(self.max_edge, 2), dtype=int),
                'edge_index2pos': Box(-1, self.max_edge, shape=(self.max_node, self.max_edge), dtype=int),
                "action_mask": Box(-1, 2, shape=(self.max_edge, ), dtype=int),
            })




    def create_feat(self):
        graph = self.wrapped.graph
        edge_index_argument, node_feats_argument, num_edges_argument, num_nodes_argument,\
        node_id_index_argument, max_node_id_argument, edge_index2pos_argument = self.edge_nets_g(graph)
        return edge_index_argument, node_feats_argument, num_edges_argument, num_nodes_argument, node_id_index_argument, max_node_id_argument, edge_index2pos_argument


    def reset(self):
        _ = self.wrapped.reset()

        # position embedding
        pe = np.zeros((self.wrapped.max_node, self.node_attack_order_embedding_dim))
        position = np.arange(0, self.wrapped.max_node, dtype=float)[..., None]
        div_term = np.exp(np.arange(0, self.node_attack_order_embedding_dim, 2).astype(np.float32) * (
                -np.math.log(10000.0) / self.node_attack_order_embedding_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe[None].transpose(1, 0, 2)

        edge_index_argument, node_feats_argument, num_edges_argument, \
        num_nodes_argument, node_id_index_argument, max_node_id_argument, edge_index2pos_argument = self.create_feat()

        if self.use_filtration:
            action_mask = create_logits_mask_by_first_edge_graph(edge_index_argument[0], num_edges_argument[0, 0],
                                                                 self.wrapped.max_node)
        else:
            action_mask = create_logits_mask_by_first_edge_graph(
                edge_index_argument, num_edges_argument[0],
                self.wrapped.max_node)
        # we have to make sure the last column stores the mask for action 1!!
        action_mask_argument = np.zeros((self.max_edge,), dtype=np.float32)
        action_mask_argument[:action_mask.shape[1]] = action_mask[0, :]


        return {

            'node_feats': node_feats_argument,
            "action_mask": action_mask_argument,
            "node_id_index": node_id_index_argument,
            'num_edges': num_edges_argument,
            'graph_property': np.asarray([self.wrapped.robustness]),
            'num_nodes': num_nodes_argument,
            'max_node_id': max_node_id_argument,
            'edge_index': edge_index_argument,
            'edge_index2pos': edge_index2pos_argument,

        }

    def step(self, action):


        orig_obs, rew, done, info = self.wrapped.step(action)

        edge_index_argument, node_feats_argument, num_edges_argument, num_nodes_argument,\
        node_id_index_argument, max_node_id_argument, edge_index2pos_argument = self.create_feat()


        if self.use_filtration:
            action_mask = create_logits_mask_by_first_edge_graph(edge_index_argument[0], num_edges_argument[0, 0],
                                                           num_nodes_argument[0, 0])
        else:
            action_mask = create_logits_mask_by_first_edge_graph(
                                                               edge_index_argument, num_edges_argument[0],
                                                               num_nodes_argument[0])

        action_mask_argument = np.zeros((self.max_edge,), dtype=np.float32)
        action_mask_argument[:action_mask.shape[1]] = action_mask[0, :]

        obs = {

            "action_mask": action_mask_argument,
            'node_feats': node_feats_argument,
            "node_id_index": node_id_index_argument,
            'num_edges': num_edges_argument,
            'num_nodes': num_nodes_argument,
            'max_node_id': max_node_id_argument,
            'edge_index': edge_index_argument,
            'edge_index2pos': edge_index2pos_argument,
            'graph_property': np.asarray([self.wrapped.robustness])
        }
        return obs, rew, done, info

    def seed(self, seed=None):

        np.random.seed(seed)
        random.seed(seed)

        self.wrapped.seed(seed)

    def render(self, mode='human'):
        pass

    def edge_nets_g(self, graph):
        """
        create obs
        when there is no edge in the lg, the filtration process stops
        since we only care the edges  of the graph
        """


        N = graph.number_of_nodes()
        E = graph.number_of_edges() * 2

        # the maximum node of the graphs in the training dataset
        max_N = self.max_node
        max_E = self.max_edge
        sp_length_lg_dim = self.sp_length_ndim

        """
        Unify of the info storage:
        all valid tensors are stored in the first valid pos, and the padded are zeros
        e.g, if the num of node is n, then valid elements of num_feats_argument are in num_feats_argument[0, 1 ,..., n-1]
        """

        if self.use_filtration:

            edge_index_argument = np.zeros((self.ob_filtration_size, max_E, 2), dtype=np.int32)
            num_edges_argument = np.zeros((self.ob_filtration_size, 1), dtype=np.int32)
            num_nodes_argument = np.zeros((self.ob_filtration_size, 1), dtype=np.int32)
            node_id_index_argument = np.zeros((self.ob_filtration_size, max_N), dtype=np.int32)
            max_node_id_argument = np.zeros((self.ob_filtration_size, 1), dtype=np.int32)
            node_feats_argument = np.zeros((self.ob_filtration_size, max_N, self.ndim), dtype=np.float32)
            edge_index2pos_argument = np.zeros((self.ob_filtration_size, max_E), dtype=np.int32)

            g_fil_nx = deepcopy(graph)

            sp_length_g = self.get_features_sp_sample(graph, list(range(graph.number_of_nodes())), sp_length_lg_dim)
            filtration_order_list = self.wrapped.filtration_order_list
            isolated_nodes = self.wrapped.isolated_nodes.copy()
            filtration_order_list = np.array(filtration_order_list, dtype=np.int32)


            """
             add the raw graph info
            """
            # edge index

            g_edge_index = self.wrapped.edge_index
            edge_index_argument[0, :E, :] = g_edge_index#deepcopy(g_edge_index) #todo: do we need a copy here if we use wrapped.edge_index?

            # node init feats
            # for the raw graph, we do not mask out any node
            node_feats_argument[0, :N, :self.node_attack_order_embedding_dim] = self.pe[np.argsort(
                np.array(filtration_order_list, dtype=np.int32)), 0, :].copy() # the first removed node has high first feats


            node_feats_argument[0, :N, self.node_attack_order_embedding_dim:-1] = sp_length_g# / (sp_length_g.max(0, keepdims=True) + 1e-7)  # the first removed node has high f
            node_feats_argument[0, :N, -1] = np.ones((N), dtype=np.int32)  # the first removed node has high f
            node_feats_argument[0, isolated_nodes, -1] = np.zeros((len(isolated_nodes)), dtype=np.int32)  # the first removed node has high f

            # node id init
            node_id_record = list(range(N))
            node_id_index_argument[0, :N] = deepcopy(np.array(node_id_record))  # 0:(N-1)\

            num_edges_argument[0, 0] = E
            num_nodes_argument[0, 0] = N
            max_node_id_argument[0, 0] = max(node_id_record) + 1
            edge_index2pos_argument[0, :E] = np.arange(0, E)



            cnt_g = 0

            for i, removed_node in enumerate(filtration_order_list[:self.wrapped.filtration_order]): # NB: use wrapped filtration_order

                g_fil_nx.remove_node(removed_node)

                # Note: do not forget to time 2 when using nx, but not times 2 when using dgl !! (undirected graph for g in nx, but directed for g in dgl)
                g_temp_E = g_fil_nx.number_of_edges() * 2
                g_temp_N = g_fil_nx.number_of_nodes()
                # create edge index for g
                if g_temp_E > 0:

                    g_edge_index = edge_to_index(g_fil_nx.edges)

                    edge_index_argument[cnt_g + 1, :g_temp_E, :] = g_edge_index
                    for idx, edge in enumerate(g_edge_index):
                        edge_index2pos_argument[cnt_g + 1, idx] = self.wrapped.edge_index_map[tuple(edge)]

                    # node_id_recored stores the remaining nodes' id of the original graph
                    node_id_record.remove(removed_node)

                    # create feats for original graph
                    num_nodes_argument[cnt_g + 1, 0] = g_temp_N
                    max_node_id_argument[cnt_g + 1, 0] = max(node_id_record) + 1
                    num_edges_argument[cnt_g + 1, 0] = g_temp_E

                    node_id_index_argument[cnt_g + 1, :g_temp_N] = deepcopy(np.array(node_id_record))

                    # note: we should copy from the raw node_feats to i+1, instead of i, since last one is stored in the relative pos, not by node_id
                    # the following is wrong, since the degree of the filtrated g is not the same with original g
                    # node_feats_argument[i + 1, :g_temp_N, :] = node_feats_argument[0,
                    #                                            np.asarray(node_id_record), :].copy()
                    # the 0-dim is fine, since the filtration order feature is the same as the original g
                    node_feats_argument[cnt_g + 1, :g_temp_N, 0:self.node_attack_order_embedding_dim] = node_feats_argument[0,
                                                                                       np.asarray(
                                                                                           node_id_record), 0:self.node_attack_order_embedding_dim].copy()

                    sp_length_g = self.get_features_sp_sample(g_fil_nx, node_id_record, sp_length_lg_dim, N)
                    node_feats_argument[cnt_g + 1, :g_temp_N, self.node_attack_order_embedding_dim:-1] = sp_length_g

                    node_feats_argument[cnt_g + 1, :g_temp_N, -1] = node_feats_argument[0,
                                                                                       np.asarray(
                                                                                           node_id_record), -1].copy()


                else:
                    del g_fil_nx
                    break

                cnt_g += 1


        else:

            edge_index_argument = np.zeros((max_E, 2), dtype=np.int32)
            num_edges_argument = np.zeros((1,), dtype=np.int32)
            num_nodes_argument = np.zeros((1,), dtype=np.int32)
            node_id_index_argument = np.zeros((max_N,), dtype=np.int32)
            max_node_id_argument = np.zeros((1), dtype=np.int32)
            node_feats_argument = np.zeros((max_N, self.ndim), dtype=np.float32)
            edge_index2pos_argument = np.zeros((max_N, max_E), dtype=np.int32)


            sp_length_g = self.get_features_sp_sample(graph, list(range(graph.number_of_nodes())), sp_length_lg_dim)

            filtration_order_list = self.wrapped.filtration_order_list
            isolated_nodes = self.wrapped.isolated_nodes


            """
             add the raw graph info
            """
            # edge index
            g_edge_index = self.wrapped.edge_index
            edge_index_argument[:E, :] = g_edge_index


            # node init feats
            node_feats_argument[:N, 0:self.node_attack_order_embedding_dim] = self.pe[np.argsort(
                np.array(filtration_order_list, dtype=np.int32)), 0, :]

            node_feats_argument[:N, self.node_attack_order_embedding_dim:-1] = sp_length_g# / (sp_length_g.max(0, keepdims=True) + 1e-7)  # the first removed node has high f
            node_feats_argument[:N, -1] = np.ones((N), dtype=np.int32)  # the first removed node has high f
            node_feats_argument[isolated_nodes, -1] = np.zeros((len(isolated_nodes)), dtype=np.int32)  # the first removed node has high f

            # node id init
            node_id_record = list(range(N))
            node_id_index_argument[:N] = deepcopy(np.array(node_id_record))
            num_edges_argument[0] = E
            num_nodes_argument[0] = N
            max_node_id_argument[0] = max(node_id_record) + 1
            edge_index2pos_argument[0] = np.arange(0, E)

        return edge_index_argument, node_feats_argument, num_edges_argument, num_nodes_argument, node_id_index_argument, max_node_id_argument, edge_index2pos_argument

    # calculate the shortest path length as a part of node features
    def get_features_sp_sample(self, G, node_set, max_sp, N=None):
        dim = max_sp + 1
        set_size = len(node_set)
        if N is None:
            N = G.number_of_nodes()
        sp_length = np.ones((N, set_size), dtype=np.int32) * -1
        for i, node in enumerate(node_set):
            for node_ngh, length in nx.shortest_path_length(G, source=node).items():
                sp_length[node_ngh, i] = length
        sp_length = np.minimum(sp_length, max_sp)
        onehot_encoding = np.eye(dim, dtype=np.float64)  # [n_features, n_features]
        features_sp = onehot_encoding[sp_length].sum(axis=1)

        return features_sp[node_set, :]
