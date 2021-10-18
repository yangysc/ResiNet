import numpy as np
import copy
import networkx as nx
import gym.spaces
from gym.spaces import MultiDiscrete
from contextlib import contextmanager
import sys, os
from glob import glob


from ray.rllib.examples.env.utils_ import objective_func, step_robustness_reward, example_graph, \
    get_removal_order

from pathlib import Path


class GraphEdgeArgumentEnv(gym.Env):
    """
    Environment wrapper for edge rewiring env
    """

    def __init__(self, max_action, filtration_order, use_stop, use_SimulatedAnnealing, is_train, dataset_type,
                 max_num_node=None,
                 max_num_edge=None,
                 alpha=0,
                 robust_measure='R',
                 single_obj=True,
                 second_obj_func='ge',
                 reward_scale=1,
                 sequentialMode=True,
                 add_penality=False,
                 attack_strategy='degree',
                 break_tie='inc_by_id',
                 verbose=False
                 ):

        # assert use_filtration is True
        assert use_stop is True
        self.is_normalize = False  # bool(is_normalize)
        self.use_stop = use_stop
        self.use_SimulatedAnnealing = use_SimulatedAnnealing
        self.graph = nx.Graph()
        self.reward = 0
        self.is_train = is_train
        # self.dataset_name = dataset
        # self.reward_func = reward_func
        self.counter = 0  # count improving robustness step
        # assert max_action == 100
        self.max_action = max_action
        # self.min_action = min_action
        self.alpha = alpha  # weighted sum of two objectivess
        # self.num_train_dataset = num_train_dataset
        self.robustness, self.robustness_pre, self.robustness_init = 0, 0, 0
        # self.use_filtration = use_filtration
        self.filtration_order_list = []  # removed order
        self.isolated_nodes = []  # removed order

        self.graph_in_mem = False
        self.attack_strategy = attack_strategy
        self.break_tie = break_tie
        # dataset_type = 'ba'

        # load dataset

        # load  data
        # if dataset_type == 'grid':
        #     self.dataset = []
        #     for k in range(5):
        #         for i in range(12, 13):
        #             for j in range(3, 4):
        #                 self.dataset.append(nx.barabasi_albert_graph(i, j))
        #     # self.max_node = 25
        #     self.data_len = len(self.dataset)
        #     self.dataset_index = 0
            # self.max_node = len(self.dataset[0])
            # self.max_edge = self.dataset[0].number_of_edges() * 2

        if dataset_type == 'example_15':
            if verbose:
                print(f'dataset: {dataset_type}')
            self.dataset = []
            # self.robustness_list = []
            self.dataset.append(example_graph())
            # self.robustness_list.append(objective_func(self.dataset[0], alpha=self.alpha))
            # self.dataset.append(nx.barabasi_albert_graph(20, 3))
            self.data_len = 1
            self.dataset_index = 0
            self.max_node = len(self.dataset[0])
            self.max_edge = self.dataset[0].number_of_edges() * 2
            self.graph_in_mem = True
            self.penalty = 1 / self.max_node / self.max_node if add_penality and is_train else 0  # the cost of each time step

        elif dataset_type.startswith('example_'):
            if verbose:
                print(f'dataset: {dataset_type}')

            self.max_node = int(dataset_type.split('_')[1])
            self.dataset = []
            # self.robustness_list = []
            # self.dataset.append(nx.barabasi_albert_graph(self.max_node, 3))

            # print("datasets/ba/ba%d/ba%d*" % (self.max_node,self.max_node))
            try:
                self.dataset.append(
                    nx.read_graphml(glob(os.path.join(Path(__file__).parent, "datasets", "ba", "ba%d" % self.max_node, "ba%d*" %self.max_node))[0],
                        node_type=int))
            except:
                print('check dataset!')
                exit(-2)
                # self.dataset.append(nx.barabasi_albert_graph(self.max_node, 2))

            # self.robustness_list.append(objective_func(self.dataset[0], alpha=self.alpha))
            self.data_len = 1
            self.dataset_index = 0

            self.max_node = len(self.dataset[0])
            self.max_edge = self.dataset[0].number_of_edges() * 2
            self.graph_in_mem = True
            self.penalty = 1 / self.max_node / self.max_node if add_penality and is_train else 0  # the cost of each time step


        elif 'ba' in dataset_type:
            if verbose:
                print('dataset: barabasi')
            assert max_num_node is not None and max_num_edge is not None, "for ba datasets, max_num_node and max_num_edge must be not None"
            self.dataset = []
            # self.robustness_list = []
            if is_train:
                path_src = os.path.join(Path(__file__).parent, "datasets", "ba", dataset_type, 'train')
            else:
                print('evaluating test data')
                path_src = os.path.join(Path(__file__).parent, "datasets", "ba", dataset_type, 'test')
            if verbose:
                print(path_src)
                # os.mkdir(os.path.join(Path(__file__).parent, "datasets", "ba", dataset_type))
                # generate 1000 datasets
            data_folders = glob(os.path.join(path_src, '[0-9b]*.graphml'))
            # num_g_each_folder = 10  # use 100 graphs from each folder (they belong to the same trajectory)
            # data_files = glob('./datasets/{}/train/before_idx_*'.format(self.dataset_name))
            self.dataset = data_folders
            # self.line_graph_size = []
            # for graph in self.graph_src:
            #     # graph = glob(os.path.join(folder, "[0-9b]*.graphml"))
            #     i = 0
            #     # while i < num_g_each_folder:
            #
            #     graph = nx.read_graphml(all_graphs[i], node_type=int)
            #     if graph.number_of_nodes() <= 20:  # only consider 20 nodes of graphs
            #         self.dataset.append(graph)
            #         self.robustness_list.append(objective_func(graph, alpha=self.alpha))
            #         self.graph_size.append(graph.number_of_nodes())
            #         # self.line_graph_size.append(graph.number_of_edges() * 2)
            #         i += 1
                    # printgraph_src
            self.data_len = len(self.dataset)
            self.dataset_index = 0
            self.graph_in_mem = False

            self.max_node = max_num_node
            self.max_edge = max_num_edge
            if verbose:
                print('The number of total graphs is %d' % self.data_len)
        elif 'EU' in dataset_type:  # test
            if verbose:
                print('dataset: EU')
            assert max_num_node is not None and max_num_edge is not None, "for ba datasets, max_num_node and max_num_edge must be not None"
            self.dataset = []
            # self.robustness_list = []

            path_src = os.path.join(Path(__file__).parent, "datasets")
            # else:
            #     path_src = os.path.join(Path(__file__).parent, "datasets", "ba", dataset_type, 'test')
            if verbose:
                print(path_src)
                # os.mkdir(os.path.join(Path(__file__).parent, "datasets", "ba", dataset_type))
                # generate 1000 datasets
            data_folders = glob(os.path.join(path_src, 'EU*.graphml'))
            # num_g_each_folder = 10  # use 100 graphs from each folder (they belong to the same trajectory)
            # data_files = glob('./datasets/{}/train/before_idx_*'.format(self.dataset_name))
            self.dataset.append(
                nx.read_graphml(data_folders[0],
                                node_type=int))
            # self.dataset = data_folders
            # self.line_graph_size = []
            # for graph in self.graph_src:
            #     # graph = glob(os.path.join(folder, "[0-9b]*.graphml"))
            #     i = 0
            #     # while i < num_g_each_folder:
            #
            #     graph = nx.read_graphml(all_graphs[i], node_type=int)
            #     if graph.number_of_nodes() <= 20:  # only consider 20 nodes of graphs
            #         self.dataset.append(graph)
            #         self.robustness_list.append(objective_func(graph, alpha=self.alpha))
            #         self.graph_size.append(graph.number_of_nodes())
            #         # self.line_graph_size.append(graph.number_of_edges() * 2)
            #         i += 1
            # printgraph_src
            self.data_len = len(self.dataset)
            self.dataset_index = 0
            # self.robustness_list.append(objective_func(self.dataset[0], alpha=self.alpha))
            self.data_len = 1
            self.dataset_index = 0

            self.max_node = len(self.dataset[0])
            self.max_edge = self.dataset[0].number_of_edges() * 2

            # print(self.max_node)
            # print(self.max_edge)
            # self.max_node = max_num_node
            # self.max_edge = max_num_edge
            # print(self.max_node)
            # print(self.max_edge)
            self.graph_in_mem = True
            self.penalty = 1 / self.max_node / self.max_node if add_penality and is_train else 0  # the cost of each time step
            if verbose:
                print('The number of total graphs is %d' % self.data_len)

        # set reward function
        self.rewards = {'step_robustness_reward': step_robustness_reward}
        self.reward_func = "step_robustness_reward"
        self.reward_function = self.rewards[self.reward_func]


        # Each action is composed of five parts: [first node, second node, third node, fourth node, prediction of termination]
        if self.use_stop:
            # self.action_space = Tuple([Discrete(2), Discrete(self.max_node ** 2), Discrete(self.max_node ** 2)])
            self.action_space = MultiDiscrete([2, self.max_edge, self.max_edge])
        else:
            # self.action_space = Tuple([Discrete(self.max_node ** 2), Discrete(self.max_node ** 2)])
            self.action_space = MultiDiscrete([self.max_edge, self.max_edge])
            # [Discrete(2), Discrete(self.max_node **2), Discrete(self.max_node **2)])
        # hidden_dim = 128
        # upacboud = np.array([1] * 2 * hidden_dim + [self.max_node ** 2] * 2)
        # # lowacboud = -np.array([10] * 4 * hidden_dim + [self.max_node] * 4 + [0])
        # self.action_space = gym.spaces.Box(low=-upacboud, high=upacboud, dtype=np.float32)
            # set max filtration order

        self.filtration_order_given_by_user = 0
        if self.graph_in_mem and filtration_order != 0:
                self.use_filtration = True

                if filtration_order != -1:
                    assert filtration_order < self.max_node, "maximum filtration order is N-1"
                    self.filtration_order = filtration_order
                else:
                    self.filtration_order = self.max_node - 1

        else: # for multiple graphs
            if filtration_order == 0:

                self.use_filtration = False
                self.filtration_order = 0
            else:
                if filtration_order == -1:
                    self.use_filtration = True
                    # we need to change the filtration value after each reset for multiple graphs setting if order== -1
                    self.filtration_order_given_by_user = filtration_order
                    self.filtration_order = 0
                else:
                    self.use_filtration = True

                    self.filtration_order_given_by_user = filtration_order
                    self.filtration_order = filtration_order


        self.scale = reward_scale

        self.graph_history = []
        self.reward_history = []
        self.action_history = []

        self.adjmat = None  # adjacent matrix
        self.node_list = []  # node ordering
        self.filtration_order_list = []  # removed order
        self.isolated_nodes = []  # removed order

        self.eps = 1e-5

        self.edge_index = None  # store the edge list here to decode actions
        self.edge_index_map = {}  # given decoded action, return the encoded action

        self.robust_measure = robust_measure
        self.single_obj = single_obj
        self.second_obj_func = second_obj_func
        self.sequentialMode = sequentialMode
        self.add_penality = add_penality



    def render(self, mode='human'):
        pass


    def step(self, action_origin, is_test=False):
        """

        :param action_origin: the action (one-hot encoded in N^4 dimension)
        :return:
        """
        ### init
        info = {}  # info we care about

        stop = False

        action_with_stop = np.asarray(action_origin, dtype=int).flatten().tolist()
        if self.use_stop and action_with_stop[0] == 0:  # action 0 == 0 means stop
            constraint = True  #self.num_vialation >= self.max_num_vialation
        else:
            constraint = False


        if constraint:  # stop
            reward = 0
            stop = True

        else:

            # in case for small graphs, after some rewiring steps, all action is invalid, so the padded action space may be chosen, out of boundary
            try:
                if self.use_stop:
                    action = self.edge_index[action_with_stop[1]][0], \
                             self.edge_index[action_with_stop[1]][1], \
                             self.edge_index[action_with_stop[2]][0], \
                             self.edge_index[action_with_stop[2]][1]  # edge_index[action_with_stop].flatten().tolist()  # [:-

                    # self.edge_index
                else:

                    action = self.edge_index[action_with_stop[0]][0], \
                             self.edge_index[action_with_stop[0]][1], \
                             self.edge_index[action_with_stop[1]][0], \
                             self.edge_index[action_with_stop[1]][1]
            except:
                stop = True
                reward = 0
            else:

                edge_1 = action[:2]
                edge_2 = action[2:]

                if len(set(edge_1).union(set(edge_2))) != 4 or not self.graph.has_edge(*edge_1) or not self.graph.has_edge(*edge_2) or self.graph.has_edge(edge_1[0], edge_2[0]) or self.graph.has_edge(edge_1[1], edge_2[1]):
                    stop = True
                    reward = 0
                else:
                    self.graph.remove_edge(*edge_1)
                    self.graph.remove_edge(*edge_2)
                    self.graph.add_edge(edge_1[0], edge_2[0])
                    self.graph.add_edge(edge_1[1], edge_2[1])

                    robustness = objective_func(self.graph,
                                                alpha=self.alpha,
                                                scale=self.scale,
                                                robust_measure=self.robust_measure,
                                                single_obj=self.single_obj,
                                                second_obj_func=self.second_obj_func,
                                                sequentialMode=self.sequentialMode,
                                                attack_strategy=self.attack_strategy
                                                )
                    reward = self.reward_function(robustness, self.robustness) * self.scale

                    self.robustness = robustness
                    self.edge_index = edge_to_index(self.graph.edges)
                    edge_cnt = 0
                    self.edge_index_map.clear()
                    for edge in self.edge_index:
                        self.edge_index_map[(edge[0], edge[1])] = edge_cnt
                        edge_cnt += 1

        self.counter += 1

        # max step
        if not stop and self.counter >= self.max_action:
            stop = True
            info['stop_reason'] = 'exceed max step'

        if not stop:
            reward -= self.penalty
        info['reward'] = reward
        info['stop'] = stop
        self.reward_history.append(reward)

        # get observation
        ob = self.get_observation()


        return ob, reward, stop, info

    def reset(self):
        """
        :reset the env and return ob
        """

        self.dataset_index = (self.dataset_index + 1) % self.data_len
        del self.graph
        if self.graph_in_mem:
            self.graph = copy.deepcopy(self.dataset[self.dataset_index])
        else:

            self.graph = nx.read_graphml(self.dataset[self.dataset_index], node_type=int)
            self.max_node = self.graph.number_of_nodes()
            self.max_edge = self.graph.number_of_edges() * 2
            self.penalty = 1 / self.max_node / self.max_node if self.add_penality and self.is_train else 0  # the cost of each time step

            self.filtration_order = self.max_node // 2 if self.filtration_order_given_by_user == -1 else self.filtration_order_given_by_user

        self.robustness = objective_func(self.graph,
                                         alpha=self.alpha,
                                         scale=self.scale,
                                         robust_measure=self.robust_measure,
                                         single_obj=self.single_obj,
                                         second_obj_func=self.second_obj_func,
                                         sequentialMode=self.sequentialMode
                                         )

        self.reward = 0
        self.counter = 0

        self.robustness_init = self.robustness
        self.adjmat = None #nx.to_numpy_array(self.graph, nodelist=range(self.max_node), dtype=np.int32)
        ob = self.get_observation()

        self.edge_index = edge_to_index(self.graph.edges)

        edge_cnt = 0
        self.edge_index_map.clear()
        for edge in self.edge_index:
            self.edge_index_map[(edge[0], edge[1])] = edge_cnt
            edge_cnt += 1

        self.filtration_order_list, self.isolated_nodes = get_removal_order(self.graph,
                                                                            # self.max_node,
                                                                            return_tuple=True,
                                                                            sequentialMode=True,
                                                                            attack_strategy=self.attack_strategy,
                                                                            break_tie=self.break_tie
                                                                            )
        return ob

    def get_observation(self):

        self.filtration_order_list, self.isolated_nodes = get_removal_order(self.graph,
                                                                            # self.max_node,
                                                                            return_tuple=True,
                                                                            sequentialMode=True,
                                                                            attack_strategy=self.attack_strategy,
                                                                            break_tie=self.break_tie
                                                                            )
        return None  # return None since we have the wrapper to return the obs.
        # Uncomment the following codes to return adjacent matrix if needed

        # NB: the positional parameter 'node_list' is essential, or the resulting node order of matrix is random
        # N = self.graph.number_of_nodes()
        # adj_mat = nx.to_numpy_array(self.graph, nodelist=range(N), dtype=np.int)
        # if self.use_filtration:
        #     self.graph_for_filtration = self.graph
        #     ob = None# np.zeros(shape=(self.observation_space.shape), dtype=np.int)
        #     # ob[0, :, :] = E
        #     # ob[0, -1, :] = np.ones(shape=(self.observation_space.shape[1],))
        #     # ob[0, :, -1] = np.ones(shape=(self.observation_space.shape[1],))
        #     # if self.is_normalize:
        #     #     ob[1:2, :, :] = self.normalize_adj(ob[0:1, :, :])
        #     # else:
        #     #     ob[1:2, :, :] = ob[0:1, :, :]
        #     # ob[0, :N, :N] = adj_mat
        #     # the filtration_order for node [0, 1, 2, 3, 4] may be [4,2, 0, 1, 3], which means the node 4 is first removed,
        #     # and the node 3 is the last one to be removed
        # self.filtration_order_list, self.isolated_nodes = get_removal_order(self.graph_for_filtration, self.filtration_order, return_tuple=True, sequentialMode=True)
        #     # fitration_order = np.array(self.filtration_order_list[:-1])
        #     # sort the order, and reture the corresponding node id
        #     # node_order = np.argsort(-fitration_order)
        #     # for i, node in enumerate(fitration_order):
        #     #     # self.graph_for_filtration.remove_edges_from(list(self.graph_for_filtration.edges(node)))
        #     #     ob[i + 1, :N, :N] = copy.deepcopy(ob[i, :N, :N])
        #     #     # if i < len(fitration_order) - 2:
        #     #     ob[i + 1, node, :N] = np.zeros((N,))
        #     #     ob[i + 1, :N, node] = np.zeros((N,))
        #     # it should be noted that the last view is always zeros, since the last remaining node has lost all edges, so it is auto-deleted by nx
        #     # ob[i + 1, :, :] = copy.deepcopy(ob[i, :, :])
        #     # return ob
        # else:
        #     ob = None
        # return ob

    def seed(self, seed=None):
        np.random.seed(seed)


def edge_to_index(edges, is_undirected=True):
    r"""
    List of G.edges to torch tensor edge_index

    Only the selected edges' edge indices are extracted.
    """

    edge_index = np.asarray(edges)
    edge_index = np.stack([np.min(edge_index, axis=1), np.max(edge_index, axis=1)], axis=1)
    edge_index = edge_index[np.lexsort((edge_index[:, 1], edge_index[:, 0]))]
    edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)

    return edge_index
