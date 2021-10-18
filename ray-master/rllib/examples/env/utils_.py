import networkx as nx
from copy import deepcopy
import numpy as np


def efficiency(graph):
    return nx.global_efficiency(graph)

def get_removal_order(graph_input, return_tuple=False, sequentialMode=True, attack_strategy='degree', break_tie='inc_by_id'):
    """
    Returns the node removal order of the given network G
    """
    Gcopy = deepcopy(graph_input)
    Vcount = len(Gcopy)
    assert break_tie in ['random', 'inc_by_id', 'dec_by_id']
    # compute G
    # if filtration_order == 0:
    #     return [], []
    # if Gcopy.number_of_nodes() == 0:
    #     return 0
    # centrality_func = nx.degree_centrality if attack_strategy == 'degree' elif  nx.betweenness_centrality
    if attack_strategy == 'degree':
        centrality_func = nx.degree_centrality
    elif attack_strategy == 'betweenness':
        centrality_func = nx.betweenness_centrality
    else:
        centrality_func = None

    if centrality_func is not None:
        if break_tie == 'inc_by_id':
            V = sorted(centrality_func(Gcopy).items(), key=lambda x: (-x[1], x[0]))
        elif break_tie == 'dec_by_id':
            V = sorted(centrality_func(Gcopy).items(), key=lambda x: (-x[1], -x[0]))
        else:
            V = sorted(centrality_func(Gcopy).items(), key=lambda x: (-x[1]))

    else:
        V = (list(Gcopy.nodes()))
        np.random.shuffle(V)

    removal_order = []
    isolated_nodes_list = []

    for i in range(0, Vcount):
        remove_v = V.pop(0)[0] if centrality_func is not None else V.pop(0)

        Gcopy.remove_node(remove_v)

        isolated_nodes = list(nx.isolates(Gcopy))
        for isolated_node in isolated_nodes:
            if isolated_node not in isolated_nodes_list:
                isolated_nodes_list.append(isolated_node)

        removal_order.append(remove_v)
        #
        if sequentialMode:
            if centrality_func is not None:
                if break_tie == 'inc_by_id':
                    V = sorted(centrality_func(Gcopy).items(), key=lambda x: (-x[1], x[0]))
                elif break_tie == 'dec_by_id':
                    V = sorted(centrality_func(Gcopy).items(), key=lambda x: (-x[1], -x[0]))
                else:
                    V = sorted(centrality_func(Gcopy).items(), key=lambda x: (-x[1]))
                # else:
                #     V = sorted(centrality_func(Gcopy).items(), key=lambda x: x[0])
            else:
                V = (list(Gcopy.nodes()))
                np.random.shuffle(V)
    del Gcopy
    if return_tuple:
        # it should be noted that the node order in isolated_nodes_list is different from the order they appears in removal_order
        # since in isolated_nodes_list, we do not sort them by the node id, instead we sort them in removal_order (this
        # is how isolated nodes are selected to be removed)
        return removal_order, isolated_nodes_list
    return removal_order




def objective_func(graph_input, alpha=0, scale=1, robust_measure='R', single_obj=True, second_obj_func='ge', sequentialMode=True, attack_strategy='degree'):
    """
    Returns the weighted sum of the robustness value and the utility value of the given network G
    alpha * robustness + (1-alpha) * utility
    """
    if alpha > 0:
        single_obj = False
    if robust_measure == 'R':  # robustness based on "Mitigation of Malicious Attacks on Networks"
        robustness = robustness_cal_robust(graph_input, sequentialMode, attack_strategy)
    elif robust_measure == "sr":  # spectral_radius
        robustness = np.real(np.max(nx.linalg.adjacency_spectrum(graph_input)))
    elif robust_measure == "ac":  # algebraic_connectivity
        robustness = nx.linalg.algebraic_connectivity(graph_input)
    else:
        robustness = 0
        print('robustness must be ["R", "sr", "ac"]')

    if single_obj:
        return robustness
    else:
        if second_obj_func == 'ge': # global efficiency
            return alpha * scale * nx.global_efficiency(graph_input) / 10 + (1-alpha) * scale * robustness
        elif second_obj_func == 'le':
            return alpha * scale * nx.local_efficiency(graph_input) + (1 - alpha) * scale * robustness
        else:
            print("second_obj_func must be ['ge', 'le']")
            exit(-13)



def robustness_cal_robust(graph_input, sequentialMode=True, attack_strategy='degree'):
    """
    Returns the robustness value of the given network G, computed using
    degree_centrality-attack strategy (or betweenness-based) and the given attack mode (sequential
    or simultaneous).
    """
    Gcopy = deepcopy(graph_input)
    Vcount = len(Gcopy)
    sequentialMode = sequentialMode
    centrality_func = nx.degree_centrality if attack_strategy == 'degree' else nx.betweenness_centrality
    V = sorted(centrality_func(Gcopy).items(),
               key=lambda x: (-x[1], x[0]))
    R = 0.0
    for i in range(1, Vcount):
        Gcopy.remove_node(V.pop(0)[0])
        if sequentialMode:
            V = sorted(centrality_func(Gcopy).items(),
                       key=lambda x: (-x[1], x[0]))
        giantComponent = max(nx.connected_components(Gcopy), key=len)
        R += len(giantComponent) / Vcount

    del Gcopy
    return R / Vcount


def step_robustness_reward(robustness, robustness_pre):
    step_reward = robustness - robustness_pre
    return step_reward



def example_graph():
    G = nx.Graph()
    # original example_15 (BA-15) graph
    edge_list = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 8], [0, 9], [0, 12],
                 [1, 2], [1, 3], [1, 7], [1, 11],
                 [2, 4], [2, 5], [2, 10],
                 [5, 6], [5, 10], [5, 12],
                 [6, 8], [6, 7], [6, 13], [6, 14],
                 [7, 9], [7, 11],
                 [9, 14],
                 [12, 13]
                 ]
    G.add_edges_from(edge_list)

    return G

