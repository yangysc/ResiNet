from gym.spaces import Tuple, Box

from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override

from torch_geometric.nn import global_add_pool
import dgl
from dgl.nn.pytorch.glob import SumPooling


tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
import torch as th
import numpy as np


from .gnnmodel import GIN






class TorchEdgeAutoregressiveBaseModel(TorchModelV2, nn.Module):
    """Edge selection network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)


        use_filtrated = model_config["custom_model_config"]["use_filtration"]
        self.filtration_order = model_config["custom_model_config"]["filtration_order"]
        self.use_filtrated = use_filtrated

        fc_size = model_config["custom_model_config"]["hidden_dim"]
        lstm_state_size = fc_size

        dim_nfeats = 8 + 6 + 1
        hidden_dim = fc_size


        # GIN , dgl
        self.context_layer = GIN(num_layers=5, num_mlp_layers=2, input_dim=dim_nfeats, hidden_dim=hidden_dim,
                                 output_dim=hidden_dim, final_dropout=0, learn_eps=True, graph_pooling_type='sum',
                                 neighbor_pooling_type='sum', mode='lstm')



        self.merger = nn.Linear(hidden_dim * 6, hidden_dim)

        self.lin_graph = None
        self.node_to_graph = SumPooling()

        self.graph_hidden_size = hidden_dim


        # Build the value layers
        if num_outputs is None:
            self._emb_graph_branch_separate = None
            self._value_branch = None
        # self._value_branch = nn.Linear(hidden_dim, 1)
        else:
            self._value_branch = nn.Sequential(SlimFC(
                in_size=hidden_dim,
                out_size=256,
                initializer=torch.nn.init.xavier_uniform_,
                activation_fn='relu'),
                SlimFC(
                    in_size=256,
                    out_size=1,
                    initializer=torch.nn.init.xavier_uniform_,
                    activation_fn=None)
            )
            vf_share_layers = self.model_config.get("vf_share_layers")
            if vf_share_layers:
                self._emb_graph_branch_separate = None
            else:
                self._emb_graph_branch_separate = GIN(num_layers=5, num_mlp_layers=2, input_dim=dim_nfeats,
                                                      hidden_dim=hidden_dim,
                                                      output_dim=hidden_dim, final_dropout=0, learn_eps=True,
                                                      graph_pooling_type='sum',
                                                      neighbor_pooling_type='sum', mode='lstm')
                self.lin_graph_separate = None
                self.node_to_graph_separate = SumPooling()


        # edge selection models given edge embedding and graph embedding
        class _EdgeSelectionAction1Model(nn.Module):
            def __init__(self, is_direct_logits, similarity_func, use_graph_embedding, transfer_func, embnode_mlp):
                nn.Module.__init__(self)
                self.embnode_mlp = embnode_mlp
                self.similarity_func = similarity_func
                self.node_gating_ac_1 = nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
                self.node_to_graph_ac_1 = nn.Linear(hidden_dim,
                                                    hidden_dim)
                # self.multiheadattn = nn.MultiheadAttention(hidden_dim, 4)
                self.conditionedheadattn = ConditionedAttn(hidden_size=hidden_dim, using_emb_action_as_v=True)
                assert similarity_func != 'dot'
                if self.similarity_func == 'dot':
                    self.select_action_func = self.dot_similarity
                else:
                    self.select_action_func = self.pointerNet_similarity

                # self.a1_logits = Attention(hidden_dim)
                # self.bn_1 = nn.BatchNorm1d(hidden_dim, affine=True)
                self.a1_hidden = nn.Sequential(
                    SlimFC(
                        in_size=hidden_dim,
                        out_size=hidden_dim,
                        activation_fn=nn.Tanh,

                        initializer=normc_init_torch(1)))

                self.a1_logits = SlimFC(
                    in_size=hidden_dim,
                    out_size=num_outputs,  # concat of [mean, std] of gaussion
                    activation_fn=None,
                    initializer=normc_init_torch(0.01))

            def forward(self_, ctx_input, emb_node, emb_graphs_filtrated, edge_index_mask):

                if use_filtrated:

                    attn_output_padded = (
                            self_.node_gating_ac_1(emb_graphs_filtrated)[..., None] * self_.node_to_graph_ac_1(
                            emb_node))
                    attn_output_padded = attn_output_padded * edge_index_mask.unsqueeze(-1)

                    attn_output = attn_output_padded.sum(1)
                else:

                    attn_output = emb_node

                # attn_output = emb_node[:, 0, :, :]
                a1_context = ctx_input  # the ctx_input is the estimated action embedding ( for inverse model like curiousty)

                # select action
                a1_logits = self_.select_action_func(a1_context, attn_output)
                # del attn_output
                # a1_logits = self_.a1_logits(self_.bn_1(self_.a1_hidden(ctx_input)))
                return a1_logits

            def dot_similarity(self, emb_estimate_action, emb_all_action):
                similarity = torch.einsum('ijk, ik->ij', emb_all_action, emb_estimate_action)
                return similarity

            def pointerNet_similarity(self, emb_estimate_action, emb_all_action):
                similarity = self.conditionedheadattn(emb_estimate_action, emb_estimate_action, emb_all_action)
                # similarity = torch.einsum('ijk, ik->ij', emb_all_action, emb_estimate_action)
                return similarity

        class _EdgeSelectionAction2Model(nn.Module):
            def __init__(self, is_direct_logits, similarity_func, use_graph_embedding, transfer_func, embnode_mlp):
                nn.Module.__init__(self)
                self.embnode_mlp = embnode_mlp
                # self.a2_logits = Attention(hidden_dim)
                # self.multiheadattn = nn.MultiheadAttention(hidden_dim, 4)
                self.node_gating_ac_2 = nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
                self.node_to_graph_ac_2 = nn.Linear(hidden_dim,
                                                    hidden_dim)
                self.conditionedheadattn = ConditionedAttn(hidden_size=hidden_dim, using_emb_action_as_v=True)
                assert similarity_func != 'dot'
                # self.bn_2 = nn.BatchNorm1d(hidden_dim, affine=True)
                if similarity_func == 'dot':
                    self.select_action_func = self.dot_similarity
                else:
                    self.select_action_func = self.pointerNet_similarity


                self.a2_hidden_action_1 = nn.Sequential(
                    SlimFC(
                        in_size=hidden_dim,
                        out_size=hidden_dim,
                        activation_fn=nn.Tanh,

                        initializer=normc_init_torch(1))

                )

                self.a2_logits = SlimFC(
                    in_size=hidden_dim * 2,
                    out_size=hidden_dim,  # concat of [mean, std] of gaussion  #
                    activation_fn=None,
                    initializer=normc_init_torch(0.01))

            def forward(self_, ctx_input, emb_node, a1_input, emb_graphs_filtrated, edge_index_mask):

                if use_filtrated:

                    attn_output_padded = (
                            self_.node_gating_ac_2(emb_graphs_filtrated)[..., None] * self_.node_to_graph_ac_2(
                            emb_node))
                    attn_output_padded = attn_output_padded * edge_index_mask.unsqueeze(-1)
                    attn_output = attn_output_padded.sum(1)
                else:
                    attn_output = emb_node

                a2_hidden = torch.cat([ctx_input, self_.a2_hidden_action_1(a1_input)],
                                      dim=-1)
                a2_context = self_.a2_logits(a2_hidden)

                # select action
                a2_logits = self_.select_action_func(a2_context, a1_input, attn_output)
                # del attn_output
                return a2_logits

            def dot_similarity(self, emb_graph, emb_preaction, emb_all_action):
                similarity = torch.einsum('ijk, ik->ij', emb_all_action, emb_preaction)
                return similarity

            def pointerNet_similarity(self, emb_graph, emb_preaction, emb_all_action):
                similarity = self.conditionedheadattn(emb_graph, emb_preaction, emb_all_action)
                # similarity = torch.einsum('ijk, ik->ij', emb_all_action, emb_estimate_action)
                return similarity


        is_direct_logits = model_config['custom_model_config']['direct_logits']
        similarity_func = model_config['custom_model_config']['similarity_func']
        use_graph_embedding = model_config['custom_model_config']['use_graph_embedding']
        transfer_func = model_config['custom_model_config']['transfer_func']
        self.action1_module = _EdgeSelectionAction1Model(is_direct_logits=is_direct_logits,
                                                         similarity_func=similarity_func,
                                                         use_graph_embedding=use_graph_embedding,
                                                         transfer_func=transfer_func, embnode_mlp=None)
        self.action2_module = _EdgeSelectionAction2Model(is_direct_logits=is_direct_logits,
                                                         similarity_func=similarity_func,
                                                         use_graph_embedding=use_graph_embedding,
                                                         transfer_func=transfer_func, embnode_mlp=None)

        # view_requirements, needed by rllib
        self.N = model_config['custom_model_config']['max_num_node'] if self.filtration_order == -1 or  self.filtration_order == 0 else self.filtration_order+1
        self.max_edge = model_config['custom_model_config']['max_num_edge']
            # exit(-1)

        if self.use_filtrated:
            # self.bn = nn.BatchNorm1d(self.num_outputs, affine=True)
            # emb_edge is the embedding for each edge in the g (each node in the lg)
            self.view_requirements['emb_edge'] = \
                ViewRequirement('emb_edge', space=Box(-200.0, 200.0, shape=(self.N, self.max_edge, lstm_state_size)),
                                used_for_training=True, used_for_compute_actions=False # num_outputs is the max edge
                                )
            # emb_node is the embedding for each node in the g (used for determining the order of the nodes in a edge
            # e.g. for the edge [0, 1] , it can also mean (1, 0]
            # self.view_requirements['emb_node'] = \
            #     ViewRequirement('emb_node', space=Box(-20.0, 20.0, shape=(self.N, self.N, self.num_outputs)),
            #                     used_for_training=True  # num_outputs is the max edge
            #                     )
            # emb_graphs_filtrated is a tensor (bs, N, d), is the graph embedding for filtrated graphs, and would be
            # used in the action models to aggregate the node embedding from different view of filtrated graphs
            self.view_requirements['emb_graphs_filtrated'] = \
                ViewRequirement('emb_graphs_filtrated', space=Box(-200.0, 200.0, shape=(self.N, lstm_state_size)),
                                used_for_training=True, used_for_compute_actions=False  # num_outputs is the max edge
                                )

            self.view_requirements['num_edges'] = \
                ViewRequirement('num_edges',
                                Box(0, self.N ** 2, shape=(self.N, 1), dtype=int), used_for_training=True, used_for_compute_actions=False
                                )

            self.view_requirements['edge_index'] = \
                ViewRequirement('edge_index', space=Box(low=-1, high=self.N ** 2, shape=(self.N, self.max_edge, 2), dtype=int),
                                used_for_training=True, used_for_compute_actions=False)

            self.view_requirements['action_mask'] = \
                ViewRequirement('action_mask',
                                space=Box(low=-1, high=1, shape=(self.max_edge, ), dtype=int),
                                used_for_training=True, used_for_compute_actions=True)
            self.view_requirements['a0_logits'] = \
                ViewRequirement('a0_logits',
                                space=Box(low=-100, high=100, shape=(2,), dtype=float),
                                used_for_training=True, used_for_compute_actions=False)
            self.view_requirements['a1_logits'] = \
                ViewRequirement('a1_logits',
                                space=Box(low=-100, high=100, shape=(self.max_edge,), dtype=float),
                                used_for_training=True, used_for_compute_actions=False)
            self.view_requirements['a2_logits'] = \
                ViewRequirement('a2_logits',
                                space=Box(low=-100, high=100, shape=(self.max_edge,), dtype=float),
                                used_for_training=True, used_for_compute_actions=False)

            self.view_requirements['edge_index_mask'] = \
                ViewRequirement('edge_index_mask',
                                space=Box(low=-1, high=1, shape=(self.N, self.max_edge), dtype=bool),
                                used_for_training=True, used_for_compute_actions=False)


        else:
            self.view_requirements['emb_edge'] = \
                ViewRequirement('emb_edge', space=Box(-200.0, 200.0, shape=( self.max_edge, lstm_state_size)),
                                used_for_training=True, used_for_compute_actions=False  # num_outputs is the max edge
                                )

            # emb_graphs_filtrated is a tensor (bs, N, d), is the graph embedding for filtrated graphs, and would be
            # used in the action models to aggregate the node embedding from different view of filtrated graphs
            self.view_requirements['emb_graphs_filtrated'] = \
                ViewRequirement('emb_graphs_filtrated', space=Box(-200.0, 200.0, shape=(lstm_state_size,)),
                                used_for_training=True, used_for_compute_actions=False  # num_outputs is the max edge
                                )

            self.view_requirements['num_edges'] = \
                ViewRequirement('num_edges',
                                Box(0, self.N ** 2, shape=(1,), dtype=int), used_for_training=True, used_for_compute_actions=False
                                )

            # view requirements for g
            self.view_requirements['edge_index'] = \
                ViewRequirement('edge_index',
                                space=Box(low=-1, high=self.N ** 2, shape=( self.max_edge, 2), dtype=int),
                                used_for_training=True, used_for_compute_actions=False)

            self.view_requirements['action_mask'] = \
                ViewRequirement('action_mask',
                                space=Box(low=-1, high=1, shape=(self.max_edge,), dtype=int),
                                used_for_training=True, used_for_compute_actions=True)

            self.view_requirements['a0_logits'] = \
                ViewRequirement('a0_logits',
                                space=Box(low=-100, high=100, shape=(2,), dtype=float),
                                used_for_training=True, used_for_compute_actions=False)
            self.view_requirements['a1_logits'] = \
                ViewRequirement('a1_logits',
                                space=Box(low=-100, high=100, shape=(self.max_edge,), dtype=float),
                                used_for_training=True, used_for_compute_actions=False)
            self.view_requirements['a2_logits'] = \
                ViewRequirement('a2_logits',
                                space=Box(low=-100, high=100, shape=(self.max_edge,), dtype=float),
                                used_for_training=True, used_for_compute_actions=False)

            self.view_requirements['edge_index_mask'] = \
                ViewRequirement('edge_index_mask',
                                space=Box(low=-1, high=1, shape=(self.max_edge,), dtype=bool),
                                used_for_training=True, used_for_compute_actions=False)

        self._context = None

        self.rawobs, self.n_feats, self.num_edges = [None] * 3

        self.emb_edge = None

        self.edge_index = None
        self.edge_index_mask = None
        self.lg_edge_index = None
        self.lg_node_feats = None


    def _create_fc_net(self, layer_dims, activation, name=None):
        """Given a list of layer dimensions (incl. input-dim), creates FC-net.

        Args:
            layer_dims (Tuple[int]): Tuple of layer dims, including the input
                dimension.
            activation (str): An activation specifier string (e.g. "relu").

        Examples:
            If layer_dims is [4,8,6] we'll have a two layer net: 4->8 (8 nodes)
            and 8->6 (6 nodes), where the second layer (6 nodes) does not have
            an activation anymore. 4 is the input dimension.
        """
        layers = []

        for i in range(len(layer_dims) - 1):
            act = activation if i < len(layer_dims) - 2 else None
            if self.framework == "torch":
                layers.append(
                    SlimFC(
                        in_size=layer_dims[i],
                        out_size=layer_dims[i + 1],
                        initializer=torch.nn.init.xavier_uniform_,
                        activation_fn=act))

        return nn.Sequential(*layers)


    @staticmethod
    def get_boundary_info(edge_index):

        e = edge_index.permute(1, 0).sort(1)[0].tolist()
        e = set([tuple(ee) for ee in e])
        return torch.tensor([ee for ee in e], dtype=torch.long)

    def forward(self, input_dict, state, seq_lens):


        device = input_dict["obs"]['edge_index'].device

        # Note: the edge_indexes is stored in the fist num_edges (not the id pos)
        # use cpu to form graph for dgl
        edge_indexes = input_dict["obs"]['edge_index'].long()
        node_feats = input_dict["obs"]['node_feats']
        bs = edge_indexes.size(0)

        # Note: the node_label is stored in the fist num_edges (not the id pos)
        node_id_index = input_dict["obs"]['node_id_index'].long()
        num_edges = input_dict["obs"]['num_edges'].long()
        # num_nodes = input_dict["obs"]['num_nodes'].long().cpu()
        num_nodes = input_dict["obs"]['num_nodes'].long()
        max_node_ids = input_dict["obs"]['max_node_id'].long()
        edge_index2pos = input_dict["obs"]['edge_index2pos'].long()


        action_mask = input_dict["obs"]['action_mask']

        # bs = 2 # for easy test
        ########### =.= ##############
        # not use for loop, only vector process
        ########### =.= ##############
        """
        g info
        """


        n_edge_of_g = num_edges[:bs].reshape(-1)
        n_node_of_g = num_nodes[:bs].reshape(-1)
        max_node_ids_reshaped = max_node_ids[:bs].reshape(-1)
        # valid graphs for g and lg (non-zeros padding), which must be equal if no edge in the graph means no node in the graph
        valid_num_edge_graph = n_edge_of_g > 0
        valid_num_node_graph = n_node_of_g > 0
        # print('encoder part 1 good')
        # [0] + [ N**2] * (bs*N -1), to align the idx for the padded output (bs, N, N*N, d)->reshape to (bs*N*N^2, d)
        # e.g. if N = 15, then emb_output_bias_for_align is [0, 225, 225, 225 ... 225], total (bs * 15-1) numbers of 225
        if self.use_filtrated:
            repeat_time = bs * node_feats.size(1)
            N = node_id_index.size(2)
        else:
            repeat_time = bs
            N = node_id_index.size(1)

        emb_output_bias_for_align = th.cat(
            [th.tensor(0, device=device)[None],
             th.tensor(edge_indexes.size(-2), device=device).repeat(repeat_time - 1)])


        emb_output_bias_for_align_cumsum = th.cumsum(emb_output_bias_for_align, dim=-1)
        # repeat for each edge
        emb_output_bias_for_align_cumsum_rep = th.repeat_interleave(emb_output_bias_for_align_cumsum,
                                                                       n_edge_of_g)  # this is

        # add node id bias for the edge_index, e.g, first graphj is 0-N-1, second graph is N->2N-1, etc
        # wrong usage : node_id_bias_for_edge_index = torch.cat([torch.tensor(0, device=device)[None], n_node_of_g[:-1]])
        # since n_node_of_g has zeros, so the last one is often zero, then skip it is useless
        # Note: we have to filter out the zero number of graph, since 1) cumsum ([1, 2, 0, 3] gives [1, 3, 3, 6], so there is a extra 3
        # 2) but torch.repeat_interleave is fine with the zero times repeats, since it would skip the zero, but the dims for repeat must be the same

        node_id_bias_for_edge_index = th.cat(
            [th.tensor(0, device=device)[None], max_node_ids_reshaped[valid_num_node_graph][:-1]])

        node_id_bias_for_edge_index_cumsum = th.cumsum(node_id_bias_for_edge_index, dim=-1)

        # repeat for ech edge of g (since we add the bias for each (u, v) tuple of each edge
        node_id_bias_for_edge_index_cumsum_rep = th.repeat_interleave(node_id_bias_for_edge_index_cumsum,
                                                                         n_edge_of_g[valid_num_edge_graph])  # this is

        # node_id_bias_for_edge_index =
        #
        # node_id_bias_for_edge_index_cumsum =
        #
        # # repeat for ech edge (since we add the bias for each node tuple of each edge
        # node_id_bias_for_edge_index_cumsum_rep = torch.repeat_interleave(torch.cumsum( torch.cat(
        #     [torch.tensor(0, device=device)[None], n_node_of_g[:-1]]), dim=-1), n_node_of_g)  # t

        # align the edge_indexes ( add position id to each edge)
        edge_indexes_reshaptoall = edge_indexes[:bs].reshape(-1, edge_indexes.size(-1))

        # edge_indexes_headn_idx [0-E1, 0-E2, 0-E3], record the real edge index, the first head of Ei in each graph
        edge_indexes_headn_idx = th.cat(
            [th.arange(stop_value, device=device) for stop_value in n_edge_of_g])  # todo:
        edge_indexes_headn_idx_padded = edge_indexes_headn_idx + emb_output_bias_for_align_cumsum_rep

        # node_id_headn_idx = torch.cat([torch.arange(stop_value) for stop_value in n_nod ,.e_of_g]).to(device)
        # select all the edge index for a batch of graphs (not add bias for batch position yet)
        edge_indexes_final_raw_index = edge_indexes_reshaptoall[edge_indexes_headn_idx_padded]
        # edge_indexes_final_for_construct_graph is used as the final edge_index for the whole graph (add position id bias).
        # the first graph is labeled as [0,N-1], and the second graph is labeled as [N, 2N -1]
        # edge_indexes_final_for_construct_graph：(total_valid_edges_in_the_batch, 2)), the node id in the second graph is added by the number of nodes in the first graph
        # and the third graph is added by cumsum(N1, N2), and so on ...

        edge_indexes_final_for_construct_graph = edge_indexes_final_raw_index + \
                                                 node_id_bias_for_edge_index_cumsum_rep[..., None].expand_as(
                                                        edge_indexes_final_raw_index)


        # edge_indexes_bias_final_for_align_edge_embedding_g = edge_indexes_final_raw_index[:, 0] * N + edge_indexes_final_raw_index[:, 1] + \
        #                                          emb_output_bias_for_align_cumsum_rep
        # + node_id_bias_for_edge_index_cumsum_rep[..., None].expand_as(edge_indexes_final_raw_index)
        # print('dgl starts')
        g_dgl = dgl.graph((edge_indexes_final_for_construct_graph[:, 0],
                           edge_indexes_final_for_construct_graph[:, 1]), num_nodes=max_node_ids_reshaped.sum(),
                          device=device)

        node_feats_bias_for_align = th.cat([th.tensor(0, device=device)[None],
             th.tensor(N, device=device).repeat(repeat_time - 1)])
        node_feats_for_bias_align_cumsum = th.cumsum(node_feats_bias_for_align, dim=-1)
        node_feats_for_bia_align_cumsum_rep = th.repeat_interleave(node_feats_for_bias_align_cumsum,
                                                                      n_node_of_g)
        # node_feats_for_bia_align_cumsum_rep = th.repeat_interleave(node_id_bias_for_edge_index_cumsum,
        #                                                               n_node_of_g)
        node_feats_headn_idx = th.cat([th.arange(stop_value) for stop_value in n_node_of_g]).to(device)
        # bias_for_node_feats_repeat = torch.cat([torch.tensor(0, device=device)[None], n_node_of_g[:-1]])

        node_feats_reshaptoall = node_feats[:bs].reshape(-1, node_feats.size(-1))
        node_feats_valid = node_feats_reshaptoall[node_feats_headn_idx + node_feats_for_bia_align_cumsum_rep]



        g_dgl = dgl.remove_self_loop(g_dgl)
        g_dgl = dgl.add_self_loop(g_dgl)  # dgl would append the self loop edge index (0-0, 1-1, N-1 - N-1)

        # set_batch must be behind the add_self_loop, otherwise the bs would be set to 1 (add_self_loop would lose the batch info)
        # see https://docs.dgl.ai/en/0.6.x/generated/dgl.add_self_loop.html
        g_dgl.set_batch_num_nodes(max_node_ids_reshaped[valid_num_node_graph])

        # add the additional node degree, since we add self-loop
        g_dgl.set_batch_num_edges(n_edge_of_g[valid_num_edge_graph] + max_node_ids_reshaped[valid_num_node_graph])


        # pad the node features
        node_id_index_reshape = node_id_index[:bs].reshape(-1)
        node_id_index_valid = node_id_index_reshape[node_feats_headn_idx + node_feats_for_bia_align_cumsum_rep]
        node_id_bias_for_node_feats_cumsum_rep = th.repeat_interleave(node_id_bias_for_edge_index_cumsum,
        n_node_of_g[valid_num_edge_graph])
        node_id_index_valid_cumsum = node_id_index_valid + node_id_bias_for_node_feats_cumsum_rep

        padded_node_feats = th.zeros((g_dgl.num_nodes(), node_feats_valid.size(-1)), device=device)
        padded_node_feats.scatter_add_(0, node_id_index_valid_cumsum[..., None].expand(-1, node_feats_valid.size(-1)).long(),
                                       node_feats_valid.float())


        deg = g_dgl.in_degrees().float().clamp(min=1).to(device)
        norm = th.pow(deg, -0.5)
        g_dgl.ndata['d'] = norm
        emb_edge_output = self.context_layer(g_dgl, padded_node_feats.to(device), max_node_ids_reshaped[valid_num_node_graph].to(device))

        # final node embedding for each graph
        emb_node_output_valid = emb_edge_output[node_id_index_valid_cumsum]
        # construct valid graph, and obtain graph embedding from it

        # g_pyg = Data(x=emb_node_output_valid, edge_index=(edge_indexes_final_for_construct_graph.t().contiguous()))
        batch = th.repeat_interleave(th.arange(valid_num_node_graph.sum(), device=device), n_node_of_g[valid_num_node_graph])
        emb_graph_out_g = global_add_pool(emb_node_output_valid, batch)

        emb_graph_g = th.zeros((repeat_time, emb_graph_out_g.size(-1)), device=device)
        emb_graph_g[valid_num_node_graph] = emb_graph_out_g
        if self.use_filtrated:
            emb_graph_g = emb_graph_g.reshape(-1, edge_indexes.size(1), emb_graph_g.size(-1))
        else:
            emb_graph_g = emb_graph_g.reshape(-1, emb_graph_g.size(-1))
        emb_graphs_filtrated = emb_graph_g
        if self.use_filtrated:
            # normalized by the valid filtrated subgraphs and add 1e-5 
            emb_graph_g = emb_graph_g.sum(1) / ((num_nodes > 0).sum(1) + 1e-5)

        else:
            emb_graph_g = emb_graph_g


        # step 3: construct edge embedding from node embedding
        node_u_latent = th.index_select(emb_edge_output, 0, edge_indexes_final_for_construct_graph[:, 0])
        node_v_latent = th.index_select(emb_edge_output, 0, edge_indexes_final_for_construct_graph[:, 1])
        node_feats_UV = th.stack([node_u_latent, node_v_latent], dim=-1)
        emb_edge_cat = th.cat(
            [node_feats_UV.min(-1)[0], node_feats_UV.max(-1)[0], node_feats_UV.mean(-1), node_feats_UV.sum(-1),
             node_feats_UV[:, :, 0] - node_feats_UV[:, :, 1], node_feats_UV[:, :, 0] * node_feats_UV[:, :, 1]], -1)
        emb_edge_output = self.merger(emb_edge_cat)
        # way 2): use for loop

        emb_edge = th.zeros((repeat_time * edge_indexes.size(-2), emb_edge_output.size(-1)),
                            device=device)  # .to(graph_mat.device)



        emb_output_bias_for_align_lg = th.cat(
            [th.tensor(0, device=device)[None],
             th.tensor(edge_indexes.size(-2), device=device).repeat(repeat_time - 1)])
        emb_output_bias_for_align_cumsum_lg = th.cumsum(emb_output_bias_for_align_lg, dim=-1)
        # repeat for each edge
        emb_output_bias_for_align_cumsum_rep_lg = th.repeat_interleave(emb_output_bias_for_align_cumsum_lg,
                                                                       n_edge_of_g)  # this is

        # edge_indexes_headn_idx [0-E1, 0-E2, 0-E3], record the real edge index, the first head of Ei in each graph
        edge_indexes_headn_idx_lg = th.cat(
            [th.arange(stop_value, device=device) for stop_value in n_edge_of_g])  # todo:
        edge_indexes_headn_idx_padded_lg = edge_indexes_headn_idx_lg + emb_output_bias_for_align_cumsum_rep_lg


        """
         align the filtrated version of emb_edge  ( this is very important)!
        """
        edge_index2pos_reshaped = edge_index2pos[:bs].reshape(-1)
        edge_index2pos_valid = edge_index2pos_reshaped[edge_indexes_headn_idx_padded_lg]
        edge_indexes_bias_final_for_align_node_embedding_lg = edge_index2pos_valid + emb_output_bias_for_align_cumsum_rep_lg

        edge_index_mask = th.zeros((repeat_time * edge_indexes.size(-2)), dtype=th.bool,
                                  device=device)
        # edge label mask
        edge_index_mask[edge_indexes_bias_final_for_align_node_embedding_lg] = True  # valid elements are not padded

        if self.use_filtrated:
            edge_index_mask = edge_index_mask.reshape(bs, edge_indexes.size(1), edge_indexes.size(2))
        else:
            edge_index_mask = edge_index_mask.reshape(bs, edge_indexes.size(-2))

        add_idx = edge_indexes_bias_final_for_align_node_embedding_lg[..., None].expand((-1, emb_edge.size(-1))).to(
                device)
        emb_edge.scatter_add_(0, add_idx, emb_edge_output)
        
        if self.use_filtrated:
            emb_edge = emb_edge.reshape(-1, edge_indexes.size(1), edge_indexes.size(-2), emb_edge.size(-1))
        else:
            emb_edge = emb_edge.reshape(-1, edge_indexes.size(-2), emb_edge.size(-1))


        self._context = emb_graph_g

        self.emb_graphs_filtrated = emb_graphs_filtrated
        if self._emb_graph_branch_separate is None:
            self._context_value = None
            self._value_out = self._value_branch(self._context) if self._value_branch is not None else None
        # self._value_out = self._value_branch(self._context)
        else:
            # calculate the graph embedding of lg using a separate gnn
            # emb_edge_ouput_value = self._emb_graph_branch_separate(lg_dgl, deg[..., None], lg_n_node_valid.to(device))
            emb_edge_ouput_separate = self._emb_graph_branch_separate(g_dgl, padded_node_feats.to(device),
                                                                      max_node_ids_reshaped[valid_num_node_graph].to(
                                                                          device))
            emb_node_output_valid = emb_edge_ouput_separate[node_id_index_valid_cumsum]
            emb_graph_out_g = global_add_pool(emb_node_output_valid, batch)


            # emb_edge.scatter_add_(0, edge_indexes_final_for_node_embedding[...,None].expand((-1, emb_edge.size(-1))), lg_output)
            emb_graph_value = th.zeros((repeat_time, emb_graph_out_g.size(-1)),
                                       device=device)
            emb_graph_value[valid_num_node_graph] = emb_graph_out_g
            if self.use_filtrated:
                emb_graph_value = emb_graph_value.reshape(-1, edge_indexes.size(1), emb_graph_value.size(-1))
                emb_graph_value = emb_graph_value.sum(1) / ((num_nodes > 0).sum(1) + 1e-5)
                self._value_out = self._value_branch(emb_graph_value) / (
                        num_nodes[:, 0] * num_nodes[:, 0]) if self._value_branch is not None else None

            else:
                emb_graph_value = emb_graph_value.reshape(-1, emb_graph_value.size(-1))
                emb_graph_value = emb_graph_value
                #NB：don't do this: self._value_out = self._value_branch(emb_graph_value) / (
                        # num_nodes * num_nodes) if self._value_branch is not None else None
                # the resultant shape is (bs, bs)...not (bs, 1)
                self._value_out = self._value_branch(emb_graph_value) / (
                        num_nodes * num_nodes) if self._value_branch is not None else None
            self._context_value = emb_graph_value

        self.edge_index = edge_indexes
        self.edge_index_mask = edge_index_mask
        self.num_edges = num_edges
        self.emb_edge = emb_edge
        self.action_mask = action_mask
        self.emb_graphs_filtrated = emb_graphs_filtrated

        return self._context, state

    @override(TorchModelV2)
    def value_function(self):
        return torch.reshape(self._value_out, [-1])

    def obtain_graph_env(self):
        return self.emb_edge, self.emb_graphs_filtrated, self.action_mask, self.edge_index_mask, self.num_edges, self.edge_index





class ConditionedAttn(nn.Module):
    def __init__(self, hidden_size, using_emb_action_as_v=False):
        super(ConditionedAttn, self).__init__()
        if using_emb_action_as_v:  # using the previous action embedding as v
            self.v = None
        else:
            self.v = nn.Parameter(torch.rand(hidden_size))
            stdv = 1. / np.math.sqrt(self.v.size(0))
            self.v.data.normal_(mean=0, std=stdv)
        self.using_emb_action_as_v = using_emb_action_as_v
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)

    def forward(self, graph_hidden, condition_hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (B,T,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(1)
        # this_batch_size = encoder_outputs.size(1)
        graph_hidden = graph_hidden.repeat(max_len, 1, 1).transpose(0, 1)
        # encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(graph_hidden, condition_hidden, encoder_outputs)  # compute attention score
        return attn_energies


    def score(self, graph_hidden, condition_hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([graph_hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        if self.using_emb_action_as_v:
            energy = torch.bmm(condition_hidden.unsqueeze(1),energy) # [B*1*T]
        else:
            v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
            energy = torch.bmm(v, energy)  # [B*1*T]

        return energy.squeeze(1) #[B*T]








class TorchEdgeAutoregressiveDiscreteActionModel(TorchModelV2, nn.Module):
    """Policy network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        with_stop = kwargs["with_stop_action"]
        fc_size = kwargs["hidden_dim"]

        lstm_state_size = fc_size


        hidden_dim = fc_size  # hidden_size

        self.num_outputs = lstm_state_size  # action_space.spaces[0].n
        self.graph_hidden_size = hidden_dim


        self.feature_dim = fc_size
        self.inverse_net_hiddens = (self.feature_dim,)
        self.inverse_net_activation = "relu"
        self.forward_net_hiddens = (self.feature_dim,)
        self.forward_net_activation = "relu"
        self.action_dim = self.feature_dim

        # create models (this defines the pretrained action selection model)
        self._curiosity_feature_net = TorchEdgeAutoregressiveBaseModel(
            obs_space,
            self.action_space,
            num_outputs,
            model_config=model_config,
            name="feature_net",
        )

        vf_share_layers = self.model_config.get("vf_share_layers")
        if vf_share_layers:
            self.value_branch = None  # nn.Linear(self.feature_dim, 1)

        else:
            self.value_branch = None


        if with_stop:
            class _Action0Model(nn.Module):
                def __init__(self, is_direct_logits, similarity_func, use_graph_embedding, transfer_func, embnode_mlp):
                    nn.Module.__init__(self)

                    self.a0_hidden = nn.Sequential(
                        SlimFC(
                            in_size=hidden_dim,
                            out_size=hidden_dim,
                            activation_fn=nn.ReLU,
                            initializer=normc_init_torch(1.414)), )

                    self.a0_logits = SlimFC(
                        in_size=hidden_dim,
                        out_size=2,  # concat of [mean, std] of gaussion
                        activation_fn=None,
                        initializer=normc_init_torch(0.01))

                def forward(self_, ctx_input, emb_node):

                    a0_logits = self_.a0_logits(self_.a0_hidden(ctx_input))

                    return a0_logits

            class _Action1Model(nn.Module):
                def __init__(self, is_direct_logits, similarity_func, use_graph_embedding, transfer_func, embnode_mlp):
                    nn.Module.__init__(self)

                    self.a1_hidden = nn.Sequential(
                        SlimFC(
                            in_size=hidden_dim * 2,
                            out_size=hidden_dim,
                            activation_fn=nn.ReLU,
                            initializer=normc_init_torch(1.414)), )

                    self.a1_logits = SlimFC(
                        in_size=hidden_dim,
                        out_size=hidden_dim,  # concat of [mean, std] of gaussion
                        activation_fn=None,
                        initializer=normc_init_torch(0.01))

                def forward(self_, ctx_input, one_hot_head_action):

                    a1_logits = self_.a1_logits(self_.a1_hidden(torch.cat([ctx_input, one_hot_head_action], -1)))

                    return a1_logits

            class _Action2Model(nn.Module):
                def __init__(self, is_direct_logits, similarity_func, use_graph_embedding, transfer_func, embnode_mlp):
                    nn.Module.__init__(self)
                    self.embnode_mlp = embnode_mlp


                    self.a2_hidden = nn.Sequential(
                        SlimFC(
                            in_size=hidden_dim * 3,
                            out_size=hidden_dim,
                            activation_fn=nn.ReLU,
                            initializer=normc_init_torch(1.414), )

                    )

                    self.a2_logits = SlimFC(
                        in_size=hidden_dim,
                        out_size=hidden_dim,  # concat of [mean, std] of gaussion
                        activation_fn=None,
                        initializer=normc_init_torch(0.01))

                def forward(self_, ctx_input, one_hot_head_action, a1_input):

                    a2_hidden = self_.a2_hidden(torch.cat([ctx_input, a1_input, one_hot_head_action], dim=-1))
                    a2_logits = self_.a2_logits(a2_hidden)

                    return a2_logits
        else:
            class _Action1Model(nn.Module):
                def __init__(self, is_direct_logits, similarity_func, use_graph_embedding, transfer_func, embnode_mlp):
                    nn.Module.__init__(self)

                    self.a1_hidden = nn.Sequential(
                        SlimFC(
                            in_size=hidden_dim,
                            out_size=hidden_dim,
                            activation_fn=nn.ReLU,

                            initializer=normc_init_torch(1)), )

                    self.a1_logits = SlimFC(
                        in_size=hidden_dim,
                        out_size=num_outputs,  # concat of [mean, std] of gaussion
                        activation_fn=None,
                        initializer=normc_init_torch(0.01))

                def forward(self_, ctx_input):
                    a1_logits = self_.a1_logits(self_.a1_hidden(ctx_input))
                    return a1_logits



            class _Action2Model(nn.Module):
                def __init__(self, is_direct_logits, similarity_func, use_graph_embedding, transfer_func, embnode_mlp):
                    nn.Module.__init__(self)
                    self.embnode_mlp = embnode_mlp


                    self.a2_hidden = nn.Sequential(
                        SlimFC(
                            in_size=hidden_dim * 2,
                            out_size=hidden_dim,
                            activation_fn=nn.Tanh,
                            initializer=normc_init_torch(1), )

                    )

                    self.a2_logits = SlimFC(
                        in_size=hidden_dim,
                        out_size=num_outputs,  # concat of [mean, std] of gaussion
                        activation_fn=None,
                        initializer=normc_init_torch(0.01))

                def forward(self_, ctx_input, a1_input):

                    a2_hidden = self_.a2_hidden(torch.cat([ctx_input, a1_input], dim=-1))


                    a2_logits = self_.a2_logits(a2_hidden)

                    # # select action
                    # a2_logits = self_.select_action_func(a2_context, a1_input, attn_output)
                    return a2_logits


        is_direct_logits = model_config['custom_model_config']['direct_logits']
        similarity_func = model_config['custom_model_config']['similarity_func']
        use_graph_embedding = model_config['custom_model_config']['use_graph_embedding']
        transfer_func = model_config['custom_model_config']['transfer_func']
        self.stop_action_embedding = nn.Embedding(2, hidden_dim)

        if with_stop:
            self.logit_branch_action0 = _Action0Model(is_direct_logits=is_direct_logits,
                                                      similarity_func=similarity_func,
                                                      use_graph_embedding=use_graph_embedding,
                                                      transfer_func=transfer_func,
                                                      embnode_mlp=None)
        self.logit_branch_action1 = _Action1Model(is_direct_logits=is_direct_logits, similarity_func=similarity_func,
                                                  use_graph_embedding=use_graph_embedding, transfer_func=transfer_func,
                                                  embnode_mlp=None)
        self.logit_branch_action2 = _Action2Model(is_direct_logits=is_direct_logits, similarity_func=similarity_func,
                                                  use_graph_embedding=use_graph_embedding, transfer_func=transfer_func,
                                                  embnode_mlp=None)

        self.action_heads_mask = torch.as_tensor([[0, 0], [1, 1]])
        # self._curiosity_feature_net.train(False)
        self.N = 15
        self.max_edge = 54
        # self.bn = nn.BatchNorm1d(self.num_outputs, affine=True)
        self.view_requirements.update(self._curiosity_feature_net.view_requirements)


        self._context = None
        self.value_context = None
        self.graph_mat, self.n_feats,self.ego_n_feats, self.ego_edge_indexes, self.node_id_indexes, self.node_label_indexes, self.num_edges, self.ego_num_edges = [None] * 8

        self.emb_edge = None

    def _create_fc_net(self, layer_dims, activation, name=None):
        """Given a list of layer dimensions (incl. input-dim), creates FC-net.

        Args:
            layer_dims (Tuple[int]): Tuple of layer dims, including the input
                dimension.
            activation (str): An activation specifier string (e.g. "relu").

        Examples:
            If layer_dims is [4,8,6] we'll have a two layer net: 4->8 (8 nodes)
            and 8->6 (6 nodes), where the second layer (6 nodes) does not have
            an activation anymore. 4 is the input dimension.
        """
        layers = []

        for i in range(len(layer_dims) - 1):
            act = activation if i < len(layer_dims) - 2 else None
            if self.framework == "torch":
                layers.append(
                    SlimFC(
                        in_size=layer_dims[i],
                        out_size=layer_dims[i + 1],
                        initializer=torch.nn.init.xavier_uniform_,
                        activation_fn=act))

        return nn.Sequential(*layers)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):

        context, state = self._curiosity_feature_net.forward(input_dict, state, seq_lens)

        self._context = context

        return self._context, state

    @override(ModelV2)
    def value_function(self):
        assert self._context is not None
        if self.value_branch is not None:
            return torch.reshape(self.value_branch(self._context), [-1])
        else:
            return self._curiosity_feature_net.value_function()


