from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, TorchDistributionWrapper
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType, List
from ray.rllib.examples.env.get_mask import create_logits_mask_by_second_edge_graph
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
import numpy as np



def batch_index_select(input, dim, index):
    # input: bs * N * d
    # index: bs * 1
    views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)

    return torch.gather(input, dim, index)



class TorchGraphEdgeEmbDistribution_SP_with_stop(TorchDistributionWrapper):
    """Action distribution P(a1, a2, a3, a4) = P(a1) * P(a2 | a1) * P(a3 |a1, a2) * P(a4 | a1, a2,a3)"""

    @override(ActionDistribution)
    def __init__(self, inputs: List[TensorType], model: TorchModelV2, emb_edge=None,
                 emb_graphs_filtrated=None, action_mask=None, edge_index_mask=None, num_edges=None, edge_index=None,
                 actions=None):
        # If inputs are not a torch Tensor, make them one and make sure they
        # are on the correct device.
        if not isinstance(inputs, torch.Tensor):
            # print('converting i')
            inputs = torch.from_numpy(inputs)
            if isinstance(model, TorchModelV2):
                inputs = inputs.to(next(model.parameters()).device)
        super().__init__(inputs, model)
        # Store the last sample here.
        self.last_sample = None
        # self.max_N = 15
        self.using_emb_estimation = False
        self.use_embedding_ascontext = True
        self.action_heads_mask = self.model.action_heads_mask.to(next(model.parameters()).device)  # if the action_head_0 sampled value is 0, means stop
        # mask out the last two actions
        if emb_edge is None:

            emb_edge, emb_graphs_filtrated, action_mask, edge_index_mask, num_edges, edge_index = model._curiosity_feature_net.obtain_graph_env()
        # store the emb_nodes and rawobs here
        # self.rawobs = rawobs

        if len(emb_graphs_filtrated.shape) == 3:
            self.use_filtration = True
        else:
            self.use_filtration = False

        # assert self.use_filtraion
        self.emb_edge = emb_edge  # .clone()
        self.edge_index_mask = edge_index_mask
        self.num_edges = num_edges
        self.edge_index = edge_index

        self.emb_graphs_filtrated = emb_graphs_filtrated
        self.action_mask = action_mask
        self.block_subaction = False

        self._a0_logits = None
        self._a1_logits = None
        self._a2_logits = None

        # do sampling here to back up current model (sgd would change the model,
        # so the pre_dist(logits, model) is wrong, kl can be inf)

        if actions is None:
            self._a0_dist = self._a0_distribution(saving=True)
            self._node_0_idx = self._a0_dist.sample()
            self._a1_dist = self._a1_distribution(self._node_0_idx, saving=True)
            self._node_1_idx = self._a1_dist.sample()
            self._a2_dist = self._a2_distribution(self._node_0_idx, self._node_1_idx, saving=True)
            self.node_2_idx = self._a2_dist.sample()
        else:
            node_0_idx, node_1_idx, node_2_idx = torch.chunk(actions.long(), 3, dim=1)

            node_0_idx = node_0_idx.squeeze(-1)
            node_1_idx = node_1_idx.squeeze(-1)
            node_2_idx = node_2_idx.squeeze(-1)
            self._a0_dist = self._a0_distribution()
            self._node_0_idx = node_0_idx
            self._a1_dist = self._a1_distribution(self._node_0_idx)
            self._node_1_idx = node_1_idx
            self._a2_dist = self._a2_distribution(self._node_0_idx, self._node_1_idx)
            self.node_2_idx = node_2_idx

    def deterministic_sample(self):
        # First, sample a1.

        a0_dist = self._a0_distribution(deterministic=True, saving=True)
        node_0_idx = a0_dist.deterministic_sample()

        a1_dist = self._a1_distribution(node_0_idx, deterministic=True, saving=True)
        node_1_idx = a1_dist.deterministic_sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(node_0_idx, node_1_idx, deterministic=True, saving=True)
        node_2_idx = a2_dist.deterministic_sample()

        #  # Sample a3 conditioned on a2, a1.

        # stop action
        # a3_dist = self._a3_distribution()
        # node_3_idx = a3_dist.deterministic_sample()
        # node_5_idx = torch.argmax(a5_vec, dim=1, keepdim=True)
        if self.block_subaction:
            actionhead_mask_expand = batch_index_select(self.action_heads_mask[None].expand(node_1_idx.size(0), 2, 2),
                                                        1,
                                                        node_0_idx).squeeze(1)
            self._action_logp = a0_dist.logp(node_0_idx) + actionhead_mask_expand[:, 0] * a1_dist.logp(node_1_idx) \
                                + actionhead_mask_expand[:, 1] * a2_dist.logp(node_2_idx)  # + a3_dist.logp(node_3_idx)
            # del actionhead_mask_expand
        else:
            self._action_logp = a0_dist.logp(node_0_idx) + a1_dist.logp(node_1_idx) \
                                + a2_dist.logp(node_2_idx)


        return torch.stack((node_0_idx, node_1_idx, node_2_idx), dim=-1)

    def sample(self):


        a0_dist = self._a0_dist
        node_0_idx = self._node_0_idx
        # First, sample a1.

        a1_dist = self._a1_dist
        node_1_idx = self._node_1_idx

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_dist
        node_2_idx = self.node_2_idx
        #  # Sample a3 conditioned on a2, a1.

        # stop action
        # a3_dist = self._a3_distribution()
        # node_3_idx = a3_dist.sample()

        if self.block_subaction:
            actionhead_mask_expand = batch_index_select(self.action_heads_mask[None].expand(node_1_idx.size(0), 2, 2),
                                                        1,
                                                        node_0_idx).squeeze(1)
            self._action_logp = a0_dist.logp(node_0_idx) + actionhead_mask_expand[:, 0] * a1_dist.logp(node_1_idx) \
                                + actionhead_mask_expand[:, 1] * a2_dist.logp(node_2_idx)  # + a3_dist.logp(node_3_idx)
            del actionhead_mask_expand
        else:
            self._action_logp = a0_dist.logp(node_0_idx) + a1_dist.logp(node_1_idx) + a2_dist.logp(
                node_2_idx)  # + a3_dist.logp(node_3_idx)


        return torch.stack((node_0_idx, node_1_idx, node_2_idx), dim=-1)

    #
    def logp(self, actions):

        node_0_idx, node_1_idx, node_2_idx = torch.chunk(actions.long(), 3, dim=1)

        node_0_idx = node_0_idx.squeeze(-1)
        node_1_idx = node_1_idx.squeeze(-1)
        node_2_idx = node_2_idx.squeeze(-1)

        a0_dist = self._a0_dist
        a1_dist = self._a1_dist
        a2_dist = self._a2_dist

        if self.block_subaction:
            actionhead_mask_expand = batch_index_select(self.action_heads_mask[None].expand(node_1_idx.size(0), 2, 2),
                                                        1,
                                                        node_0_idx).squeeze(1)

            logp = a0_dist.logp(node_0_idx) + \
                   actionhead_mask_expand[:, 0] * a1_dist.logp(node_1_idx) + \
                   actionhead_mask_expand[:, 1] * a2_dist.logp(node_2_idx)
            # del actionhead_mask_expand
            return logp
        else:
            logp = a0_dist.logp(node_0_idx) + a1_dist.logp(node_1_idx) + a2_dist.logp(node_2_idx)
        return logp  # +


    def sampled_action_logp(self):
        return self._action_logp
        # return torch.exp(self._action_logp)

    def entropy(self, actions):

        a0_dist = self._a0_dist

        node_0_idx, node_1_idx, node_2_idx = torch.chunk(actions.long(), 3, dim=1)
        node_0_idx = node_0_idx.squeeze(-1)
        node_1_idx = node_1_idx.squeeze(-1)
        # node_2_idx = node_2_idx.squeeze(-1)
        # node_0_idx = a0_dist.sample()

        # First, sample a1.

        a1_dist = self._a1_dist #self._a1_distribution(node_0_idx)
        # node_1_idx = a1_dist.sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_dist # self._a2_distribution(node_0_idx, node_1_idx)
        # node_2_idx = a2_dist.sample()

        #  # Sample a3 conditioned on a2, a1.

        # # stop action
        # a3_dist = self._a3_distribution()
        if self.block_subaction:
            actionhead_mask_expand = batch_index_select(self.action_heads_mask[None].expand(node_1_idx.size(0), 2, 2),
                                                        1,
                                                        node_0_idx).squeeze(1)
            total_entropy = a0_dist.entropy() + actionhead_mask_expand[:, 0] * a1_dist.entropy() \
                                + actionhead_mask_expand[:, 1] * a2_dist.entropy()  # + a3_dist.logp(node_3_idx)
            # del actionhead_mask_expand
            return total_entropy
        else:
            return a0_dist.entropy() + a1_dist.entropy() + a2_dist.entropy()  # + a3_dist.entropy()  #todo: multi mask or not

    def kl(self, other, actions):

        node_0_idx, node_1_idx, node_2_idx = torch.chunk(actions.long(), 3, dim=1)
        node_0_idx = node_0_idx.squeeze(-1)
        node_1_idx = node_1_idx.squeeze(-1)

        a0_dist = self._a0_distribution()
        # node_0_idx = a0_dist.sample()

        # First, sample a1.

        a1_dist = self._a1_distribution(node_0_idx)
        # node_1_idx = a1_dist.sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(node_0_idx, node_1_idx)
        # node_2_idx = a2_dist.sample()

        #  # Sample a3 conditioned on a2, a1.ff

        # stop action
        # a3_dist = self._a3_distribution()
        a0_dist_other = other._a0_distribution()
        a1_dist_other = other._a1_distribution(node_0_idx)
        a2_dist_other = other._a2_distribution(node_0_idx, node_1_idx)

        a0_terms = a0_dist.kl(a0_dist_other)
        a1_terms = a1_dist.kl(a1_dist_other)
        a2_terms = a2_dist.kl(a2_dist_other)

        # a3_terms = a3_dist.kl(other._a3_distribution())
        if self.block_subaction:
            actionhead_mask_expand = batch_index_select(self.action_heads_mask[None].expand(node_1_idx.size(0), 2, 2),
                                                        1,
                                                        node_0_idx).squeeze(1)
            total_kl = a0_terms + actionhead_mask_expand[:, 0] * a1_terms \
                            + actionhead_mask_expand[:, 1] * a2_terms  # + a3_dist.logp(node_3_idx)
            del actionhead_mask_expand
            return total_kl
        else:
            total_kl = a0_terms + a1_terms + a2_terms  # + a3_terms

            return total_kl


    def _a0_distribution(self, deterministic=False, saving=False):

        if self._a0_logits is None or deterministic:


            a0_intent_vector = self.model.logit_branch_action0(self.inputs,
                                                               self.emb_edge)
            similarity_node_0 = a0_intent_vector

            a0_logits = similarity_node_0
            # try:
            a0_dist = TorchCategorical(a0_logits)
            if saving:
                self._a0_dist = a0_dist
                self._a0_logits = a0_logits
            return a0_dist
        else:
            return self._a0_dist



    def _a1_distribution(self, node_0_idx, deterministic=False, saving=False):

        if self._a1_logits is None or deterministic:

            one_hot_head_action = self.model.stop_action_embedding(node_0_idx)

            valid_mask = self.action_mask

            a1_intent_vector = self.model.logit_branch_action1(self.inputs,
                                                               one_hot_head_action)

            a1_edge_logits = self.model._curiosity_feature_net.action1_module(a1_intent_vector,
                                                                              self.emb_edge,
                                                                              self.emb_graphs_filtrated,
                                                                              self.edge_index_mask
                                                                              )

            similarity_node_1 = a1_edge_logits
            similarity_node_1 = similarity_node_1 + torch.clamp(torch.log(valid_mask),
                                                                min=FLOAT_MIN)

            a1_logits = similarity_node_1
            # try:
            a1_dist = TorchCategorical(a1_logits)
            if saving:
                self._a1_logits = a1_logits
                self._a1_dist = a1_dist
            # del valid_mask
            return a1_dist
        else:
            return self._a1_dist


    def _a2_distribution(self, *action, deterministic=False, saving=False):

        if self._a2_logits is None or deterministic:
            # split action into four nodes
            node_0_idx, node_1_idx = action
            # one_hot_head_action = torch.zeros(self.inputs.size(0), self.emb_edge.size(-1),
            #                                   device=node_0_idx.device)

            one_hot_head_action = self.model.stop_action_embedding(node_0_idx)
            # if self.use_embedding_ascontext:
            if self.use_filtration:
                emb_node_1 = batch_index_select(self.emb_edge[:, 0, :, :], 1, node_1_idx.long()).squeeze(1)
            else:
                emb_node_1 = batch_index_select(self.emb_edge, 1, node_1_idx.long()).squeeze(1)
            a1_vec = emb_node_1

            a2_intent_vector = self.model.logit_branch_action2(self.inputs,
                                                               one_hot_head_action,
                                                               a1_vec)


            a2_edge_logits = self.model._curiosity_feature_net.action2_module(a2_intent_vector,
                                                                              self.emb_edge, a1_vec,
                                                                              self.emb_graphs_filtrated,
                                                                              self.edge_index_mask
                                                                              )

            similarity_node_2 = a2_edge_logits

            if self.use_filtration:
                valid_mask_2 = create_logits_mask_by_second_edge_graph(self.edge_index[:, 0, :, :].cpu().long().numpy(),
                                                                       self.num_edges[:, 0, 0].cpu().numpy(),
                                                                       self.emb_edge.size(2), node_1_idx.cpu().numpy())
            else:
                valid_mask_2 = create_logits_mask_by_second_edge_graph(self.edge_index.cpu().long().numpy(),
                                                                       self.num_edges[:, 0].cpu().numpy(),
                                                                       self.emb_edge.size(1), node_1_idx.cpu().numpy())

            similarity_node_2 = similarity_node_2 + torch.clamp(torch.log(torch.from_numpy(valid_mask_2).float().to(similarity_node_2.device)),
                                                                min=FLOAT_MIN)
            a2_logits = similarity_node_2

            a2_dist = TorchCategorical(a2_logits)
            if saving:
                self._a2_logits = a2_logits
                self._a2_dist = a2_dist

            return a2_dist
        else:
            return self._a2_dist



    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config['custom_model_config']['hidden_dim']  # controls model output feature vector size

