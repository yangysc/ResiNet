"""
PyTorch policy class used for PPO.
"""
import gym
import logging
from typing import Dict, List, Type, Union, Callable, Optional, Set
# from typing import Any, Callable,, Optional, Tuple, \

from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.examples.models.autoregressive_action_dist import batch_index_select
import ray
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, \
    LearningRateSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    explained_variance, sequence_mask
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.utils import force_list

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


def ppo_surrogate_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # exit(-1)
    logits, state = model(train_batch)
    # try:
    curr_action_dist = dist_class(logits, model, actions=train_batch[SampleBatch.ACTIONS])
    # except:
    #     curr_action_dist = dist_class(logits, model, actions=train_batch[SampleBatch.ACTIONS])
    # curr_action_dist_same_action_for_kl = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch["seq_lens"])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch["seq_lens"],
            max_seq_len,
            time_major=model.is_time_major())
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    # prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS],
    #                               model,
    #
    #                               train_batch['emb_edge'],
    #                               # train_batch['num_edges'],
    #                               # train_batch['edge_index_mask'],
    #                               # train_batch['edge_index'],
    #                               # train_batch['lg_node_feats'],
    #                               # train_batch['lg_edge_index'],
    #                               train_batch['emb_graphs_filtrated'],
    #                               train_batch['action_mask'],
    #                               train_batch['edge_index_mask']
    #                               )

    # ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS].long()) -
    #     train_batch[SampleBatch.ACTION_LOGP])
    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS].long()) -
    #     train_batch[SampleBatch.ACTION_LOGP])
    log_ratio = curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) - train_batch[SampleBatch.ACTION_LOGP]  # NB: log error
    # log_ratio_not_same_dist = curr_action_dist_same_action_for_kl.logp(train_batch[SampleBatch.ACTIONS].long()) - train_batch[SampleBatch.ACTION_LOGP]  # NB: log error
    ratio = torch.exp(log_ratio)

    # calculate kl divergence
    pre_dist_a0 = TorchCategorical(train_batch['a0_logits'])
    pre_dist_a1 = TorchCategorical(train_batch['a1_logits'])
    pre_dist_a2 = TorchCategorical(train_batch['a2_logits'])
    # node_0_idx, node_1_idx, node_2_idx = torch.chunk(train_batch[SampleBatch.ACTIONS].long(), 3, dim=1)
    approx_kl = pre_dist_a0.kl(curr_action_dist._a0_dist) + pre_dist_a1.kl(curr_action_dist._a1_dist) + pre_dist_a2.kl(curr_action_dist._a2_dist)  # todo: block kl
    # actionhead_mask_expand = batch_index_select(model.action_heads_mask[None].to(ratio.device).expand(node_1_idx.size(0), 2, 2),
    #                                             1,
    #                                             node_0_idx).squeeze(1)
    # actionhead_mask_expand = actionhead_mask_expand.to(ratio.device)
    # approx_kl = pre_dist_a0.kl(curr_action_dist._a0_dist) + \
    #             actionhead_mask_expand[:, 0] * pre_dist_a1.kl(curr_action_dist._a1_dist) \
    #             + actionhead_mask_expand[:, 1] * pre_dist_a2.kl(curr_action_dist._a2_dist) #todo: block kl
    # torch.distributions.kl.kl_divergence(pre_dist_a0, curr_action_dist._a0_dist)
    # approx_kl = prev_action_dist.kl(curr_action_dist, train_batch[SampleBatch.ACTIONS].long())
    # approx_kl = torch.mean(ratio - 1 - log_ratio)
    mean_kl_loss = reduce_mean_valid(approx_kl) # numpy overflows if kl has inf
    # surrogate_loss = torch.where(
    #     (approx_kl >= policy_kl_range) & (ratio > 1),
    #     ratio * train_batch[Postprocessing.ADVANTAGES] - policy_params * approx_kl,
    #     ratio * train_batch[Postprocessing.ADVANTAGES]
    # )
    # mean_policy_loss = reduce_mean_valid(-surrogate_loss)
    # if torch.isinf(mean_kl):
    #     print('kl inf')
    curr_entropy = curr_action_dist.entropy(train_batch[SampleBatch.ACTIONS])
    mean_entropy = reduce_mean_valid(curr_entropy)

    # surrogate_loss = torch.min(
    #     train_batch[Postprocessing.ADVANTAGES] * ratio,
    #     train_batch[Postprocessing.ADVANTAGES] * torch.clamp(
    #         ratio, 1 - policy.config["clip_param"],
    #         1 + policy.config["clip_param"]))
    if policy.config["ppo_alg"] == 'ppo':
        mean_policy_loss = -torch.min(
            train_batch[Postprocessing.ADVANTAGES] * ratio,
            train_batch[Postprocessing.ADVANTAGES] * torch.clamp(
                ratio, 1 - policy.config["clip_param"],
                       1 + policy.config["clip_param"]))
    elif policy.config["ppo_alg"] == 'dcppo':

        dual_clip = policy.config["dual_clip_param"]
        clip1 = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * ratio,
            train_batch[Postprocessing.ADVANTAGES] * torch.clamp(
                ratio, 1 - policy.config["clip_param"],
                       1 + policy.config["clip_param"]))
        clip2 = torch.max(clip1, dual_clip * train_batch[Postprocessing.ADVANTAGES])
        mean_policy_loss = -torch.where(train_batch[Postprocessing.ADVANTAGES] < 0, clip2, clip1)
    elif policy.config["ppo_alg"] == 'tppo':
        policy_params = 20 # 20 for discrete
        policy_kl_range = 0.0008
        mean_policy_loss = -torch.where(
            (approx_kl >= policy_kl_range) & (ratio > 1),
            ratio * train_batch[Postprocessing.ADVANTAGES] - policy_params * approx_kl,
            ratio * train_batch[Postprocessing.ADVANTAGES]
        )


    surrogate_loss = mean_policy_loss
    mean_policy_loss = reduce_mean_valid(mean_policy_loss)
    # Compute a value function loss.
    if policy.config["use_critic"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]
        value_fn_out = model.value_function()
        vf_loss1 = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_clipped = prev_value_fn_out + torch.clamp(
            value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"])
        vf_loss2 = torch.pow(
            vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss = torch.max(vf_loss1, vf_loss2)
        mean_vf_loss = reduce_mean_valid(vf_loss)

        # value_fn_out = model.value_function()
        # vf_loss1 = torch.pow(
        #     value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        #
        # vf_loss = torch.clamp(
        #     vf_loss1, -policy.config["vf_clip_param"],
        #     policy.config["vf_clip_param"])
        #
        # mean_vf_loss = reduce_mean_valid(vf_loss)
    # Ignore the value function.
    else:
        vf_loss = mean_vf_loss = 0.0


    total_loss = reduce_mean_valid(surrogate_loss +
                                   # policy.kl_coeff * mean_kl +
                                    -
                                   policy.entropy_coeff * curr_entropy)
    # Optional vf loss (or in a separate term due to separate
    # optimizers/networks).
    loss_wo_vf = total_loss
    if not policy.config["_separate_vf_optimizer"]:
        total_loss += policy.config["vf_loss_coeff"] * mean_vf_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], model.value_function())

    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss
    if policy.config["_separate_vf_optimizer"]:
        return loss_wo_vf, mean_vf_loss
    else:
        return total_loss






def kl_and_loss_stats(policy: Policy,
                      train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Stats function for PPO. Returns a dict with important KL and loss stats.

    Args:
        policy (Policy): The Policy to generate stats for.
        train_batch (SampleBatch): The SampleBatch (already) used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """
    return {
        "cur_kl_coeff": policy.kl_coeff,
        "cur_lr": policy.cur_lr,
        "total_loss": torch.mean(
            torch.stack(policy.get_tower_stats("total_loss"))),
        "policy_loss": torch.mean(
            torch.stack(policy.get_tower_stats("mean_policy_loss"))),
        "vf_loss": torch.mean(
            torch.stack(policy.get_tower_stats("mean_vf_loss"))),
        "vf_explained_var": torch.mean(
            torch.stack(policy.get_tower_stats("vf_explained_var"))),
        "kl": torch.mean(torch.stack(policy.get_tower_stats("mean_kl_loss"))),
        "entropy": torch.mean(
            torch.stack(policy.get_tower_stats("mean_entropy"))),
        "entropy_coeff": policy.entropy_coeff,
    }


def vf_preds_fetches(
        policy: Policy, input_dict: Dict[str, TensorType],
        state_batches: List[TensorType], model: ModelV2,
        action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:
    """Defines extra fetches per action computation.

    Args:
        policy (Policy): The Policy to perform the extra action fetch on.
        input_dict (Dict[str, TensorType]): The input dict used for the action
            computing forward pass.
        state_batches (List[TensorType]): List of state tensors (empty for
            non-RNNs).
        model (ModelV2): The Model object of the Policy.
        action_dist (TorchDistributionWrapper): The instantiated distribution
            object, resulting from the model's outputs and the given
            distribution class.

    Returns:
        Dict[str, TensorType]: Dict with extra tf fetches to perform per
            action computation.
    """
    # Return value function outputs. VF estimates will hence be added to the
    # SampleBatches produced by the sampler(s) to generate the train batches
    # going into the loss function.
    # emb_edge, num_edges, edge_index_mask, edge_index, lg_node_feats, lg_edge_index, emb_graphs_filtrated, action_mask = model._curiosity_feature_net.obtain_graph_env()
    emb_edge, emb_graphs_filtrated, action_mask, edge_index_mask, num_edges, edge_index = model._curiosity_feature_net.obtain_graph_env()
    vf_preds = model.value_function()
    return {
        SampleBatch.VF_PREDS: vf_preds,
        "edge_index_mask": edge_index_mask,
        "edge_index": edge_index,
        "emb_edge": emb_edge,
        # 'node_feats': node_feats,
        # "node_id_index": node_id_index,
        "num_edges": num_edges,
        # "num_nodes": num_nodes,
        # "max_node_id": max_node_id,
        # "edge_index2pos": edge_index2pos,
        "action_mask": action_mask,
        # "lg_node_feats": lg_node_feats,
        # "lg_edge_index": lg_edge_index,
        "emb_graphs_filtrated": emb_graphs_filtrated,

        'a0_logits': action_dist._a0_logits,
        'a1_logits': action_dist._a1_logits,
        'a2_logits': action_dist._a2_logits,

    }
def vf_preds_fetches_withoud_saving(
        policy: Policy, input_dict: Dict[str, TensorType],
        state_batches: List[TensorType], model: ModelV2,
        action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:
    """Defines extra fetches per action computation.

    Args:
        policy (Policy): The Policy to perform the extra action fetch on.
        input_dict (Dict[str, TensorType]): The input dict used for the action
            computing forward pass.
        state_batches (List[TensorType]): List of state tensors (empty for
            non-RNNs).
        model (ModelV2): The Model object of the Policy.
        action_dist (TorchDistributionWrapper): The instantiated distribution
            object, resulting from the model's outputs and the given
            distribution class.

    Returns:
        Dict[str, TensorType]: Dict with extra tf fetches to perform per
            action computation.
    """
    # Return value function outputs. VF estimates will hence be added to the
    # SampleBatches produced by the sampler(s) to generate the train batches
    # going into the loss function.
    # emb_edge, num_edges, edge_index_mask, edge_index, lg_node_feats, lg_edge_index, emb_graphs_filtrated, action_mask = model._curiosity_feature_net.obtain_graph_env()
    edge_index, node_feats, node_id_index, num_edges, num_nodes, \
    max_node_id, edge_index2pos, action_mask = model._curiosity_feature_net.obtain_graph_env()
    vf_preds = model.value_function()
    return {
        SampleBatch.VF_PREDS: vf_preds,
        # "rawobs": rawobs,
        "edge_index": edge_index,
        'node_feats': node_feats,
        "node_id_index": node_id_index,
        "num_edges": num_edges,
        "num_nodes": num_nodes,
        "max_node_id": max_node_id,
        "edge_index2pos": edge_index2pos,
        "action_mask": action_mask,
        # "lg_node_feats": lg_node_feats,
        # "lg_edge_index": lg_edge_index,
        # "emb_graphs_filtrated": emb_graphs_filtrated,

        'a0_logits': action_dist._a0_logits,
        'a1_logits': action_dist._a1_logits,
        'a2_logits': action_dist._a2_logits,

    }

class KLCoeffMixin:
    """Assigns the `update_kl()` method to the PPOPolicy.

    This is used in PPO's execution plan (see ppo.py) for updating the KL
    coefficient after each learning step based on `config.kl_target` and
    the measured KL value (from the train_batch).
    """

    def __init__(self, config):
        # The current KL value (as python float).
        self.kl_coeff = config["kl_coeff"]
        # Constant target value.
        self.kl_target = config["kl_target"]

    def update_kl(self, sampled_kl):
        # Update the current KL value based on the recently measured value.
        # if sampled_kl > 2.0 * self.kl_target:
        #     self.kl_coeff *= 1.5
        # elif sampled_kl < 0.5 * self.kl_target:
        #     self.kl_coeff *= 0.5
        # Return the current KL value.
        if sampled_kl > 1.5 * self.kl_target:
            self.kl_coeff *= 2.0
        elif sampled_kl < 0.66666 * self.kl_target:  # (paper says: "sampled_kl < self.kl_target / 1.5")
            self.kl_coeff *= 0.5
        return self.kl_coeff


class ValueNetworkMixin:
    """Assigns the `_value()` method to the PPOPolicy.

    This way, Policy can call `_value()` to get the current VF estimate on a
    single(!) observation (as done in `postprocess_trajectory_fn`).
    Note: When doing this, an actual forward pass is being performed.
    This is different from only calling `model.value_function()`, where
    the result of the most recent forward pass is being used to return an
    already calculated tensor.
    """

    def __init__(self, obs_space, action_space, config):
        # When doing GAE, we need the value function estimate on the
        # observation.
        if config["use_gae"]:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.

            def value(**input_dict):
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.value_function()[0].item()

        # When not doing GAE, we do not require the value function's output.
        else:

            def value(*args, **kwargs):
                return 0.0

        self._value = value



class Optimizer_fn_Mixin:
    def __init__(self):
        pass


def setup_mixins(policy: Policy, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 config: TrainerConfigDict) -> None:
    """Call all mixin classes' constructors before PPOPolicy initialization.

    Args:
        policy (Policy): The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config (TrainerConfigDict): The Policy's config.
    """
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    Optimizer_fn_Mixin.__init__(policy)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"], config["_lr_vf"], config["lr_vf_schedule"])
    # LearningRateSchedule.__init__(policy, config["_lr_vf"], config["lr_schedule"])


'_curiosity_feature_net.context_layer.'
def optimizer_fn(policy: Policy, config: TrainerConfigDict) -> torch.optim.Adam:
    # print("two optimizers")
    # exit(1)
    if config['_separate_vf_optimizer']:
        policy_weight = [item[1] for item in filter(lambda p: (p[0][:37] == '_curiosity_feature_net.context_layer.' or p[0][23:29] == 'action' or p[0][23:29] == 'merger' or p[0][:12] == 'logit_branch' or p[0][:4] == 'stop') and p[1].requires_grad, policy.model.named_parameters())]
        value_weight = [item[1] for item in filter(lambda p: (p[0][40:50] == '_separate.' or p[0][23:36] == '_value_branch') and p[1].requires_grad, policy.model.named_parameters())]
        optimizers = force_list([
            torch.optim.Adam(policy_weight, lr=config["lr"], eps=1e-5), torch.optim.Adam(value_weight, lr=config["_lr_vf"], eps=1e-5)])
        return optimizers
    else:
        return force_list([
            torch.optim.Adam(policy.model.parameters(), lr=config["lr"], eps=1e-5)])


# Build a child class of `TorchPolicy`, given the custom functions defined
# above.
PPOTorchPolicy = build_policy_class(
    name="PPOTorchPolicy",
    framework="torch",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    optimizer_fn=optimizer_fn,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=compute_gae_for_sample_batch,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin#,Optimizer_fn_Mixin
    ],
)
