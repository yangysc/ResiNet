from ray.rllib.policy.torch_policy import TorchPolicy
import numpy as np
from ray.rllib.utils.torch_ops import convert_to_non_torch_type, convert_to_torch_tensor
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from collections import deque
from .utils import *
import time
import copy
import functools
import gym
import logging
import math
import numpy as np
import os
import time
import threading
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union, \
    TYPE_CHECKING

torch, nn = try_import_torch()
import torch.distributions as td
from torch.cuda.amp import autocast, GradScaler
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.typing import ModelGradients, ModelWeights, TensorType, \
    TensorStructType, TrainerConfigDict
from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.sample_batch import SampleBatch


class CustomTorchPolicy(TorchPolicy):
    """Example of a random policy
    If you are using tensorflow/pytorch to build custom policies,
    you might find `build_tf_policy` and `build_torch_policy` to
    be useful.
    Adopted from examples from https://docs.ray.io/en/master/rllib-concepts.html
    """

    def __init__(  self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            config: TrainerConfigDict,
            *,
            model: ModelV2,
            loss: Callable[[
                Policy, ModelV2, Type[TorchDistributionWrapper], SampleBatch
            ], Union[TensorType, List[TensorType]]],
            action_distribution_class: Type[TorchDistributionWrapper],
            action_sampler_fn: Optional[Callable[[
                TensorType, List[TensorType]
            ], Tuple[TensorType, TensorType]]] = None,
            action_distribution_fn: Optional[Callable[[
                Policy, ModelV2, TensorType, TensorType, TensorType
            ], Tuple[TensorType, Type[TorchDistributionWrapper], List[
                TensorType]]]] = None,
            max_seq_len: int = 20,
            get_batch_divisibility_req: Optional[Callable[[Policy],
                                                          int]] = None,):





        TorchPolicy.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            config=config,
            model=model,
            loss=loss,
            action_distribution_class=action_distribution_class,
            action_sampler_fn=action_sampler_fn,
            action_distribution_fn=action_distribution_fn,
            max_seq_len=max_seq_len,
            get_batch_divisibility_req=get_batch_divisibility_req,
        )

        self.framework = "torch"

    def init_training(self):
        """ Init once only for the policy - Surely there should be a bette way to do this """
        aux_params = set(self.model.aux_vf.parameters())
        value_params = set(self.model.value_fc.parameters())
        network_params = set(self.model.parameters())
        aux_optim_params = list(network_params - value_params)
        ppo_optim_params = list(network_params - aux_params - value_params)
        if not self.config['single_optimizer']:
            self.optimizer = torch.optim.Adam(ppo_optim_params, lr=self.config['lr'],
                                              weight_decay=self.config['l2_reg'])
        else:
            self.optimizer = torch.optim.Adam(network_params, lr=self.config['lr'], weight_decay=self.config['l2_reg'])
        self.aux_optimizer = torch.optim.Adam(aux_optim_params, lr=self.config['aux_lr'],
                                              weight_decay=self.config['l2_reg'])
        self.value_optimizer = torch.optim.Adam(value_params, lr=self.config['value_lr'],
                                                weight_decay=self.config['l2_reg'])
        self.max_reward = self.config['env_config']['return_max']
        self.rewnorm = RewardNormalizer(cliprew=self.max_reward)  ## TODO: Might need to go to custom state
        self.reward_deque = deque(maxlen=100)
        self.best_reward = -np.inf
        self.best_weights = None
        self.time_elapsed = 0
        self.batch_end_time = time.time()
        self.timesteps_total = 0
        self.best_rew_tsteps = 0

        nw = self.config['num_workers'] if self.config['num_workers'] > 0 else 1
        nenvs = nw * self.config['num_envs_per_worker']
        nsteps = self.config['rollout_fragment_length']
        n_pi = self.config['n_pi']
        self.nbatch = nenvs * nsteps
        self.actual_batch_size = self.nbatch // self.config['updates_per_batch']
        self.accumulate_train_batches = int(np.ceil(self.actual_batch_size / self.config['max_minibatch_size']))
        self.mem_limited_batch_size = self.actual_batch_size // self.accumulate_train_batches
        if self.nbatch % self.actual_batch_size != 0 or self.nbatch % self.mem_limited_batch_size != 0:
            print("#################################################")
            print("WARNING: MEMORY LIMITED BATCHING NOT SET PROPERLY")
            print("#################################################")
        replay_shape = (n_pi, nsteps, nenvs)
        self.return_selector = Returnselector(nenvs, self.observation_space, self.action_space, replay_shape,
                                              skips=self.config['skips'],
                                              n_pi=n_pi,
                                              num_return=self.config['num_return'],
                                              flat_buffer=self.config['flattened_buffer'])
        self.save_success = 0
        self.target_timesteps = 8_000_000
        self.buffer_time = 20  # TODO: Could try to do a median or mean time step check instead
        self.max_time = self.config['max_time']
        self.maxrewep_lenbuf = deque(maxlen=100)
        self.gamma = self.config['gamma']
        self.adaptive_discount_tuner = AdaptiveDiscountTuner(self.gamma, momentum=0.98, eplenmult=3)

        self.lr = self.config['lr']
        self.ent_coef = self.config['entropy_coeff']

        self.last_dones = np.zeros((nw * self.config['num_envs_per_worker'],))
        self.make_distr = dist_build(self.action_space)
        self.return_completed = 0
        self.amp_scaler = GradScaler()

        self.update_lr()

    def to_tensor(self, arr):
        return torch.from_numpy(arr).to(self.device)

    @override(TorchPolicy)
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        return {'values': model._value.tolist()}

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        """Fused compute gradients and apply gradients call.
        Either this or the combination of compute/apply grads must be
        implemented by subclasses.
        Returns:
            grad_info: dictionary of extra metadata from compute_gradients().
        Examples:
            >>> batch = ev.sample()
            >>> ev.learn_on_batch(samples)
        Reference: https://github.com/ray-project/ray/blob/master/rllib/policy/policy.py#L279-L316
        """
        ## Config data values
        nbatch = self.nbatch
        nbatch_train = self.mem_limited_batch_size
        gamma, lam = self.gamma, self.config['lambda']
        nsteps = self.config['rollout_fragment_length']
        nenvs = nbatch // nsteps
        ts = (nenvs, nsteps)
        mb_dones = unroll(samples['dones'], ts)

        ## Reward Normalization - No reward norm works well for many envs
        if self.config['standardize_rewards']:
            mb_origrewards = unroll(samples['rewards'], ts)
            mb_rewards = np.zeros_like(mb_origrewards)
            mb_rewards[0] = self.rewnorm.normalize(mb_origrewards[0], self.last_dones, self.config["reset_returns"])
            for ii in range(1, nsteps):
                mb_rewards[ii] = self.rewnorm.normalize(mb_origrewards[ii], mb_dones[ii - 1],
                                                        self.config["reset_returns"])
            self.last_dones = mb_dones[-1]
        else:
            mb_rewards = unroll(samples['rewards'], ts)

        # Weird hack that helps in many envs (Yes keep it after reward normalization)
        rew_scale = self.config["scale_reward"]
        if rew_scale != 1.0:
            mb_rewards *= rew_scale

        should_skip_train_step = self.best_reward_model_select(samples)
        if should_skip_train_step:
            self.update_batch_time()
            return {}  # Not doing last optimization step - This is intentional due to noisy gradients

        obs = samples['obs']

        ## Value prediction
        next_obs = unroll(samples['new_obs'], ts)[-1]
        last_values, _ = self.model.vf_pi(next_obs, ret_numpy=True, no_grad=True, to_torch=True)
        values = samples['values']

        ## GAE
        mb_values = unroll(values, ts)
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextvalues = last_values
            else:
                nextvalues = mb_values[t + 1]
            nextnonterminal = 1.0 - mb_dones[t]
            delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        ## Data from config
        cliprange, vfcliprange = self.config['clip_param'], self.config['vf_clip_param']
        max_grad_norm = self.config['grad_clip']
        ent_coef, vf_coef = self.ent_coef, self.config['vf_loss_coeff']

        logp_actions = samples['action_logp']  ## np.isclose seems to be True always, otherwise compute again if needed
        noptepochs = self.config['num_sgd_iter']
        actions = samples['actions']
        returns = roll(mb_returns)

        advs = returns - values
        normalized_advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)

        ## Train multiple epochs
        optim_count = 0
        inds = np.arange(nbatch)
        for _ in range(noptepochs):
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (self.to_tensor(arr[mbinds]) for arr in
                          (obs, returns, actions, values, logp_actions, normalized_advs))
                optim_count += 1
                apply_grad = (optim_count % self.accumulate_train_batches) == 0
                self._batch_train(apply_grad, self.accumulate_train_batches,
                                  cliprange, vfcliprange, max_grad_norm, ent_coef, vf_coef, *slices)

        ## Distill with aux head
        should_return = self.return_selector.update(unroll(obs, ts), mb_returns)
        if should_return:
            self.aux_train()

        self.update_gamma(samples)
        self.update_lr()
        self.update_ent_coef()

        self.update_batch_time()
        return {}

    def update_batch_time(self):
        self.time_elapsed += time.time() - self.batch_end_time
        self.batch_end_time = time.time()

    def _batch_train(self, apply_grad, num_accumulate,
                     cliprange, vfcliprange, max_grad_norm,
                     ent_coef, vf_coef,
                     obs, returns, actions, values, logp_actions_old, advs):

        if not self.config['pi_phase_mixed_precision']:
            loss, vf_loss = self._calc_pi_vf_loss(apply_grad, num_accumulate,
                                                  cliprange, vfcliprange, max_grad_norm,
                                                  ent_coef, vf_coef,
                                                  obs, returns, actions, values, logp_actions_old, advs)

            loss.backward()
            vf_loss.backward()
            if apply_grad:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if not self.config['single_optimizer']:
                    self.value_optimizer.step()
                    self.value_optimizer.zero_grad()
        else:
            with autocast():
                loss, vf_loss = self._calc_pi_vf_loss(apply_grad, num_accumulate,
                                                      cliprange, vfcliprange, max_grad_norm,
                                                      ent_coef, vf_coef,
                                                      obs, returns, actions, values, logp_actions_old, advs)

            self.amp_scaler.scale(loss).backward(retain_graph=True)
            self.amp_scaler.scale(vf_loss).backward()

            if apply_grad:
                self.amp_scaler.step(self.optimizer)
                if not self.config['single_optimizer']:
                    self.amp_scaler.step(self.value_optimizer)
                self.amp_scaler.update()

                self.optimizer.zero_grad()
                if not self.config['single_optimizer']:
                    self.value_optimizer.zero_grad()

    def _calc_pi_vf_loss(self, apply_grad, num_accumulate,
                         cliprange, vfcliprange, max_grad_norm,
                         ent_coef, vf_coef,
                         obs, returns, actions, values, logp_actions_old, advs):

        vpred, pi_logits = self.model.vf_pi(obs, ret_numpy=False, no_grad=False, to_torch=False)
        pd = self.make_distr(pi_logits)
        logp_actions = pd.log_prob(actions[..., None]).squeeze(1)
        entropy = torch.mean(pd.entropy())

        vf_loss = .5 * torch.mean(torch.pow((vpred - returns), 2)) * vf_coef

        ratio = torch.exp(logp_actions - logp_actions_old)
        pg_losses1 = -advs * ratio
        pg_losses2 = -advs * torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
        pg_loss = torch.mean(torch.max(pg_losses1, pg_losses2))

        loss = pg_loss - entropy * ent_coef

        loss = loss / num_accumulate
        vf_loss = vf_loss / num_accumulate
        return loss, vf_loss

    def aux_train(self):
        nbatch_train = self.mem_limited_batch_size
        return_epochs = self.config['return_epochs']
        replay_shape = self.return_selector.vtarg_replay.shape
        replay_pi = np.empty((*replay_shape, self.return_selector.ac_space.n), dtype=np.float32)

        for nnpi in range(self.return_selector.n_pi):
            for ne in range(self.return_selector.nenvs):
                _, replay_pi[nnpi, :, ne] = self.model.vf_pi(self.return_selector.exp_replay[nnpi, :, ne],
                                                             ret_numpy=True, no_grad=True, to_torch=True)

        # Tune vf and pi heads to older predictions with (augmented?) observations
        num_accumulate = self.config['aux_num_accumulates']
        num_rollouts = self.config['aux_mbsize']
        for ep in range(return_epochs):
            counter = 0
            for slices in self.return_selector.make_minibatches(replay_pi, num_rollouts):
                counter += 1
                apply_grad = (counter % num_accumulate) == 0
                self.tune_policy(slices[0], self.to_tensor(slices[1]), self.to_tensor(slices[2]),
                                 apply_grad, num_accumulate)
        self.return_completed += 1
        self.return_selector.return_done()

    def tune_policy(self, obs, target_vf, target_pi, apply_grad, num_accumulate):
        if self.config['augment_buffer']:
            obs_aug = np.empty(obs.shape, obs.dtype)
            aug_idx = np.random.randint(self.config['augment_randint_num'], size=len(obs))
            obs_aug[aug_idx == 0] = pad_and_random_crop(obs[aug_idx == 0], 64, 10)
            obs_aug[aug_idx == 1] = random_cutout_color(obs[aug_idx == 1], 10, 30)
            obs_aug[aug_idx >= 2] = obs[aug_idx >= 2]
            obs_in = self.to_tensor(obs_aug)
        else:
            obs_in = self.to_tensor(obs)

        if not self.config['aux_phase_mixed_precision']:
            loss, vf_loss = self._aux_calc_loss(obs_in, target_vf, target_pi, num_accumulate)
            loss.backward()
            vf_loss.backward()

            if apply_grad:
                if not self.config['single_optimizer']:
                    self.aux_optimizer.step()
                    self.value_optimizer.step()
                else:
                    self.optimizer.step()


        else:
            with autocast():
                loss, vf_loss = self._aux_calc_loss(obs_in, target_vf, target_pi, num_accumulate)

            self.amp_scaler.scale(loss).backward(retain_graph=True)
            self.amp_scaler.scale(vf_loss).backward()

            if apply_grad:
                if not self.config['single_optimizer']:
                    self.amp_scaler.step(self.aux_optimizer)
                    self.amp_scaler.step(self.value_optimizer)
                else:
                    self.amp_scaler.step(self.optimizer)

                self.amp_scaler.update()

        if apply_grad:
            if not self.config['single_optimizer']:
                self.aux_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()

    def _aux_calc_loss(self, obs_in, target_vf, target_pi, num_accumulate):
        vpred, pi_logits = self.model.vf_pi(obs_in, ret_numpy=False, no_grad=False, to_torch=False)
        aux_vpred = self.model.aux_value_function()
        aux_loss = .5 * torch.mean(torch.pow(aux_vpred - target_vf, 2))

        target_pd = self.make_distr(target_pi)
        pd = self.make_distr(pi_logits)
        pi_loss = td.kl_divergence(target_pd, pd).mean()

        loss = pi_loss + aux_loss
        vf_loss = .5 * torch.mean(torch.pow(vpred - target_vf, 2))

        loss = loss / num_accumulate
        vf_loss = vf_loss / num_accumulate

        return loss, vf_loss

    def best_reward_model_select(self, samples):
        self.timesteps_total += len(samples['dones'])

        ## Best reward model selection
        eprews = [info['episode']['r'] for info in samples['infos'] if 'episode' in info]
        self.reward_deque.extend(eprews)
        mean_reward = safe_mean(eprews) if len(eprews) >= 100 else safe_mean(self.reward_deque)
        if self.best_reward < mean_reward:
            self.best_reward = mean_reward
            self.best_weights = self.get_weights()["current_weights"]
            self.best_rew_tsteps = self.timesteps_total

        if self.timesteps_total > self.target_timesteps or (self.time_elapsed + self.buffer_time) > self.max_time:
            if self.best_weights is not None:
                self.set_model_weights(self.best_weights)
                return True

        return False

    def update_lr(self):
        if self.config['lr_schedule'] == 'linear':
            self.lr = linear_schedule(initial_val=self.config['lr'],
                                      final_val=self.config['final_lr'],
                                      current_steps=self.timesteps_total,
                                      total_steps=self.target_timesteps)

        elif self.config['lr_schedule'] == 'exponential':
            self.lr = 0.997 * self.lr

        for g in self.optimizer.param_groups:
            g['lr'] = self.lr
        if self.config['same_lr_everywhere']:
            for g in self.value_optimizer.param_groups:
                g['lr'] = self.lr
            for g in self.aux_optimizer.param_groups:
                g['lr'] = self.lr

    def update_ent_coef(self):
        if self.config['entropy_schedule']:
            self.ent_coef = linear_schedule(initial_val=self.config['entropy_coeff'],
                                            final_val=self.config['final_entropy_coeff'],
                                            current_steps=self.timesteps_total,
                                            total_steps=self.target_timesteps)

    def update_gamma(self, samples):
        if self.config['adaptive_gamma']:
            epinfobuf = [info['episode'] for info in samples['infos'] if 'episode' in info]
            self.maxrewep_lenbuf.extend([epinfo['l'] for epinfo in epinfobuf if epinfo['r'] >= self.max_reward])
            sorted_nth = lambda buf, n: np.nan if len(buf) < 100 else sorted(self.maxrewep_lenbuf.copy())[n]
            target_horizon = sorted_nth(self.maxrewep_lenbuf, 80)
            self.gamma = self.adaptive_discount_tuner.update(target_horizon)

    def get_custom_state_vars(self):
        return {
            "time_elapsed": self.time_elapsed,
            "timesteps_total": self.timesteps_total,
            "best_weights": self.best_weights,
            "reward_deque": self.reward_deque,
            "batch_end_time": self.batch_end_time,
            "gamma": self.gamma,
            "maxrewep_lenbuf": self.maxrewep_lenbuf,
            "lr": self.lr,
            "ent_coef": self.ent_coef,
            "rewnorm": self.rewnorm,
            "best_rew_tsteps": self.best_rew_tsteps,
            "best_reward": self.best_reward,
            "last_dones": self.last_dones,
            "return_completed": self.return_completed,
        }

    def set_custom_state_vars(self, custom_state_vars):
        self.time_elapsed = custom_state_vars["time_elapsed"]
        self.timesteps_total = custom_state_vars["timesteps_total"]
        self.best_weights = custom_state_vars["best_weights"]
        self.reward_deque = custom_state_vars["reward_deque"]
        self.batch_end_time = custom_state_vars["batch_end_time"]
        self.gamma = self.adaptive_discount_tuner.gamma = custom_state_vars["gamma"]
        self.maxrewep_lenbuf = custom_state_vars["maxrewep_lenbuf"]
        self.lr = custom_state_vars["lr"]
        self.ent_coef = custom_state_vars["ent_coef"]
        self.rewnorm = custom_state_vars["rewnorm"]
        self.best_rew_tsteps = custom_state_vars["best_rew_tsteps"]
        self.best_reward = custom_state_vars["best_reward"]
        self.last_dones = custom_state_vars["last_dones"]
        self.return_completed = custom_state_vars["return_completed"]

    @override(TorchPolicy)
    def get_weights(self):
        weights = {}
        weights["current_weights"] = {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }
        return weights

    @override(TorchPolicy)
    def set_weights(self, weights):
        self.set_model_weights(weights["current_weights"])

    def set_optimizer_state(self, optimizer_state, aux_optimizer_state, value_optimizer_state, amp_scaler_state):
        optimizer_state = convert_to_torch_tensor(optimizer_state, device=self.device)
        self.optimizer.load_state_dict(optimizer_state)

        aux_optimizer_state = convert_to_torch_tensor(aux_optimizer_state, device=self.device)
        self.aux_optimizer.load_state_dict(aux_optimizer_state)

        value_optimizer_state = convert_to_torch_tensor(value_optimizer_state, device=self.device)
        self.value_optimizer.load_state_dict(value_optimizer_state)

        amp_scaler_state = convert_to_torch_tensor(amp_scaler_state, device=self.device)
        self.amp_scaler.load_state_dict(amp_scaler_state)

    def set_model_weights(self, model_weights):
        model_weights = convert_to_torch_tensor(model_weights, device=self.device)
        self.model.load_state_dict(model_weights)
