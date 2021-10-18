import argparse
from typing import Dict
import numpy as np
import torch as th
import os

import ray
from ray import tune
from ray.rllib.examples.env.parametric_actions_graph import ParametricActionsEdgeGraph
from ray.rllib.examples.models.autoregressive_action_model import TorchEdgeAutoregressiveDiscreteActionModel
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env import BaseEnv
from ray.rllib.examples.env.utils_ import efficiency, objective_func
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.examples.models.autoregressive_action_dist import TorchGraphEdgeEmbDistribution_SP_with_stop
from ray.rllib.models import ModelCatalog
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.ppo import ddppo
from ray.rllib.utils.sgd import standardized





class MyCallbacks(DefaultCallbacks):
    """
    Custom callbacks
    """
    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          result: dict, **kwargs) -> None:
        # normalize advantages on the minibatch
        train_batch['advantages'] = standardized(train_batch['advantages'])

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        # second_obj_func
        effi_val = efficiency(base_env.vector_env.envs[env_index].wrapped.graph)
        episode.custom_metrics["global_efficiency"] = effi_val

        # robustness based on R
        robust_val = objective_func(base_env.vector_env.envs[env_index].wrapped.graph,
                       alpha=base_env.vector_env.envs[env_index].wrapped.alpha,
                       scale=base_env.vector_env.envs[env_index].wrapped.scale,
                       robust_measure=base_env.vector_env.envs[env_index].wrapped.robust_measure,
                       single_obj=base_env.vector_env.envs[env_index].wrapped.single_obj,
                       second_obj_func=base_env.vector_env.envs[env_index].wrapped.second_obj_func,
                       sequentialMode=base_env.vector_env.envs[env_index].wrapped.sequentialMode,
                       attack_strategy=base_env.vector_env.envs[env_index].wrapped.attack_strategy)

        episode.custom_metrics["robustness_%s_%s" % (base_env.vector_env.envs[env_index].wrapped.robust_measure, base_env.vector_env.envs[env_index].wrapped.second_obj_func)] = robust_val


if __name__ == "__main__":

    ModelCatalog.register_custom_model(
        "autoedgeregressivedecouplediscretedgraph_model", TorchEdgeAutoregressiveDiscreteActionModel)  # TorchGraphNotNodeEmbRNNModel

    ModelCatalog.register_custom_action_dist(
        "graph_edgeEmb_pretrained_with_stop_dist", TorchGraphEdgeEmbDistribution_SP_with_stop)



    assert th.cuda.device_count() > 0, "The code requires gpu"
    print('running with gpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="DDPPO")  # try PG, PPO, IMPALA
    parser.add_argument("--ppo_alg", type=str, default="ppo", choices=["ppo", 'tppo', 'dcppo'])  # try PG, PPO, IMPALA
    parser.add_argument("--torch", action="store_true", default=True)
    parser.add_argument("--num-cpus", type=int, default=16)
    parser.add_argument("--num-gpus", type=int, default=1)

    parser.add_argument("--as-test", action="store_true")
    parser.add_argument("--max_action", type=int, default=20, help="the maximum episode length")
    parser.add_argument("--filtration_order", type=int, default=-2, help="-1 means the whole size, -2 means top k% nodes, -3 means a grid search")
    parser.add_argument("--alpha", type=float, default=0, help="-1 means grid search")
    parser.add_argument("--filtration_order_ratio", type=float, default=0.1, help="the ration of number of filtrated graphs")
    parser.add_argument("--stop-iters", type=int, default=1000)
    parser.add_argument("--stop-timesteps", type=int, default=500000)
    # parser.add_argument("--use_filtrated", type=bool, default=True, help="whether to use filtrated graph to augument the graph")
    parser.add_argument("--stop-reward", type=float, default=50)
    parser.add_argument("--dataset", type=str,
                        default=r"example_15", choices=["example_15", "example_50", "example_100",
                                                        "ba_small_30", "ba_200",  "ba_mixed", "EU"])
    # parser.add_argument("--num_train_dataset", type=int,
    #                     default=1000, help="number of training samples")
    parser.add_argument("--max_num_node", type=int,
                        default=15, help="maximum number of nodes in a single graph during training")
    parser.add_argument("--max_num_edge", type=int,
                        default=54, help="maximum number of edges in a single graph during training")

    parser.add_argument("--with_stop_action", type=bool, default=True, help="if the policy has stop action")
    parser.add_argument("--with_SimulatedAnnealing", type=bool, default=False, help="if the env use SA to decide to accept the current action or not, (not used)")
    # parser.add_argument("--model_dir", type=str, default=18)

    parser.add_argument("--cwd-path", type=str, default='/', help="model and tensorboard log sava directory")
    parser.add_argument("--tasks-per-gpu", type=int, default=1, help='how manys tasks on a single gpu')
    parser.add_argument("--gpus_per_instance", type=int, default=1, help='how manys gpus on a single instance')
    parser.add_argument("--bs", type=int, default=4096, help='batch size')
    parser.add_argument("--mini-bs", type=int, default=256, help='minibatch batch size')
    parser.add_argument("--hidden_dim", type=int, default=64, help='hidden_dim')
    parser.add_argument("--separate_vf_optimizer", type=bool, default=True, help='separate_vf_optimizer or not')
    parser.add_argument("--disable_preprocessor_api", type=bool, default=True, help='disable_preprocessor_api')
    parser.add_argument("--lr", type=float, default=7e-4, help='learning rate')
    parser.add_argument("--vf_lr", type=float, default=3e-4, help='vf learning rate')
    parser.add_argument("--dual_clip_param", type=float, default=5, help='dual_clip_param')

    # different robustness measure
    parser.add_argument("--robust-measure", type=str, default='R', choices=['-1', 'R', "sr", "ac"])
    parser.add_argument("--attack_strategy", type=str, default='degree', choices=['-1', 'degree', 'betweenness', 'null'])
    parser.add_argument("--break_tie", type=str, default='inc_by_id', choices=['-1', 'inc_by_id', 'random', 'dec_by_id'])

    parser.add_argument("--sequentialMode", type=bool, default=True)
    # parser.add_argument("--single-obj",  action='store_true')
    parser.add_argument("--add-penality", type=bool, default=False)
    # parser.add_argument("--penality", type=float, default=)
    # parser.add_argument("--robust-measure", type=str, default='R', choices=['R', "sr", "ac"])
    # parser.add_argument("--single-obj", type=bool, default=True, help="True: only optimize robustness")
    parser.add_argument("--second-obj-func", type=str, default='ge', choices=['-1', 'ge', 'le'],help="'ge': global efficiency")
    parser.add_argument("--reward_scale", type=float, default=1)

    parser.add_argument("--project", type=str, default='example_project', help='wandb name')
    parser.add_argument("--is_train", default=True, help="if false, disable reward penalty", action='store_true')
    parser.add_argument("--is_restore", help="if resume training", action='store_true')
    parser.add_argument("--no_wandb", help="no wandb", action='store_false')
    # parser.add_argument("--wandb", help="no wandb", action='store_false')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_freq", type=int, default=0, help="model save frequency")


    args = parser.parse_args()
    # print(args.as_test)
    # exit(10)
    restore_path = ""
    if args.is_restore:
        assert restore_path != "", "musth provide restore path"


    if int(args.alpha) == -1:
        args.alpha = tune.grid_search([0, 0.5])
        # args.alpha = tune.grid_search(list(np.linspace(0, 1, 51))[25:])
    if int(args.seed) == -1:
        args.seed = tune.grid_search([0, 1, 2])

    # set the max_num_node and max_num_edge for the model ViewRequirement,

    # datasets for the inductive setting,  change the value according to your graph
    # the max_num_edge means the number of directed edge; double it if training an undirected graph

    if args.dataset == 'ba_small_30':
        args.max_num_node = 30
        args.max_num_edge = 112
    elif args.dataset == 'ba_mixed':
        args.max_num_node = 200
        args.max_num_edge = 792

    # datasets for the transductive setting;
    elif args.dataset == 'example_15':
        args.max_num_node = 15
        args.max_num_edge = 54
    elif args.dataset == 'example_50':
        args.max_num_node = 50
        args.max_num_edge = 96 * 2
    elif args.dataset == 'example_100':
        args.max_num_node = 100
        args.max_num_edge = 196 * 2

    elif args.dataset == 'EU':
        args.max_num_node = 217
        args.max_num_edge = 320 * 2
    else:
        print('use the max_num_node/edge provided  by the user')

    # set the hyparameters
    if args.filtration_order == -3:
        args.filtration_order = tune.grid_search(list(range(args.max_num_node)))
    elif args.filtration_order == -2:
        try:
            max_node = int(args.dataset.split('_')[1])
            args.filtration_order = int(max_node * args.filtration_order_ratio)
        except:
            args.filtration_order = -1  # change to the maximum number of nodes
    print(f"filtration_order is {args.filtration_order}")

    if args.second_obj_func == '-1':
        args.second_obj_func = tune.grid_search(['ge', 'le'])
    if args.break_tie == '-1':
        args.break_tie = tune.grid_search(['inc_by_id', 'random', 'dec_by_id'])

    if args.robust_measure == '-1':
        args.robust_measure = tune.grid_search(['R', "sr", "ac"])

    if args.attack_strategy == '-1':
        args.attack_strategy = tune.grid_search(['degree', 'betweenness'])

    cwd_path = args.cwd_path
    print('cwd path', cwd_path)
    logdir = cwd_path + '/log'
    print(logdir)
    if not os.path.isdir(logdir):
        os.mkdir(logdir)


    ray.init(num_gpus=args.num_gpus, local_mode=False, _plasma_directory=logdir)#, object_store_memory=10**9 * 20)
    config = ddppo.DEFAULT_CONFIG.copy()

    # adjust the cpu/gpu resources according to the real hardware
    gpu_count = args.gpus_per_instance
    num_workers = gpu_count * args.tasks_per_gpu
    num_gpus = 0  # Driver GPU
    num_gpus_per_worker = float(1/args.tasks_per_gpu)#(gpu_count - num_gpus) / (num_workers+3)
    num_envs_per_worker = 16
    num_cpus_per_worker = 1
    remote_worker_envs = False
    sgd_minibatch_size = args.mini_bs#tune.grid_search([256, 256]) ##tune.grid_search([32, 32])#32
    # train_batch_size = 256#4096#1024#tune.grid_search([2048, 4096])
    rollout_fragment_length = round(args.bs / num_workers / num_envs_per_worker)
    config.update({
        "env": ParametricActionsEdgeGraph,
        "env_config": {
            "filtration_order": args.filtration_order, #tune.grid_search([14]), #tune.grid_search(list(range(6, 15))),
            "max_action": args.max_action,
            "with_stop_action": args.with_stop_action,
            "with_SimulatedAnnealing": args.with_SimulatedAnnealing,
            "dataset_type": args.dataset,
            "max_num_node": args.max_num_node,
            "max_num_edge": args.max_num_edge,
            "robust_measure": args.robust_measure,
            # "single_obj": args.single_obj,
            "second_obj_func": args.second_obj_func,
            "reward_scale": args.reward_scale,
            "sequentialMode": args.sequentialMode,
            "add_penality": args.add_penality,
            "is_train": args.is_train,
            "attack_strategy": args.attack_strategy,
            "break_tie": args.break_tie,
            "alpha": args.alpha #tune.grid_search([0.05, 0.15, 0.25, 0.35, 0.45]),
        }})

    config['env_config'].update({"single_obj": tune.sample_from(lambda spec: bool(np.abs(spec.config.env_config.alpha) < 1e-5))})

    config.update({
        "callbacks": MyCallbacks,
        # "gamma": 0.5,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set sto > 0.
        "num_gpus": num_gpus,  # 0.35, # must set to one, cannot be 2
        "model": {

            "custom_action_dist": "graph_edgeEmb_pretrained_with_stop_dist" if args.with_stop_action else "graph_edgeEmb_pretrained_dist",
            "custom_model": "autoedgeregressivedecouplediscretedgraph_model",

            "custom_model_config": {
                "use_graph_embedding": True,
                "transfer_func": 'relu',
                'similarity_func': 'attn',
                'direct_logits': True,
                "hidden_dim": args.hidden_dim,
                "filtration_order": tune.sample_from(lambda spec: spec.config.env_config.filtration_order),
                "use_filtration": tune.sample_from(lambda spec: spec.config.env_config.filtration_order != 0),
                "with_stop_action": args.with_stop_action,
                "max_action": args.max_action,
                "with_SimulatedAnnealing": args.with_SimulatedAnnealing,
                "dataset_type": args.dataset,
                "max_num_node": args.max_num_node,
                "max_num_edge": args.max_num_edge,
            },
            #

            "vf_share_layers": False,#tune.grid_search([True, False ]),

        },

        "num_workers":  num_workers,            # Number of environments to evaluate vectorwise per worker. This enables
        # model inference batching, which can improve performance for inference
        # bottlenecked workloads.
        "num_envs_per_worker": num_envs_per_worker,
        # "remote_worker_envs": remote_worker_envs,
        "num_cpus_per_worker": num_cpus_per_worker,
        "num_gpus_per_worker": num_gpus_per_worker,
        "_separate_vf_optimizer": args.separate_vf_optimizer,
        "dual_clip_param": args.dual_clip_param,

        "seed": args.seed,
        "sgd_minibatch_size": sgd_minibatch_size,
        "grad_clip": 0.5,#tune.grid_search([1, 5]),
        "gamma": 0.99, #tune.grid_search([0.995,  0.95]),
        "lambda": 0.95,#tune.grid_search([0.999, 0.95]),
        "clip_param": 0.2, #tune.grid_search([0.1, 0.2, 0.3]),
        "rollout_fragment_length": rollout_fragment_length,

        "lr": args.lr,
        "kl_coeff": 0,

        "lr_schedule": [
            [0, args.lr],
            [args.stop_timesteps, 0],
        ],
        "entropy_coeff_schedule": [
            [0, 0.01],
            [args.stop_timesteps, 0],
        ],
        "vf_loss_coeff": 1,#tune.grid_search([1e-1,1e-2]),

        "lr_vf_schedule": [
            [0, args.vf_lr],
            [args.stop_timesteps, 0],
        ],
        "entropy_coeff": 0.01,
        "vf_clip_param": 1 * args.reward_scale,

        # ppo or dcppo or tppo
        "ppo_alg": args.ppo_alg,
        # DPPO
        "keep_local_weights_in_sync": True,
        "_disable_preprocessor_api": args.disable_preprocessor_api,


        "framework": "torch",
    })

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }


    if args.is_restore:
        results = tune.run(args.run, stop=stop, config=config, verbose=3, checkpoint_freq=args.checkpoint_freq,
                           local_dir=logdir, restore=restore_path, checkpoint_at_end=True)
    else:

        results = tune.run(args.run, stop=stop, config=config, verbose=3, checkpoint_freq=args.checkpoint_freq,
                           local_dir=logdir, checkpoint_at_end=True)
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()
