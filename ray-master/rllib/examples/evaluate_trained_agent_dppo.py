"""
test trained models
"""

import argparse
import os
import numpy as np
import torch as th
import ray
from ray import tune
from ray.rllib.examples.env.parametric_actions_graph import ParametricActionsEdgeGraph
from ray.rllib.examples.models.autoregressive_action_model import TorchEdgeAutoregressiveDiscreteActionModel
from ray.rllib.agents.ppo import ddppo
from ray.rllib.examples.models.autoregressive_action_dist import TorchGraphEdgeEmbDistribution_SP_with_stop
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import DDPPOTrainer





if __name__ == "__main__":

    restore_path = ""

    assert restore_path != "", "must provide the location of a trained model"

    ModelCatalog.register_custom_model(
        "autoedgeregressivedecouplediscretedgraph_model",
        TorchEdgeAutoregressiveDiscreteActionModel)

    ModelCatalog.register_custom_action_dist(
        "graph_edgeEmb_pretrained_with_stop_dist", TorchGraphEdgeEmbDistribution_SP_with_stop)

    assert th.cuda.device_count() > 0, "run with gpus"
    print('running with gpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="DDPPO")
    parser.add_argument("--ppo_alg", type=str, default="dcppo",
                        choices=["ppo", 'tppo', 'dcppo'])
    parser.add_argument("--num-cpus", type=int, default=60)
    parser.add_argument("--num-gpus", type=int, default=8)

    parser.add_argument("--as-test", action="store_true")
    parser.add_argument("--max_action", type=int, default=20, help="the maximum episode length")
    parser.add_argument("--filtration_order", type=int, default=-2,
                        help="-1 means the whole size, and -2 means k% nodes")
    parser.add_argument("--alpha", type=float, default=0, help="-1 means grid search")
    parser.add_argument("--filtration_order_ratio", type=float, default=0.3,
                        help="the ration of number of filtrated graphs")
    parser.add_argument("--stop-iters", type=int, default=1000)
    parser.add_argument("--stop-timesteps", type=int, default=1500000)
    parser.add_argument("--stop-reward", type=float, default=50)
    parser.add_argument("--dataset", type=str,
                        default="example_15",
                        choices=["example_15", "example_50", "example_100", "ba_small_30", "ba_mixed" ])
    parser.add_argument("--max_num_node", type=int,
                        default=15, help="maximum number of nodes in a single graph during training")
    parser.add_argument("--max_num_edge", type=int,
                        default=54,
                        help="maximum number of edges in a single graph during training (68 for ba_small)")
    parser.add_argument("--with_stop_action", type=bool, default=True, help="if the policy has stop action")
    parser.add_argument("--with_SimulatedAnnealing", type=bool, default=False,
                        help="if the env use SA to decide to accept the current action or not")
    parser.add_argument("--cwd-path", type=str, default='./')
    parser.add_argument("--tasks-per-gpu", type=int, default=1, help='how manys tasks on a single gpu')
    parser.add_argument("--gpus_per_instance", type=int, default=1, help='how manys gpus on a single instance')
    parser.add_argument("--bs", type=int, default=4096, help='batch size')
    parser.add_argument("--mini-bs", type=int, default=128, help='minibatch batch size')
    parser.add_argument("--hidden_dim", type=int, default=128, help='hidden_dim')
    parser.add_argument("--separate_vf_optimizer", type=bool, default=True, help='separate_vf_optimizer or not')
    parser.add_argument("--disable_preprocessor_api", type=bool, default=True, help='disable_preprocessor_api')
    parser.add_argument("--lr", type=float, default=3e-4, help='learning rate')
    parser.add_argument("--dual_clip_param", type=float, default=5, help='dual_clip_param')
    parser.add_argument("--kl_target", type=float, default=0.4, help='dual_clip_param')

    # different robustness measure
    parser.add_argument("--attack_strategy", type=str, default='degree', choices=['-1', 'degree', 'betweenness'])
    parser.add_argument("--break_tie", type=str, default='inc_by_id', choices=['-1', 'inc_by_id', 'random', 'dec_by_id'])

    parser.add_argument("--robust-measure", type=str, default='R', choices=['R', "sr", "ac"])
    parser.add_argument("--sequentialMode", type=bool, default=True)
    parser.add_argument("--add-penality", type=bool, default=False)

    parser.add_argument("--single-obj", type=bool, default=True, help="True: only optimize robustness")
    parser.add_argument("--second-obj-func", type=str, default='ge', choices=['ge', 'le'], help="'ge': global efficiency")
    parser.add_argument("--reward_scale", type=float, default=1)
    parser.add_argument("--is_train", type=bool, default=True, help="if false, disable reward penalty")
    parser.add_argument("--test_num", type=int, default=1, help="how many samples are tested")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_savegraph_acs", help="no savegraph and actions", action='store_false')
    parser.add_argument("--is_test", help="is_test", action='store_false')


    args = parser.parse_args()

    if args.filtration_order == -2:
        try:
            max_node = int(args.dataset.split('_')[1])
            args.filtration_order = int(max_node * args.filtration_order_ratio)
        except:
            args.filtration_order = -1  # change to the maximum number of nodes
    print(f"filtration_order is {args.filtration_order}")

    if args.alpha == -1:
        args.alpha = tune.grid_search(list(np.linspace(0, 1, 51))[25:])

    # set the max_num_node and max_num_edge for the model ViewRequirement
    if args.dataset == 'ba_small':
        args.max_num_node = 20
        args.max_num_edge = 68
    elif args.dataset == 'ba_small_30':
        args.max_num_node = 30
        args.max_num_edge = 112
    elif args.dataset == 'example_15':
        args.max_num_node = 15
        args.max_num_edge = 54
    elif args.dataset == 'example_50':
        args.max_num_node = 50
        args.max_num_edge = 96 * 2
    elif args.dataset == 'ba_mixed':
        args.max_num_node = 200
        args.max_num_edge = 792
    elif args.dataset == 'example_100':
        args.max_num_node = 100
        args.max_num_edge = 196 * 2
    elif args.dataset == 'EU':
        args.max_num_node = 217
        args.max_num_edge = 320 * 2
    else:
        print('use the max_num_node/edge provided  by the user')

    cwd_path = args.cwd_path
    print('cwd path', cwd_path)
    logdir = cwd_path #+ '/log'
    print(logdir)
    if not os.path.isdir(logdir):
        os.mkdir(logdir)

    ray.init(local_mode=False)
    tune.register_env("env", ParametricActionsEdgeGraph)


    ModelCatalog.register_custom_action_dist(
        "graph_edgeEmb_pretrained_with_stop_dist", TorchGraphEdgeEmbDistribution_SP_with_stop)
    gpu_count = args.gpus_per_instance
    num_workers = gpu_count * args.tasks_per_gpu
    num_gpus = 0  # Driver GPU
    num_gpus_per_worker = float(1 / args.tasks_per_gpu)  # (gpu_count - num_gpus) / (num_workers+3)
    num_envs_per_worker = 16
    num_cpus_per_worker = 1
    remote_worker_envs = False
    sgd_minibatch_size = args.mini_bs  # tune.grid_search([256, 256]) ##tune.grid_search([32, 32])#32
    # train_batch_size = 256#4096#1024#tune.grid_search([2048, 4096])
    rollout_fragment_length = round(args.bs / num_workers / num_envs_per_worker)
    config = ddppo.DEFAULT_CONFIG.copy()

    config.update({
        "env": ParametricActionsEdgeGraph,
        "env_config": {
            "filtration_order": args.filtration_order,
            # tune.grid_search([14]), #tune.grid_search(list(range(6, 15))),
            "max_action": args.max_action,
            "with_stop_action": args.with_stop_action,
            "with_SimulatedAnnealing": args.with_SimulatedAnnealing,
            "dataset_type": args.dataset,
            "max_num_node": args.max_num_node,
            "max_num_edge": args.max_num_edge,
            "robust_measure": args.robust_measure,
            "single_obj": args.single_obj,
            "second_obj_func": args.second_obj_func,
            "reward_scale": args.reward_scale,
            "sequentialMode": args.sequentialMode,
            "add_penality": args.add_penality,
            "is_train": not args.is_test,
            "attack_strategy": args.attack_strategy,
            "break_tie": args.break_tie,

            "alpha": args.alpha  # tune.grid_search([0.05, 0.15, 0.25, 0.35, 0.45]),
        }})
    config.update({
        # "gamma": 0.5,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,  # 0.35,
        "model": {
            "custom_action_dist": "graph_edgeEmb_pretrained_with_stop_dist",
            "custom_model": "autoedgeregressivedecouplediscretedgraph_model",
            "custom_model_config": {
                "use_graph_embedding": True,
                "transfer_func": 'relu',
                'similarity_func': 'attn',
                'direct_logits': True,
                "hidden_dim": args.hidden_dim,
                "filtration_order": args.filtration_order,
                "use_filtration": tune.sample_from(lambda spec: spec.config.env_config.filtration_order != 0),
                "with_stop_action": args.with_stop_action,
                "max_action": args.max_action,
                "with_SimulatedAnnealing": args.with_SimulatedAnnealing,
                "dataset_type": args.dataset,
                "max_num_node": args.max_num_node,
                "max_num_edge": args.max_num_edge,
            },
        },

        # this to 0 will force rollouts to be done in the trainer actor.
        "num_workers": 2,
        # Number of environments to evaluate vectorwise per worker. This enables
        # model inference batching, which can improve performance for inference
        # bottlenecked workloads.
        "num_envs_per_worker": 2,

        "num_gpus_per_worker": 1,
        # "num_gpus_per_worker": 0.3,
        # #
        "sgd_minibatch_size": 200,
        "num_sgd_iter": 10, #tune.grid_search([1, 10, 20]),
        # "num_sgd_iter": 6, #tune.grid_search([1, 10, 20]),
        # "num_workers": 0,
        "grad_clip": 0.5, #tune.grid_search([40]),
        "gamma": 0.995,  # tune.grid_search([0.99, 0.98, 0.97, 0.96, 0.95]),
        "lambda": 0.98,  # tune.grid_search([0.99, 0.98, 0.97,]),
        "clip_param": 0.2,  # tune.grid_search([0.1, 0.2]),
        "rollout_fragment_length": 10,
        "explore": False,

        "lr":7e-4,
        "kl_coeff": 0.5, #tune.grid_search([0.5, 1]),
        "lr_schedule": [
            [0,1e-4],
            [args.stop_timesteps, 1e-6],
        ],
        "entropy_coeff_schedule": [
            [0, 0.01],
            [args.stop_timesteps, 0.001],
        ],
        "vf_loss_coeff": 0.01, #tune.grid_search([1e-1, 1, 0.5]),
        "entropy_coeff": 0.01,
        # ppo or dcppo or tppo
        "ppo_alg": args.ppo_alg,
        # DPPO
        "keep_local_weights_in_sync": True,
        "_disable_preprocessor_api": args.disable_preprocessor_api,
        "_separate_vf_optimizer": args.separate_vf_optimizer,


        "framework": "torch",# if args.torch else "tf",
    })
    agent = DDPPOTrainer(config, env="env")


    agent.restore(restore_path)

    # evaluate the trained model
    env = ParametricActionsEdgeGraph(config["env_config"])

    seed = args.seed
    env.seed(seed)
    obs = env.reset()
    if args.test_num == -1:
        args.test_num = env.wrapped.data_len

    for i in range(args.test_num):
        episode_reward = 0
        # env.seed(seed)
        obs = env.reset()

        cnt = 0


        done = False
        rew_list = []

        while not done:

            a = agent.compute_single_action(obs)

            edge_index = env.wrapped.edge_index.copy()

            obs, reward, done, env_info = env.step(a)
            decoded_action = a[-2] // obs['num_edges'].shape[0], a[-2] % obs['num_edges'].shape[0], a[-1] // \
                             obs['num_edges'].shape[0], a[-1] % obs['num_edges'].shape[0]
            print(f'encoded action is [{a}], reward is {reward:.3f}')
            decoded_action = np.concatenate([edge_index[a[1]], edge_index[a[2]]]).tolist()
            print(f'decoded action is {decoded_action}')

            rew_list.append(reward)
            episode_reward += reward
            cnt += 1

        print('[%d] th episode reward is %f:' % (i, episode_reward))
        print(np.cumsum(rew_list))

    agent.cleanup()

    del agent
    del env

    ray.shutdown()

