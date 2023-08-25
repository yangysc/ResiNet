import logging
from typing import Tuple, Callable, List, Optional

import ray
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID, ASYNC_RESET_RETURN
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import MultiEnvDict, EnvType, EnvID, MultiAgentDict

logger = logging.getLogger(__name__)


@PublicAPI
class RemoteVectorEnv(BaseEnv):
    """Vector env that executes envs in remote workers.

    This provides dynamic batching of inference as observations are returned
    from the remote simulator actors. Both single and multi-agent child envs
    are supported, and envs can be stepped synchronously or async.

    You shouldn't need to instantiate this class directly. It's automatically
    inserted when you use the `remote_worker_envs` option for Trainers.
    """

    def __init__(self,
                 make_env: Callable[[int], EnvType],
                 num_envs: int,
                 multiagent: bool,
                 remote_env_batch_wait_ms: int,
                 existing_envs: Optional[List[ray.actor.ActorHandle]] = None):
        # Could be creating local or remote envs.
        self.make_env = make_env
        self.make_env_creates_actors = False
        # Already existing env objects (generated by the RolloutWorker).
        self.existing_envs = existing_envs or []
        self.num_envs = num_envs
        self.multiagent = multiagent
        self.poll_timeout = remote_env_batch_wait_ms / 1000

        self.actors = None  # lazy init
        self.pending = None  # lazy init

    @override(BaseEnv)
    def poll(self) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict,
                            MultiEnvDict, MultiEnvDict]:
        if self.actors is None:
            # `self.make_env` already produces Actors: Use it directly.
            if len(self.existing_envs) > 0 and isinstance(
                    self.existing_envs[0], ray.actor.ActorHandle):
                self.make_env_creates_actors = True
                self.actors = []
                while len(self.actors) < self.num_envs:
                    self.actors.append(self.make_env(len(self.actors)))
            # `self.make_env` produces gym.Envs (or other similar types, such
            # as MultiAgentEnv): Need to auto-wrap it here. The problem with
            # this is that custom methods wil get lost.
            else:

                def make_remote_env(i):
                    logger.info("Launching env {} in remote actor".format(i))
                    if self.multiagent:
                        return _RemoteMultiAgentEnv.remote(self.make_env, i)
                    else:
                        return _RemoteSingleAgentEnv.remote(self.make_env, i)

                self.actors = [
                    make_remote_env(i) for i in range(self.num_envs)
                ]

        if self.pending is None:
            self.pending = {a.reset.remote(): a for a in self.actors}

        # each keyed by env_id in [0, num_remote_envs)
        obs, rewards, dones, infos = {}, {}, {}, {}
        ready = []

        # Wait for at least 1 env to be ready here
        while not ready:
            ready, _ = ray.wait(
                list(self.pending),
                num_returns=len(self.pending),
                timeout=self.poll_timeout)

        # Get and return observations for each of the ready envs
        env_ids = set()
        for obj_ref in ready:
            actor = self.pending.pop(obj_ref)
            env_id = self.actors.index(actor)
            env_ids.add(env_id)
            ret = ray.get(obj_ref)
            # Our sub-envs are simple Actor-turned gym.Envs or MultiAgentEnvs.
            if self.make_env_creates_actors:
                rew, done, info = None, None, None
                if self.multiagent:
                    if isinstance(ret, tuple) and len(ret) == 4:
                        ob, rew, done, info = ret
                    else:
                        ob = ret
                else:
                    if isinstance(ret, tuple) and len(ret) == 4:
                        ob = {_DUMMY_AGENT_ID: ret[0]}
                        rew = {_DUMMY_AGENT_ID: ret[1]}
                        done = {_DUMMY_AGENT_ID: ret[2], "__all__": ret[2]}
                        info = {_DUMMY_AGENT_ID: ret[3]}
                    else:
                        ob = {_DUMMY_AGENT_ID: ret}

                if rew is None:
                    rew = {agent_id: 0 for agent_id in ob.keys()}
                    done = {"__all__": False}
                    info = {agent_id: {} for agent_id in ob.keys()}
            # Our sub-envs are auto-wrapped and already behave like multi-agent
            # envs.
            else:
                ob, rew, done, info = ret
            obs[env_id] = ob
            rewards[env_id] = rew
            dones[env_id] = done
            infos[env_id] = info

        logger.debug("Got obs batch for actors {}".format(env_ids))
        return obs, rewards, dones, infos, {}

    @override(BaseEnv)
    @PublicAPI
    def send_actions(self, action_dict: MultiEnvDict) -> None:
        for env_id, actions in action_dict.items():
            actor = self.actors[env_id]
            obj_ref = actor.step.remote(actions)
            self.pending[obj_ref] = actor

    @override(BaseEnv)
    @PublicAPI
    def try_reset(self,
                  env_id: Optional[EnvID] = None) -> Optional[MultiAgentDict]:
        actor = self.actors[env_id]
        obj_ref = actor.reset.remote()
        self.pending[obj_ref] = actor
        return ASYNC_RESET_RETURN

    @override(BaseEnv)
    @PublicAPI
    def stop(self) -> None:
        if self.actors is not None:
            for actor in self.actors:
                actor.__ray_terminate__.remote()

    @override(BaseEnv)
    @PublicAPI
    def get_unwrapped(self):
        return self.actors


@ray.remote(num_cpus=0)
class _RemoteMultiAgentEnv:
    """Wrapper class for making a multi-agent env a remote actor."""

    def __init__(self, make_env, i):
        self.env = make_env(i)

    def reset(self):
        obs = self.env.reset()
        # each keyed by agent_id in the env
        rew = {agent_id: 0 for agent_id in obs.keys()}
        info = {agent_id: {} for agent_id in obs.keys()}
        done = {"__all__": False}
        return obs, rew, done, info

    def step(self, action_dict):
        return self.env.step(action_dict)


@ray.remote(num_cpus=0)
class _RemoteSingleAgentEnv:
    """Wrapper class for making a gym env a remote actor."""

    def __init__(self, make_env, i):
        self.env = make_env(i)

    def reset(self):
        obs = {_DUMMY_AGENT_ID: self.env.reset()}
        rew = {agent_id: 0 for agent_id in obs.keys()}
        done = {"__all__": False}
        info = {agent_id: {} for agent_id in obs.keys()}
        return obs, rew, done, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action[_DUMMY_AGENT_ID])
        obs, rew, done, info = [{
            _DUMMY_AGENT_ID: x
        } for x in [obs, rew, done, info]]
        done["__all__"] = done[_DUMMY_AGENT_ID]
        return obs, rew, done, info