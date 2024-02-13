import multiprocessing

import numpy as np

from ssd_meltingpot.substrate import env_creator
from utils import to_mean_info_dict


class DummyVectorEnv:
    # Vectorized environment based on loop
    def __init__(self, envs):
        self.envs = envs
        self.env_num = len(self.envs)
        self.agents = self.envs[0]._agent_ids
        self.obs_keys = [key for key in self.observation_space["player_0"].keys()]

    def reset(self):
        # bug of meltingpot 2.2 =_=!
        observations = {
            agent_id: {
                key: np.zeros(
                    (self.env_num, *self.observation_space[agent_id][key].shape)) for key in self.obs_keys} for
            agent_id in self.agents}
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            for agent_id in self.agents:
                for key in self.obs_keys:
                    observations[agent_id][key][i] = obs[agent_id][key]
        return observations

    def seed(self, seed=None):
        for env in self.envs:
            env.seed(seed)

    def step(self, actions):
        # bug of meltingpot 2.2 =_=!
        observations = {
            agent_id: {
                key: np.zeros((self.env_num, *self.observation_space[agent_id][key].shape)) for key in self.obs_keys}
            for agent_id in self.agents}
        rewards = {agent_id: np.zeros(self.env_num) for agent_id in self.agents}
        dones = {agent_id: np.zeros(self.env_num) for agent_id in self.agents}
        infos = []
        for i, env in enumerate(self.envs):
            if self.env_num > 1:
                input_actions = {agent_id: actions[agent_id][i] for agent_id in self.agents}
            else:
                input_actions = {agent_id: int(actions[agent_id]) for agent_id in self.agents}
            obs, reward, done, info = env.step(input_actions)
            for agent_id in self.agents:
                for key in self.obs_keys:
                    observations[agent_id][key][i] = obs[agent_id][key]
                rewards[agent_id][i] = reward[agent_id]
                dones[agent_id][i] = done["__all__"]
                infos.append(info)

        infos = to_mean_info_dict(infos)
        return observations, rewards, dones, infos

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space

    def close(self):
        for env in self.envs:
            env.close()


def _worker(substrate_name, roles, scale_factor, env_id, conn):
    try:
        env_config = {"substrate": substrate_name, "roles": roles, "scaled": scale_factor}
        env = env_creator(env_config)
        while True:
            try:
                cmd, data = conn.recv()
            except EOFError:
                conn.close()
                break
            if cmd == "reset":
                obs, _ = env.reset()
                conn.send((env_id, obs))
            elif cmd == "step":
                obs, reward, done, info = env.step(data)
                conn.send((env_id, obs, reward, done, info))
            elif cmd == "close":
                env.close()
                conn.close()
                break
            elif cmd == "seed":
                if data:
                    env.seed(data)
                else:
                    env.seed()
    except KeyboardInterrupt:
        conn.close()


class SubprocVectorEnv:
    def __init__(self, env_config):
        # env_config = {"substrate": substrate_name, "roles": player_roles, "scaled": scale_factor,
        #               "env_num": args.env_parallel_num}
        self.substrate_name = env_config["substrate"]
        self.roles = env_config["roles"]
        self.scale_factor = env_config["scaled"]
        self.env_num = env_config["env_num"]
        self.sample_env = env_creator(env_config)
        self.observation_space = self.sample_env.observation_space
        self.action_space = self.sample_env.action_space
        self.agents = self.sample_env._agent_ids
        del self.sample_env
        self.obs_keys = [key for key in self.observation_space["player_0"].keys()]
        self.processes = []
        self.main_conns = []
        self.sub_conns = []

        for i in range(self.env_num):
            main_conn, sub_conn = multiprocessing.Pipe()
            self.main_conns.append(main_conn)
            self.sub_conns.append(sub_conn)
            p = multiprocessing.Process(target=_worker,
                                        args=(self.substrate_name, self.roles, self.scale_factor, i, sub_conn),
                                        daemon=True)
            p.start()

    def reset(self):
        observations = {
            agent_id: {
                key: np.zeros(
                    (self.env_num, *self.observation_space[agent_id][key].shape)) for key in self.obs_keys}
            for agent_id in self.agents}

        for conn in self.main_conns:
            conn.send(("reset", None))
        for conn in self.main_conns:
            try:
                env_id, obs = conn.recv()
            except EOFError:
                raise "reset error"
            else:
                for agent_id in self.agents:
                    for key in self.obs_keys:
                        observations[agent_id][key][env_id] = obs[agent_id][key]
        return observations

    def step(self, actions):
        observations = {
            agent_id: {
                key: np.zeros((self.env_num, *self.observation_space[agent_id][key].shape)) for key in self.obs_keys}
            for agent_id in self.agents}
        rewards = {agent_id: np.zeros(self.env_num) for agent_id in self.agents}
        dones = {agent_id: np.zeros(self.env_num) for agent_id in self.agents}
        infos = []

        for i in range(self.env_num):
            if self.env_num > 1:
                input_actions = {agent_id: actions[agent_id][i] for agent_id in self.agents}
            else:
                input_actions = {agent_id: int(actions[agent_id]) for agent_id in self.agents}
            self.main_conns[i].send(("step", input_actions))

        for conn in self.main_conns:
            try:
                env_id, obs, reward, done, info = conn.recv()
            except EOFError:
                raise "step error!"
            else:
                for agent_id in self.agents:
                    for key in self.obs_keys:
                        observations[agent_id][key][env_id] = obs[agent_id][key]
                    rewards[agent_id][env_id] = reward[agent_id]
                    dones[agent_id][env_id] = done["__all__"]
                    infos.append(info)
        infos = to_mean_info_dict(infos)
        return observations, rewards, dones, infos

    def seed(self, seed=None):
        for conn in self.main_conns:
            conn.send(("seed", seed))

    def close(self):
        for conn in self.main_conns:
            conn.send(("close", None))
