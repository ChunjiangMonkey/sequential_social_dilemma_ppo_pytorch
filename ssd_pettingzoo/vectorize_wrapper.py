import multiprocessing

import numpy as np

from utils import to_mean_info_dict


class DummyVectorEnv:
    # Vectorized environment based on loop
    def __init__(self, envs):
        self.envs = envs
        self.env_num = len(self.envs)
        self.agents = self.envs[0].agents
        self.obs_keys = [key for key in self.observation_space.keys()]

    def reset(self):
        observations = {
            agent_id: {key: np.zeros((self.env_num, *self.observation_space[key].shape)) for key in self.obs_keys} for
            agent_id in self.agents}
        for i, env in enumerate(self.envs):
            obs = env.reset()
            for agent_id in self.agents:
                for key in self.obs_keys:
                    observations[agent_id][key][i] = obs[agent_id][key]
        return observations

    def seed(self, seed=None):
        for env in self.envs:
            env.seed(seed)

    def step(self, actions):
        observations = {
            agent_id: {key: np.zeros((self.env_num, *self.observation_space[key].shape)) for key in self.obs_keys} for
            agent_id in self.agents}
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
                dones[agent_id][i] = done[agent_id]
                infos.append(info)

        infos = to_mean_info_dict(infos)
        return observations, rewards, dones, infos

    @property
    def observation_space(self):
        return self.envs[0].get_observation_space()

    @property
    def action_space(self):
        return self.envs[0].get_action_space()

    def rgb_state(self, env_id):
        rgb_state = self.envs[env_id].full_map_to_colors()
        return rgb_state

    def close(self):
        for env in self.envs:
            env.close()


def _worker(env, env_id, conn):
    try:
        while True:
            try:
                cmd, data = conn.recv()
            except EOFError:
                conn.close()
                break
            if cmd == "reset":
                obs = env.reset()
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
            elif cmd == "rgb_state":
                rgb_state = env.full_map_to_colors()
                conn.send((env_id, rgb_state))
    except KeyboardInterrupt:
        conn.close()


class SubprocVectorEnv:
    # Vectorized environment based on multiprocess
    def __init__(self, envs):
        self.envs = envs
        self.env_num = len(self.envs)
        self.agents = self.envs[0].agents
        self.obs_keys = [key for key in self.observation_space.keys()]
        self.processes = []
        self.main_conns = []
        self.sub_conns = []
        for i in range(self.env_num):
            main_conn, sub_conn = multiprocessing.Pipe()
            self.main_conns.append(main_conn)
            self.sub_conns.append(sub_conn)
            p = multiprocessing.Process(target=_worker, args=(self.envs[i], i, sub_conn), daemon=True)
            p.start()

    def reset(self):
        observations = {agent_id: {key: np.zeros((self.env_num, *self.observation_space[key].shape)) for key in self.obs_keys} for agent_id in self.agents}

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
        observations = {agent_id: {key: np.zeros((self.env_num, *self.observation_space[key].shape)) for key in self.obs_keys} for agent_id in self.agents}
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
                    dones[agent_id][env_id] = done[agent_id]
                    infos.append(info)
        infos = to_mean_info_dict(infos)
        return observations, rewards, dones, infos

    def seed(self, seed=None):
        for conn in self.main_conns:
            try:
                conn.send(("seed", seed))
            except EOFError:
                raise "step error!"
            else:
                pass

    @property
    def observation_space(self):
        return self.envs[0].get_observation_space()

    @property
    def action_space(self):
        return self.envs[0].get_action_space()

    def rgb_state(self, env_id):
        self.main_conns[env_id].send(("rgb_state", None))
        try:
            env_id, rgb_state = self.main_conns[env_id].recv()
        except EOFError:
            raise "step error!"
        else:
            return rgb_state

    def close(self):
        for conn in self.main_conns:
            conn.send(("close", None))
