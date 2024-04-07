import numpy as np

from env_utils.dict_utils import create_dict_with_same_keys, reset_dict


class DataCollector:
    def __init__(self, env_num, agent_names, infos):
        self.env_num = env_num
        self.agent_names = agent_names
        self._info_step = create_dict_with_same_keys(infos)
        self._info_episode = create_dict_with_same_keys(infos)
        self.episode = 0
        self.custom_initial()

    def custom_initial(self):
        raise NotImplementedError

    def update_step_info(self, actions, infos):
        for key in infos.keys():
            if isinstance(self._info_step[key], dict):
                for sub_key in infos[key].keys():
                    self._info_step[key][sub_key].append(infos[key][sub_key])
            else:
                self._info_step[key].append(infos[key])
        self.custom_update_step_info(actions, infos)

    def custom_update_step_info(self, actions, infos):
        raise NotImplementedError

    def update_episode_info(self):
        # episode data is the mean or sum of step data
        for key in self._info_episode.keys():
            if isinstance(self._info_episode[key], dict):
                for sub_key in self._info_episode[key].keys():
                    self._info_episode[key][sub_key].append(np.sum(self._info_step[key][sub_key]))
                    self._info_step[key][sub_key] = []
            else:
                self._info_episode[key].append(np.mean(self._info_step[key]))
                self._info_step[key] = []
        self.episode += 1

    @property
    def info_episode(self):
        info_dict = {}
        for key in self._info_episode.keys():
            # for agent data
            if isinstance(self._info_episode[key], dict):
                for sub_key in self._info_episode[key].keys():
                    info_dict[f"{key}_{sub_key}"] = self._info_episode[key][sub_key][self.episode - 1]
            # for env data
            else:
                info_dict[key] = self._info_episode[key][self.episode - 1]
        return info_dict

    @property
    def info_all_episode(self):
        info_dict = {}
        for key in self._info_episode.keys():
            # for agent data
            if isinstance(self._info_episode[key], dict):
                for sub_key in self._info_episode[key].keys():
                    info_dict[f"{key}_{sub_key}"] = self._info_episode[key][sub_key][self.episode]
                    self._info_episode[key][sub_key] = []
            # for env data
            else:
                info_dict[key] = self._info_episode[key][self.episode]
        return info_dict

    def clear(self):
        self.episode = 0
        self._info_step = reset_dict(self._info_step)
        self._info_episode = reset_dict(self._info_episode)


class CleanupDataCollector(DataCollector):
    def custom_initial(self):
        for key in self._info_step.keys():
            # for agent data
            if isinstance(self._info_step[key], dict):
                self._info_step[key]["fire_num"] = []
                self._info_step[key]["clean_num"] = []
        for key in self._info_episode.keys():
            # for agent data
            if isinstance(self._info_episode[key], dict):
                self._info_episode[key]["fire_num"] = []
                self._info_episode[key]["clean_num"] = []

    def custom_update_step_info(self, actions, infos):
        for agent in self.agent_names:
            action = actions[agent]
            fire_num = np.sum(action == 7)
            clean_num = np.sum(action == 8)
            self._info_step[agent]["fire_num"].append(fire_num / self.env_num)
            self._info_step[agent]["clean_num"].append(clean_num / self.env_num)


class HarvestDataCollector(DataCollector):
    def custom_initial(self):
        for key in self._info_step.keys():
            # for agent data
            if isinstance(self._info_step[key], dict):
                self._info_step[key]["fire_num"] = []

        for key in self._info_episode.keys():
            # for agent data
            if isinstance(self._info_episode[key], dict):
                self._info_episode[key]["fire_num"] = []

    def custom_update_step_info(self, actions, infos):
        for agent in self.agent_names:
            action = actions[agent].cpu().numpy()
            fire_num = np.sum(action == 7)
            self._info_step[agent]["fire_num"].append(fire_num / self.env_num)
