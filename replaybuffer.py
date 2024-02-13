import numpy as np
import torch


class VectorizedReplayBuffer:
    def __init__(self, args):
        self.env_parallel_num = args.env_parallel_num
        self.batch_size = args.batch_size * args.env_parallel_num
        self.input_c = args.input_c
        self.input_h = args.input_h
        self.input_w = args.input_w
        self.action_dim = args.action_dim
        self.train_episode = args.train_episode
        self.train_inner_steps = args.train_inner_steps
        self.episode_num = 0
        self.buffer = None
        # self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.device = torch.device(args.device)
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {
            "obs": np.zeros((self.batch_size, self.train_inner_steps, self.input_c, self.input_h, self.input_w)),
            "v": np.zeros((self.batch_size, self.train_inner_steps + 1)),
            "a": np.zeros((self.batch_size, self.train_inner_steps)),
            "a_logprob": np.zeros((self.batch_size, self.train_inner_steps)),
            "r": np.zeros((self.batch_size, self.train_inner_steps)),
            "dw": np.zeros((self.batch_size, self.train_inner_steps)),
        }
        self.episode_num = 0

    def store_transition(self, step, obs, v, a, a_logprob, r, dw):
        start_index = self.episode_num * self.env_parallel_num
        end_index = (self.episode_num + 1) * self.env_parallel_num
        self.buffer['obs'][start_index:end_index, step] = obs
        self.buffer['v'][start_index:end_index, step] = v
        self.buffer['a'][start_index:end_index, step] = a
        self.buffer['a_logprob'][start_index:end_index, step] = a_logprob
        self.buffer['r'][start_index:end_index, step] = r
        self.buffer['dw'][start_index:end_index, step] = dw

    def store_last_value(self, step, v):
        start_index = self.episode_num * self.env_parallel_num
        end_index = (self.episode_num + 1) * self.env_parallel_num
        self.buffer['v'][start_index:end_index, step] = v
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a':
                batch[key] = torch.tensor(self.buffer[key][:, :self.train_inner_steps], dtype=torch.long).to(self.device)
            elif key == 'v':
                batch[key] = torch.tensor(self.buffer[key][:, :self.train_inner_steps + 1], dtype=torch.float32).to(self.device)
            else:
                batch[key] = torch.tensor(self.buffer[key][:, :self.train_inner_steps], dtype=torch.float32).to(self.device)
        return batch


class ReplayBuffer:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.input_c = args.input_c
        self.input_h = args.input_h
        self.input_w = args.input_w
        self.action_dim = args.action_dim
        self.train_episode = args.train_episode
        self.train_inner_steps = args.train_inner_steps
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.buffer = None
        # self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.device = torch.device(args.device)
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {
            "obs": np.zeros((self.batch_size, self.train_inner_steps, self.input_c, self.input_h, self.input_w)),
            "v": np.zeros((self.batch_size, self.train_inner_steps + 1)),
            "a": np.zeros((self.batch_size, self.train_inner_steps)),
            "a_logprob": np.zeros((self.batch_size, self.train_inner_steps)),
            "r": np.zeros((self.batch_size, self.train_inner_steps)),
            "dw": np.zeros((self.batch_size, self.train_inner_steps)),
        }
        self.episode_num = 0

    def store_transition(self, step, obs, v, a, a_logprob, r, dw):
        self.buffer['obs'][self.episode_num][step] = obs
        self.buffer['v'][self.episode_num][step] = v
        self.buffer['a'][self.episode_num][step] = a
        self.buffer['a_logprob'][self.episode_num][step] = a_logprob
        self.buffer['r'][self.episode_num][step] = r
        self.buffer['dw'][self.episode_num][step] = dw

    def store_last_value(self, step, v):
        self.buffer['v'][self.episode_num][step] = v
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a':
                batch[key] = torch.tensor(self.buffer[key][:, :self.train_inner_steps], dtype=torch.long).to(
                    self.device)
            elif key == 'v':
                batch[key] = torch.tensor(self.buffer[key][:, :self.train_inner_steps + 1], dtype=torch.float32).to(
                    self.device)
            else:
                batch[key] = torch.tensor(self.buffer[key][:, :self.train_inner_steps], dtype=torch.float32).to(
                    self.device)
        return batch
