import argparse
import os

import numpy as np
import setproctitle
import tqdm
import wandb
from matplotlib import pyplot as plt

from normalization import Normalization, RewardScaling
from ppo_agent import PPO_discrete
from replaybuffer import VectorizedReplayBuffer
from ssd_pettingzoo.clean_up import CleanupEnv
from ssd_pettingzoo.vectorize_wrapper import SubprocVectorEnv


def get_args():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--exp_name", type=str, default="clean_up_baseline_local", help="Name of this experiment")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device")
    parser.add_argument("--train_episode", type=int, default=4, help=" Maximum number of training episode")
    parser.add_argument("--train_inner_steps", type=int, default=512,
                        help=" Maximum number of steps per episode")
    parser.add_argument("--save_freq", type=int, default=128, help="The number of episodes per saving")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="The number of episodes for sampling for one training")
    parser.add_argument("--mini_batch_size", type=int, default=1,
                        help="The number of episodes used for one training epoch")
    parser.add_argument("--use_rnn", default=True, action="store_true", help="Whether to use RNN")
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Whether to use wandb")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of NNs")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=16, help="The number of update for one training")
    parser.add_argument("--use_rgb_norm", type=bool, default=True, help="map 0-255 to 0-1")
    parser.add_argument("--network_layer", type=int, default=2,
                        help="The layer number of policy network and value network")
    parser.add_argument("--env_parallel_num", default=4, type=int,
                        help="The number of parallel envs (i.e. the size of 'batch' in  the input shape of NN (batch, channel, height, width))")
    parser.add_argument("--use_collective_reward", default=False, action="store_true",
                        help="Whether to use collective reward")
    parser.add_argument("--use_inequity_averse_reward", default=False, action="store_true",
                        help="Whether to use inequity averaging reward")
    # PPO tricks
    parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick: advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick: state normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="Trick:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=False, help="Trick: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick: tanh activation function")
    args = parser.parse_args()
    return args


class Runner:
    def __init__(self, args):
        self.args = args
        setproctitle.setproctitle(f"{self.args.exp_name}")

        env = [CleanupEnv(use_collective_reward=args.use_collective_reward,
                          inequity_averse_reward=args.use_inequity_averse_reward) for _ in
               range(args.env_parallel_num)]
        self.env = SubprocVectorEnv(env)

        self.args.input_h, self.args.input_w, self.args.input_c = self.env.observation_space["curr_obs"].shape
        self.args.action_dim = self.env.action_space.n

        self.policies = {agent_id: PPO_discrete(args) for agent_id in self.env.agents}
        self.replay_buffers = {agent_id: VectorizedReplayBuffer(args) for agent_id in self.env.agents}

        # checkpoints folder
        self.checkpoint_path = os.path.join("checkpoints", args.exp_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        if args.use_wandb:
            self.init_wandb()

        if args.use_state_norm:
            self.state_norm = Normalization(shape=(args.env_parallel_num, args.input_c, args.input_h, args.input_w))
        if args.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=(args.env_parallel_num, 1), gamma=args.gamma)

        self.total_steps = 0

        # true reward without any shaping
        self.returns = {agent_id: [] for agent_id in self.env.agents}
        # the number of collected apples
        self.collected_apple_nums = {agent_id: [] for agent_id in self.env.agents}
        # the number of clean action
        self.clean_nums = {agent_id: [] for agent_id in self.env.agents}
        # the number of fire action
        self.fire_nums = {agent_id: [] for agent_id in self.env.agents}
        # mean apple num
        self.env_apple_nums = []
        # mean waste density
        self.env_waste_densities = []

    def init_wandb(self):
        # replace with your own wandb key
        wandb_config = {keys: value for keys, value in self.args.__dict__.items()}
        wandb.init(project='ssd_prosocial_baseline', name=self.args.exp_name, config=wandb_config)

    def train(self):
        for episode_i in tqdm.tqdm(range(self.args.train_episode)):
            episode_returns, env_apple_num, env_waste_density, infos = self.run_episode(evaluate=False)
            for agent_id in self.env.agents:
                # train
                if self.replay_buffers[agent_id].episode_num == self.args.batch_size:
                    self.policies[agent_id].train(self.replay_buffers[agent_id], self.total_steps)
                    self.replay_buffers[agent_id].reset_buffer()
                if episode_i % self.args.save_freq == 0 and episode_i > 0:
                    model_name = f"episode_{episode_i}_agent_{agent_id}.pth"
                    path = os.path.join(self.checkpoint_path, model_name)
                    self.policies[agent_id].save_model(path)

            # process data
            env_apple_num = env_apple_num / self.args.train_inner_steps
            env_waste_density = env_waste_density / self.args.train_inner_steps
            if self.args.use_wandb:
                log_dict = {}
                for agent_id in self.env.agents:
                    log_dict[f"return_{agent_id}"] = episode_returns[agent_id]
                    log_dict[f"collected_apple_num_{agent_id}"] = infos[agent_id]["collected_apple_num"]
                    log_dict[f"fire_num_{agent_id}"] = infos[agent_id]["fire_num"]
                    log_dict[f"clean_num_{agent_id}"] = infos[agent_id]["clean_num"]
                log_dict["env_apple_num"] = env_apple_num
                log_dict["env_waste_density"] = env_waste_density
                wandb.log(log_dict)

            for agent_id in self.env.agents:
                self.collected_apple_nums[agent_id].append(infos[agent_id]["collected_apple_num"])
                self.fire_nums[agent_id].append(infos[agent_id]["fire_num"])
                self.clean_nums[agent_id].append(infos[agent_id]["clean_num"])
                self.returns[agent_id].append(episode_returns[agent_id])
            self.env_apple_nums.append(env_apple_num)
            self.env_waste_densities.append(env_waste_density)

        return self.returns, self.collected_apple_nums

    def run_episode(self, evaluate=False):
        actions = {}
        action_logprobs = {}
        values = {}
        obs_rgb = {}

        obs = self.env.reset()
        episode_returns = {}
        env_apple_num = 0
        env_waste_density = 0

        for agent_id in self.env.agents:
            episode_returns[agent_id] = 0
            if self.args.use_rnn:
                self.policies[agent_id].actor_critic.actor_rnn_hidden = None
                self.policies[agent_id].actor_critic.critic_rnn_hidden = None
            obs_rgb[agent_id] = obs[agent_id]["curr_obs"].transpose(0, 3, 1, 2)
            if self.args.use_rgb_norm:
                obs_rgb[agent_id] = obs_rgb[agent_id] / 255
            if self.args.use_state_norm:
                obs_rgb[agent_id] = self.state_norm(obs_rgb[agent_id])
            if self.args.use_reward_scaling:
                self.reward_scaling.reset()

        for step in range(self.args.train_inner_steps):
            self.total_steps += 1
            for agent_id in self.env.agents:
                a, a_logprob = self.policies[agent_id].choose_action(obs_rgb[agent_id])
                v = self.policies[agent_id].get_value(obs_rgb[agent_id])
                actions[agent_id] = a
                action_logprobs[agent_id] = a_logprob
                values[agent_id] = v

            next_obs, rewards, dones, infos = self.env.step(actions)

            next_obs_rgb = {}
            for agent_id in self.env.agents:
                next_obs_rgb[agent_id] = next_obs[agent_id]["curr_obs"].transpose(0, 3, 1, 2)

                if self.args.use_rgb_norm:
                    next_obs_rgb[agent_id] = next_obs_rgb[agent_id] / 255
                if self.args.use_state_norm:
                    next_obs_rgb[agent_id] = self.state_norm(next_obs_rgb[agent_id])
                if not evaluate:
                    if self.args.use_reward_scaling:
                        rewards[agent_id] = self.reward_scaling(rewards[agent_id])

                    self.replay_buffers[agent_id].store_transition(step, obs_rgb[agent_id],
                                                                   values[agent_id],
                                                                   actions[agent_id],
                                                                   action_logprobs[agent_id],
                                                                   rewards[agent_id],
                                                                   dones[agent_id])
                obs_rgb[agent_id] = next_obs_rgb[agent_id]
                episode_returns[agent_id] += infos[agent_id]["reward"]
            env_apple_num += infos["apple_num"]
            env_waste_density += infos["waste_density"]

        for agent_id in self.env.agents:
            v = self.policies[agent_id].get_value(obs_rgb[agent_id])
            self.replay_buffers[agent_id].store_last_value(step + 1, v)
        return episode_returns, env_apple_num, env_waste_density, infos


if __name__ == '__main__':
    args = get_args()
    runner = Runner(args)
    return_list, collected_apple_nums = runner.train()
    for key in return_list.keys():
        return_list[key] = np.array(return_list[key])
        collected_apple_nums[key] = np.array(collected_apple_nums[key])
        plt.plot(return_list[key])
        plt.show()
        plt.plot(collected_apple_nums[key])
        plt.show()
