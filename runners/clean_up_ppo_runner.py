import os

import setproctitle
import tqdm
import wandb

from agents.normalization import Normalization, RewardScaling
from agents.ppo_agent import PPO_discrete
from agents.replaybuffer import VectorizedReplayBuffer
from envs.clean_up import CleanupEnv
from envs.env_wrapper import SubprocVectorWrapper


class Runner:
    def __init__(self, args):
        self.args = args
        setproctitle.setproctitle(f"{self.args.exp_name}")

        env = [CleanupEnv(use_collective_reward=args.use_collective_reward,
                          inequity_averse_reward=args.use_inequity_averse_reward) for _ in range(args.env_parallel_num)]
        self.env = SubprocVectorWrapper(env)

        self.args.input_h, self.args.input_w, self.args.input_c = self.env.observation_space["curr_obs"].shape
        self.args.action_dim = self.env.action_space.n

        self.policies = {agent_id: PPO_discrete(args) for agent_id in self.env.agents}
        self.replay_buffers = {agent_id: VectorizedReplayBuffer(args) for agent_id in self.env.agents}
        self.data_collector = None

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

    def init_wandb(self):
        # replace with your own wandb key
        os.environ['WANDB_API_KEY'] = self.args.wandb_api_key
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
                log_dict = self.data_collector.info_episode
                wandb.log(log_dict)

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
        step = 0
        while step < self.args.train_inner_steps:
            # for step in range(self.args.train_inner_steps):
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
            step += 1
        for agent_id in self.env.agents:
            v = self.policies[agent_id].get_value(obs_rgb[agent_id])
            self.replay_buffers[agent_id].store_last_value(step + 1, v)
        return episode_returns, env_apple_num, env_waste_density, infos
