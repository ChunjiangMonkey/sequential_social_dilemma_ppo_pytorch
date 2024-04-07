import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SequentialSampler


def orthogonal_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRUCell):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


def conv2d_size_out(size, kernel_size=3, stride=1, padding=0):
    return (size - kernel_size + 2 * padding) // stride + 1


class ActorCriticTwoLayerCNN(nn.Module):
    def __init__(self, args):
        super(ActorCriticTwoLayerCNN, self).__init__()
        self.output_channel = 32
        self.fc_hidden_dim = 64
        conv_output_h = conv2d_size_out(conv2d_size_out(args.input_h))
        conv_output_w = conv2d_size_out(conv2d_size_out(args.input_w))
        self.fc_input_dim = conv_output_h * conv_output_w * self.output_channel
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        self.actor_cnn = nn.Sequential(nn.Conv2d(args.input_c, 16, kernel_size=3, stride=1),
                                       self.activate_func,
                                       nn.Conv2d(16, self.output_channel, kernel_size=3, stride=1),
                                       self.activate_func,
                                       nn.Flatten())
        self.actor_fc = nn.Sequential(nn.Linear(self.fc_input_dim, self.fc_hidden_dim),
                                      self.activate_func,
                                      nn.Linear(self.fc_hidden_dim, args.action_dim))

        self.critic_cnn = nn.Sequential(nn.Conv2d(args.input_c, 16, kernel_size=3, stride=1),
                                        self.activate_func,
                                        nn.Conv2d(16, self.output_channel, kernel_size=3, stride=1),
                                        self.activate_func,
                                        nn.Flatten())
        self.critic_fc = nn.Sequential(nn.Linear(self.fc_input_dim, self.fc_hidden_dim),
                                       self.activate_func,
                                       nn.Linear(self.fc_hidden_dim, 1))

    def forward_v(self, s):
        return self.critic_fc(self.critic_cnn(s))

    def forward_a(self, s):
        s = self.actor_fc(self.actor_cnn(s))
        return torch.softmax(s, dim=-1)


class ActorCriticOneLayerCNN(nn.Module):
    def __init__(self, args):
        super(ActorCriticOneLayerCNN, self).__init__()
        self.output_channel = 16
        conv_output_h = conv2d_size_out(args.input_h)
        conv_output_w = conv2d_size_out(args.input_w)
        self.fc_input_dim = conv_output_h * conv_output_w * self.output_channel
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        self.actor_cnn = nn.Conv2d(args.input_c, self.output_channel, kernel_size=3, stride=1)
        self.actor_fc = nn.Linear(self.fc_input_dim, args.action_dim)
        self.critic_cnn = nn.Conv2d(args.input_c, self.output_channel, kernel_size=3, stride=1)
        self.critic_fc = nn.Linear(self.fc_input_dim, 1)

    def forward_v(self, s):
        x = self.activate_func(self.critic_cnn(s))
        x = x.view(-1, self.fc_input_dim)
        return self.critic_fc(x)

    def forward_a(self, s):
        x = self.activate_func(self.actor_cnn(s))
        x = x.view(-1, self.fc_input_dim)
        x = self.actor_fc(x)
        return torch.softmax(x, dim=-1)


class ActorCriticTwoLayerRNN(nn.Module):
    def __init__(self, args):
        super(ActorCriticTwoLayerRNN, self).__init__()
        self.actor_rnn_hidden = None
        self.critic_rnn_hidden = None
        self.output_channel = 32
        self.rnn_hidden_dim = 64
        conv_output_h = conv2d_size_out(conv2d_size_out(args.input_h))
        conv_output_w = conv2d_size_out(conv2d_size_out(args.input_w))
        self.rnn_input_dim = conv_output_h * conv_output_w * self.output_channel
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        self.actor_cnn = nn.Sequential(nn.Conv2d(args.input_c, 16, kernel_size=3, stride=1),
                                       self.activate_func,
                                       nn.Conv2d(16, self.output_channel, kernel_size=3, stride=1),
                                       self.activate_func,
                                       nn.Flatten())
        self.actor_gru = nn.GRUCell(self.rnn_input_dim, self.rnn_hidden_dim)
        self.actor_fc = nn.Sequential(self.activate_func,
                                      nn.Linear(self.rnn_hidden_dim, args.action_dim))

        self.critic_cnn = nn.Sequential(nn.Conv2d(args.input_c, 16, kernel_size=3, stride=1),
                                        self.activate_func,
                                        nn.Conv2d(16, self.output_channel, kernel_size=3, stride=1),
                                        self.activate_func,
                                        nn.Flatten())
        self.critic_gru = nn.GRUCell(self.rnn_input_dim, self.rnn_hidden_dim)
        self.critic_fc = nn.Sequential(self.activate_func,
                                       nn.Linear(self.rnn_hidden_dim, 1))

    def forward_v(self, s):
        s = self.critic_cnn(s)
        self.critic_rnn_hidden = self.critic_gru(s, self.critic_rnn_hidden)
        s = self.critic_fc(self.critic_rnn_hidden)
        return s

    def forward_a(self, s):
        s = self.actor_cnn(s)
        self.actor_rnn_hidden = self.actor_gru(s, self.actor_rnn_hidden)
        s = self.actor_fc(self.actor_rnn_hidden)
        return torch.softmax(s, dim=-1)


class ActorCriticOneLayerRNN(nn.Module):
    def __init__(self, args):
        super(ActorCriticOneLayerRNN, self).__init__()
        self.actor_rnn_hidden = None
        self.critic_rnn_hidden = None
        self.output_channel = 16
        self.rnn_hidden_dim = 64
        conv_output_h = conv2d_size_out(args.input_h)
        conv_output_w = conv2d_size_out(args.input_w)
        self.rnn_input_dim = conv_output_h * conv_output_w * self.output_channel
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        self.actor_cnn = nn.Conv2d(args.input_c, self.output_channel, kernel_size=3, stride=1)
        self.actor_gru = nn.GRUCell(self.rnn_input_dim, self.rnn_hidden_dim)
        self.actor_fc = nn.Linear(self.rnn_hidden_dim, args.action_dim)

        self.critic_cnn = nn.Conv2d(args.input_c, self.output_channel, kernel_size=3, stride=1)
        self.critic_gru = nn.GRUCell(self.rnn_input_dim, self.rnn_hidden_dim)
        self.critic_fc = nn.Linear(self.rnn_hidden_dim, 1)

    def forward_v(self, s):
        x = self.activate_func(self.critic_cnn(s))
        x = x.view(-1, self.rnn_input_dim)
        self.critic_rnn_hidden = self.critic_gru(x, self.critic_rnn_hidden)
        x = self.critic_fc(self.activate_func(self.critic_rnn_hidden))
        return x

    def forward_a(self, s):
        x = self.activate_func(self.actor_cnn(s))
        x = x.view(-1, self.rnn_input_dim)
        self.actor_rnn_hidden = self.actor_gru(x, self.actor_rnn_hidden)
        x = self.actor_fc(self.activate_func(self.actor_rnn_hidden))
        return torch.softmax(x, dim=-1)


class PPO_discrete:
    def __init__(self, args):
        self.args = args
        self.use_rnn = args.use_rnn
        self.batch_size = args.batch_size * args.env_parallel_num  # batch size = episode_num * env_parallel_num
        self.mini_batch_size = args.mini_batch_size * args.env_parallel_num
        self.train_inner_steps = args.train_inner_steps
        self.train_episode = args.train_episode
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.device = torch.device(args.device)

        if args.network_layer == 1:
            if args.use_rnn:
                self.actor_critic = ActorCriticOneLayerRNN(args).to(self.device)
            else:
                self.actor_critic = ActorCriticOneLayerCNN(args).to(self.device)
        else:
            if args.use_rnn:
                self.actor_critic = ActorCriticTwoLayerRNN(args).to(self.device)
            else:
                self.actor_critic = ActorCriticTwoLayerCNN(args).to(self.device)

        if args.use_orthogonal_init:
            self.actor_critic.apply(orthogonal_init)

        if self.set_adam_eps:
            self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=1e-5)
        else:
            self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr)

    def choose_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        if obs.dim() != 4:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            dist = Categorical(probs=self.actor_critic.forward_a(obs))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
            return torch.squeeze(a).cpu().numpy(), torch.squeeze(a_logprob).cpu().numpy()

    def get_value(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        if obs.dim() != 4:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            v = self.actor_critic.forward_v(obs)
        return torch.squeeze(v).cpu().numpy()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()
        adv = []
        gae = 0
        with torch.no_grad():
            deltas = batch['r'] + self.gamma * batch['v'][:, 1:] * (1 - batch['dw']) - batch['v'][:, :-1]
            for t in reversed(range(self.train_inner_steps)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)
            v_target = adv + batch['v'][:, :-1]
            if self.use_adv_norm:
                if "cuda" in str(self.device):
                    adv_copy = copy.deepcopy(adv.cpu().numpy())
                else:
                    adv_copy = copy.deepcopy(adv.numpy())
                adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))
        actor_inputs, critic_inputs = self.get_inputs(batch)

        for k in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                if self.use_rnn:
                    self.actor_critic.critic_rnn_hidden = None
                    self.actor_critic.actor_rnn_hidden = None
                    probs_now, values_now = [], []
                    for t in range(self.train_inner_steps):
                        prob = self.actor_critic.forward_a(actor_inputs[index, t])
                        probs_now.append(prob)
                        v = self.actor_critic.forward_v(critic_inputs[index, t])
                        values_now.append(v)
                    probs_now = torch.squeeze(torch.stack(probs_now, dim=1), -1)
                    values_now = torch.squeeze(torch.stack(values_now, dim=1), -1)

                else:
                    probs_now = self.actor_critic.forward_a(
                        actor_inputs[index].reshape(self.mini_batch_size * self.train_inner_steps,
                                                    self.args.input_c, self.args.input_h, self.args.input_w))
                    probs_now = probs_now.reshape(self.mini_batch_size, self.train_inner_steps, -1)
                    values_now = self.actor_critic.forward_v(
                        critic_inputs[index].reshape(self.mini_batch_size * self.train_inner_steps,
                                                     self.args.input_c, self.args.input_h, self.args.input_w))
                    values_now = values_now.reshape(self.mini_batch_size, self.train_inner_steps)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()
                a_logprob_now = dist_now.log_prob(batch['a'][index])
                ratios = torch.exp(a_logprob_now - batch['a_logprob'][index].detach())
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                critic_loss = (values_now - v_target[index]) ** 2
                ac_loss = actor_loss + critic_loss
                self.optimizer.zero_grad()
                ac_loss.mean().backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = self.lr * (1 - total_steps / self.args.max_train_steps)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch):
        return batch['obs'], batch['obs']

    def save_model(self, filepath):
        torch.save({"actor_critic": self.actor_critic.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
