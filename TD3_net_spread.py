import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, maxaction, user_num):
		super(Actor, self).__init__()
		self.fc_t_sprend = nn.Linear(1, user_num)
		self.fc_loc_sprend = nn.Linear(2, user_num)
		self.l1 = nn.Linear(state_dim - 3 + user_num * 2, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.l3 = nn.Linear(net_width, action_dim)

		self.maxaction = maxaction

	def forward(self, state):
		t = self.fc_t_sprend(state[:, -1:])
		loc = self.fc_loc_sprend(state[:, -3:-1])
		a = torch.cat([t, loc, state[:, :-3]], 1)
		a = torch.tanh(self.l1(a))
		a = torch.tanh(self.l2(a))
		a = torch.tanh(self.l3(a)) * self.maxaction
		return a


class Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, net_width,user_num):
		super(Q_Critic, self).__init__()
		self.fc_t_sprend = nn.Linear(1, user_num)
		self.fc_loc_sprend = nn.Linear(2, user_num)
		self.l1 = nn.Linear(state_dim - 3 + user_num * 2+action_dim, net_width)
		# Q1 architecture
		self.l2 = nn.Linear(net_width, net_width)
		self.l3 = nn.Linear(net_width, 1)

		# Q2 architecture
		self.fc_t_sprend2 = nn.Linear(1, user_num)
		self.fc_loc_sprend2 = nn.Linear(2, user_num)
		self.l4 = nn.Linear(state_dim - 3 + user_num * 2+action_dim, net_width)
		self.l5 = nn.Linear(net_width, net_width)
		self.l6 = nn.Linear(net_width, 1)


	def forward(self, state, action):
		t = self.fc_t_sprend(state[:, -1:])
		loc = self.fc_loc_sprend(state[:, -3:-1])
		sa = torch.cat([t, loc, state[:, :-3], action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		t2 = self.fc_t_sprend2(state[:, -1:])
		loc2 = self.fc_loc_sprend2(state[:, -3:-1])
		sa2 = torch.cat([t2, loc2, state[:, :-3], action], 1)

		q2 = F.relu(self.l4(sa2))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		t = self.fc_t_sprend(state[:, -1:])
		loc = self.fc_loc_sprend(state[:, -3:-1])
		sa = torch.cat([t, loc, state[:, :-3], action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1



class TD3(object):
	def __init__(
		self,
		env_with_Dead,
		state_dim,
		action_dim,
		max_action,
        user_num,
		train_path,
		gamma=0.99,
		net_width=128,
		a_lr=1e-4,
		c_lr=1e-4,
		Q_batchsize = 256
	):
		self.writer = SummaryWriter(train_path)
		max_action = torch.tensor(max_action, dtype=torch.float32).to(device)
		self.actor = Actor(state_dim, action_dim, net_width, max_action,user_num).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = Q_Critic(state_dim, action_dim, net_width,user_num).to(device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.env_with_Dead = env_with_Dead
		self.action_dim = action_dim
		self.max_action = max_action
		self.gamma = gamma
		self.policy_noise = 0.2*max_action
		self.noise_clip = 0.5*max_action
		self.tau = 0.005
		self.Q_batchsize = Q_batchsize
		self.delay_counter = -1
		self.delay_freq = 1
		self.q_iteration = 0
		self.a_iteration = 0


	def select_action(self, state):#only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			a = self.actor(state)
		return a.cpu().numpy().flatten()

	def train(self,replay_buffer):
		self.delay_counter += 1
		with torch.no_grad():
			s, a, r, s_prime, dead_mask = replay_buffer.sample(self.Q_batchsize)
			noise = torch.max(-self.noise_clip, torch.min(torch.randn_like(a) * self.policy_noise, self.noise_clip))
			smoothed_target_a = torch.max(-self.max_action,torch.min(
					self.actor_target(s_prime) + noise, self.max_action))

		# Compute the target Q value
		target_Q1, target_Q2 = self.q_critic_target(s_prime, smoothed_target_a)
		target_Q = torch.min(target_Q1, target_Q2)
		'''DEAD OR NOT'''
		if self.env_with_Dead:
			target_Q = r + (1 - dead_mask) * self.gamma * target_Q  # env with dead
		else:
			target_Q = r + self.gamma * target_Q  # env without dead


		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)

		# Compute critic loss
		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		self.writer.add_scalar('q_loss', q_loss, self.q_iteration)
		# Optimize the q_critic
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()
		self.q_iteration += 1
		if self.delay_counter == self.delay_freq:
			# Update Actor
			a_loss = -self.q_critic.Q1(s,self.actor(s)).mean()
			self.writer.add_scalar('a_loss', a_loss, self.a_iteration)
			self.actor_optimizer.zero_grad()
			a_loss.backward()
			self.actor_optimizer.step()
			self.a_iteration += 1
			# Update the frozen target models
			for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			self.delay_counter = -1


	def save(self,episode,model_path):
		torch.save(self.actor.state_dict(), model_path+"td3_actor{}.pth".format(episode))
		torch.save(self.q_critic.state_dict(), model_path+"td3_q_critic{}.pth".format(episode))


	def load(self,episode):

		self.actor.load_state_dict(torch.load("td3_actor{}.pth".format(episode)))
		self.q_critic.load_state_dict(torch.load("td3_q_critic{}.pth".format(episode)))



