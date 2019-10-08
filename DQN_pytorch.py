#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 05/10/2019 22:47 
@Author: XinZhi Yao 
"""

import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CarPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATE = env.observation_space.shape[0]


class Net(nn.Module):
    # construct two neural networks use this structure
    # Q-target and Q-eval
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATE, 10)
        # random initialization weight
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # calculate action value for selecting action
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.estimate_net, self.reality_net = Net(), Net()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        # N_STATE = env.observation_space.shape[0]
        self.memory = np.zeros(MEMORY_CAPACITY, N_STATE * 2 + 2)  # initialize memory
        self.optimizer = optim.Adam(self.estimate_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        """
        take action based on observations
        :param x: observations
        :return: action
        """
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < EPSILON:  # greedy police
            actions_value = self.estimate_net(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0, 0]
        else: # randomly select an action
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        """
        construct memory store
        :param s: status value at now
        :param a: action value at now
        :param r: reward at now
        :param s_: status value at next step
        :return: None
        """
        transition = np.hstack(s, [a, r], s_)
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        """
        learning method of Q-Learning
        :return: None
        """
        # target net update (Fix Q-targets)
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.reality_net.load_state_dict(self.estimate_net.state_dict())
        self.learn_step_counter += 1
        # eval net update
        # randomly take out memories
        sample_index = np.random.choice(MEMORY_CAPACITY,  )
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATE]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATE:N_STATE+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATE + 1:N_STATE + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATE:]))

        """
        # Q(s,a) means the value produced by the state s and action a
        # equivalent to the value in Q-learning table
        
        # input s and a, get value(s, a)
        # or input s, get action-value {a1,...,ak}, and
        # select the action having maximum action-value
        # according to the Q-learning principle.
        """
        """
        # Gathers values along an axis specified by dim.
        # Take out the value of the precious action.
        # Q estimate for update neural network Q(s2,a1) Q(s2,a2)
        # Q estimate network with the latest parameters
        """
        q_eval = self.estimate_net(b_s).gether(dim=1, index=b_a)
        """
        # Named Q-reality bacause it is rewarded 
        # and actually ia an estimate 
        # Q reality network have early parameters
        # Q reality Q(s',a1) Q(s',a2) Next step s' estimate
        # At this time, Q-reality net not been update
        # and the actual behavior has not yet occurred
        """
        q_next = self.reality_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0]

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\nCollectiong experience...')
for i_episode in range(400):
    s = env.reset()
    while True:
        # rendering environment
        env.render()

        # taking action
        a = dqn.choose_action(s)

        # action feedback
        s_, r, done, info = env.step(a)

        # modify the reward function
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.8
        r = r1 + r2

        # saving memory (Experience replay)
        dqn.store_transition(s, a, r, s_)

        # start learning after building a memory store
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if done:
            break
        s = s_