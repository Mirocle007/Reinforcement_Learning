"""
Independent of the environment, the agent make all decisions.
"""
 

import numpy as np

import pandas as pd


class Agent(object):
    def __init__(self, opt):
        self.actions = list(range(len(opt.action_space)))
        self.lr = opt.learning_rate
        self.gamma = opt.gamma
        self.epsilon = opt.epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
    
    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action
    
    def learn(self, state, action, reward, new_state, done):
        self.check_state_exist(new_state)
        q_predict = self.q_table.loc[state, action]
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * self.q_table.loc[new_state, :].max()
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)
    
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )

