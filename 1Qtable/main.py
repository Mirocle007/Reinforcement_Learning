"""
An example for RL using Qtable method.
An agent "I" is on the random position of a one dimensional world, the target point is on the
Right of the world.
Run this program and see how the agent will find the target point.
Referenced the tutorial of morvanzhou: https://morvanzhou.github.io/tutorials/
"""

import random
import time

import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n_states", dest="n_states", type=int, default=6, help="the length of the 1d world")
parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.9, help="greedy police")
parser.add_argument("--lr", dest="alpha", type=float, default=0.1, help="learning rate")
parser.add_argument("--gamma", dest="gamma", type=float, default=0.9, help="discount factor of Q value")
parser.add_argument("--episodes", dest="episodes", type=int, default=20, help="maximum episodes")
parser.add_argument("--fresh_time", dest="fresh_time", type=float, default=0.3, help="fresh time for one move")
opt = parser.parse_args()

ACTIONS = ['left', 'right']  # available actions


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    return table


class Env(object):
    def __init__(self, opt):
        self.opt = opt
    
    def get_state(self):
        return self.state

    def reset(self):
        self.state = random.randint(0, opt.n_states-2)
        self.done = False
    
    def step(self, action):
        if action == 'right':    # move right
            if self.state == self.opt.n_states - 2:   # terminate
                self.state += 1
                self.done = True
                reward = 1
            else:
                self.state += 1
                reward = 0
        else:   # move left
            if self.state == 0:
                self.done = True  # reach the wall
                reward = -1
            else:
                self.state -= 1
                reward = 0
        return self.state, reward, self.done

    def render(self, episode, step_counter):
        env_list = ["-"]*(self.opt.n_states-1) + ["T"]
        if self.done:
            interaction = "Episode {}: total_steps = {}".format(episode+1, step_counter)
            print("\r{}".format(interaction), end="")
            time.sleep(2)
            print("\r{}".format(" "*50), end="")
        else:
            env_list[self.state] = "I"
            interaction = "".join(env_list)
            print("\r{}".format(interaction), end="")
            time.sleep(opt.fresh_time)


    def choose_action(self, q_table):
        state_actions = q_table.iloc[self.state, :]
        if (np.random.uniform() > self.opt.epsilon) or ((state_actions == 0).all()):
            action_name = np.random.choice(ACTIONS)
        else:
            action_name = state_actions.idxmax()
        return action_name


def train(env, opt):
    # train to get a good qtable
    q_table = build_q_table(opt.n_states, ACTIONS)
    for episode in range(opt.episodes):
        step_counter = 0
        env.reset()
        env.render(episode, step_counter)
        while not env.done:
            pre_state = env.get_state()
            action = env.choose_action(q_table)
            state, reward, done = env.step(action)
            env.render(episode, step_counter)
            q_predict = q_table.loc[pre_state, action]
            if env.done:
                q_target = reward     # next state is terminal
            else:
                q_target = reward + opt.gamma * q_table.iloc[state, :].max()   # next state is not terminal

            q_table.loc[pre_state, action] += opt.alpha * (q_target - q_predict)  # update
            step_counter += 1
    return q_table


if __name__ == "__main__":
    env = Env(opt)
    env.reset()
    q_table = train(env, opt)
    print('\r\nQ-table:\n')
    print(q_table)
