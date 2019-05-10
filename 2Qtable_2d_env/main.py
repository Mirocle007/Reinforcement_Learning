"""
Main function of the 2Qtable_2d_env, combind the agent and the environment.
"""


import argparse

from maze_env import Maze
from agent import Agent


parser = argparse.ArgumentParser()
parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.9, help="greedy police")
parser.add_argument("--lr", dest="learning_rate", type=float, default=0.1, help="learning rate")
parser.add_argument("--gamma", dest="gamma", type=float, default=0.9, help="discount factor of Q value")
parser.add_argument("--episodes", dest="episodes", type=int, default=100, help="maximum episodes")
parser.add_argument("--fresh_time", dest="fresh_time", type=float, default=0.1, help="fresh time with unit ms for one move")
parser.add_argument("--maze_h", dest="maze_h", type=int, default=8, help="the height of the maze(unit:unit)")
parser.add_argument("--maze_w", dest="maze_w", type=int, default=8, help="the width of the maze(unit:unit)")
parser.add_argument("--hole_num", dest="hole_num", type=int, choices=range(6), default=2, help="the number of holes")
parser.add_argument("--action_space", nargs="+", default=["u", "d", "l", "r"], help="all available actions")
opt = parser.parse_args()


def train():
    for episode in range(opt.episodes):
        state = env.reset()
        while True:
            env.render()
            action = agent.choose_action(str(state))
            new_state, reward, done = env.step(action)
            agent.learn(str(state), action, reward, str(new_state), done)
            state = new_state
            if done:
                break
        
    print("game over")
    print(agent.q_table)
    env.destroy()


if __name__ == "__main__":
    env = Maze(opt)
    agent = Agent(opt)
    env.after(10, train)
    env.mainloop()
