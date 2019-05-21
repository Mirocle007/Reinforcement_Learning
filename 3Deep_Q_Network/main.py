"""
Main function of the 2Qtable_2d_env, combind the agent and the environment.
"""


import argparse
from tqdm import tqdm

from maze_env import Maze
from agent import Agent


def str2bool(s):
    if s.lower() in "trueyes":
        return True
    else:
        return False


parser = argparse.ArgumentParser()
parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.9, help="greedy police")
parser.add_argument("--lr", dest="learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--gamma", dest="gamma", type=float, default=0.9, help="discount factor of Q value")
parser.add_argument("--episodes", dest="episodes", type=int, default=100, help="maximum episodes")
parser.add_argument("--fresh_time", dest="fresh_time", type=float, default=0.1, help="fresh time with unit ms for one move")
parser.add_argument("--maze_h", dest="maze_h", type=int, default=8, help="the height of the maze(unit:unit)")
parser.add_argument("--maze_w", dest="maze_w", type=int, default=8, help="the width of the maze(unit:unit)")
parser.add_argument("--hole_num", dest="hole_num", type=int, choices=range(16), default=2, help="the number of holes")
parser.add_argument("--action_space", nargs="+", default=["u", "d", "l", "r"], help="all available actions")
parser.add_argument("--n_state", dest="n_state", default=2, 
                    help="number of input size of the nework, and alse the number of representing state")
parser.add_argument("--replace_target_iter", dest="replace_target_iter", type=int, 
                    default=64, help="number of replacing target iter")
parser.add_argument("--memory_size", dest="memory_size", type=int, 
                    default=1000, help="memory size")
parser.add_argument("--batch_size", dest="batch_size", type=int, 
                    default=32, help="batch_size")
parser.add_argument("--output_graph", dest="output_graph", type=str2bool, 
                    default=False, help="output_graph")

opt = parser.parse_args()


def train():
    for episode in tqdm(range(opt.episodes)):
        state = env.reset()
        while True:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state)
            agent.learn()
            state = next_state
            if done:
                break
        
    print("game over")
    env.destroy()


if __name__ == "__main__":
    env = Maze(opt)
    agent = Agent(opt)
    env.after(10, train)
    env.mainloop()
    agent.plot_cost()
