"""
Maze environment for reinforcement learning, with the python package tkinter.
Red rectangle:      explorer.
Black rectangle:    hells       [reward = -1]
Yellow bin circle:  paradise    [reward = +1]
All other state:    ground      [reward = 0]
Alse referenced the tutorial of morvanzhou: https://morvanzhou.github.io/tutorials/, 
but make some change, initial state has became random state
"""


import numpy as np
import random
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40  # pixels


def is_different(*numbers):
    length = len(numbers)
    return len(set(numbers)) == length


class Maze(tk.Tk, object):
    def __init__(self, opt):
        super(Maze, self).__init__()
        self.action_space = opt.action_space
        self.maze_w = opt.maze_w
        self.maze_h = opt.maze_h
        self.hole_num = opt.hole_num
        self.fresh_time = opt.fresh_time
        self.n_actions = len(self.action_space)
        self.title("{}X{} MAZE".format(self.maze_h, self.maze_w))
        self.geometry("{}x{}".format(self.maze_h * UNIT, self.maze_w * UNIT))
        self.origin_x = UNIT/2 + random.randint(0, self.maze_w - 1) * UNIT
        self.origin_y = UNIT/2 + random.randint(0, self.maze_h - 1) * UNIT
        self.origin = [self.origin_x, self.origin_y]
        seed = 1  # 736
        if opt.play:
            seed = int(input("Please enter the seed: "))
        while True:
            random.seed(seed)
            for i in range(self.hole_num):
                self.__setattr__("hole{}_center".format(i),
                                 [UNIT/2 + random.randint(0, self.maze_w - 1) * UNIT,
                                  UNIT/2 + random.randint(0, self.maze_h - 1) * UNIT])
            self.oval_center = [UNIT/2 + random.randint(0, self.maze_w - 1) * UNIT,
                                UNIT/2 + random.randint(0, self.maze_h - 1) * UNIT]
            self.sum_centers = [sum(self.__getattribute__("hole{}_center".format(i)))
                                 for i in range(self.hole_num)]
            if opt.play:
                break
            if len(self.sum_centers) != self.hole_num:
                continue
            self.sub_centers = [(self.__getattribute__("hole{}_center".format(i))[0] - 
                                 self.__getattribute__("hole{}_center".format(i))[1]) 
                                 for i in range(self.hole_num)]
            
            self.sum_centers.extend([sum(self.origin), sum(self.oval_center)])
            if is_different(*self.sum_centers):
                break
            else:
                self.sub_centers.extend([self.origin[0] - self.origin[1], self.oval_center[0] - self.oval_center[1]])
                if is_different(*self.sub_centers):
                    break
            seed += 1
        print("Remember the seed: {}".format(seed))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg="white",
                                height=self.maze_h * UNIT,
                                width=self.maze_w * UNIT)
        # create grids
        for c in range(0, self.maze_w * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.maze_h * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.maze_h * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.maze_w * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create holes
        for i in range(self.hole_num):
            hole_center = self.__getattribute__("hole{}_center".format(i))
            self.__setattr__("hole{}".format(i), 
                             self.canvas.create_rectangle(
                                hole_center[0] - 15, hole_center[1] - 15,
                                hole_center[0] + 15, hole_center[1] + 15,
                                fill="black"))

        # create oval
        self.oval = self.canvas.create_oval(
            self.oval_center[0] - 15, self.oval_center[1] - 15,
            self.oval_center[0] + 15, self.oval_center[1] + 15,
            fill="yellow"
        )

        # create red rect
        self.rect = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] -15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill="red"
        )

        # pack all
        self.canvas.pack()
    
    def reset(self):
        while True:
            self.origin_x = UNIT/2 + random.randint(0, self.maze_w - 1) * UNIT
            self.origin_y = UNIT/2 + random.randint(0, self.maze_h - 1) * UNIT
            self.origin = [self.origin_x, self.origin_y]
            self.sum_centers = [sum(self.__getattribute__("hole{}_center".format(i)))
                                 for i in range(self.hole_num)]
            self.sub_centers = [(self.__getattribute__("hole{}_center".format(i))[0] - 
                                 self.__getattribute__("hole{}_center".format(i))[1]) 
                                 for i in range(self.hole_num)]
            
            if is_different(self.sum_centers.extend([sum(self.origin), 
                             sum(self.oval_center)])):
                break
            else:
                if is_different(self.sub_centers.extend([self.origin[0] - self.origin[1],
                                self.oval_center[0] - self.oval_center[1]])):
                    break
        self.update()
        time.sleep(self.fresh_time*2)
        self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill="red"
        )
        rect_coords = self.canvas.coords(self.rect)
        observation = (np.array(rect_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(self.maze_h * UNIT)
        return observation
    
    def step(self, action):
        state = self.canvas.coords(self.rect)
        base_action = [0, 0]
        done = False
        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
            else:
                reward = -1
                done = True
        elif action == 1:  # down
            if state[1] < (self.maze_h * UNIT - UNIT):
                base_action[1] += UNIT
            else:
                reward = -1
                done = True
        elif action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
            else:
                reward = -1
                done = True
        elif action == 3:  # right
            if state[0] < self.maze_w * UNIT -UNIT:
                base_action[0] += UNIT
            else:
                reward = -1
                done = True
        
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        state = self.canvas.coords(self.rect)
        # reward function
        if state == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif state in [self.canvas.coords(self.__getattribute__("hole{}".format(i))) for i in range(self.hole_num)]:
            reward = -1
            done = True
        else:
            reward = 0
        observation = (np.array(state[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(self.maze_h * UNIT)
        return observation, reward, done
    
    def render(self):
        time.sleep(self.fresh_time)
        self.update()


def update():
    for t in range(10):
        state = env.reset()
        while True:
            env.render()
            a = 0
            s, r, done = env.step(a)
            if done:
                break
    env.destroy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh_time", dest="fresh_time", type=float, default=0.1, help="fresh time for one move")
    parser.add_argument("--maze_h", dest="maze_h", type=int, default=8, help="the height of the maze(unit:unit)")
    parser.add_argument("--maze_w", dest="maze_w", type=int, default=8, help="the width of the maze(unit:unit)")
    parser.add_argument("--hole_num", dest="hole_num", type=int, choices=range(5), default=2, help="the number of holes")
    parser.add_argument("--action_space", nargs="+", default=["u", "d", "l", "r"], help="all available actions")
    opt = parser.parse_args()
    env = Maze(opt)
    env.after(10, update)
    env.mainloop()

