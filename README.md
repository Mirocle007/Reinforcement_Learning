# Reinforcement_Learning
codes while learning reinforcement learning

note: The environment is in python3.6.7


## 1. 1Qtable_1d_env

You can run main.py with parameters like this:<br>
```Shell
$ python main.py --n_state 20 --fresh_time 0.001 --lr 0.9 --episodes 100
```

To find other parameters, you can run the command: 
```Shell
$ python main.py --help
```


## 2. 2Qtable_2d_env

This directory is the extendtion of 1d_env. Thus we set up a maze of 2 dimension.
we can change the number of holes, and the position of holes, target and initial agent is random. 
You can run main.py with parameters like this:<br>
```Shell
$ python main.py --maze_h 10 --maze_w 10 --hole_num 5
```

To find other parameters, you can run the command: 
```Shell
$ python main.py --help
```

## 3. 3Deep_Q_Network
This directory is the same as 2Qtable_2d_env except that this agent uses a neural network(a simple two layer network).You can run main.py like this:
```Shell
python main.py --episodes 10000 --fresh_time 0.1 --hole_num 10 --lr 0.01 --output_graph t
```

To use tensorbord, you can run this in another terminal:
```Shell
tensorboard --logdir=logs
```