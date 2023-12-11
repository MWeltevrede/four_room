# 4-room
A simple 4-room grid world environment to test generalisation behaviour of RL agents. 

Requires Minigrid:
```
pip install minigrid
```
and NetworkX
```
pip install networkx
```
## Usage
From the parent directory of this repository you can import the environment with:
```
import gymnasium as gym
import dill
from four_room.env import FourRoomsEnv
from four_room.env_wrappers import gym_wrapper

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('four_room/configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
                                agent_pos=train_config['agent positions'], 
                                goal_pos=train_config['goal positions'], 
                                doors_pos=train_config['topologies'], 
                                agent_dir=train_config['agent directions']))
```
Similarly for the reachable testing config (```'four_room/configs/fourrooms_test_100_config.pl'```) and the unreachable testing config (```'four_room/configs/fourrooms_test_0_config.pl'```).

Utility functions for turning observations into state or images can be found in ```utils.py``` and code for finding the optimal trajectories/q-values for any state can be found in ```shortest_path.py```.
