from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
import imageio
import numpy as np
from pyvirtualdisplay import Display  
import dill
import gymnasium as gym
from four_room.utils import obs_to_state
from four_room.shortest_path import find_all_action_values

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

size = 11

# with open('configs/train.pl', 'rb') as file:
with open('train11.pl', 'rb') as file:
    train_config = dill.load(file)

env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
                                                    agent_pos= train_config['agent positions'],
                                                    goal_pos = train_config['goal positions'],
                                                    doors_pos = train_config['topologies'],
                                                    agent_dir = train_config['agent directions'],
                                                    size=size,
                                                    render_mode="rgb_array"),
                                                    original_obs=True)

with Display(visible=False) as disp:
    images = []
    for i in range(len(train_config['topologies'])):
        obs, _ = env.reset()
        img = env.render()
        done = False
        while not done:
            images.append(img)
            # retrieve your action here
            state = obs_to_state(obs)
            q = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99, size)
            action = np.array(q).argmax()
            # action = env.action_space.sample() # for example, just sample a random action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            img = env.render()

                


    imageio.mimsave('rendered_episode.gif', [np.array(img) for i, img in enumerate(images) if i%1 == 0], duration=200)
