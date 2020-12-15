import imageio
import gym
import numpy as np
import copy
import sys
from copy import deepcopy
from .sokoban_env import SokobanEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from .render_utils import room_to_rgb
import time
import os
from PIL import Image


class AgentDemoWrapper(gym.Wrapper):
    def __init__(self, env, agent_path=None, tempdir_path=None):
        self.alg = PPO('MlpPolicy', env, verbose=0)
        if agent_path is not None:
            load_path = agent_path
            self.alg.set_parameters(load_path, exact_match=True)

        if tempdir_path is None:
            tempdir_path = 'temp'

        try:
            os.mkdir(tempdir_path)
        except:
            pass
        self.save_dir = tempdir_path
        self.max_attempt = 1000
        super(AgentDemoWrapper, self).__init__(env)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def generate_episode_gif(self, init_map):
        images = []
        done = False
        obs = self.env.manual_reset(init_map)
        im = room_to_rgb(obs)
        images.append(im)
        while not done:
            action, _ = self.alg.predict(obs, deterministic=True)
            obs, _, done, _ = self.env.step(action)
            im = room_to_rgb(obs)
            images.append(im)

        im_name = '{}/agent_episode.gif'.format(self.save_dir)
        imageio.mimsave(im_name, images, 'GIF', fps=2)