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


class ALGDemoWrapper(gym.Wrapper):
    def __init__(self, env, alg_path=None, alg_version=0, tempdir_path=None):
        self.alg = PPO('MlpPolicy', env, verbose=0)
        if alg_path is not None:
            load_path = alg_path + str(alg_version)
            self.alg.set_parameters(load_path, exact_match=True)

        if tempdir_path is None:
            tempdir_path = 'temp'

        try:
            os.mkdir(tempdir_path)
        except:
            pass
        self.save_dir = tempdir_path
        self.max_attempt = 1000
        self.version = alg_version
        super(ALGDemoWrapper, self).__init__(env)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def generate_episode_gif(self):
        attempt = 0
        while True:
            images = []
            done = False
            obs = self.env.reset()
            im = room_to_rgb(obs)
            images.append(im)
            while not done:
                action, _ = self.alg.predict(obs, deterministic=True)
                obs, _, done, info = self.env.step(action)
                im = room_to_rgb(obs)
                images.append(im)

            if info['train_result'] == 0:
                im_name = '{}/alg_episode_v{}.gif'.format(
                    self.save_dir, self.version)
                imageio.mimsave(im_name, images, 'GIF', fps=2)
                return True, obs

            attempt += 1
            if attempt >= self.max_attempt:
                print('Time out. Wasn\'t able to generate good map.')
                return False, None
