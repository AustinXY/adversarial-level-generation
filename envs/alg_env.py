import gym
from gym.utils import seeding
from gym.spaces import MultiDiscrete
from gym.spaces import Box
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
import numpy as np
import copy
import sys
from copy import deepcopy
from .sokoban_env import SokobanEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import time


class ALGEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'np_array']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 num_boxes=4,
                 reset=True,
                 log_interval=1000,
                 alg_version=0,
                 train_mode='cnn',
                 agent_lb_path=None,
                 agent_ub_path=None,
                 init_probs=[0.5, 0.5, 0.5]):

        assert train_mode in TRAIN_MODES
        self.train_mode = train_mode
        if log_interval > 0:
            self.log_train_info = True
        else:
            self.log_train_info = False

        # 0: basic playable map
        # 1: playble map
        # 2: hardness adjustable map
        self.alg_version = alg_version
        if alg_version == 0:
            pass
        else:
            env_li = [lambda: SokobanEnv(dim_room=dim_room, max_steps=50, num_boxes=num_boxes, train_mode=train_mode, log_train_info=False)]
            self.soko_env = DummyVecEnv(env_li)
            self.agent_ub = PPO.load(agent_ub_path, env=self.soko_env)
            print('loaded', agent_ub_path, 'as ub')
            if alg_version == 2:
                self.agent_lb = PPO.load(agent_lb_path, env=self.soko_env)
                print('loaded', agent_lb_path, 'as lb')

        # General Configuration
        self.dim_room = dim_room
        self.num_boxes = num_boxes
        self.num_players = 1

        # Training hyperperams
        self.max_prefer_subs = dim_room[0] * dim_room[1] // 2
        self.place_target_prob = init_probs[0]
        self.place_box_prob = init_probs[1]
        self.place_player_prob = init_probs[2]

        # Log info
        self.start_time = time.time()
        self.train_result_summary = {-1: 0, 0: 0, 1: 0, 2: 0}
        self.fail_type_summary = {-1: 0, 0: 0, 1: 0, 2: 0}
        # self.sample_map = False
        self.episode_reward = 0
        self.total_reward_per_log_interval = 0
        self.total_steps_per_log_interval = 0
        self.total_subs_per_log_interval = 0
        self.log_interval = log_interval
        self.reseted = False
        self.train_counter = 0

        # Env properties
        self.map = None

        # Penalties and Rewards
        self.penalty_sub_wrong_tile = -5
        self.penalty_exc_btp_tiles = -10
        self.penalty_bad_map_design = -50
        self.penalty_generation_fail = -50
        self.penalty_exc_subs = -10

        self.reward_neighbor_valid_tiles = 2
        self.reward_place_btp_tiles = 5
        self.reward_basic_playable = 40

        if alg_version == 1:
            # too hard or unsolvable
            self.penalty_agent_ub_thou = -30
            self.reward_agent_ub_solvable = 50
        elif alg_version == 2:
            self.penalty_agent_lb_solvable = -30
            self.penalty_agent_ub_thou = -30
            self.reward_agent_ub_solvable = 10
            self.reward_agent_lb_thou = 50

        # Generation Track
        self.placed_player = 0
        self.placed_boxes = 0
        self.placed_target = 0
        self.env_steps = 0

        # Env Settings
        self.viewer = None
        self.max_steps = dim_room[0] * dim_room[1]
        self.action_space = MultiDiscrete([dim_room[0], dim_room[1], 5])

        if train_mode == 'cnn':
            self.scale = 6
            screen_height, screen_width = (dim_room[0] * self.scale, dim_room[1] * self.scale)
            self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        else:
            self.observation_space = Box(low=0, high=6, shape=(dim_room[0], dim_room[1]), dtype=np.uint8)

        if reset:
            # Initialize Room
            _ = self.reset()

    def random_init_map(self):
        room = np.zeros((self.dim_room[0], self.dim_room[1]), dtype=np.uint8)
        for _ in range(self.num_boxes):
            if np.random.rand(1) < self.place_target_prob:
                x, y = np.random.randint(1, self.dim_room[0]-1, size=2)
                room[x, y] = 2
            if np.random.rand(1) < self.place_box_prob:
                x, y = np.random.randint(1, self.dim_room[0]-1, size=2)
                room[x, y] = 4

        for _ in range(self.num_players):
            if np.random.rand(1) < self.place_player_prob:
                x, y = np.random.randint(1, self.dim_room[0]-1, size=2)
                room[x, y] = 5

        self.placed_target += np.count_nonzero(room==2)
        self.placed_boxes += np.count_nonzero(room==4)
        self.placed_player += np.count_nonzero(room==5)

        return room

    def reset(self):
        self.placed_player = 0
        self.placed_boxes = 0
        self.placed_target = 0
        self.map = self.random_init_map()
        self.env_steps = 0
        self.episode_subs = 0
        self.episode_reward = 0
        self.reseted = True

        if self.train_mode == 'cnn':
            starting_observation = self.render('tiny_rgb_array', scale=self.scale)
        else:
            starting_observation = self.render('np_array')
        return starting_observation

    def soko_agent_test(self):
        reward = 0

        # v1
        if self.alg_version == 1:
            done = False
            obs = self.soko_env.env_method('manual_reset', self.map)
            while not done:
                action, _ = self.agent_ub.predict(obs, deterministic=True)
                obs, _, done, info = self.soko_env.step(action)

            # agent_ub solvable
            if info[0]["all_boxes_on_target"]:
                reward += self.reward_agent_ub_solvable
                train_result = 0  # good map
            else:
                reward += self.penalty_agent_ub_thou
                train_result = 2  # thou map

        # v2
        else:
            done = False
            obs = self.soko_env.env_method('manual_reset', self.map)
            while not done:
                action, _ = self.agent_ub.predict(obs, deterministic=True)
                obs, _, done, info = self.soko_env.step(action)

            # agent_ub thou
            if not info[0]["all_boxes_on_target"]:
                reward += self.penalty_agent_ub_thou
                train_result = 2  # thou

            # agent_ub solvable
            else:
                reward += self.reward_agent_ub_solvable
                done = False
                obs = self.soko_env.env_method('manual_reset', self.map)
                while not done:
                    action, _ = self.agent_lb.predict(obs, deterministic=True)
                    obs, _, done, info = self.soko_env.step(action)

                # agent_lb solvable
                if info[0]["all_boxes_on_target"]:
                    reward += self.penalty_agent_lb_solvable
                    train_result = 1  # too easy
                else:
                    reward += self.reward_agent_lb_thou
                    train_result = 0  # good map

        return reward, train_result

    def good_map_reward(self, attempts):
        return (self.reward_make_agent_work_max + self.reward_make_agent_work_dec / 2) * attempts

    def step(self, action):
        '''
        Tile type:
            0: Wall
            1: Floor
            2: Target
            3: Box On Target
            4: Box
            5: Player
            6: Player On Target
        act:
            0: Finish Generation
            1: Floor
            2: Box Target
            3: Box
            4: Player
        '''
        x, y, act = action
        reward = 0
        done = False
        self.env_steps += 1
        # not finish generation
        if act != 0:
            if self.map[x][y] != 0:
                reward += self.penalty_sub_wrong_tile

            # is wall tile, can substitute
            else:
                for (_x, _y) in [(x-1, y), (x, y-1), (x, y+1), (x+1, y)]:
                    if _x in range(self.dim_room[0]) and _y in range(self.dim_room[1]):
                        if self.map[_x, _y] != 0:
                            reward += self.reward_neighbor_valid_tiles

                if act == 1:
                    self.map[x][y] = 1
                    self.episode_subs += 1
                    if self.episode_subs >= self.max_prefer_subs:
                        reward += self.penalty_exc_subs
                        # print(self.episode_subs)

                # place box target
                elif act == 2:
                    if self.placed_target >= self.num_boxes:
                        reward += self.penalty_exc_btp_tiles
                    else:
                        self.placed_target += 1
                        self.map[x][y] = 2
                        self.episode_subs += 1
                        reward += self.reward_place_btp_tiles
                        if self.episode_subs >= self.max_prefer_subs:
                            reward += self.penalty_exc_subs
                            # print(self.episode_subs)

                # place box
                elif act == 3:
                    if self.placed_boxes >= self.num_boxes:
                        reward += self.penalty_exc_btp_tiles
                    else:
                        self.placed_boxes += 1
                        self.map[x][y] = 4
                        self.episode_subs += 1
                        reward += self.reward_place_btp_tiles
                        if self.episode_subs >= self.max_prefer_subs:
                            reward += self.penalty_exc_subs
                            # print(self.episode_subs)

                # place player
                elif act == 4:
                    if self.placed_player >= self.num_players:
                        reward += self.penalty_exc_btp_tiles
                    else:
                        self.placed_player += 1
                        self.map[x][y] = 5
                        self.episode_subs += 1
                        reward += self.reward_place_btp_tiles
                        if self.episode_subs >= self.max_prefer_subs:
                            reward += self.penalty_exc_subs
                            # print(self.episode_subs)

            if self.is_maxsteps():
                done = True
                _train_result = -1
                _fail_type = 2

        # finished generation
        else:
            done = True
            _train_result = -1  # not used for training
            _fail_type = -1  # not failed
            if (self.placed_player != self.num_players or
                self.placed_boxes != self.num_boxes or
                self.placed_target != self.num_boxes):
                reward += self.penalty_generation_fail
                _fail_type = 0  # wrong number btp tiles
            else:
                if not self.basic_playable(self.map):
                    reward += self.penalty_bad_map_design
                    _fail_type = 1  # not basic playable
                else:
                    reward += self.reward_basic_playable
                    if self.alg_version == 0:
                        _train_result = 0
                    else:
                        _train_reward, _train_result = self.soko_agent_test()
                        reward += _train_reward

        self.episode_reward += reward

        # Convert the observation to RGB frame
        if self.train_mode == 'cnn':
            observation = self.render(mode='tiny_rgb_array', scale=self.scale)
        else:
            observation = self.render(mode='np_array')

        info = {
            "coordinate": (x, y),
            "action": act,
            "curr_steps": self.env_steps,
        }

        if self.reseted:
            self.reseted = False
            self.train_counter += 1

        if done:
            info["total_steps"] = self.env_steps
            info["train_result"] = _train_result
            info['fail_type'] = _fail_type

            self.train_result_summary[_train_result] += 1
            self.fail_type_summary[_fail_type] += 1
            self.total_reward_per_log_interval += self.episode_reward
            self.total_steps_per_log_interval += self.env_steps
            self.total_subs_per_log_interval += self.episode_subs

            # if _fail_type == -1 and self.sample_map:
            #     print('Sample map:')
            #     print(self.map)
            #     print('*********************************************')
                # self.sample_map = False

            if self.log_train_info and self.train_counter % self.log_interval == 0:
                end_time = time.time()
                duration = end_time - self.start_time
                avg_reward = self.total_reward_per_log_interval / self.log_interval
                avg_steps = self.total_steps_per_log_interval / self.log_interval
                avg_subs = self.total_subs_per_log_interval / self.log_interval
                print('[{}] Summary'.format(self.train_counter))
                print('Duration: %.2fs' % (duration))
                print('Average reward current log interval: ', avg_reward)
                print('Average steps current log interval: ', avg_steps)
                print('Average subs current log interval: ', avg_subs)

                print('Good Map                  :', self.train_result_summary[0])
                if self.alg_version == 2:
                    print('Too easy map              :', self.train_result_summary[1])
                if self.alg_version != 0:
                    print('Too hard or unsolvable map:', self.train_result_summary[2])
                print('Not for training map      :', self.train_result_summary[-1])

                print('Generated wrong number of btp tiles:', self.fail_type_summary[0])
                print('Generated not basic playable map   :', self.fail_type_summary[1])
                print('Unable to finish by max step       :', self.fail_type_summary[2])
                print('Succeeded generate map for training:', self.fail_type_summary[-1])
                print('*********************************************')

                self.total_reward_per_log_interval = 0
                self.total_steps_per_log_interval = 0
                self.total_subs_per_log_interval = 0
                self.train_result_summary = {-1: 0, 0: 0, 1: 0, 2: 0}
                self.fail_type_summary = {-1: 0, 0: 0, 1: 0, 2: 0}
                self.sample_map = True
                self.start_time = time.time()

        return observation, reward, done, info

    def render(self, mode=None, close=None, scale=16):
        if mode is None:
            if self.train_mode == 'cnn':
                mode = 'human'
            else:
                mode = 'np_array'
        assert mode in RENDERING_MODES

        if 'rgb_array' in mode:
            img = self.get_image(mode, scale)
            return img

        elif 'np_array' in mode:
            return self.map

        elif 'human' in mode:
            from gym.envs.classic_control import rendering
            if self.viewer is None or not self.viewer.isopen:
                self.viewer = rendering.SimpleImageViewer()
            img = self.get_image(mode, scale)
            self.viewer.imshow(img)
            return self.viewer.isopen

        else:
            super(ALGEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode, scale=1):
        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.map, scale=scale)
        else:
            img = room_to_rgb(self.map)
        return img

    def basic_playable(self, room):
        # # player can reach all boxes and all targets
        # for player_coord in np.argwhere(room==5):
        #     des = np.concatenate((np.argwhere(room==2), np.argwhere(room==4)), axis=0)
        #     if not self.contaminate(room, player_coord, des):
        #         return False

        # player can reach all none wall tiles

        if not self.contaminate_room(room):
            return False

        # no three walls around any box
        if self.box_stuck(room):
            return False
        return True

    def box_stuck(self, room):
        room = deepcopy(room)
        room = np.pad(room, 1, 'constant', constant_values=0)
        for (x, y) in np.argwhere(room==4):
            if (room[x-1, y] == room[x, y-1] == 0
                or room[x-1, y] == room[x, y+1] == 0
                or room[x+1, y] == room[x, y-1] == 0
                or room[x+1, y] == room[x, y-1] == 0):
                return True
            num_wall = 0
            for (_x, _y) in [(x-1, y), (x, y-1), (x, y+1), (x+1, y)]:
                if room[_x, _y] == 0:
                    num_wall += 1
            if num_wall >= 3:
                return True
        return False

    # player can reach any none wall tile within room
    def contaminate_room(self, room):
        room = deepcopy(room)
        room = np.pad(room, 1, 'constant', constant_values=0)
        (x, y) = np.argwhere(room==5)[0]
        room[room != 0] = 1
        room[x, y] = 5
        fixpoint = False
        while not fixpoint:
            fixpoint = True
            for (x, y) in np.argwhere(room==5):
                for (_x, _y) in [(x-1, y), (x, y-1), (x, y+1), (x+1, y)]:
                    if room[_x, _y] not in [0, 5]:
                        room[_x, _y] = 5
                        fixpoint = False
        for i in [1, 2, 4]:
            if i in room:
                return False
        return True

    def contaminate(self, room, src, des):
        room = deepcopy(room)
        (x, y) = src
        src_tile = room[x, y]
        room[room != 0] = 1
        room[x, y] = src_tile
        fixpoint = False
        while not fixpoint:
            fixpoint = True
            for (x, y) in np.argwhere(room==src_tile):
                for (_x, _y) in [(x-1, y), (x, y-1), (x, y+1), (x+1, y)]:
                    if _x in range(self.dim_room[0]) and _y in range(self.dim_room[1]):
                        if room[_x, _y] not in [0, src_tile]:
                            room[_x, _y] = src_tile
                            fixpoint = False
        reachable = True
        for (x, y) in des:
            if room[x, y] != src_tile:
                reachable = False
                break
        return reachable

    def is_maxsteps(self):
        return self.env_steps >= self.max_steps

    def deconstruct_map(self, obs_map):
        state_map = copy.deepcopy(obs_map)
        fix_map = copy.deepcopy(obs_map)
        state_map[state_map == 6] = 5
        fix_map[(fix_map == 3) | (fix_map == 6)] = 2
        fix_map[(fix_map == 4) | (fix_map == 5)] = 1
        return fix_map, state_map

    def assemble_map(self, state_map, fix_map):
        obs_map = copy.deepcopy(state_map)
        obs_map[(obs_map == 5) & (fix_map == 2)] = 6
        return obs_map

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

TRAIN_MODES = ['cnn', 'mlp']
RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw', 'np_array']
