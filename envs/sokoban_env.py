import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from .room_utils import generate_room
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
import numpy as np
import copy
import time

class SokobanEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw', 'np_array']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=4,
                 num_gen_steps=None,
                 reset=True,
                 log_train_info=True,
                 train_mode='cnn'):

        self.train_mode = train_mode
        self.manual_mode = False
        self.log_train_info = log_train_info

        # Training hyperperams
        self.map_max_usage = 50

        # Log info
        self.start_time = time.time()
        self.episode_reward = 0
        self.episode_moves = 0
        self.episode_pushes = 0
        self.total_reward_per_log_interval = 0
        self.total_steps_per_log_interval = 0
        self.total_moves_per_log_interval = 0
        self.total_pushes_per_log_interval = 0
        self.total_solved_map_log_interval = 0
        self.log_interval = 1000
        self.reseted = False
        self.train_counter = 0

        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0

        # Penalties and Rewards
        self.penalty_for_step = -0.5
        self.penalty_box_off_target = -10
        self.penalty_no_action = -10

        self.reward_box_on_target = 10
        self.reward_finished = 100
        self.reward_pushed_box = 1
        self.reward_last = 0

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        if train_mode == 'cnn':
            self.scale = 6
            screen_height, screen_width = (dim_room[0] * self.scale, dim_room[1] * self.scale)
            self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        else:
            self.observation_space = Box(low=0, high=6, shape=(dim_room[0], dim_room[1]), dtype=np.uint8)
        self.fix_room = None
        self.num_map_used = 0
        self.update_map = True

        if reset:
            # Initialize Room
            _ = self.reset()

    def reset(self):
        if not self.manual_mode:
            if self.num_map_used >= self.map_max_usage:
                self.update_map = True

            if self.update_map:
                try:
                    self.fix_room = generate_room(
                        dim=self.dim_room,
                        num_steps=self.num_gen_steps,
                        num_boxes=self.num_boxes,
                        second_player=False
                    )
                    # print('generated new room')
                    self.num_map_used = 0
                    self.update_map = False
                except (RuntimeError, RuntimeWarning) as e:
                    print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                    print("[SOKOBAN] Retry . . .")
                    return self.reset()
            self.room_fixed, self.room_state, _ = copy.deepcopy(self.fix_room)

        else:
            self.room_fixed, self.room_state = self.deconstruct_map(
                self.alg_map)

        self.episode_reward = 0
        self.env_steps = 0
        self.episode_moves = 0
        self.episode_pushes = 0
        self.reseted = True
        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.reward_last = 0
        self.boxes_on_target = 0

        if self.train_mode == 'cnn':
            starting_observation = self.render('tiny_rgb_array', scale=self.scale)
        else:
            starting_observation = self.render('np_array')
        return starting_observation

    def step(self, action):
        '''
        Tile type:
            0: Wall
            1: Floor
            2: Box Target
            3: Box On Target
            4: Box
            5: Player
            6: Player On Target
        '''
        assert action in ACTION_LOOKUP

        self.env_steps += 1
        self.new_box_position = None
        self.old_box_position = None
        self.moved_box = False

        self.moved_player, self.moved_box = self._push(action)
        if self.moved_player:
            self.episode_moves += 1
        if self.moved_box:
            self.episode_pushes += 1

        self._calc_reward()
        done = self._check_if_done()

        self.episode_reward += self.reward_last

        # Convert the observation to RGB frame
        if self.train_mode == 'cnn':
            observation = self.render(mode='tiny_rgb_array', scale=self.scale)
        else:
            observation = self.render(mode='np_array')

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": self.moved_player,
            "action.moved_box": self.moved_box,
        }

        if self.reseted:
            self.reseted = False
            self.train_counter += 1

        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

            self.total_reward_per_log_interval += self.episode_reward
            self.total_steps_per_log_interval += self.env_steps
            self.total_moves_per_log_interval += self.episode_moves
            self.total_pushes_per_log_interval += self.episode_pushes
            self.num_map_used += 1

            # if solved map, can play with new map
            if info["all_boxes_on_target"]:
                self.total_solved_map_log_interval += 1
                self.update_map = True

            if self.train_counter % self.log_interval == 0 and self.log_train_info:
                end_time = time.time()
                duration = end_time - self.start_time
                avg_reward = self.total_reward_per_log_interval / self.log_interval
                avg_steps = self.total_steps_per_log_interval / self.log_interval
                avg_moves = self.total_moves_per_log_interval / self.log_interval
                avg_pushes = self.total_pushes_per_log_interval / self.log_interval
                print('[{}] Summary'.format(self.train_counter))
                print('Duration: %.2fs' % (duration))
                print('Average reward current log interval:', avg_reward)
                print('Average steps current log interval:', avg_steps)
                print('Average moves current log interval:', avg_moves)
                print('Average pushses current log interval:', avg_pushes)

                print('Number of solved maps:', self.total_solved_map_log_interval)
                print('*********************************************')

                self.total_reward_per_log_interval = 0
                self.total_steps_per_log_interval = 0
                self.total_moves_per_log_interval = 0
                self.total_pushes_per_log_interval = 0
                self.total_solved_map_log_interval = 0
                self.start_time = time.time()

        return observation, self.reward_last, done, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[action % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if not (new_box_position[0] in range(self.dim_room[0])
                and new_box_position[1] in range(self.dim_room[0])):
            return False, False

        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]

        if can_push_box:
            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[action % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if new_position[0] in range(self.dim_room[0]) and new_position[1] in range(self.dim_room[1]):
            if self.room_state[new_position[0], new_position[1]] in [1, 2]:
                self.player_position = new_position
                self.room_state[(new_position[0], new_position[1])] = 5
                self.room_state[current_position[0], current_position[1]] = \
                    self.room_fixed[current_position[0], current_position[1]]

            return True

        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        if self.moved_box:
            self.reward_last += self.reward_pushed_box

        if not self.moved_player:
            self.reward_last += self.penalty_no_action

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target

        game_won = self._check_if_all_boxes_on_target()
        if game_won:
            self.reward_last += self.reward_finished

        self.boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return self.env_steps >= self.max_steps

    def destroy_map(self):
        self.update_map = True

    def manual_reset(self, alg_map):
        self.alg_map = copy.deepcopy(alg_map)
        self.manual_mode = True
        self.update_map = False
        return self.reset()

    def deconstruct_map(self, obs_map):
        state_map = copy.deepcopy(obs_map)
        fix_map = copy.deepcopy(obs_map)
        state_map[state_map == 6] = 5
        fix_map[(fix_map == 3) | (fix_map == 6)] = 2
        fix_map[(fix_map == 4) | (fix_map == 5)] = 1
        # fix_map[fix_map == 6] = 2
        return fix_map, state_map

    def assemble_map(self, state_map, fix_map):
        obs_map = copy.deepcopy(state_map)
        obs_map[(obs_map == 5) & (fix_map == 2)] = 6
        return obs_map

    def render(self, mode='human', close=None, scale=16):
        assert mode in RENDERING_MODES

        if 'rgb_array' in mode:
            img = self.get_image(mode, scale)
            return img

        elif 'np_array' in mode:
            # clear_output(wait=True)
            obs_map = self.assemble_map(self.room_state, self.room_fixed)
            # print(obs_map)
            return obs_map

        elif 'human' in mode:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            img = self.get_image(mode, scale)
            self.viewer.imshow(img)
            return self.viewer.isopen

        elif 'raw' in mode:
            arr_walls = (self.room_fixed == 0).view(np.int8)
            arr_goals = (self.room_fixed == 2).view(np.int8)
            arr_boxes = ((self.room_state == 4) + (self.room_state == 3)).view(np.int8)
            arr_player = (self.room_state == 5).view(np.int8)

            return arr_walls, arr_goals, arr_boxes, arr_player

        else:
            super(SokobanEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode, scale=1):
        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)
        else:
            img = room_to_rgb(self.room_state, self.room_fixed)
        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP

TRAIN_MODES = ['cnn', 'mlp']

ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    # 5: 'move up',
    # 6: 'move down',
    # 7: 'move left',
    # 8: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw', 'np_array']
