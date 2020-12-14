import gym
from stable_baselines3 import PPO
from envs import SokobanEnv, room_utils
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import os.path
from tqdm import tqdm
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

version_li = ['1b_0', '1b_1']
num_boxes = 1
dim_room = (7, 7)
max_steps = 50
num_tests = 1000
train_mode = 'mlp'
load_dir = '../demo_checkpoints'
num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
env_li = [lambda: SokobanEnv(dim_room=dim_room, max_steps=max_steps,
                             num_boxes=num_boxes, train_mode=train_mode, log_train_info=False)]
soko_env = DummyVecEnv(env_li)
print('created soko env')

# agent_li = []
for version in version_li:
    agent = PPO("MlpPolicy", soko_env, verbose=0)
    # agent = PPO.load('../models/soko_v'+version+'.zip', env=soko_env)
    # print('loaded', version, 'model')
    # agent_li.append(agent)

for _ in range(3):
    num_solved_li = []
    num_unique_solved_li = []
    for _ in range(len(version_li)):
        num_solved_li.append(0)
        num_unique_solved_li.append(0)
    for _ in tqdm(range(num_tests)):
        unique_solver_idx = -1
        success = False
        while not success:
            success = True
            try:
                fix_room = room_utils.generate_room(
                    dim=dim_room,
                    num_steps=num_gen_steps,
                    num_boxes=num_boxes,
                    second_player=False
                )
                _, state, _  = fix_room
            except:
                success = False
        for i in range(len(version_li)):
            version = version_li[i]
            load_path = '{}/agent_v{}.zip'.format(load_dir, version)
            agent.set_parameters(load_path, exact_match=True)
            # agent = agent_li[i]
            done = False
            obs = np.expand_dims(soko_env.env_method('manual_reset', state)[0], axis=0)
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, done, info = soko_env.step(action)

            # solved
            if info[0]["all_boxes_on_target"]:
                num_solved_li[i] += 1
                if unique_solver_idx == -1:
                    unique_solver_idx = i
                else:
                    unique_solver_idx = -1

        if unique_solver_idx != -1:
            num_unique_solved_li[unique_solver_idx] += 1

    for i in range(len(version_li)):
        print('{} solved {}, uniquely solved {}'.format(
            version_li[i], num_solved_li[i], num_unique_solved_li[i]))
