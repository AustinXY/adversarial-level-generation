import gym
from stable_baselines3 import PPO
from envs import SokobanEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import os
import os.path

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def make_env(rank, seed=0):
    num_boxes = 1
    dim_room = (7, 7)
    train_mode = 'mlp'
    max_steps = 20

    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = SokobanEnv(dim_room=dim_room, max_steps=max_steps,
                         num_boxes=num_boxes, train_mode=train_mode)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def main():
    num_cpu = 1
    load_version = ''
    save_version = '1b_0'
    load_dir = '../demo_checkpoints'
    save_dir = '../models'
    timesteps_per_checkpoint = int(1e6)
    num_checkpoints = int(1e1)  # controlling performance level of agent

    try:
        os.mkdir(save_dir)
    except OSError as error:
        pass

    soko_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    print('created soko env')

    train_policy = 'MlpPolicy'
    load_path = '{}/agent_v{}.zip'.format(load_dir, load_version)
    if os.path.exists(load_path):
        agent = PPO(train_policy, soko_env, verbose=0)
        agent.set_parameters(load_path, exact_match=True)
        print('loaded agent checkpoint' + load_path)
    else:
        agent = PPO(train_policy, soko_env, verbose=0)
        print('created agent model')

    save_path = '{}/agent_v{}.zip'.format(save_dir, save_version)
    for _ in range(num_checkpoints):
        agent.learn(total_timesteps=timesteps_per_checkpoint)
        agent.save(save_path)
        print('saved soko checkpoint' + save_path)


if __name__ == '__main__':
    main()
