import gym
from stable_baselines3 import PPO
from envs import ALGEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import os
import os.path
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# env_li = [lambda: ALGEnv(dim_room=dim_room, num_boxes=num_boxes, train_mode=train_mode, alg_version=alg_version, agent_lb_path=None, agent_ub_path=None)]
# alg_env = DummyVecEnv(env_li)


def make_env(rank, seed=0):
    num_boxes = 1
    alg_version = 2
    dim_room = (7, 7)
    train_mode = 'mlp'
    agent_lb_path = '../demo_checkpoints/agent_v1b_3.zip'
    agent_ub_path = '../demo_checkpoints/agent_v1b_2.zip'

    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ALGEnv(dim_room=dim_room, num_boxes=num_boxes, train_mode=train_mode,
                     alg_version=alg_version, agent_lb_path=agent_lb_path, agent_ub_path=agent_ub_path)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    num_cpu = 9
    load_version = '1b_v2'
    save_version = '1b_v2'
    save_dir = '../models'
    load_dir = '../demo_checkpoints'
    timesteps_per_checkpoint = int(1e6)
    num_checkpoints = int(1e1)  # controlling performance level of agent

    try:
        os.mkdir(save_dir)
    except OSError as error:
        pass

    alg_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    print('created alg env')

    load_path = '{}/alg_v{}.zip'.format(load_dir, load_version)
    if os.path.exists(load_path):
        alg = PPO("MlpPolicy", alg_env, verbose=0)
        alg.set_parameters(load_path, exact_match=True)
        # alg = PPO.load(load_path, env=alg_env)
        print('loaded alg checkpoint' + load_path)
    else:
        alg = PPO("MlpPolicy", alg_env, verbose=0)
        print('created alg model')

    save_path = '{}/alg_v{}.zip'.format(save_dir, save_version)
    for _ in range(num_checkpoints):
        alg.learn(total_timesteps=timesteps_per_checkpoint)
        alg.save(save_path)
        print('saved alg checkpoint' + save_path)


if __name__ == '__main__':
    main()
