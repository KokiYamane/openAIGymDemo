# import random
# import pyglet
import gym
from PIL import Image
import time
import torch
# from pyglet.window import key
# from stable_baselines.gail import generate_expert_traj
import sys
sys.path.append('.')
sys.path.append('..')
from scripts.DQN import CNNQNetwork
from scripts.replay_buffer import PrioritizedReplayBuffer


def wrap_monitor(env):
    """ Gymの環境をmp4に保存するために，環境をラップする関数． """
    return gym.wrappers.Monitor(env, '/tmp/monitor', video_callable=lambda x: True, force=True)


def save_gif(frames, path, duration=20):
    frames[0].save(path,
                   save_all=True,
                   append_images=frames[1:],
                   duration=duration,
                   loop=0)


def main():
    # envids = [spec.id for spec in gym.envs.registry.all()]
    # print(envids)
    # env = gym.make('Acrobot-v1')
    # env = wrap_monitor(env)
    # env = gym.make('CartPole-v0')
    # env.reset()

    modelpath = 'model_entire_best.pt'
    model = torch.load(modelpath)
    model.eval()


    # CartPoleをランダムに動かす
    frames = []
    env = gym.make('CartPole-v0')
    observation = env.reset()
    observation = torch.from_numpy(observation).float()
    for step in range(0, 200):
        image = env.render(mode='rgb_array')
        frames.append(Image.fromarray(image))
        # action = env.action_space.sample()
        action = model.act(observation)
        observation, reward, done, info = env.step(action)
        observation = torch.from_numpy(observation).float()

    save_gif(frames, 'result.gif', duration=20)


if __name__ == '__main__':
    main()
